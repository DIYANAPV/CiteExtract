"""Write CheckCitation verdicts back onto the user's uploaded PDF.

Produces a second PDF file where every detected in-text citation marker
(e.g. ``[12]`` or ``(Smith, 2020)``) is visually highlighted and carries a
click-to-expand sticky note with the verdict label + explanation.

The original PDF bytes are never modified — we write a fresh copy.

Only meaningful for PDF inputs. LaTeX / BibTeX / text uploads have no
PDF to annotate, so the higher layer (``app.py``) gates the call.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    # Type-only import — fitz is loaded lazily inside ``annotate_pdf`` and
    # ``_draw_highlight_with_note``. Importing here lets type checkers
    # resolve ``fitz.Rect`` annotations without paying the import cost
    # (or failing the import) at module load time.
    import fitz

log = logging.getLogger(__name__)


# RGB in [0, 1]. PDF highlights tint underlying glyphs, so these are a hue
# brighter than the deep CSS verdict colors in app.py — but the same family,
# so a reader recognises a "red FABRICATED" highlight from the UI's red
# verdict pill at a glance.
#
# UI mapping (hex from `:root` design tokens in app.py):
#   FABRICATED          ↔ --verdict-err          #991B1B
#   UNVERIFIABLE        ↔ --verdict-info         #525252
#   VALID               ↔ --verdict-ok           #15803D
#   CONTRADICTS_CLAIM   ↔ --verdict-warn         #B45309
#
# The actual highlight values are intentionally desaturated so the
# underlying citation marker stays legible through the highlight overlay.
VERDICT_COLORS = {
    "FABRICATED":   (0.95, 0.40, 0.40),   # soft red
    "UNVERIFIABLE": (0.65, 0.65, 0.68),   # neutral gray
    "VALID":        (0.40, 0.78, 0.55),   # soft green
    # Claim-dimension overlay: VALID metadata + CONTRADICTS claim → amber.
    "CONTRADICTS_CLAIM": (0.96, 0.65, 0.30),
}


def color_for_verdict(metadata_verdict: str, claim_verdict: str | None = None) -> tuple[float, float, float]:
    """Pick the highlight color for a citation marker in the annotated PDF.

    Metadata FABRICATED / UNVERIFIABLE always wins (the cited paper itself
    is the issue). Otherwise, if the claim agent contradicted the citing
    sentence, use the amber CONTRADICTS_CLAIM overlay so reviewers see
    the per-claim finding without expanding the sticky-note.
    """
    if metadata_verdict in ("FABRICATED", "UNVERIFIABLE"):
        return VERDICT_COLORS[metadata_verdict]
    if claim_verdict == "CONTRADICTS":
        return VERDICT_COLORS["CONTRADICTS_CLAIM"]
    return VERDICT_COLORS.get(metadata_verdict or "VALID", VERDICT_COLORS["VALID"])


@dataclass
class AnnotationStats:
    """Return payload from annotate_pdf: how the run went.

    Each ``skipped_*`` counter names a distinct failure mode so callers can
    distinguish "GROBID handed us a marker the PDF text-layer doesn't
    contain" (``skipped_not_found``) from "two citations resolved to the
    same on-page rect and only the first got the icon"
    (``skipped_collision``) — they have different fixes.
    """
    annotated: int = 0                 # successfully highlighted + noted
    skipped_no_marker: int = 0         # citation had no marker text
    skipped_not_found: int = 0         # marker text not locatable on any page
    skipped_collision: int = 0         # marker found, but every rect already used by an earlier citation
    skipped_no_verdict: int = 0        # cit.ref_id has no entry in PaperReport.verdicts
    pages: int = 0


def annotate_pdf(
    source_pdf_path: str,
    paper_report,
    parsed,
    output_path: str,
    comp_report=None,
) -> AnnotationStats:
    """Write ``output_path`` = source PDF + highlights + sticky notes.

    Args:
        source_pdf_path: Original uploaded PDF.
        paper_report:    PaperReport with verdicts.
        parsed:          ParsedPaper (we use ``citations`` for markers + context).
        output_path:     Destination path for the annotated PDF.
        comp_report:     Optional ComprehensionReport — when present we attach
                         the per-citation top retrieved passages to each note,
                         so a reader can audit the agent's claim verdict
                         against the actual evidence without leaving the PDF.

    Returns:
        AnnotationStats describing what landed vs. what couldn't be located.

    Raises:
        ImportError:  if PyMuPDF is missing (caller should translate to a
                      friendly UI message).
        RuntimeError: for corrupt / unparseable PDFs.
    """
    import fitz  # PyMuPDF

    src = Path(source_pdf_path)
    if not src.exists():
        raise RuntimeError(f"Source PDF not found: {source_pdf_path}")

    verdicts_by_ref = {v.ref_id: v for v in paper_report.verdicts}
    refs_by_id = {r.ref_id: r for r in (parsed.references if parsed else [])}
    citations = list(parsed.citations) if parsed else []

    # Index comprehension results by (ref_id, citing_sentence) so each
    # citation marker gets exactly the passages that fed its claim verdict
    # — not a generic per-paper mash-up.
    comp_by_key: dict[tuple[str, str], object] = {}
    if comp_report is not None:
        for r in comp_report.results:
            comp_by_key[(r.ref_id, r.citing_sentence)] = r

    try:
        doc = fitz.open(str(src))
    except Exception as e:
        raise RuntimeError(f"Could not open PDF: {e}") from e

    stats = AnnotationStats(pages=doc.page_count)

    # Track already-annotated marker positions so a repeated marker like "[12]"
    # that appears on pages 3 and 7 annotates both occurrences, not the same one
    # twice.
    used_positions: set[tuple[int, int, int]] = set()

    try:
        for cit in citations:
            verdict = verdicts_by_ref.get(cit.ref_id)
            if verdict is None:
                # Orphan citation — typically a GROBID bibr target that didn't
                # resolve to a kept reference, or a ref_id that got remapped
                # by reference deduplication without the citation being
                # updated. Count it so the UI can warn instead of silently
                # losing this marker on the annotated PDF.
                stats.skipped_no_verdict += 1
                log.debug(
                    f"Skipping citation with no verdict: ref_id={cit.ref_id!r} "
                    f"marker={cit.marker!r}"
                )
                continue
            color = color_for_verdict(
                verdict.verdict,
                getattr(verdict, "claim_verdict", None),
            )

            marker = (cit.marker or "").strip()
            if not marker:
                stats.skipped_no_marker += 1
                continue

            target, reason = _find_citation_target(doc, cit, marker, used_positions)
            if target is None:
                if reason == "all_collided":
                    stats.skipped_collision += 1
                else:
                    stats.skipped_not_found += 1
                continue

            page_idx, rect = target
            ref = refs_by_id.get(cit.ref_id)
            comp = comp_by_key.get((cit.ref_id, cit.citing_sentence or ""))
            _draw_highlight_with_note(
                doc[page_idx], rect, verdict, ref, comp, cit, color,
            )
            used_positions.add((page_idx, int(rect.x0), int(rect.y0)))
            stats.annotated += 1

        # garbage=4 + deflate keeps the resulting file size reasonable
        doc.save(output_path, garbage=4, deflate=True)
    finally:
        doc.close()

    log.info(
        f"PDF annotated: {stats.annotated} markers highlighted, "
        f"{stats.skipped_not_found} not locatable, "
        f"{stats.skipped_collision} location collisions, "
        f"{stats.skipped_no_verdict} without verdicts, "
        f"{stats.skipped_no_marker} without marker text"
    )
    return stats


def _find_citation_target(
    doc, cit, marker: str, used_positions: set[tuple[int, int, int]],
) -> tuple[Optional[tuple[int, fitz.Rect]], str]:
    """Locate the best on-page rect to annotate this citation.

    Returns ``(target, reason)`` where:
      - ``target`` is ``(page_index, rect)`` on success, ``None`` on failure.
      - ``reason`` is one of:
          * ``"found"``        — caller should annotate ``target``.
          * ``"all_collided"`` — the marker text exists in the PDF but every
            occurrence was already claimed by an earlier citation in this
            run. Signals a multi-target marker that needs the Phase 2 group
            merge, not a search miss.
          * ``"not_in_text"``  — the marker string never matched any page.
            Typically means the GROBID bibr text and the rendered glyphs
            disagree (line wrap, comma variant, partial wrap).

    Strategy (in order, each skipping positions already annotated):
      1. Locate the citing sentence on some page; pick the marker rect closest
         to it on that page. This handles repeated markers correctly.
      2. Fall back to the first not-yet-used marker occurrence anywhere in
         the document.
    """
    sentence_prefix = _safe_sentence_prefix(cit.citing_sentence)
    # Tracks "did search_for(marker) ever return >0 rects on any page". Lets
    # the caller distinguish the two failure modes when we return None.
    any_marker_match = False

    # Strategy 1: sentence-first — find the page holding this citing sentence
    if sentence_prefix:
        for page_idx, page in enumerate(doc):
            sentence_rects = page.search_for(sentence_prefix)
            if not sentence_rects:
                continue
            marker_rects = page.search_for(marker)
            if not marker_rects:
                continue
            any_marker_match = True
            s0 = sentence_rects[0]
            for rect in sorted(
                marker_rects,
                key=lambda r: abs(r.y0 - s0.y0) + abs(r.x0 - s0.x0),
            ):
                key = (page_idx, int(rect.x0), int(rect.y0))
                if key not in used_positions:
                    return (page_idx, rect), "found"

    # Strategy 2: first unused occurrence anywhere in the document
    for page_idx, page in enumerate(doc):
        marker_rects = page.search_for(marker)
        if marker_rects:
            any_marker_match = True
        for rect in marker_rects:
            key = (page_idx, int(rect.x0), int(rect.y0))
            if key not in used_positions:
                return (page_idx, rect), "found"

    return None, ("all_collided" if any_marker_match else "not_in_text")


def _safe_sentence_prefix(sentence: Optional[str], length: int = 40) -> str:
    """First ~40 chars of the citing sentence, trimmed so a PDF search won't
    choke on line breaks / trailing whitespace. Empty for no-context citations.
    """
    if not sentence:
        return ""
    text = " ".join(sentence.split())  # collapse whitespace / newlines
    return text[:length].strip()


def _draw_highlight_with_note(page, rect, verdict, ref, comp, cit, color) -> None:
    """Color-highlight the citation marker + attach a sticky-note nearby.

    The note carries four sections (in fixed order):
      1. Title of the cited paper.
      2. Claim agent's opinion — the per-citation explanation written by the
         LLM when it inspected the citing sentence against retrieved passages.
         When no per-sentence verdict exists (quick mode, or ref didn't reach
         the claim agent), falls back to the metadata explanation.
      3. Abstract of the cited paper.
      4. Top retrieved passages (up to 3) the agent saw, with section names.

    Verdict labels (FABRICATED / SUPPORTED / etc.) are intentionally omitted
    from the body — they're already conveyed by the highlight color and by
    the on-page web report; reprinting them in the popup just adds noise.

    A bound popup window is attached via ``set_popup`` so PDF readers that
    honor it (Adobe, Foxit, macOS Preview) open the popup at a stable size
    and keep it open until clicked closed — addresses the "tooltip vanishes
    on mouse-out" complaint with hover-only viewers like Chrome's.
    """
    import fitz

    # Highlight (kept on the citation marker; lighter so glyphs stay legible)
    highlight = page.add_highlight_annot(rect)
    highlight.set_colors(stroke=color)
    highlight.set_opacity(0.35)
    highlight.update()

    # ── Section 1: title + byline ─────────────────────────────────────
    title_line = (ref.title.strip() if ref and ref.title else "Cited paper")
    title_line = title_line.replace("\n", " ").strip()

    byline = ""
    if ref:
        authors = "; ".join(a for a in ref.authors if a) if ref.authors else ""
        year = f" ({ref.year})" if ref.year else ""
        byline = f"{authors}{year}".strip()

    # ── Section 2: claim agent's opinion (per-citation if we have it) ──
    # `verdict.per_sentence_claim_verdicts` is keyed by the literal citing
    # sentence — that's the LLM's actual reasoning for *this* marker. If
    # present, use its `explanation` (free-text reasoning) and any quote it
    # latched onto. Otherwise fall back to the rolled-up metadata reason.
    claim_opinion = ""
    claim_quote = ""
    if cit and cit.citing_sentence:
        psv = (getattr(verdict, "per_sentence_claim_verdicts", None) or {})
        per = psv.get(cit.citing_sentence)
        if per:
            claim_opinion = (per.explanation or "").strip()
            claim_quote = (per.evidence_quote or "").strip()
    if not claim_opinion:
        claim_opinion = (
            getattr(verdict, "claim_explanation", None)
            or getattr(verdict, "metadata_explanation", None)
            or getattr(verdict, "explanation", None)
            or ""
        ).strip()

    # ── Section 3: abstract ───────────────────────────────────────────
    abstract = ""
    ex = getattr(verdict, "existence", None)
    if ex and getattr(ex, "abstract", None):
        abstract = _strip_inline_tags(ex.abstract).strip().replace("\n", " ")

    # ── Section 4: top retrieved passages ─────────────────────────────
    passage_lines: list[str] = []
    if comp is not None:
        for i, sc in enumerate(getattr(comp, "top_passages", [])[:3], 1):
            chunk = getattr(sc, "chunk", None)
            text = (getattr(chunk, "text", "") or "").strip().replace("\n", " ")
            section = (getattr(chunk, "section_name", "") or "").strip()
            if not text:
                continue
            head = f"[{i}] {section}" if section else f"[{i}]"
            passage_lines.append(f"{head}\n{_truncate(text, 700)}")

    parts = [title_line]
    if byline:
        parts.append(byline)
    if claim_opinion:
        parts.append("")
        parts.append("Claim agent's opinion:")
        parts.append(_truncate(claim_opinion, 1200))
        if claim_quote:
            parts.append(f'  > "{_truncate(claim_quote, 500)}"')
    if abstract:
        parts.append("")
        parts.append("Cited paper abstract:")
        parts.append(_truncate(abstract, 2000))
    if passage_lines:
        parts.append("")
        parts.append("Retrieved passages:")
        for pl in passage_lines:
            parts.append(pl)
    body = "\n".join(parts)

    # Cap so PDF reader popup widgets don't choke. Modern desktop viewers
    # (Acrobat, Preview, Foxit, Okular) render 5000+ chars without issue.
    # Web viewers like PDF.js truncate harder, but we can't help that.
    if len(body) > 5000:
        body = body[:4997].rstrip() + "..."

    # Push the icon clear of the marker rect so clicks land on the icon,
    # not the underlying citation hyperlink. ~14pt right + 4pt up keeps it
    # on the same baseline but visibly separate.
    icon_x = rect.x1 + 14
    icon_y = max(rect.y0 - 4, 12)
    note_point = fitz.Point(icon_x, icon_y)

    note = page.add_text_annot(note_point, body, icon="Comment")
    note.set_info(title=_truncate(title_line, 80))

    # Attach a fixed popup window so the popup opens at a predictable
    # position and size, and stays open until the user clicks it closed
    # in viewers that honor the popup annotation (most desktop readers).
    page_rect = page.rect
    popup_w = min(360, page_rect.width * 0.45)
    popup_h = 280
    # Prefer the right margin; fall back to left if there's no room.
    popup_x0 = icon_x + 8
    if popup_x0 + popup_w > page_rect.width - 8:
        popup_x0 = max(8, rect.x0 - popup_w - 8)
    popup_y0 = max(8, icon_y - 12)
    if popup_y0 + popup_h > page_rect.height - 8:
        popup_y0 = max(8, page_rect.height - popup_h - 8)
    popup_rect = fitz.Rect(popup_x0, popup_y0, popup_x0 + popup_w, popup_y0 + popup_h)
    try:
        note.set_popup(popup_rect)
    except Exception:
        # Older PyMuPDF may name it differently; best-effort, the note still
        # works as a regular sticky note without an explicit popup rect.
        pass

    # `fill` controls the icon body color in most readers (default is the
    # bright yellow sticky-note look). Use a desaturated tint of the verdict
    # color so the icon is recognizable but not glaring.
    soft = tuple(min(1.0, 0.55 + 0.45 * c) for c in color)
    note.set_colors(stroke=color, fill=soft)
    note.update()


_INLINE_TAG_RE = re.compile(r"<[^>]+>")


def _strip_inline_tags(s: str) -> str:
    """Drop `<em>`, `<jats:p>`, etc. that some DBs ship inside abstracts.
    Without this the popup shows literal angle-bracket tags as text."""
    return _INLINE_TAG_RE.sub(" ", s or "")


def _truncate(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[: n - 3].rstrip() + "..."
