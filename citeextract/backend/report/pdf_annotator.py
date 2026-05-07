
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import fitz

log = logging.getLogger(__name__)


VERDICT_COLORS = {
    "FABRICATED":   (0.95, 0.40, 0.40),
    "UNVERIFIABLE": (0.65, 0.65, 0.68),
    "VALID":        (0.40, 0.78, 0.55),
    "CONTRADICTS_CLAIM": (0.96, 0.65, 0.30),
}


def color_for_verdict(metadata_verdict: str, claim_verdict: str | None = None) -> tuple[float, float, float]:
    if metadata_verdict in ("FABRICATED", "UNVERIFIABLE"):
        return VERDICT_COLORS[metadata_verdict]
    if claim_verdict == "CONTRADICTS":
        return VERDICT_COLORS["CONTRADICTS_CLAIM"]
    return VERDICT_COLORS.get(metadata_verdict or "VALID", VERDICT_COLORS["VALID"])


@dataclass
class AnnotationStats:
    annotated: int = 0
    skipped_no_marker: int = 0
    skipped_not_found: int = 0
    skipped_collision: int = 0
    skipped_no_verdict: int = 0
    pages: int = 0


def annotate_pdf(
    source_pdf_path: str,
    paper_report,
    parsed,
    output_path: str,
    comp_report=None,
) -> AnnotationStats:
    import fitz

    src = Path(source_pdf_path)
    if not src.exists():
        raise RuntimeError(f"Source PDF not found: {source_pdf_path}")

    verdicts_by_ref = {v.ref_id: v for v in paper_report.verdicts}
    refs_by_id = {r.ref_id: r for r in (parsed.references if parsed else [])}
    citations = list(parsed.citations) if parsed else []

    comp_by_key: dict[tuple[str, str], object] = {}
    if comp_report is not None:
        for r in comp_report.results:
            comp_by_key[(r.ref_id, r.citing_sentence)] = r

    try:
        doc = fitz.open(str(src))
    except Exception as e:
        raise RuntimeError(f"Could not open PDF: {e}") from e

    stats = AnnotationStats(pages=doc.page_count)

    used_positions: set[tuple[int, int, int]] = set()

    try:
        for cit in citations:
            verdict = verdicts_by_ref.get(cit.ref_id)
            if verdict is None:
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
    sentence_prefix = _safe_sentence_prefix(cit.citing_sentence)
    any_marker_match = False

    target = _find_anchored_match(doc, marker, sentence_prefix, used_positions)
    if target is not None:
        return target, "found"

    for page_idx, page in enumerate(doc):
        marker_rects = page.search_for(marker)
        if marker_rects:
            any_marker_match = True
        for rect in marker_rects:
            key = (page_idx, int(rect.x0), int(rect.y0))
            if key not in used_positions:
                return (page_idx, rect), "found"

    normalized = _normalize_unicode_marker(marker)
    if normalized != marker:
        target = _find_anchored_match(
            doc, normalized, sentence_prefix, used_positions,
        )
        if target is not None:
            log.info(
                "annotator: unicode-normalized fallback matched %r → %r",
                marker[:60], normalized[:60],
            )
            return target, "found"

    surname = _surname_from_marker(marker)
    if surname is not None and sentence_prefix:
        target = _find_surname_near_sentence(
            doc, surname, sentence_prefix, used_positions,
        )
        if target is not None:
            log.info(
                "annotator: surname-only fallback matched %r → %r",
                marker[:60], surname,
            )
            return target, "found"

    return None, ("all_collided" if any_marker_match else "not_in_text")


def _find_anchored_match(
    doc, marker: str, sentence_prefix: str,
    used_positions: set[tuple[int, int, int]],
) -> Optional[tuple[int, fitz.Rect]]:
    if not sentence_prefix or not marker:
        return None
    for page_idx, page in enumerate(doc):
        sentence_rects = page.search_for(sentence_prefix)
        if not sentence_rects:
            continue
        marker_rects = page.search_for(marker)
        if not marker_rects:
            continue
        s0 = sentence_rects[0]
        for rect in sorted(
            marker_rects,
            key=lambda r: abs(r.y0 - s0.y0) + abs(r.x0 - s0.x0),
        ):
            key = (page_idx, int(rect.x0), int(rect.y0))
            if key not in used_positions:
                return page_idx, rect
    return None


_SURNAME_BAND_POINTS = 30.0


def _find_surname_near_sentence(
    doc, surname: str, sentence_prefix: str,
    used_positions: set[tuple[int, int, int]],
) -> Optional[tuple[int, fitz.Rect]]:
    for page_idx, page in enumerate(doc):
        sentence_rects = page.search_for(sentence_prefix)
        if not sentence_rects:
            continue
        s0 = sentence_rects[0]
        for rect in sorted(
            page.search_for(surname),
            key=lambda r: abs(r.y0 - s0.y0) + abs(r.x0 - s0.x0),
        ):
            if abs(rect.y0 - s0.y0) > _SURNAME_BAND_POINTS:
                continue
            key = (page_idx, int(rect.x0), int(rect.y0))
            if key not in used_positions:
                return page_idx, rect
    return None


_LATIN_ATOM_MAP = {
    "ı": "i", "İ": "I",
    "ø": "o", "Ø": "O",
    "æ": "ae", "Æ": "AE",
    "œ": "oe", "Œ": "OE",
    "đ": "d", "Đ": "D",
    "ł": "l", "Ł": "L",
    "ß": "ss",
    "ð": "d", "Ð": "D",
    "þ": "th", "Þ": "Th",
}


def _normalize_unicode_marker(marker: str) -> str:
    import unicodedata
    if not marker:
        return ""
    decomposed = unicodedata.normalize("NFKD", marker)
    no_diacritics = "".join(c for c in decomposed if not unicodedata.combining(c))
    return "".join(_LATIN_ATOM_MAP.get(c, c) for c in no_diacritics)


_SURNAME_RE = re.compile(
    r"[A-Za-zÀ-ſ][A-Za-zÀ-ſ'\-]+",
    re.UNICODE,
)
_SURNAME_STOPWORDS = frozenset({
    "and", "or", "the", "in", "of", "et", "etal",
})


def _surname_from_marker(marker: str) -> Optional[str]:
    if not marker:
        return None
    text = marker.lstrip(" ,;([{")
    if not text or text[0].isdigit():
        return None
    m = _SURNAME_RE.match(text)
    if m is None:
        return None
    token = m.group()
    if token.lower() in _SURNAME_STOPWORDS or len(token) < 4:
        return None
    return token


def _safe_sentence_prefix(sentence: Optional[str], length: int = 40) -> str:
    if not sentence:
        return ""
    text = " ".join(sentence.split())
    return text[:length].strip()


def _draw_highlight_with_note(page, rect, verdict, ref, comp, cit, color) -> None:
    import fitz

    highlight = page.add_highlight_annot(rect)
    highlight.set_colors(stroke=color)
    highlight.set_opacity(0.35)
    highlight.update()

    title_line = (ref.title.strip() if ref and ref.title else "Cited paper")
    title_line = title_line.replace("\n", " ").strip()

    byline = ""
    if ref:
        authors = "; ".join(a for a in ref.authors if a) if ref.authors else ""
        year = f" ({ref.year})" if ref.year else ""
        byline = f"{authors}{year}".strip()

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

    abstract = ""
    ex = getattr(verdict, "existence", None)
    if ex and getattr(ex, "abstract", None):
        abstract = _strip_inline_tags(ex.abstract).strip().replace("\n", " ")

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

    if len(body) > 5000:
        body = body[:4997].rstrip() + "..."

    icon_x = rect.x1 + 14
    icon_y = max(rect.y0 - 4, 12)
    note_point = fitz.Point(icon_x, icon_y)

    note = page.add_text_annot(note_point, body, icon="Comment")
    note.set_info(title=_truncate(title_line, 80))

    page_rect = page.rect
    popup_w = min(360, page_rect.width * 0.45)
    popup_h = 280
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
        pass

    soft = tuple(min(1.0, 0.55 + 0.45 * c) for c in color)
    note.set_colors(stroke=color, fill=soft)
    note.update()


_INLINE_TAG_RE = re.compile(r"<[^>]+>")


def _strip_inline_tags(s: str) -> str:
    return _INLINE_TAG_RE.sub(" ", s or "")


def _truncate(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[: n - 3].rstrip() + "..."
