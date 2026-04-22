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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# RGB in [0, 1]. Chosen to read clearly on white backgrounds and match the
# verdict colors used in the web UI.
VERDICT_COLORS = {
    "FABRICATED":    (0.95, 0.30, 0.30),   # red
    "MISREPRESENTED":(0.95, 0.60, 0.20),   # amber
    "UNVERIFIABLE":  (0.60, 0.60, 0.65),   # gray
    "VALID":         (0.25, 0.75, 0.45),   # green
}


@dataclass
class AnnotationStats:
    """Return payload from annotate_pdf: how the run went."""
    annotated: int = 0                 # successfully highlighted + noted
    skipped_no_marker: int = 0         # citation had no marker text
    skipped_not_found: int = 0         # marker text not locatable on any page
    pages: int = 0


def annotate_pdf(
    source_pdf_path: str,
    paper_report,
    parsed,
    output_path: str,
) -> AnnotationStats:
    """Write ``output_path`` = source PDF + highlights + sticky notes.

    Args:
        source_pdf_path: Original uploaded PDF.
        paper_report:    PaperReport with verdicts.
        parsed:          ParsedPaper (we use ``citations`` for markers + context).
        output_path:     Destination path for the annotated PDF.

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
    citations = list(parsed.citations) if parsed else []

    try:
        doc = fitz.open(str(src))
    except Exception as e:
        raise RuntimeError(f"Could not open PDF: {e}") from e

    stats = AnnotationStats(pages=doc.page_count)

    try:
        for cit in citations:
            verdict = verdicts_by_ref.get(cit.ref_id)
            if verdict is None:
                continue  # no verdict for this ref — shouldn't happen, but skip safely
            color = VERDICT_COLORS.get(verdict.verdict)
            if color is None:
                continue  # unknown verdict label, skip

            marker = (cit.marker or "").strip()
            if not marker:
                stats.skipped_no_marker += 1
                continue

            if _place_annotation_on_first_match(doc, cit, verdict, marker, color):
                stats.annotated += 1
            else:
                stats.skipped_not_found += 1

        # garbage=4 + deflate keeps the resulting file size reasonable
        doc.save(output_path, garbage=4, deflate=True)
    finally:
        doc.close()

    log.info(
        f"PDF annotated: {stats.annotated} markers highlighted, "
        f"{stats.skipped_not_found} not locatable, "
        f"{stats.skipped_no_marker} without marker text"
    )
    return stats


def _place_annotation_on_first_match(doc, cit, verdict, marker: str, color) -> bool:
    """Find the best occurrence of ``marker`` for this citation and annotate it.

    Strategy: search every page for the marker string. When a page has only one
    occurrence, use it. When there are multiple, use the citing sentence prefix
    to pick the one physically closest to where the sentence appears. When all
    else fails, fall back to the first occurrence on the first page that has any.
    Returns True if we placed at least one annotation.
    """
    sentence_prefix = _safe_sentence_prefix(cit.citing_sentence)

    for page in doc:
        rects = page.search_for(marker)
        if not rects:
            continue

        target_rect = rects[0]
        if len(rects) > 1 and sentence_prefix:
            sentence_rects = page.search_for(sentence_prefix)
            if sentence_rects:
                s0 = sentence_rects[0]
                target_rect = min(
                    rects, key=lambda r: abs(r.y0 - s0.y0) + abs(r.x0 - s0.x0)
                )

        _draw_highlight_with_note(page, target_rect, verdict, color)
        return True

    return False


def _safe_sentence_prefix(sentence: Optional[str], length: int = 40) -> str:
    """First ~40 chars of the citing sentence, trimmed so a PDF search won't
    choke on line breaks / trailing whitespace. Empty for no-context citations.
    """
    if not sentence:
        return ""
    text = " ".join(sentence.split())  # collapse whitespace / newlines
    return text[:length].strip()


def _draw_highlight_with_note(page, rect, verdict, color) -> None:
    """One colored highlight over the marker + one sticky-note beside it.

    The note is placed just to the right of the marker so clicking it in a
    viewer doesn't occlude the paper text.
    """
    import fitz

    # Highlight
    highlight = page.add_highlight_annot(rect)
    highlight.set_colors(stroke=color)
    highlight.update()

    # Sticky-note (text annotation)
    note_point = fitz.Point(rect.x1 + 2, rect.y0)
    body = f"{verdict.verdict} — {(verdict.explanation or '').strip()}"
    # PDF annotations can render awkwardly when very long; cap it
    if len(body) > 600:
        body = body[:597] + "..."
    note = page.add_text_annot(note_point, body, icon="Comment")
    note.set_info(title=f"CheckCitation · {verdict.verdict}")
    note.set_colors(stroke=color)
    note.update()
