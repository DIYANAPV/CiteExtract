
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

from citeextract.report.pdf_annotator import annotate_pdf
from citeextract_ui.render.helpers import _esc, _safe_stem


def _try_annotate_pdf(
    source_path: str, paper_report, parsed, comp_report=None,
    paper_stem: str = "",
) -> tuple[Optional[str], str]:
    def status(msg: str, tone: str = "info") -> str:
        bg = {"info": "#f3f4f6", "warn": "#fef3c7"}.get(tone, "#f3f4f6")
        fg = {"info": "#4b5563", "warn": "#92400e"}.get(tone, "#4b5563")
        return (
            f'<div style="font-size:14px;color:{fg};background:{bg};'
            f'padding:6px 10px;border-radius:6px;display:inline-block;">'
            f'{_esc(msg)}</div>'
        )

    if not source_path or Path(source_path).suffix.lower() != ".pdf":
        return None, status("Annotated PDF export is available for PDF uploads only.")

    if paper_report is None or parsed is None:
        return None, status("No verdicts to annotate.")

    stem = paper_stem or _safe_stem(source_path)
    out_tmp = tempfile.NamedTemporaryFile(
        suffix=".pdf", prefix=f"{stem}_annotated_",
        delete=False, mode="wb",
    )
    out_tmp.close()
    try:
        stats = annotate_pdf(
            source_path, paper_report, parsed, out_tmp.name,
            comp_report=comp_report,
        )
    except Exception as e:
        logging.getLogger(__name__).warning(f"Annotated PDF failed: {e}")
        Path(out_tmp.name).unlink(missing_ok=True)
        return None, status(
            f"Annotated PDF could not be generated: {e}",
            tone="warn",
        )

    if stats.annotated == 0:
        Path(out_tmp.name).unlink(missing_ok=True)
        return None, status(
            "Annotated PDF skipped: no citation markers could be located on any page.",
            tone="warn",
        )

    total_cits = len(parsed.citations) if parsed else 0
    if total_cits and stats.annotated < total_cits * 0.8:
        missing = total_cits - stats.annotated
        return out_tmp.name, status(
            f"Annotated {stats.annotated}/{total_cits} citation markers; "
            f"{missing} could not be placed on the PDF (see server logs for breakdown).",
            tone="warn",
        )
    return out_tmp.name, ""
