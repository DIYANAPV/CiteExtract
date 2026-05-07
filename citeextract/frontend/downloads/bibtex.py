
from __future__ import annotations

import tempfile
import time
from typing import Optional

from citeextract_ui.render.helpers import _safe_stem


_PROBLEM_VERDICTS = {"FABRICATED", "UNVERIFIABLE"}


def _bibtex_escape(text: str) -> str:
    return str(text).replace("{", "").replace("}", "").replace("\n", " ").strip()


def _format_problem_bibtex_entry(v, ref) -> Optional[str]:
    if v.verdict not in _PROBLEM_VERDICTS and v.claim_verdict != "CONTRADICTS":
        return None
    if ref is None:
        return None

    key = v.ref_id
    fields: list[tuple[str, str]] = []
    if ref.title:
        fields.append(("title", _bibtex_escape(ref.title)))
    if ref.authors:
        fields.append(("author", _bibtex_escape(" and ".join(ref.authors))))
    if ref.year:
        fields.append(("year", str(ref.year)))
    if ref.venue:
        fields.append(("howpublished", _bibtex_escape(ref.venue)))
    if ref.doi:
        fields.append(("doi", _bibtex_escape(ref.doi)))
    if ref.url:
        fields.append(("url", _bibtex_escape(ref.url)))

    note_parts = [f"CiteExtract verdict: {v.verdict}"]
    if v.explanation:
        note_parts.append(_bibtex_escape(v.explanation))
    ex = v.existence
    if ex:
        if ex.status == "NOT_FOUND" and ex.databases_checked:
            note_parts.append(
                f"Not found in: {', '.join(ex.databases_checked)}"
            )
        elif ex.status == "FOUND" and ex.matched_title:
            note_parts.append(
                f"DB match: '{_bibtex_escape(ex.matched_title)[:120]}' "
                f"via {ex.source or 'unknown'}"
            )
    fields.append(("note", " | ".join(note_parts)))

    body = ",\n  ".join(f'{k} = {{{v}}}' for k, v in fields)
    return f"@misc{{{key},\n  {body}\n}}"


def _build_problematic_bibtex(paper_report, references, source_label: str = "") -> str:
    if paper_report is None:
        return ""
    ref_map = {r.ref_id: r for r in references}
    entries = []
    for v in paper_report.verdicts:
        entry = _format_problem_bibtex_entry(v, ref_map.get(v.ref_id))
        if entry:
            entries.append(entry)
    if not entries:
        return ""
    header_bits = [
        "% CiteExtract: problematic references",
        "% Exported: " + time.strftime("%Y-%m-%d %H:%M:%S"),
    ]
    if source_label:
        header_bits.append(f"% Source: {source_label}")
    header_bits.append(
        f"% Entries: {len(entries)} "
        "(FABRICATED, UNVERIFIABLE verdicts + references with CONTRADICTS claims)"
    )
    header_bits.append("% Each entry carries a `note = {...}` field with the audit trail.")
    return "\n".join(header_bits) + "\n\n" + "\n\n".join(entries) + "\n"


def _write_problematic_bibtex(
    paper_report, references, source_label: str = "",
    paper_stem: str = "",
) -> Optional[str]:
    bib = _build_problematic_bibtex(paper_report, references, source_label)
    if not bib:
        return None
    stem = paper_stem or _safe_stem(source_label or "")
    tmp = tempfile.NamedTemporaryFile(
        suffix=".bib", prefix=f"{stem}_problems_", delete=False, mode="w",
        encoding="utf-8",
    )
    tmp.write(bib)
    tmp.close()
    return tmp.name


def _write_batch_problem_bibtex(per_paper: list[dict]) -> Optional[str]:
    chunks = []
    for p in per_paper:
        if p["status"] != "ok" or p["paper_report"] is None:
            continue
        bib = _build_problematic_bibtex(
            p["paper_report"], p["parsed"].references, source_label=p["name"],
        )
        if bib:
            chunks.append(bib)
    if not chunks:
        return None
    tmp = tempfile.NamedTemporaryFile(
        suffix=".bib", prefix="citeextract_batch_problems_", delete=False, mode="w",
        encoding="utf-8",
    )
    tmp.write("\n\n".join(chunks))
    tmp.close()
    return tmp.name
