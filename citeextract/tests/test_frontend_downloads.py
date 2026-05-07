
from __future__ import annotations

from citeextract.classification.classifier import CitationVerdict
from citeextract.models.reference import Reference
from citeextract.models.report import PaperReport
from citeextract.models.verdict import ExistenceResult
from citeextract_ui.downloads.bibtex import (
    _PROBLEM_VERDICTS,
    _bibtex_escape,
    _build_problematic_bibtex,
    _format_problem_bibtex_entry,
)


def _verdict(verdict: str = "FABRICATED", ref_id: str = "1", explanation: str = "Not found.") -> CitationVerdict:
    return CitationVerdict(
        ref_id=ref_id, verdict=verdict, mode="quick",
        action="remove_citation", explanation=explanation,
    )


def _ref(ref_id: str = "1") -> Reference:
    return Reference(
        ref_id=ref_id, title="A Cited Paper", authors=["Doe, J."], year=2020,
        venue="Journal of X", doi="10.1234/abc",
        source_format="grobid",
    )


def _report(verdicts: list[CitationVerdict]) -> PaperReport:
    return PaperReport(
        input_file="-", input_format="pdf", mode="quick", timestamp="-",
        total_references=len(verdicts), verdicts=verdicts,
    )


def test_problem_verdicts_set_contents():
    assert _PROBLEM_VERDICTS == {"FABRICATED", "UNVERIFIABLE"}


def test_bibtex_escape_strips_braces_and_newlines():
    assert _bibtex_escape("a{b}c") == "abc"
    assert _bibtex_escape("line1\nline2") == "line1 line2"
    assert _bibtex_escape("  trailing  ") == "trailing"


def test_format_problem_bibtex_entry_returns_none_for_valid():
    assert _format_problem_bibtex_entry(_verdict("VALID"), _ref()) is None


def test_format_problem_bibtex_entry_returns_none_for_no_ref():
    assert _format_problem_bibtex_entry(_verdict("FABRICATED"), None) is None


def test_format_problem_bibtex_entry_emits_misc_for_fabricated():
    entry = _format_problem_bibtex_entry(_verdict("FABRICATED"), _ref())
    assert entry is not None
    assert entry.startswith("@misc{1,")
    assert entry.rstrip().endswith("}")
    assert "title = {A Cited Paper}" in entry
    assert "author = {Doe, J.}" in entry
    assert "year = {2020}" in entry
    assert "doi = {10.1234/abc}" in entry
    assert "CiteExtract verdict: FABRICATED" in entry


def test_build_problematic_bibtex_returns_empty_when_no_problems():
    assert _build_problematic_bibtex(_report([_verdict("VALID")]), [_ref()]) == ""


def test_build_problematic_bibtex_includes_header_and_entries():
    bib = _build_problematic_bibtex(_report([_verdict("FABRICATED")]), [_ref()], source_label="paper.pdf")
    assert "% CiteExtract: problematic references" in bib
    assert "% Source: paper.pdf" in bib
    assert "@misc{1," in bib


def test_build_problematic_bibtex_drops_none_paper_report():
    assert _build_problematic_bibtex(None, []) == ""
