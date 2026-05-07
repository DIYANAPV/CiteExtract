
from __future__ import annotations

from citeextract.classification.classifier import CitationVerdict
from citeextract.models.citation import Citation
from citeextract.models.parsed_paper import ParsedPaper
from citeextract.models.reference import Reference
from citeextract.models.report import PaperReport
from citeextract.report import pdf_annotator
from citeextract.report.pdf_annotator import annotate_pdf


def test_truncate_short_string_unchanged():
    assert pdf_annotator._truncate("hello", 100) == "hello"


def test_truncate_long_string_gets_ellipsis():
    s = "x" * 200
    out = pdf_annotator._truncate(s, 50)
    assert len(out) == 50
    assert out.endswith("...")


def test_section_caps_fit_a_real_abstract_plus_passages():
    import inspect
    body_src = inspect.getsource(pdf_annotator)

    assert "_truncate(claim_opinion, 1200)" in body_src
    assert "_truncate(claim_quote, 500)" in body_src
    assert "_truncate(abstract, 2000)" in body_src
    assert "_truncate(text, 700)" in body_src
    assert "len(body) > 5000" in body_src


def _make_pdf(tmp_path, body_text: str, name: str = "src.pdf") -> str:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), body_text, fontsize=11)
    out = tmp_path / name
    doc.save(str(out))
    doc.close()
    return str(out)


def _verdict(ref_id: str = "1", verdict: str = "VALID") -> CitationVerdict:
    return CitationVerdict(
        ref_id=ref_id, verdict=verdict, mode="quick",
        action="no_action", explanation="ok",
    )


def _citation(
    ref_id: str = "1",
    marker: str = "[1]",
    position: int = 0,
    sentence: str = "We build on [1] in this work.",
) -> Citation:
    return Citation(
        ref_id=ref_id, marker=marker, position=position,
        citing_sentence=sentence, context_before="", context_after="",
    )


def _ref(ref_id: str = "1", title: str = "T") -> Reference:
    return Reference(
        ref_id=ref_id, title=title, authors=["A"], year=2020,
        source_format="grobid",
    )


def _parsed(citations: list[Citation], references: list[Reference]) -> ParsedPaper:
    return ParsedPaper(
        references=references, citations=citations,
        has_body_text=True, body_text="...", input_format="pdf",
        metadata={}, warnings=[],
    )


def _report(verdicts: list[CitationVerdict]) -> PaperReport:
    return PaperReport(
        input_file="-", input_format="pdf", mode="quick", timestamp="-",
        total_references=len(verdicts), verdicts=verdicts,
    )


class TestObservabilityCounters:

    def test_no_verdict_for_ref_id_increments_counter(self, tmp_path):
        pdf = _make_pdf(tmp_path, "Body with [1] in it.")
        out = str(tmp_path / "out.pdf")
        cit = _citation(ref_id="orphan")
        parsed = _parsed([cit], [_ref(ref_id="orphan")])
        report = _report([])

        stats = annotate_pdf(pdf, report, parsed, out)

        assert stats.skipped_no_verdict == 1
        assert stats.annotated == 0
        assert stats.skipped_collision == 0
        assert stats.skipped_not_found == 0

    def test_repeated_marker_at_same_position_collides(self, tmp_path):
        pdf = _make_pdf(tmp_path, "Methods build on [1] for evaluation.")
        out = str(tmp_path / "out.pdf")
        cit_a = _citation(ref_id="a", marker="[1]", position=18)
        cit_b = _citation(ref_id="b", marker="[1]", position=18)
        parsed = _parsed(
            [cit_a, cit_b],
            [_ref(ref_id="a"), _ref(ref_id="b")],
        )
        report = _report([_verdict("a"), _verdict("b")])

        stats = annotate_pdf(pdf, report, parsed, out)

        assert stats.annotated == 1
        assert stats.skipped_collision == 1
        assert stats.skipped_not_found == 0

    def test_marker_absent_from_pdf_increments_not_found(self, tmp_path):
        pdf = _make_pdf(tmp_path, "Body without any citation marker.")
        out = str(tmp_path / "out.pdf")
        cit = _citation(ref_id="1", marker="[42]")
        parsed = _parsed([cit], [_ref()])
        report = _report([_verdict()])

        stats = annotate_pdf(pdf, report, parsed, out)

        assert stats.skipped_not_found == 1
        assert stats.skipped_collision == 0
        assert stats.annotated == 0

    def test_empty_marker_increments_no_marker(self, tmp_path):
        pdf = _make_pdf(tmp_path, "Body with [1] somewhere.")
        out = str(tmp_path / "out.pdf")
        cit = _citation(marker="")
        parsed = _parsed([cit], [_ref()])
        report = _report([_verdict()])

        stats = annotate_pdf(pdf, report, parsed, out)

        assert stats.skipped_no_marker == 1
        assert stats.annotated == 0


class TestBasicAnnotation:

    def test_single_citation_lands_one_icon(self, tmp_path):
        pdf = _make_pdf(tmp_path, "We build on [1] in this work.")
        out = str(tmp_path / "out.pdf")
        parsed = _parsed([_citation()], [_ref()])
        report = _report([_verdict()])

        stats = annotate_pdf(pdf, report, parsed, out)

        assert stats.annotated == 1
        assert stats.skipped_no_verdict == 0
        assert stats.skipped_collision == 0
        assert stats.skipped_not_found == 0
        assert stats.skipped_no_marker == 0

    def test_repeated_marker_at_distinct_positions_both_annotated(self, tmp_path):
        pdf = _make_pdf(
            tmp_path,
            "First we build on [1] in this work. Later we extend [1] further.",
        )
        out = str(tmp_path / "out.pdf")
        cit_a = _citation(
            ref_id="1", marker="[1]", position=18,
            sentence="First we build on [1] in this work.",
        )
        cit_b = _citation(
            ref_id="1", marker="[1]", position=53,
            sentence="Later we extend [1] further.",
        )
        parsed = _parsed([cit_a, cit_b], [_ref()])
        report = _report([_verdict()])

        stats = annotate_pdf(pdf, report, parsed, out)

        assert stats.annotated == 2
        assert stats.skipped_collision == 0


class TestSurnameFromMarker:

    def test_simple_paren_author_year(self):
        from citeextract.report.pdf_annotator import _surname_from_marker
        assert _surname_from_marker("(Smith, 2020)") == "Smith"

    def test_truncated_trailing_half(self):
        from citeextract.report.pdf_annotator import _surname_from_marker
        assert _surname_from_marker("Wager and Middleton, 2008)") == "Wager"

    def test_narrative_form(self):
        from citeextract.report.pdf_annotator import _surname_from_marker
        assert _surname_from_marker("Yamada et al. (2025)") == "Yamada"

    def test_numeric_marker_returns_none(self):
        from citeextract.report.pdf_annotator import _surname_from_marker
        assert _surname_from_marker("[42]") is None

    def test_short_surname_returns_none(self):
        from citeextract.report.pdf_annotator import _surname_from_marker
        assert _surname_from_marker("(Liu, 2023)") is None
        assert _surname_from_marker("(Wu et al., 2024)") is None

    def test_stopword_first_token_returns_none(self):
        from citeextract.report.pdf_annotator import _surname_from_marker
        assert _surname_from_marker("and Smith, 2020)") is None

    def test_hyphenated_surname_kept_whole(self):
        from citeextract.report.pdf_annotator import _surname_from_marker
        assert _surname_from_marker("(Müller-Schmidt, 2019)") == "Müller-Schmidt"

    def test_empty_marker_returns_none(self):
        from citeextract.report.pdf_annotator import _surname_from_marker
        assert _surname_from_marker("") is None


class TestNormalizeUnicodeMarker:

    def test_turkish_chars_stripped(self):
        from citeextract.report.pdf_annotator import _normalize_unicode_marker
        assert _normalize_unicode_marker("(Taşkın, 2025)") == "(Taskin, 2025)"

    def test_german_umlaut_stripped(self):
        from citeextract.report.pdf_annotator import _normalize_unicode_marker
        assert _normalize_unicode_marker("(Müller, 2020)") == "(Muller, 2020)"

    def test_french_accent_stripped(self):
        from citeextract.report.pdf_annotator import _normalize_unicode_marker
        assert _normalize_unicode_marker("(Céspedes, 2025)") == "(Cespedes, 2025)"

    def test_ascii_marker_unchanged(self):
        from citeextract.report.pdf_annotator import _normalize_unicode_marker
        assert _normalize_unicode_marker("(Smith, 2020)") == "(Smith, 2020)"

    def test_empty_returns_empty(self):
        from citeextract.report.pdf_annotator import _normalize_unicode_marker
        assert _normalize_unicode_marker("") == ""


class TestSurnameOnlyFallback:

    def test_recovers_when_literal_marker_missing_but_surname_present(
        self, tmp_path,
    ):
        pdf = _make_pdf(
            tmp_path,
            "Studies by Wager show evidence of citation drift in this area.",
        )
        out = str(tmp_path / "out.pdf")
        cit = _citation(
            ref_id="1",
            marker="Wager and Middleton, 2008)",
            position=11,
            sentence="Studies by Wager show evidence of citation drift in this area.",
        )
        parsed = _parsed([cit], [_ref()])
        report = _report([_verdict()])

        stats = annotate_pdf(pdf, report, parsed, out)

        assert stats.annotated == 1, "Surname fallback should rescue this citation"
        assert stats.skipped_not_found == 0
