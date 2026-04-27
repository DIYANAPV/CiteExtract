"""Tests for src/report/pdf_annotator.py.

Two concerns are pinned here:

1. **Sticky-note size caps.** Sticky notes need to fit a real abstract
   (≈ 1500–2500 chars) plus retrieved passages plus the claim agent's
   opinion. An earlier 2400-char total cap silently truncated abstracts
   in half — these tests make sure that doesn't come back.

2. **Phase 1 observability counters.** Every silently-dropped citation
   should now land in a typed ``AnnotationStats`` counter so the UI can
   warn instead of producing a quietly-incomplete PDF. Behavior is
   otherwise unchanged — these tests pin down the counter semantics so
   Phase 2 (multi-target merge) and Phase 3 (search fallback) can be
   evaluated against them.

Phase 1 fixture PDFs are written on the fly with PyMuPDF so the tests
don't depend on GROBID, the ``data/`` directory, or any fixture file.
"""

from __future__ import annotations

from src.classification.classifier import CitationVerdict
from src.models.citation import Citation
from src.models.parsed_paper import ParsedPaper
from src.models.reference import Reference
from src.models.report import PaperReport
from src.report import pdf_annotator
from src.report.pdf_annotator import annotate_pdf


# ---------------------------------------------------------------------------
# Truncation / size cap regression guards
# ---------------------------------------------------------------------------


def test_truncate_short_string_unchanged():
    assert pdf_annotator._truncate("hello", 100) == "hello"


def test_truncate_long_string_gets_ellipsis():
    s = "x" * 200
    out = pdf_annotator._truncate(s, 50)
    assert len(out) == 50
    assert out.endswith("...")


def test_section_caps_fit_a_real_abstract_plus_passages():
    """A typical abstract (~2000 chars) + 3 long passages must not be
    silently halved by the per-section caps."""
    import inspect
    body_src = inspect.getsource(pdf_annotator)

    # Hard floors — if any of these regress, the test fails.
    assert "_truncate(claim_opinion, 1200)" in body_src
    assert "_truncate(claim_quote, 500)" in body_src
    assert "_truncate(abstract, 2000)" in body_src
    assert "_truncate(text, 700)" in body_src
    assert "len(body) > 5000" in body_src


# ---------------------------------------------------------------------------
# Builders for Phase 1 tests — keep tests focused on the data point under
# test, not Pydantic ceremony. Defaults are chosen so the citation, the
# reference, and the verdict all share ``ref_id="1"`` unless overridden.
# ---------------------------------------------------------------------------


def _make_pdf(tmp_path, body_text: str, name: str = "src.pdf") -> str:
    """Write a one-page PDF whose text layer matches ``body_text``."""
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


# ---------------------------------------------------------------------------
# Phase 1 — observability counters
# ---------------------------------------------------------------------------


class TestObservabilityCounters:
    """Every silently-dropped citation should now land in a typed counter."""

    def test_no_verdict_for_ref_id_increments_counter(self, tmp_path):
        """Citation whose ref_id has no matching verdict was previously
        dropped via a silent ``continue`` with no stats trail."""
        pdf = _make_pdf(tmp_path, "Body with [1] in it.")
        out = str(tmp_path / "out.pdf")
        cit = _citation(ref_id="orphan")
        parsed = _parsed([cit], [_ref(ref_id="orphan")])
        report = _report([])  # no verdicts at all

        stats = annotate_pdf(pdf, report, parsed, out)

        assert stats.skipped_no_verdict == 1
        assert stats.annotated == 0
        assert stats.skipped_collision == 0
        assert stats.skipped_not_found == 0

    def test_repeated_marker_at_same_position_collides(self, tmp_path):
        """Two Citations sharing (marker, position) — the PDF only has one
        rect for ``[1]``, so the second one hits the new collision counter
        instead of being miscounted as ``not_found``. This is the smoking
        gun for the multi-target citations Phase 2 will fix."""
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
        """Marker text genuinely not in the PDF text-layer — the path
        Phase 3 will add a fallback for. Stays in ``skipped_not_found``."""
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
        """Existing counter still works — guards against a refactor that
        accidentally reroutes the empty-marker branch."""
        pdf = _make_pdf(tmp_path, "Body with [1] somewhere.")
        out = str(tmp_path / "out.pdf")
        cit = _citation(marker="")
        parsed = _parsed([cit], [_ref()])
        report = _report([_verdict()])

        stats = annotate_pdf(pdf, report, parsed, out)

        assert stats.skipped_no_marker == 1
        assert stats.annotated == 0


class TestBasicAnnotation:
    """Sanity check that the simple case still works after the refactor."""

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
        """A real repeat — same ``[1]`` marker cited from two different
        sentences/positions in the body. Each Citation should land its
        own icon on the corresponding rect, exercising the ``used_positions``
        bookkeeping in the success path."""
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

        # Both occurrences exist in the PDF — both should be claimed.
        assert stats.annotated == 2
        assert stats.skipped_collision == 0
