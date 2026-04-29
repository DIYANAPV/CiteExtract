"""Tests for metadata validation (L3) and mismatch detection."""

import pytest

from src.models.reference import Reference
from src.models.verdict import ExistenceResult
from src.verification.metadata import validate_metadata, FieldComparison


def _make_ref(**kwargs) -> Reference:
    defaults = dict(
        ref_id="1", title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"], year=2017,
        venue="NeurIPS", doi="10.1234/test", raw_text="...", source_format="test",
    )
    defaults.update(kwargs)
    return Reference(**defaults)


def _make_exist(**kwargs) -> ExistenceResult:
    defaults = dict(
        ref_id="1", status="FOUND", source="semantic_scholar",
        matched_title="Attention Is All You Need",
        matched_authors=["Ashish Vaswani", "Noam Shazeer"],
        matched_year=2017, matched_venue="NeurIPS",
        matched_doi="10.1234/test", databases_checked=["semantic_scholar"],
        flags=[],
    )
    defaults.update(kwargs)
    return ExistenceResult(**defaults)


class TestFieldComparison:
    def test_all_match(self):
        ref = _make_ref()
        exist = _make_exist()
        result = validate_metadata(ref, exist)
        assert result.has_metadata_mismatch is False
        assert result.is_retracted is False
        title_cmp = next(c for c in result.comparisons if c.field == "title")
        assert title_cmp.status == "MATCH"

    def test_year_close_match_flagged(self):
        """Year diff of 1 is CLOSE_MATCH (preprint vs publication), not MISMATCH."""
        ref = _make_ref(year=2017)
        exist = _make_exist(matched_year=2018)
        result = validate_metadata(ref, exist)
        year_cmp = next(c for c in result.comparisons if c.field == "year")
        assert year_cmp.status == "CLOSE_MATCH"
        assert any("year_close_match" in f for f in result.flags)

    def test_year_hard_mismatch_flagged(self):
        """Year diff of 3+ is a hard MISMATCH even with strong title + authors."""
        ref = _make_ref(year=2015)
        exist = _make_exist(matched_year=2018)
        result = validate_metadata(ref, exist)
        year_cmp = next(c for c in result.comparisons if c.field == "year")
        assert year_cmp.status == "MISMATCH"
        assert any("year_mismatch" in f for f in result.flags)

    def test_year_diff_two_with_strong_match_is_close(self):
        """When title and authors both agree, a 2-year gap relaxes from
        MISMATCH to CLOSE_MATCH — versioned arXiv repost or slow journal
        pipeline pattern. Without this, the verdict would flag a real
        citation as broken just because v1 and v2 of the same arXiv
        paper crossed a year boundary."""
        ref = _make_ref(year=2020)  # title + authors match defaults
        exist = _make_exist(matched_year=2022)
        result = validate_metadata(ref, exist)
        year_cmp = next(c for c in result.comparisons if c.field == "year")
        assert year_cmp.status == "CLOSE_MATCH"
        assert any("2-year gap" in f for f in result.flags)

    def test_year_diff_two_without_strong_match_stays_mismatch(self):
        """Same 2-year gap but authors disagree → still a MISMATCH.
        The leniency is gated on strong title + author agreement so that
        two genuinely different papers with similar titles can't sneak
        through just because they're 2 years apart."""
        ref = _make_ref(year=2020, authors=["Completely Different Author"])
        exist = _make_exist(matched_year=2022)
        result = validate_metadata(ref, exist)
        year_cmp = next(c for c in result.comparisons if c.field == "year")
        assert year_cmp.status == "MISMATCH"

    def test_year_none_is_missing(self):
        ref = _make_ref(year=None)
        exist = _make_exist(matched_year=2017)
        result = validate_metadata(ref, exist)
        year_cmp = next(c for c in result.comparisons if c.field == "year")
        assert year_cmp.status == "MISSING"

    def test_title_near_match_flagged(self):
        ref = _make_ref(title="Attention Is All You Need")
        exist = _make_exist(matched_title="Attention Is All We Need")
        result = validate_metadata(ref, exist)
        title_cmp = next(c for c in result.comparisons if c.field == "title")
        assert title_cmp.status == "MATCH"
        assert any("not exact match" in f for f in result.flags)

    def test_authors_mismatch(self):
        ref = _make_ref(authors=["John Smith"])
        exist = _make_exist(matched_authors=["Alice Wang"])
        result = validate_metadata(ref, exist)
        auth_cmp = next(c for c in result.comparisons if c.field == "authors")
        assert auth_cmp.status == "MISMATCH"

    def test_doi_mismatch(self):
        ref = _make_ref(doi="10.1234/aaa")
        exist = _make_exist(matched_doi="10.1234/bbb")
        result = validate_metadata(ref, exist)
        doi_cmp = next(c for c in result.comparisons if c.field == "doi")
        assert doi_cmp.status == "MISMATCH"

    def test_retracted(self):
        ref = _make_ref()
        exist = _make_exist(retraction_status=True)
        result = validate_metadata(ref, exist)
        assert result.is_retracted is True


class TestMismatchDetection:
    def test_all_match_no_mismatch(self):
        result = validate_metadata(_make_ref(), _make_exist())
        assert result.has_metadata_mismatch is False

    def test_one_field_mismatch(self):
        """Title matches, one other field wrong → mismatch detected."""
        ref = _make_ref(year=2017)
        exist = _make_exist(matched_year=2099)
        result = validate_metadata(ref, exist)
        assert result.has_metadata_mismatch is True

    def test_two_field_mismatch(self):
        """Title matches, 2+ other fields wrong → mismatch detected."""
        ref = _make_ref(authors=["John Smith"], year=2017)
        exist = _make_exist(matched_authors=["Alice Wang"], matched_year=2099)
        result = validate_metadata(ref, exist)
        assert result.has_metadata_mismatch is True

    def test_three_field_mismatch(self):
        ref = _make_ref(authors=["A"], year=2017, venue="X")
        exist = _make_exist(matched_authors=["B"], matched_year=2099, matched_venue="Y")
        result = validate_metadata(ref, exist)
        assert result.has_metadata_mismatch is True


class TestAuthorTruncation:
    """Author list truncation should be CLOSE_MATCH, not MISMATCH."""

    def test_truncated_authors_not_mismatch(self):
        """5 of 12 correct authors — truncation, not fabrication."""
        ref = _make_ref(authors=[
            "Tom Brown", "Benjamin Mann", "Nick Ryder",
            "Melanie Subbiah", "Jared Kaplan",
        ])
        exist = _make_exist(matched_authors=[
            "Tom Brown", "Benjamin Mann", "Nick Ryder",
            "Melanie Subbiah", "Jared Kaplan", "Prafulla Dhariwal",
            "Arvind Neelakantan", "Pranav Shyam", "Girish Sastry",
            "Amanda Askell", "Sandhini Agarwal", "Ariel Herbert-Voss",
        ])
        result = validate_metadata(ref, exist)
        auth_cmp = next(c for c in result.comparisons if c.field == "authors")
        assert auth_cmp.status == "CLOSE_MATCH"
        assert result.has_metadata_mismatch is False

    def test_blended_authors_still_mismatch(self):
        """3 of 5 correct, 2 fabricated — NOT truncation, IS fabrication."""
        ref = _make_ref(authors=[
            "Tom Brown", "Benjamin Mann", "Nick Ryder",
            "Fake Author One", "Fake Author Two",
        ])
        exist = _make_exist(matched_authors=[
            "Tom Brown", "Benjamin Mann", "Nick Ryder",
            "Melanie Subbiah", "Jared Kaplan", "Prafulla Dhariwal",
            "Arvind Neelakantan", "Pranav Shyam", "Girish Sastry",
            "Amanda Askell", "Sandhini Agarwal", "Ariel Herbert-Voss",
        ])
        result = validate_metadata(ref, exist)
        auth_cmp = next(c for c in result.comparisons if c.field == "authors")
        assert auth_cmp.status == "MISMATCH"
        assert result.has_metadata_mismatch is True


class TestVenuePreprint:
    """Preprint vs. publication venue should be CLOSE_MATCH, not MISMATCH."""

    def test_arxiv_vs_conference(self):
        """Ref says conference, DB says arXiv — same paper, different stage."""
        ref = _make_ref(venue="International Conference on Machine Learning")
        exist = _make_exist(matched_venue="arXiv (Cornell University)")
        result = validate_metadata(ref, exist)
        venue_cmp = next(c for c in result.comparisons if c.field == "venue")
        assert venue_cmp.status == "CLOSE_MATCH"
        assert result.has_metadata_mismatch is False

    def test_preprint_vs_journal(self):
        """Ref says arXiv preprint, DB says journal — same paper."""
        ref = _make_ref(venue="arXiv preprint")
        exist = _make_exist(matched_venue="Trans. Mach. Learn. Res.")
        result = validate_metadata(ref, exist)
        venue_cmp = next(c for c in result.comparisons if c.field == "venue")
        assert venue_cmp.status == "CLOSE_MATCH"
        assert result.has_metadata_mismatch is False

    def test_genuine_venue_mismatch(self):
        """Two different conferences — real mismatch."""
        ref = _make_ref(venue="Nature")
        exist = _make_exist(matched_venue="NeurIPS")
        result = validate_metadata(ref, exist)
        venue_cmp = next(c for c in result.comparisons if c.field == "venue")
        assert venue_cmp.status == "MISMATCH"
        assert result.has_metadata_mismatch is True


class TestMetadataScore:
    def test_perfect_score(self):
        result = validate_metadata(_make_ref(), _make_exist())
        assert result.metadata_score >= 0.9  # title + authors + venue all match

    def test_low_score_on_mismatches(self):
        ref = _make_ref(authors=["A"], venue="X")
        exist = _make_exist(matched_authors=["B"], matched_venue="Y")
        result = validate_metadata(ref, exist)
        assert result.metadata_score < 0.7
