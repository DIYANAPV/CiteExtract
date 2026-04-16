"""Tests for fuzzy matching utilities — pure unit tests, no network."""

import pytest

from src.verification.matching import (
    normalize_title,
    title_similarity,
    is_title_match,
    normalize_author,
    author_similarity,
    is_author_truncation,
    compare_year,
)


# ---- Title normalization ----

class TestNormalizeTitle:
    def test_basic(self):
        assert normalize_title("Attention Is All You Need") == "attention is all you need"

    def test_unicode(self):
        result = normalize_title("Résumé of Naïve Bayes")
        assert "resume" in result
        assert "naive" in result

    def test_keep_subtitle_when_main_part_short(self):
        # Short main part ("BERT") should NOT be stripped — it would lose meaning
        result = normalize_title("BERT: Pre-training of Transformers")
        assert "bert" in result
        assert "pre training" in result

    def test_strip_subtitle_when_main_part_long(self):
        # Long main part (>30 chars) with long subtitle CAN be stripped
        result = normalize_title(
            "A comprehensive analysis of deep learning methods: "
            "Applications in natural language processing and beyond"
        )
        assert "comprehensive" in result

    def test_version_marker(self):
        result = normalize_title("Some Paper v2")
        assert "v2" not in result

    def test_smart_quotes(self):
        a = normalize_title('\u201cSome Title\u201d')
        b = normalize_title('"Some Title"')
        assert a == b

    def test_em_dash(self):
        a = normalize_title("Word\u2014Another")
        assert "word" in a and "another" in a

    def test_empty(self):
        assert normalize_title("") == ""
        assert normalize_title(None) == ""  # type: ignore


# ---- Title similarity ----

class TestTitleSimilarity:
    def test_identical(self):
        assert title_similarity("Attention Is All You Need", "Attention Is All You Need") == 1.0

    def test_case_insensitive(self):
        assert title_similarity("attention is all you need", "ATTENTION IS ALL YOU NEED") == 1.0

    def test_minor_difference(self):
        sim = title_similarity("Attention Is All You Need", "Attention Is All We Need")
        assert 0.85 < sim < 1.0

    def test_completely_different(self):
        sim = title_similarity("Attention Is All You Need", "Random Forest Classification")
        assert sim < 0.5

    def test_empty(self):
        assert title_similarity("", "Something") == 0.0
        assert title_similarity("Something", "") == 0.0


# ---- Title match ----

class TestIsTitleMatch:
    def test_exact_match(self):
        matched, sim, flags = is_title_match("Attention Is All You Need", "Attention Is All You Need")
        assert matched is True
        assert sim == 1.0
        assert len(flags) == 0

    def test_near_match_flagged(self):
        matched, sim, flags = is_title_match(
            "Attention Is All You Need", "Attention Is All We Need"
        )
        assert matched is True
        assert sim < 1.0
        assert len(flags) == 1
        assert "title_not_exact_match" in flags[0]

    def test_below_threshold(self):
        matched, sim, flags = is_title_match(
            "Attention Is All You Need", "Random Forest for Classification"
        )
        assert matched is False

    def test_none_input(self):
        matched, sim, flags = is_title_match(None, "Something")
        assert matched is False


# ---- Author normalization ----

class TestNormalizeAuthor:
    def test_basic(self):
        assert normalize_author("Smith") == "smith"

    def test_hyphens_dots_spaces(self):
        assert normalize_author("J.-P. Doe") == "jpdoe"

    def test_empty(self):
        assert normalize_author("") == ""


# ---- Author similarity ----

class TestAuthorSimilarity:
    def test_identical(self):
        sim = author_similarity(
            ["John Smith", "Jane Doe"], ["John Smith", "Jane Doe"]
        )
        assert sim == 1.0

    def test_partial_overlap(self):
        sim = author_similarity(
            ["John Smith", "Jane Doe"], ["John Smith", "Bob Brown"]
        )
        assert 0.3 < sim < 0.7  # 1 out of 3 unique surnames match

    def test_no_overlap(self):
        sim = author_similarity(["Alice Wang"], ["Bob Brown"])
        assert sim == 0.0

    def test_initials_vs_full(self):
        # "Smith" is the surname in both cases
        sim = author_similarity(["J. Smith"], ["John Smith"])
        assert sim == 1.0  # both have surname "smith"

    def test_empty(self):
        assert author_similarity([], ["John Smith"]) == 0.0
        assert author_similarity(["John Smith"], []) == 0.0


# ---- Author truncation detection ----


class TestAuthorTruncation:
    """Tests for is_author_truncation — detecting BibTeX 'et al.' patterns.

    Truncation = True means the ref is a valid subset of the DB authors
    and the low Jaccard score is NOT evidence of fabrication.
    """

    # --- Legitimate truncation (should detect) ---

    def test_truncated_all_correct(self):
        """5 of 12 authors listed, all correct — classic truncation."""
        ref = ["Tom Brown", "Benjamin Mann", "Nick Ryder",
               "Melanie Subbiah", "Jared Kaplan"]
        db = ref + ["Prafulla Dhariwal", "Arvind Neelakantan",
                     "Pranav Shyam", "Girish Sastry", "Amanda Askell",
                     "Sandhini Agarwal", "Ariel Herbert-Voss"]
        assert is_author_truncation(ref, db) is True

    def test_truncated_name_format_differences(self):
        """Ref uses 'First Last', DB uses 'Last, First' — still a subset."""
        ref = ["Aaron Grattafiori", "Abhimanyu Dubey"]
        db = ["Grattafiori, Aaron", "Dubey, Abhimanyu", "Jauhri, Abhinav",
              "Pandey, Abhinav", "Kadian, Abhishek"]
        assert is_author_truncation(ref, db) is True

    def test_truncated_with_one_name_variation(self):
        """4 of 5 match, 1 has a transliteration issue — still truncation."""
        ref = ["Tom Brown", "Benjamin Mann", "Nick Ryder",
               "Melanie Subbiah", "Fake Author"]
        db = ref[:4] + ["Jared Kaplan", "Prafulla Dhariwal",
                         "Arvind Neelakantan", "Pranav Shyam",
                         "Girish Sastry", "Amanda Askell"]
        # containment = 4/5 = 0.80, exactly at threshold
        assert is_author_truncation(ref, db) is True

    # --- Fabrication (should NOT detect truncation) ---

    def test_blended_all_wrong(self):
        """All authors from a different paper — not truncation."""
        ref = ["Alice Wang", "Bob Chen", "Charlie Davis"]
        db = ["Tom Brown", "Benjamin Mann", "Nick Ryder",
              "Melanie Subbiah", "Jared Kaplan", "Prafulla Dhariwal",
              "Arvind Neelakantan"]
        assert is_author_truncation(ref, db) is False

    def test_blended_one_of_three(self):
        """Only 1 of 3 ref authors in DB — fabrication, not truncation."""
        ref = ["Tom Brown", "Alice Wang", "Bob Chen"]
        db = ["Tom Brown", "Benjamin Mann", "Nick Ryder",
              "Melanie Subbiah", "Jared Kaplan", "Prafulla Dhariwal",
              "Arvind Neelakantan"]
        assert is_author_truncation(ref, db) is False

    def test_blended_three_of_five(self):
        """3 of 5 correct, 2 fake — below 0.80 containment threshold."""
        ref = ["Tom Brown", "Benjamin Mann", "Nick Ryder",
               "Alice Wang", "Bob Chen"]
        db = ["Tom Brown", "Benjamin Mann", "Nick Ryder",
              "Melanie Subbiah", "Jared Kaplan", "Prafulla Dhariwal",
              "Arvind Neelakantan"]
        # containment = 3/5 = 0.60, below 0.80 threshold
        assert is_author_truncation(ref, db) is False

    # --- Not truncation by size (similar-sized lists) ---

    def test_similar_size_lists(self):
        """Even if all match, ratio < 2x — Jaccard handles these fine."""
        ref = ["John Smith", "Jane Doe"]
        db = ["John Smith", "Jane Doe", "Bob Brown"]
        assert is_author_truncation(ref, db) is False

    def test_empty_lists(self):
        assert is_author_truncation([], ["John Smith"]) is False
        assert is_author_truncation(["John Smith"], []) is False


# ---- Year comparison ----

class TestCompareYear:
    def test_match(self):
        result = compare_year(2020, 2020)
        assert result["match"] is True
        assert result["flag"] is None

    def test_close_match(self):
        """Year difference of 1 is a close match (preprint vs publication)."""
        result = compare_year(2023, 2024)
        assert result["match"] is True
        assert result["close_match"] is True
        assert "year_close_match" in result["flag"]
        assert "2023" in result["flag"]
        assert "2024" in result["flag"]

    def test_mismatch(self):
        """Year difference of 2+ is a hard mismatch."""
        result = compare_year(2020, 2024)
        assert result["match"] is False
        assert "year_mismatch" in result["flag"]
        assert "2020" in result["flag"]
        assert "2024" in result["flag"]

    def test_none_ref(self):
        result = compare_year(None, 2020)
        assert result["match"] is True  # can't compare, don't penalize
        assert result["flag"] is None

    def test_none_db(self):
        result = compare_year(2020, None)
        assert result["match"] is True
        assert result["flag"] is None
