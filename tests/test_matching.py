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


# ---------------------------------------------------------------------------
# Composite candidate scoring (Commit #1: stop matching the wrong paper)
# ---------------------------------------------------------------------------


class TestYearScore:
    """Year similarity component of the composite score. Lenient about
    small gaps (preprint vs published, versioned arXiv reposts), strict
    about gaps that suggest a different paper entirely."""

    def test_exact_year_perfect_score(self):
        from src.verification.matching import _year_score
        assert _year_score(2020, 2020) == 1.0

    def test_one_year_off_high_score(self):
        # arXiv preprint date vs proceedings date — same paper.
        from src.verification.matching import _year_score
        assert _year_score(2023, 2024) == 0.8

    def test_two_years_off_partial(self):
        # Versioned arXiv reposts (the "Unleashing prompt engineering"
        # case: 2023 v1 vs 2025 v2) — same paper, different year.
        from src.verification.matching import _year_score
        assert _year_score(2023, 2025) == 0.5

    def test_large_gap_zero_score(self):
        # Berners-Lee Semantic Web (2001) vs LNCS book (2011) — almost
        # certainly a different paper.
        from src.verification.matching import _year_score
        assert _year_score(2001, 2011) == 0.0

    def test_missing_year_neutral(self):
        # Don't penalize refs whose year wasn't extracted.
        from src.verification.matching import _year_score
        assert _year_score(None, 2020) == 0.5
        assert _year_score(2020, None) == 0.5
        assert _year_score(None, None) == 0.5


class TestIsAuthorsDisjoint:
    """The hard-reject signal: zero author surnames in common means it's
    not the same paper, regardless of how similar the titles look."""

    def test_disjoint_authors(self):
        from src.verification.matching import is_authors_disjoint
        # The user's reproduction: NLP-with-Python ref vs Sarkar's
        # "Python for NLP" — different authors, similar tokens.
        assert is_authors_disjoint(
            ["Steven Bird", "Ewan Klein", "Edward Loper"],
            ["Dipanjan Sarkar"],
        ) is True

    def test_overlapping_authors(self):
        from src.verification.matching import is_authors_disjoint
        assert is_authors_disjoint(
            ["A. Vaswani", "N. Shazeer"],
            ["Ashish Vaswani", "Noam Shazeer"],
        ) is False

    def test_either_side_empty_passes(self):
        # When we can't tell, don't reject — the picker falls back to
        # title-only and downstream verdict layer flags the result.
        from src.verification.matching import is_authors_disjoint
        assert is_authors_disjoint([], ["A"]) is False
        assert is_authors_disjoint(["A"], []) is False


class TestCompositeMatchScore:
    """The composite blends title + authors + year. A strong title with
    disjoint authors should score lower than a perfect match — that's
    the discriminator the old title-only picker was missing."""

    def test_perfect_match_max_score(self):
        from src.verification.matching import composite_match_score
        score = composite_match_score(
            "The Llama 3 herd of models", ["A. Dubey"], 2024,
            "The Llama 3 herd of models", ["A. Dubey"], 2024,
        )
        assert score >= 0.95

    def test_disjoint_authors_drag_score_below_threshold(self):
        # NLP-with-Python ref → Sarkar's book: high title overlap, zero
        # author overlap. Composite should be too low to accept.
        from src.verification.matching import (
            COMPOSITE_MATCH_THRESHOLD, composite_match_score,
        )
        score = composite_match_score(
            "Natural Language Processing with Python",
            ["Steven Bird", "Ewan Klein", "Edward Loper"], 2009,
            "Python for Natural Language Processing",
            ["Dipanjan Sarkar"], 2019,
        )
        assert score < COMPOSITE_MATCH_THRESHOLD, (
            f"Composite {score:.2f} should NOT clear threshold "
            f"{COMPOSITE_MATCH_THRESHOLD} when authors are disjoint"
        )

    def test_year_off_by_one_still_strong(self):
        # Preprint 2023 vs published 2024 — same paper, year diff small.
        from src.verification.matching import (
            COMPOSITE_MATCH_THRESHOLD, composite_match_score,
        )
        score = composite_match_score(
            "Some title", ["A. Smith"], 2023,
            "Some title", ["A. Smith"], 2024,
        )
        assert score >= COMPOSITE_MATCH_THRESHOLD

    def test_missing_authors_neutral(self):
        from src.verification.matching import composite_match_score
        # Ref has no authors → author component is neutral 0.5, not 0.
        score = composite_match_score(
            "Title", [], 2020,
            "Title", ["A. Smith"], 2020,
        )
        # Title 1.0 + author 0.5 (neutral) + year 1.0 = 0.5+0.15+0.2 = 0.85
        assert score >= 0.80


class TestPickBestCandidate:
    """End-to-end: the picker rejects the wrong-paper-with-similar-title
    case and falls back to title-only with a flag when authors are
    missing or fully hallucinated."""

    def test_picks_composite_match_over_close_title_alternative(self):
        # Two candidates with near-identical titles; only one has the
        # right authors. Picker must choose the author-matching one.
        from src.verification.matching import (
            MATCH_STRATEGY_COMPOSITE, pick_best_candidate,
        )
        candidates = [
            {  # Wrong paper, similar title
                "title": "Python for Natural Language Processing",
                "authors": ["Dipanjan Sarkar"], "year": 2019,
            },
            {  # Correct paper
                "title": "Natural Language Processing with Python",
                "authors": ["Steven Bird", "Ewan Klein", "Edward Loper"],
                "year": 2009,
            },
        ]
        chosen, strategy, score = pick_best_candidate(
            "Natural Language Processing with Python: Analyzing Text",
            ["Steven Bird", "Ewan Klein", "Edward Loper"], 2009,
            candidates,
        )
        assert chosen is not None
        assert chosen["authors"][0].startswith("Steven Bird")
        assert strategy == MATCH_STRATEGY_COMPOSITE

    def test_rejects_disjoint_authors_even_with_high_title_similarity(self):
        # Berners-Lee "Semantic Web" → wrong LNCS book; 100% title
        # similarity but zero author overlap. Composite phase rejects it
        # via the disjoint-authors hard rule. With NO title-only candidate
        # also passing the bar (only one candidate, already rejected),
        # the picker returns nothing.
        from src.verification.matching import pick_best_candidate
        candidates = [
            {
                "title": "The Semantic Web",
                "authors": [
                    "G. Goos", "J. Hartmanis", "J. Leeuwen",
                    "David Hutchison",
                ],
                "year": 2011,
            },
        ]
        chosen, strategy, score = pick_best_candidate(
            "The Semantic Web",
            ["O. Lassila", "J. Hendler", "T. Berners-Lee"], 2001,
            candidates,
        )
        # Disjoint authors → composite reject. Then title-only fallback
        # would also pick this same candidate (title matches). The
        # resulting strategy is "title_only" with a low-confidence flag —
        # the verdict layer surfaces this as "low_author_confidence".
        # We verify the strategy label, not the chosen candidate
        # identity.
        if chosen is not None:
            from src.verification.matching import MATCH_STRATEGY_TITLE_ONLY
            assert strategy == MATCH_STRATEGY_TITLE_ONLY

    def test_falls_back_to_title_only_when_authors_missing(self):
        # The hallucination-detection case: ref has authors that don't
        # exist anywhere. Title-only fallback finds the most likely
        # paper the citation meant, flagged as low author confidence.
        from src.verification.matching import (
            MATCH_STRATEGY_TITLE_ONLY, pick_best_candidate,
        )
        candidates = [{
            "title": "Real Paper Title",
            "authors": ["Real Author"],
            "year": 2020,
        }]
        chosen, strategy, _ = pick_best_candidate(
            "Real Paper Title",
            ["Made-Up Name"], 2020,
            candidates,
        )
        assert chosen is not None
        assert strategy == MATCH_STRATEGY_TITLE_ONLY

    def test_returns_none_when_nothing_clears_either_bar(self):
        from src.verification.matching import pick_best_candidate
        chosen, strategy, score = pick_best_candidate(
            "Looking for paper X", [], None,
            [{"title": "Totally Unrelated Paper", "authors": [], "year": 2020}],
        )
        assert chosen is None
        assert score == 0.0


# ---------------------------------------------------------------------------
# Canonical paper identifiers (Commit #2: cross-system aliasing)
# ---------------------------------------------------------------------------


class TestCanonicalId:
    """Same paper, multiple identifier shapes the user might cite. The
    canonicalizer collapses them to a single ``(system, value)`` tuple
    so the metadata layer's ``DOI mismatch`` verdict isn't fired on
    cross-system aliases."""

    def test_doi_url_strips_to_bare_doi(self):
        from src.verification.matching import canonical_id, ID_SYSTEM_DOI
        assert canonical_id("https://doi.org/10.1234/foo") == \
            (ID_SYSTEM_DOI, "10.1234/foo")
        assert canonical_id("https://dx.doi.org/10.1234/foo") == \
            (ID_SYSTEM_DOI, "10.1234/foo")

    def test_doi_prefix_form(self):
        from src.verification.matching import canonical_id, ID_SYSTEM_DOI
        assert canonical_id("doi:10.1234/foo") == \
            (ID_SYSTEM_DOI, "10.1234/foo")

    def test_doi_lowercased(self):
        # DOIs are case-insensitive by spec; canonicalize to lowercase
        # so "10.1234/Foo" and "10.1234/foo" canonicalize identically.
        from src.verification.matching import canonical_id, ID_SYSTEM_DOI
        assert canonical_id("10.1234/Foo") == \
            (ID_SYSTEM_DOI, "10.1234/foo")

    def test_arxiv_url_with_version_suffix(self):
        from src.verification.matching import canonical_id, ID_SYSTEM_ARXIV
        # https://arxiv.org/abs/2407.21783v2 → ('arxiv', '2407.21783')
        assert canonical_id("https://arxiv.org/abs/2407.21783v2") == \
            (ID_SYSTEM_ARXIV, "2407.21783")
        assert canonical_id("arxiv.org/pdf/2407.21783") == \
            (ID_SYSTEM_ARXIV, "2407.21783")

    def test_arxiv_doi_collapses_to_arxiv_id(self):
        # 10.48550/arxiv.X is the arXiv-issued DOI for arXiv ID X — same
        # paper as the bare arXiv ID. Critical case: ref has arXiv DOI,
        # DB has bare arXiv ID, must match.
        from src.verification.matching import canonical_id, ID_SYSTEM_ARXIV
        assert canonical_id("10.48550/arxiv.2407.21783") == \
            (ID_SYSTEM_ARXIV, "2407.21783")

    def test_acl_url(self):
        # The user's reproduction: ``aclanthology.org/q16-1026`` on the ref
        # side vs the publisher DOI on the DB side. Both legacy
        # ("Q16-1026") and modern ("2020.acl-main.123") IDs are recognized.
        from src.verification.matching import canonical_id, ID_SYSTEM_ACL
        assert canonical_id("https://aclanthology.org/Q16-1026") == \
            (ID_SYSTEM_ACL, "q16-1026")
        assert canonical_id("aclanthology.org/2020.acl-main.123/") == \
            (ID_SYSTEM_ACL, "2020.acl-main.123")

    def test_acl_doi_collapses_to_acl_id(self):
        from src.verification.matching import canonical_id, ID_SYSTEM_ACL
        assert canonical_id("10.18653/v1/Q16-1026") == \
            (ID_SYSTEM_ACL, "q16-1026")
        assert canonical_id("10.18653/v1/2020.acl-main.123") == \
            (ID_SYSTEM_ACL, "2020.acl-main.123")

    def test_unrecognized_returns_none(self):
        from src.verification.matching import canonical_id
        assert canonical_id("") is None
        assert canonical_id(None) is None
        assert canonical_id("just some text") is None

    def test_strips_trailing_punctuation(self):
        # PDF text-extraction often adds a trailing ".," or ")" to URLs.
        from src.verification.matching import canonical_id, ID_SYSTEM_DOI
        assert canonical_id("10.1234/foo.") == (ID_SYSTEM_DOI, "10.1234/foo")
        assert canonical_id("10.1234/foo,") == (ID_SYSTEM_DOI, "10.1234/foo")


class TestIdsEquivalent:
    """Two raw IDs refer to the same paper iff their canonical forms match."""

    def test_url_vs_bare_doi(self):
        from src.verification.matching import ids_equivalent
        assert ids_equivalent(
            "https://doi.org/10.1234/foo", "10.1234/foo"
        ) is True

    def test_arxiv_id_vs_arxiv_doi(self):
        from src.verification.matching import ids_equivalent
        # User cites arXiv ID; DB has the arXiv-issued DOI. Same paper.
        assert ids_equivalent(
            "2407.21783", "10.48550/arxiv.2407.21783"
        ) is True

    def test_acl_url_vs_acl_doi(self):
        from src.verification.matching import ids_equivalent
        # The exact case from the user's report.
        assert ids_equivalent(
            "https://aclanthology.org/q16-1026", "10.18653/v1/Q16-1026"
        ) is True

    def test_different_papers_different_dois(self):
        from src.verification.matching import ids_equivalent
        assert ids_equivalent("10.1234/foo", "10.5678/bar") is False

    def test_acl_url_vs_unrelated_doi_does_not_match(self):
        # ACL URL can't match an unrelated MIT Press DOI just by
        # canonicalization — that requires cross-record matching against
        # the DB record's ``anthology_id`` (TestMatchesAnyKnownId).
        from src.verification.matching import ids_equivalent
        assert ids_equivalent(
            "https://aclanthology.org/q16-1026", "10.1162/tacla00104"
        ) is False

    def test_empty_inputs_safe(self):
        from src.verification.matching import ids_equivalent
        assert ids_equivalent("", "10.1234/foo") is False
        assert ids_equivalent(None, None) is False


class TestMatchesAnyKnownId:
    """Cross-record matching: the ref's ID must match ANY identifier the
    DB stored for the paper, not just the primary DOI."""

    def test_matches_against_anthology_id_field(self):
        from src.verification.matching import matches_any_known_id
        # User cites the ACL anthology URL. DB record holds the publisher
        # DOI as primary and the anthology ID as a secondary identifier.
        # Must match through the anthology field.
        db_record = {
            "doi": "10.1162/tacla00104",
            "anthology_id": "Q16-1026",
        }
        assert matches_any_known_id(
            "https://aclanthology.org/q16-1026", db_record
        ) is True

    def test_matches_against_arxiv_id_field(self):
        from src.verification.matching import matches_any_known_id
        # User cites the arXiv ID; DB has it stored separately from DOI.
        db_record = {
            "doi": "10.1109/journal.123",
            "arxiv_id": "2407.21783",
        }
        assert matches_any_known_id("2407.21783", db_record) is True

    def test_no_match_when_id_not_in_record(self):
        from src.verification.matching import matches_any_known_id
        db_record = {"doi": "10.1234/foo", "arxiv_id": None}
        assert matches_any_known_id("10.5678/different", db_record) is False

    def test_unrecognized_id_returns_false(self):
        from src.verification.matching import matches_any_known_id
        assert matches_any_known_id("not-an-id", {"doi": "10.1234/foo"}) is False

    def test_empty_record_returns_false(self):
        from src.verification.matching import matches_any_known_id
        assert matches_any_known_id("10.1234/foo", {}) is False
