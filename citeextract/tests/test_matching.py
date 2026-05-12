
import pytest

from citeextract.verification.matching import (
    normalize_title,
    title_similarity,
    is_title_match,
    normalize_author,
    author_similarity,
    is_author_truncation,
    compare_year,
)


class TestNormalizeTitle:
    def test_basic(self):
        assert normalize_title("Attention Is All You Need") == "attention is all you need"

    def test_unicode(self):
        result = normalize_title("Résumé of Naïve Bayes")
        assert "resume" in result
        assert "naive" in result

    def test_keep_subtitle_when_main_part_short(self):
        result = normalize_title("BERT: Pre-training of Transformers")
        assert "bert" in result
        assert "pre training" in result

    def test_strip_subtitle_when_main_part_long(self):
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
        assert normalize_title(None) == ""


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


class TestNormalizeAuthor:
    def test_basic(self):
        assert normalize_author("Smith") == "smith"

    def test_hyphens_dots_spaces(self):
        assert normalize_author("J.-P. Doe") == "jpdoe"

    def test_empty(self):
        assert normalize_author("") == ""


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
        assert 0.3 < sim < 0.7

    def test_no_overlap(self):
        sim = author_similarity(["Alice Wang"], ["Bob Brown"])
        assert sim == 0.0

    def test_initials_vs_full(self):
        sim = author_similarity(["J. Smith"], ["John Smith"])
        assert sim == 1.0

    def test_empty(self):
        assert author_similarity([], ["John Smith"]) == 0.0
        assert author_similarity(["John Smith"], []) == 0.0


class TestAuthorTruncation:


    def test_truncated_all_correct(self):
        ref = ["Tom Brown", "Benjamin Mann", "Nick Ryder",
               "Melanie Subbiah", "Jared Kaplan"]
        db = ref + ["Prafulla Dhariwal", "Arvind Neelakantan",
                     "Pranav Shyam", "Girish Sastry", "Amanda Askell",
                     "Sandhini Agarwal", "Ariel Herbert-Voss"]
        assert is_author_truncation(ref, db) is True

    def test_truncated_name_format_differences(self):
        ref = ["Aaron Grattafiori", "Abhimanyu Dubey"]
        db = ["Grattafiori, Aaron", "Dubey, Abhimanyu", "Jauhri, Abhinav",
              "Pandey, Abhinav", "Kadian, Abhishek"]
        assert is_author_truncation(ref, db) is True

    def test_truncated_with_one_name_variation(self):
        ref = ["Tom Brown", "Benjamin Mann", "Nick Ryder",
               "Melanie Subbiah", "Fake Author"]
        db = ref[:4] + ["Jared Kaplan", "Prafulla Dhariwal",
                         "Arvind Neelakantan", "Pranav Shyam",
                         "Girish Sastry", "Amanda Askell"]
        assert is_author_truncation(ref, db) is True


    def test_blended_all_wrong(self):
        ref = ["Alice Wang", "Bob Chen", "Charlie Davis"]
        db = ["Tom Brown", "Benjamin Mann", "Nick Ryder",
              "Melanie Subbiah", "Jared Kaplan", "Prafulla Dhariwal",
              "Arvind Neelakantan"]
        assert is_author_truncation(ref, db) is False

    def test_blended_one_of_three(self):
        ref = ["Tom Brown", "Alice Wang", "Bob Chen"]
        db = ["Tom Brown", "Benjamin Mann", "Nick Ryder",
              "Melanie Subbiah", "Jared Kaplan", "Prafulla Dhariwal",
              "Arvind Neelakantan"]
        assert is_author_truncation(ref, db) is False

    def test_blended_three_of_five(self):
        ref = ["Tom Brown", "Benjamin Mann", "Nick Ryder",
               "Alice Wang", "Bob Chen"]
        db = ["Tom Brown", "Benjamin Mann", "Nick Ryder",
              "Melanie Subbiah", "Jared Kaplan", "Prafulla Dhariwal",
              "Arvind Neelakantan"]
        assert is_author_truncation(ref, db) is False


    def test_similar_size_lists(self):
        ref = ["John Smith", "Jane Doe"]
        db = ["John Smith", "Jane Doe", "Bob Brown"]
        assert is_author_truncation(ref, db) is False

    def test_empty_lists(self):
        assert is_author_truncation([], ["John Smith"]) is False
        assert is_author_truncation(["John Smith"], []) is False


class TestCompareYear:
    def test_match(self):
        result = compare_year(2020, 2020)
        assert result["match"] is True
        assert result["flag"] is None

    def test_close_match(self):
        result = compare_year(2023, 2024)
        assert result["match"] is True
        assert result["close_match"] is True
        assert "year_close_match" in result["flag"]
        assert "2023" in result["flag"]
        assert "2024" in result["flag"]

    def test_mismatch(self):
        result = compare_year(2020, 2024)
        assert result["match"] is False
        assert "year_mismatch" in result["flag"]
        assert "2020" in result["flag"]
        assert "2024" in result["flag"]

    def test_none_ref(self):
        result = compare_year(None, 2020)
        assert result["match"] is True
        assert result["flag"] is None

    def test_none_db(self):
        result = compare_year(2020, None)
        assert result["match"] is True
        assert result["flag"] is None

    def test_diff_two_default_is_mismatch(self):
        result = compare_year(2020, 2022)
        assert result["match"] is False
        assert result["close_match"] is False
        assert "year_mismatch" in result["flag"]

    def test_diff_two_strong_match_relaxes_to_close(self):
        result = compare_year(2020, 2022, strong_match=True)
        assert result["match"] is True
        assert result["close_match"] is True
        assert "year_close_match" in result["flag"]
        assert "2-year gap" in result["flag"]

    def test_diff_one_strong_match_unchanged(self):
        result = compare_year(2023, 2024, strong_match=True)
        assert result["match"] is True
        assert result["close_match"] is True
        assert "year_close_match" in result["flag"]
        assert "preprint-vs-publication" in result["flag"]

    def test_diff_three_strong_match_still_mismatch(self):
        result = compare_year(2020, 2023, strong_match=True)
        assert result["match"] is False
        assert result["close_match"] is False
        assert "year_mismatch" in result["flag"]


class TestYearScore:

    def test_exact_year_perfect_score(self):
        from citeextract.verification.matching import _year_score
        assert _year_score(2020, 2020) == 1.0

    def test_one_year_off_high_score(self):
        from citeextract.verification.matching import _year_score
        assert _year_score(2023, 2024) == 0.8

    def test_two_years_off_partial(self):
        from citeextract.verification.matching import _year_score
        assert _year_score(2023, 2025) == 0.5

    def test_large_gap_zero_score(self):
        from citeextract.verification.matching import _year_score
        assert _year_score(2001, 2011) == 0.0

    def test_missing_year_neutral(self):
        from citeextract.verification.matching import _year_score
        assert _year_score(None, 2020) == 0.5
        assert _year_score(2020, None) == 0.5
        assert _year_score(None, None) == 0.5


class TestIsAuthorsDisjoint:

    def test_disjoint_authors(self):
        from citeextract.verification.matching import is_authors_disjoint
        assert is_authors_disjoint(
            ["Steven Bird", "Ewan Klein", "Edward Loper"],
            ["Dipanjan Sarkar"],
        ) is True

    def test_overlapping_authors(self):
        from citeextract.verification.matching import is_authors_disjoint
        assert is_authors_disjoint(
            ["A. Vaswani", "N. Shazeer"],
            ["Ashish Vaswani", "Noam Shazeer"],
        ) is False

    def test_either_side_empty_passes(self):
        from citeextract.verification.matching import is_authors_disjoint
        assert is_authors_disjoint([], ["A"]) is False
        assert is_authors_disjoint(["A"], []) is False


class TestCompositeMatchScore:

    def test_perfect_match_max_score(self):
        from citeextract.verification.matching import composite_match_score
        score = composite_match_score(
            "The Llama 3 herd of models", ["A. Dubey"], 2024,
            "The Llama 3 herd of models", ["A. Dubey"], 2024,
        )
        assert score >= 0.95

    def test_disjoint_authors_drag_score_below_threshold(self):
        from citeextract.verification.matching import (
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
        from citeextract.verification.matching import (
            COMPOSITE_MATCH_THRESHOLD, composite_match_score,
        )
        score = composite_match_score(
            "Some title", ["A. Smith"], 2023,
            "Some title", ["A. Smith"], 2024,
        )
        assert score >= COMPOSITE_MATCH_THRESHOLD

    def test_missing_authors_neutral(self):
        from citeextract.verification.matching import composite_match_score
        score = composite_match_score(
            "Title", [], 2020,
            "Title", ["A. Smith"], 2020,
        )
        assert score >= 0.80


class TestPickBestCandidate:

    def test_picks_composite_match_over_close_title_alternative(self):
        from citeextract.verification.matching import (
            MATCH_STRATEGY_COMPOSITE, pick_best_candidate,
        )
        candidates = [
            {
                "title": "Python for Natural Language Processing",
                "authors": ["Dipanjan Sarkar"], "year": 2019,
            },
            {
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
        from citeextract.verification.matching import pick_best_candidate
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
        if chosen is not None:
            from citeextract.verification.matching import MATCH_STRATEGY_TITLE_ONLY
            assert strategy == MATCH_STRATEGY_TITLE_ONLY

    def test_falls_back_to_title_only_when_authors_missing(self):
        from citeextract.verification.matching import (
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
        from citeextract.verification.matching import pick_best_candidate
        chosen, strategy, score = pick_best_candidate(
            "Looking for paper X", [], None,
            [{"title": "Totally Unrelated Paper", "authors": [], "year": 2020}],
        )
        assert chosen is None
        assert score == 0.0


class TestCanonicalId:

    def test_doi_url_strips_to_bare_doi(self):
        from citeextract.verification.matching import canonical_id, ID_SYSTEM_DOI
        assert canonical_id("https://doi.org/10.1234/foo") == \
            (ID_SYSTEM_DOI, "10.1234/foo")
        assert canonical_id("https://dx.doi.org/10.1234/foo") == \
            (ID_SYSTEM_DOI, "10.1234/foo")

    def test_doi_prefix_form(self):
        from citeextract.verification.matching import canonical_id, ID_SYSTEM_DOI
        assert canonical_id("doi:10.1234/foo") == \
            (ID_SYSTEM_DOI, "10.1234/foo")

    def test_doi_lowercased(self):
        from citeextract.verification.matching import canonical_id, ID_SYSTEM_DOI
        assert canonical_id("10.1234/Foo") == \
            (ID_SYSTEM_DOI, "10.1234/foo")

    def test_arxiv_url_with_version_suffix(self):
        from citeextract.verification.matching import canonical_id, ID_SYSTEM_ARXIV
        assert canonical_id("https://arxiv.org/abs/2407.21783v2") == \
            (ID_SYSTEM_ARXIV, "2407.21783")
        assert canonical_id("arxiv.org/pdf/2407.21783") == \
            (ID_SYSTEM_ARXIV, "2407.21783")

    def test_arxiv_doi_collapses_to_arxiv_id(self):
        from citeextract.verification.matching import canonical_id, ID_SYSTEM_ARXIV
        assert canonical_id("10.48550/arxiv.2407.21783") == \
            (ID_SYSTEM_ARXIV, "2407.21783")

    def test_acl_url(self):
        from citeextract.verification.matching import canonical_id, ID_SYSTEM_ACL
        assert canonical_id("https://aclanthology.org/Q16-1026") == \
            (ID_SYSTEM_ACL, "q16-1026")
        assert canonical_id("aclanthology.org/2020.acl-main.123/") == \
            (ID_SYSTEM_ACL, "2020.acl-main.123")

    def test_acl_doi_collapses_to_acl_id(self):
        from citeextract.verification.matching import canonical_id, ID_SYSTEM_ACL
        assert canonical_id("10.18653/v1/Q16-1026") == \
            (ID_SYSTEM_ACL, "q16-1026")
        assert canonical_id("10.18653/v1/2020.acl-main.123") == \
            (ID_SYSTEM_ACL, "2020.acl-main.123")

    def test_unrecognized_returns_none(self):
        from citeextract.verification.matching import canonical_id
        assert canonical_id("") is None
        assert canonical_id(None) is None
        assert canonical_id("just some text") is None

    def test_strips_trailing_punctuation(self):
        from citeextract.verification.matching import canonical_id, ID_SYSTEM_DOI
        assert canonical_id("10.1234/foo.") == (ID_SYSTEM_DOI, "10.1234/foo")
        assert canonical_id("10.1234/foo,") == (ID_SYSTEM_DOI, "10.1234/foo")


class TestIdsEquivalent:

    def test_url_vs_bare_doi(self):
        from citeextract.verification.matching import ids_equivalent
        assert ids_equivalent(
            "https://doi.org/10.1234/foo", "10.1234/foo"
        ) is True

    def test_arxiv_id_vs_arxiv_doi(self):
        from citeextract.verification.matching import ids_equivalent
        assert ids_equivalent(
            "2407.21783", "10.48550/arxiv.2407.21783"
        ) is True

    def test_acl_url_vs_acl_doi(self):
        from citeextract.verification.matching import ids_equivalent
        assert ids_equivalent(
            "https://aclanthology.org/q16-1026", "10.18653/v1/Q16-1026"
        ) is True

    def test_different_papers_different_dois(self):
        from citeextract.verification.matching import ids_equivalent
        assert ids_equivalent("10.1234/foo", "10.5678/bar") is False

    def test_acl_url_vs_unrelated_doi_does_not_match(self):
        from citeextract.verification.matching import ids_equivalent
        assert ids_equivalent(
            "https://aclanthology.org/q16-1026", "10.1162/tacla00104"
        ) is False

    def test_empty_inputs_safe(self):
        from citeextract.verification.matching import ids_equivalent
        assert ids_equivalent("", "10.1234/foo") is False
        assert ids_equivalent(None, None) is False


class TestMatchesAnyKnownId:

    def test_matches_against_anthology_id_field(self):
        from citeextract.verification.matching import matches_any_known_id
        db_record = {
            "doi": "10.1162/tacla00104",
            "anthology_id": "Q16-1026",
        }
        assert matches_any_known_id(
            "https://aclanthology.org/q16-1026", db_record
        ) is True

    def test_matches_against_arxiv_id_field(self):
        from citeextract.verification.matching import matches_any_known_id
        db_record = {
            "doi": "10.1109/journal.123",
            "arxiv_id": "2407.21783",
        }
        assert matches_any_known_id("2407.21783", db_record) is True

    def test_no_match_when_id_not_in_record(self):
        from citeextract.verification.matching import matches_any_known_id
        db_record = {"doi": "10.1234/foo", "arxiv_id": None}
        assert matches_any_known_id("10.5678/different", db_record) is False

    def test_unrecognized_id_returns_false(self):
        from citeextract.verification.matching import matches_any_known_id
        assert matches_any_known_id("not-an-id", {"doi": "10.1234/foo"}) is False

    def test_empty_record_returns_false(self):
        from citeextract.verification.matching import matches_any_known_id
        assert matches_any_known_id("10.1234/foo", {}) is False
