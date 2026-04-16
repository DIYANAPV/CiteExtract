"""Tests for format-aware canonical author comparison."""

import pytest

from src.verification.author_comparison import (
    canonicalize_author,
    compare_authors,
    _extract_surname_and_initial,
)


# ---------------------------------------------------------------------------
# Canonicalization tests — different name formats
# ---------------------------------------------------------------------------

class TestCanonicalization:
    """Test that author names from different formats canonicalize correctly."""

    def test_apa_format(self):
        # APA: "Surname, A. B."
        c = canonicalize_author("Huang, H.")
        assert c.surname == "huang"
        assert c.first_initial == "h"

    def test_apa_full_initials(self):
        c = canonicalize_author("Smith, J. A.")
        assert c.surname == "smith"
        assert c.first_initial == "j"

    def test_vancouver_format(self):
        # Vancouver: "Surname AB"
        c = canonicalize_author("Huang H")
        assert c.surname == "huang"
        assert c.first_initial == "h"

    def test_vancouver_two_initials(self):
        c = canonicalize_author("LeCun YA")
        assert c.surname == "lecun"
        assert c.first_initial == "y"

    def test_ieee_format(self):
        # IEEE: "A. B. Surname"
        c = canonicalize_author("H. Huang")
        assert c.surname == "huang"
        assert c.first_initial == "h"

    def test_ieee_two_initials(self):
        c = canonicalize_author("Y. A. LeCun")
        assert c.surname == "lecun"
        assert c.first_initial == "y"

    def test_chicago_format(self):
        # Chicago: "Surname, Firstname"
        c = canonicalize_author("Huang, Hai")
        assert c.surname == "huang"
        assert c.first_initial == "h"

    def test_western_order(self):
        # "Firstname Surname"
        c = canonicalize_author("Hai Huang")
        assert c.surname == "huang"
        assert c.first_initial == "h"

    def test_full_name_western(self):
        c = canonicalize_author("Yann LeCun")
        assert c.surname == "lecun"
        assert c.first_initial == "y"

    def test_single_name(self):
        c = canonicalize_author("Bengio")
        assert c.surname == "bengio"
        assert c.first_initial == ""

    def test_unicode_author(self):
        c = canonicalize_author("Müller, K.")
        assert c.surname == "muller"
        assert c.first_initial == "k"

    def test_hyphenated_surname(self):
        c = canonicalize_author("García-López, M.")
        assert c.surname == "garcialopez"
        assert c.first_initial == "m"

    def test_bibtex_format(self):
        # BibTeX: "Last, First"
        c = canonicalize_author("LeCun, Yann")
        assert c.surname == "lecun"
        assert c.first_initial == "y"


# ---------------------------------------------------------------------------
# Cross-format matching — same person in different formats
# ---------------------------------------------------------------------------

class TestCrossFormatMatching:
    """Test that the same person in different citation formats gets matched."""

    def test_vancouver_vs_fullname(self):
        result = compare_authors(
            ["Huang H", "LeCun Y", "Balestriero R"],
            ["Hai Huang", "Yann LeCun", "Randall Balestriero"],
        )
        assert result.matched_count == 3
        assert result.unmatched_count == 0

    def test_apa_vs_fullname(self):
        result = compare_authors(
            ["Huang, H.", "LeCun, Y.", "Balestriero, R."],
            ["Hai Huang", "Yann LeCun", "Randall Balestriero"],
        )
        assert result.matched_count == 3
        assert result.unmatched_count == 0

    def test_ieee_vs_fullname(self):
        result = compare_authors(
            ["H. Huang", "Y. LeCun", "R. Balestriero"],
            ["Hai Huang", "Yann LeCun", "Randall Balestriero"],
        )
        assert result.matched_count == 3
        assert result.unmatched_count == 0

    def test_bibtex_vs_database(self):
        result = compare_authors(
            ["Huang, Hai", "LeCun, Yann"],
            ["Hai Huang", "Yann LeCun"],
        )
        assert result.matched_count == 2
        assert result.unmatched_count == 0


# ---------------------------------------------------------------------------
# Truncation detection
# ---------------------------------------------------------------------------

class TestTruncation:
    def test_truncated_list(self):
        result = compare_authors(
            ["Brown, T.", "Mann, B.", "Ryder, N."],
            ["Tom Brown", "Benjamin Mann", "Nick Ryder", "Melanie Subbiah",
             "Jared Kaplan", "Prafulla Dhariwal", "Arvind Neelakantan",
             "Pranav Shyam", "Girish Sastry", "Amanda Askell"],
        )
        assert result.matched_count == 3
        assert result.is_truncated is True

    def test_truncation_expected_apa(self):
        # APA allows et al. after 20 authors
        ref = ["Smith, J."]
        db = ["John Smith"] + [f"Author{i} Name{i}" for i in range(25)]
        result = compare_authors(ref, db, citation_format="apa")
        assert result.truncation_expected is True
        assert "APA" in result.explanation

    def test_truncation_expected_vancouver(self):
        # Vancouver allows et al. after 6 authors
        ref = ["Smith J", "Jones A"]
        db = ["John Smith", "Alice Jones", "Bob Brown", "Carol Davis",
              "Dan Evans", "Eve Foster", "Frank Green", "Grace Hall"]
        result = compare_authors(ref, db, citation_format="vancouver")
        assert result.truncation_expected is True
        assert "Vancouver" in result.explanation

    def test_no_truncation_when_equal(self):
        result = compare_authors(
            ["Smith, J.", "Jones, A."],
            ["John Smith", "Alice Jones"],
        )
        assert result.is_truncated is False


# ---------------------------------------------------------------------------
# Unmatched authors (potential fabrication signal)
# ---------------------------------------------------------------------------

class TestUnmatchedAuthors:
    def test_completely_wrong_authors(self):
        result = compare_authors(
            ["Zhang, W.", "Chen, L.", "Wang, H."],
            ["John Smith", "Alice Jones", "Bob Brown"],
        )
        assert result.unmatched_count == 3
        assert result.matched_count == 0

    def test_partially_wrong(self):
        result = compare_authors(
            ["Smith, J.", "FakeAuthor, X.", "Jones, A."],
            ["John Smith", "Alice Jones", "Bob Brown"],
        )
        assert result.matched_count == 2
        assert result.unmatched_count == 1
        unmatched = [m for m in result.per_author if m.status == "unmatched"]
        assert unmatched[0].ref_author == "FakeAuthor, X."

    def test_empty_ref_authors(self):
        result = compare_authors([], ["John Smith", "Alice Jones"])
        assert result.matched_count == 0
        assert result.ref_count == 0


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_dict_structure(self):
        result = compare_authors(
            ["Huang H", "LeCun Y"],
            ["Hai Huang", "Yann LeCun", "Randall Balestriero"],
            citation_format="vancouver",
        )
        d = result.to_dict()
        assert d["matched"] == 2
        assert d["unmatched"] == 0
        assert d["ref_count"] == 2
        assert d["db_count"] == 3
        assert d["format_used"] == "vancouver"
        assert len(d["per_author"]) == 2
        assert all(a["status"] == "match" for a in d["per_author"])

    def test_to_dict_with_unmatched(self):
        result = compare_authors(
            ["Smith, J.", "Fake, X."],
            ["John Smith"],
        )
        d = result.to_dict()
        assert "unmatched_authors" in d
        assert "Fake, X." in d["unmatched_authors"]
