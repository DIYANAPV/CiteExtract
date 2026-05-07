
import pytest

from citeextract.models.citation import Citation
from citeextract.models.reference import Reference
from citeextract.parsers.marker_rule_check import (
    annotate_citation_confidence,
    compute_rule_ref_id,
    normalize_marker_key,
    numeric_fallback,
    parse_marker_surface,
    rule_vote,
    unique_rule_match,
)


def _ref(ref_id, authors=None, year=None):
    return Reference(
        ref_id=ref_id,
        authors=authors or [],
        year=year,
        source_format="test",
    )


class TestParseMarkerSurface:
    def test_numeric_single(self):
        s = parse_marker_surface("[5]")
        assert s.num == 5 and s.surname is None and s.year is None

    def test_numeric_list(self):
        s = parse_marker_surface("[1, 2, 3]")
        assert s.num == 1

    def test_numeric_range(self):
        s = parse_marker_surface("[3-5]")
        assert s.num == 3

    def test_numeric_bare(self):
        s = parse_marker_surface("7")
        assert s.num == 7

    def test_author_year_paren(self):
        s = parse_marker_surface("(Smith, 2020)")
        assert s.surname == "smith" and s.year == 2020

    def test_author_year_et_al(self):
        s = parse_marker_surface("(Smith et al., 2020)")
        assert s.surname == "smith" and s.year == 2020

    def test_harvard_no_comma(self):
        s = parse_marker_surface("(Smith et al. 2020)")
        assert s.surname == "smith" and s.year == 2020

    def test_narrative(self):
        s = parse_marker_surface("Smith (2020)")
        assert s.surname == "smith" and s.year == 2020

    def test_narrative_et_al(self):
        s = parse_marker_surface("Smith et al. (2020)")
        assert s.surname == "smith" and s.year == 2020

    def test_two_authors_and(self):
        s = parse_marker_surface("Smith and Jones (2020)")
        assert s.surname == "smith" and s.year == 2020

    def test_two_authors_ampersand(self):
        s = parse_marker_surface("Smith & Jones, 2020")
        assert s.surname == "smith" and s.year == 2020

    def test_year_suffix_letter(self):
        s = parse_marker_surface("(Smith, 2020a)")
        assert s.surname == "smith" and s.year == 2020

    def test_unicode_surname(self):
        s = parse_marker_surface("(Müller, 2019)")
        assert s.surname == "müller" and s.year == 2019

    def test_non_citation_bracketed(self):
        s = parse_marker_surface("[sic]")
        assert s.year is None

    def test_empty(self):
        s = parse_marker_surface("")
        assert s.surname is None and s.year is None and s.num is None


class TestRuleVote:
    def test_match_both(self):
        ref = _ref("1", authors=["John Smith"], year=2020)
        assert rule_vote("(Smith, 2020)", ref) is True

    def test_year_mismatch(self):
        ref = _ref("1", authors=["John Smith"], year=2019)
        assert rule_vote("(Smith, 2020)", ref) is False

    def test_surname_mismatch(self):
        ref = _ref("1", authors=["Jane Jones"], year=2020)
        assert rule_vote("(Smith, 2020)", ref) is False

    def test_numeric_position_match(self):
        ref = _ref("5")
        assert rule_vote("[5]", ref) is True

    def test_numeric_position_mismatch(self):
        ref = _ref("3")
        assert rule_vote("[5]", ref) is False

    def test_ref_missing_year_year_in_marker(self):
        ref = _ref("1", authors=["John Smith"], year=None)
        assert rule_vote("(Smith, 2020)", ref) is True

    def test_ref_missing_authors(self):
        ref = _ref("1", authors=[], year=2020)
        assert rule_vote("(Smith, 2020)", ref) is True

    def test_ref_missing_both(self):
        ref = _ref("1", authors=[], year=None)
        assert rule_vote("(Smith, 2020)", ref) is None

    def test_ref_none(self):
        assert rule_vote("(Smith, 2020)", None) is None

    def test_multi_author_first_surname_matches(self):
        ref = _ref("1", authors=["John Smith", "Jane Jones"], year=2020)
        assert rule_vote("Smith and Jones (2020)", ref) is True

    def test_et_al_match(self):
        ref = _ref("1", authors=["John Smith", "Jane Jones", "Al Brown"], year=2020)
        assert rule_vote("(Smith et al., 2020)", ref) is True


class TestUniqueRuleMatch:
    def test_single_hit(self):
        refs = {
            "1": _ref("1", authors=["John Smith"], year=2020),
            "2": _ref("2", authors=["Jane Jones"], year=2019),
        }
        assert unique_rule_match("(Smith, 2020)", refs) == "1"

    def test_zero_hits(self):
        refs = {"1": _ref("1", authors=["Jane Jones"], year=2019)}
        assert unique_rule_match("(Smith, 2020)", refs) is None

    def test_ambiguous_two_hits(self):
        refs = {
            "1": _ref("1", authors=["John Smith"], year=2020),
            "2": _ref("2", authors=["Alice Smith"], year=2020),
        }
        assert unique_rule_match("(Smith, 2020)", refs) is None


class TestNumericFallback:
    def test_in_range(self):
        refs = [_ref("a"), _ref("b"), _ref("c")]
        assert numeric_fallback("[2]", refs) == "b"

    def test_out_of_range(self):
        refs = [_ref("a"), _ref("b")]
        assert numeric_fallback("[5]", refs) is None

    def test_non_numeric(self):
        refs = [_ref("a")]
        assert numeric_fallback("(Smith, 2020)", refs) is None


class TestComputeRuleRefId:
    def test_author_year_unique(self):
        refs = {
            "1": _ref("1", authors=["John Smith"], year=2020),
            "2": _ref("2", authors=["Jane Jones"], year=2019),
        }
        assert compute_rule_ref_id("(Smith, 2020)", refs) == "1"

    def test_numeric_positional(self):
        refs_list = [_ref("a"), _ref("b"), _ref("c")]
        refs_by_id = {r.ref_id: r for r in refs_list}
        assert compute_rule_ref_id("[2]", refs_by_id, refs_list) == "b"

    def test_unresolvable(self):
        refs = {"1": _ref("1", authors=["Jane Jones"], year=2019)}
        assert compute_rule_ref_id("(Smith, 2020)", refs) is None


class TestNormalizeMarkerKey:
    def test_trims_whitespace(self):
        assert normalize_marker_key("  (Smith, 2020) ") == "(smith, 2020)"

    def test_collapses_inner_whitespace(self):
        assert normalize_marker_key("(Smith   et  al., 2020)") == "(smith et al., 2020)"

    def test_unicode_quotes(self):
        assert normalize_marker_key("(O’Brien, 2020)") == "(o'brien, 2020)"

    def test_em_dash(self):
        assert normalize_marker_key("[1—2]") == "[1-2]"

    def test_empty(self):
        assert normalize_marker_key("") == ""


def _cite(ref_id, marker):
    return Citation(ref_id=ref_id, citing_sentence="x", marker=marker)


class TestAnnotateCitationConfidence:
    def test_empty_citations(self):
        assert annotate_citation_confidence([], {}) == []

    def test_all_three_agree_high(self):
        refs = {"1": _ref("1", authors=["John Smith"], year=2020)}
        c = _cite("1", "(Smith, 2020)")
        annotate_citation_confidence(
            [c], refs, marker_to_ref_norm={"(smith, 2020)": "1"}
        )
        assert c.ref_id == "1"
        assert c.link_confidence == "high"
        assert c.grobid_ref_id == "1"
        assert c.rule_ref_id == "1"
        assert c.llm_ref_id == "1"

    def test_grobid_and_rule_agree_no_llm_high(self):
        refs = {"1": _ref("1", authors=["John Smith"], year=2020)}
        c = _cite("1", "(Smith, 2020)")
        annotate_citation_confidence([c], refs)
        assert c.link_confidence == "high"
        assert c.ref_id == "1"

    def test_single_vote_rule_undecidable_high(self):
        refs = {"1": _ref("1", authors=[], year=None)}
        c = _cite("1", "[1]")
        annotate_citation_confidence([c], refs)
        assert c.link_confidence == "high"
        assert c.ref_id == "1"

    def test_disagreement_rule_tiebreaks_to_llm(self):
        refs = {
            "1": _ref("1", authors=["Alice Jones"], year=2019),
            "2": _ref("2", authors=["John Smith"], year=2020),
        }
        c = _cite("1", "(Smith, 2020)")
        annotate_citation_confidence(
            [c], refs, marker_to_ref_norm={"(smith, 2020)": "2"}
        )
        assert c.ref_id == "2"
        assert c.link_confidence == "medium"

    def test_disagreement_rule_search_finds_third_ref(self):
        refs = {
            "1": _ref("1", authors=["Alice Jones"], year=2019),
            "2": _ref("2", authors=["Bob Brown"], year=2018),
            "3": _ref("3", authors=["John Smith"], year=2020),
        }
        c = _cite("1", "(Smith, 2020)")
        annotate_citation_confidence(
            [c], refs, marker_to_ref_norm={"(smith, 2020)": "2"}
        )
        assert c.ref_id == "3"
        assert c.link_confidence == "medium"

    def test_numeric_fallback_fix3(self):
        refs = {
            "1": _ref("1"),
            "2": _ref("2"),
            "3": _ref("3"),
        }
        c = _cite("1", "[3]")
        annotate_citation_confidence(
            [c], refs, marker_to_ref_norm={"[3]": "2"}
        )
        assert c.ref_id == "3"
        assert c.link_confidence == "medium"

    def test_single_vote_rule_rejects_downgrades_to_low(self):
        refs = {
            "1": _ref("1", authors=["Jane Jones"], year=2019),
        }
        c = _cite("1", "(Smith, 2020)")
        annotate_citation_confidence([c], refs)
        assert c.ref_id == "1"
        assert c.link_confidence == "low"

    def test_single_vote_rule_rejects_finds_alt(self):
        refs = {
            "1": _ref("1", authors=["Jane Jones"], year=2019),
            "2": _ref("2", authors=["John Smith"], year=2020),
        }
        c = _cite("1", "(Smith, 2020)")
        annotate_citation_confidence([c], refs)
        assert c.ref_id == "2"
        assert c.link_confidence == "medium"

    def test_disagreement_all_fixes_fail_keeps_grobid_low(self):
        refs = {
            "1": _ref("1", authors=["Alice Jones"], year=2019),
            "2": _ref("2", authors=["Bob Brown"], year=2018),
        }
        c = _cite("1", "(Nobody, 1990)")
        annotate_citation_confidence(
            [c], refs, marker_to_ref_norm={"(nobody, 1990)": "2"}
        )
        assert c.ref_id == "1"
        assert c.link_confidence == "low"

    def test_votes_stamped_for_debugging(self):
        refs = {"1": _ref("1", authors=["John Smith"], year=2020)}
        c = _cite("1", "(Smith, 2020)")
        annotate_citation_confidence(
            [c], refs, marker_to_ref_norm={"(smith, 2020)": "1"}
        )
        assert c.grobid_ref_id == "1"
        assert c.rule_ref_id == "1"
        assert c.llm_ref_id == "1"

    def test_llm_via_normalized_marker_key(self):
        refs = {"1": _ref("1", authors=["John O'Brien"], year=2020)}
        c = _cite("1", "(O’Brien, 2020)")
        annotate_citation_confidence(
            [c], refs, marker_to_ref_norm={"(o'brien, 2020)": "1"}
        )
        assert c.llm_ref_id == "1"
        assert c.link_confidence == "high"
