
from __future__ import annotations

import pytest

from citeextract.models.reference import Reference
from citeextract.parsers.citation_recovery import (
    BibrSpan,
    RecoveryStats,
    _key_to_synthetic_marker,
    _same_appearance,
    _spans_overlap,
    recover_citations,
)


def _ref(ref_id: str, surname: str, year: int, *extra_authors: str) -> Reference:
    return Reference(
        ref_id=ref_id,
        title=f"{surname} et al. paper",
        authors=[surname, *extra_authors],
        year=year,
        source_format="grobid",
    )


def _refs(*pairs: tuple[str, str, int]) -> dict[str, Reference]:
    return {ref_id: _ref(ref_id, surname, year) for ref_id, surname, year in pairs}


class TestSpansOverlap:
    def test_disjoint_returns_false(self):
        assert _spans_overlap(0, 5, 10, 15) is False

    def test_touching_endpoints_is_not_overlap(self):
        assert _spans_overlap(0, 5, 5, 10) is False

    def test_partial_overlap_returns_true(self):
        assert _spans_overlap(0, 6, 5, 10) is True

    def test_one_contains_the_other_returns_true(self):
        assert _spans_overlap(0, 50, 10, 20) is True


class TestSameAppearance:

    def test_overlapping_spans_collapse(self):
        body = "irrelevant"
        assert _same_appearance(0, 10, 5, 15, body) is True

    def test_close_spans_with_comma_gap_collapse(self):
        body = "(Foo, 2024, Yamada et al., 2025)"
        a_pos, a_end = 1, 10
        b_pos, b_end = 12, 32
        assert _same_appearance(a_pos, a_end, b_pos, b_end, body) is True

    def test_distant_spans_do_not_collapse(self):
        body = "Smith (2020) showed X.   Smith (2020) extended this further."
        assert _same_appearance(0, 12, 25, 37, body) is False

    def test_sentence_boundary_in_gap_keeps_distinct(self):
        body = "(Smith, 2020). (Smith, 2020) again"
        assert _same_appearance(0, 13, 15, 28, body) is False

    def test_gap_too_large_keeps_distinct(self):
        body = "(Foo, 2024)" + " " * 20 + "(Foo, 2024)"
        assert _same_appearance(0, 11, 31, 42, body) is False


class TestKeyToSyntheticMarker:
    def test_solo_author(self):
        assert _key_to_synthetic_marker("smith2020") == "Smith 2020"

    def test_etal_author(self):
        assert _key_to_synthetic_marker("zhuetal2025") == "Zhu et al. 2025"

    def test_year_with_letter_suffix(self):
        assert _key_to_synthetic_marker("gaoetal2025a") == "Gao et al. 2025a"

    def test_unparseable_falls_back_to_raw(self):
        assert _key_to_synthetic_marker("neurips") == "neurips"


class TestResolvedBibrPassesThrough:
    def test_resolved_bibr_emits_one_citation_unchanged(self):
        text = "We build on (Smith, 2020) for analysis."
        bibrs = [BibrSpan(position=13, marker="(Smith, 2020)", target_ref_id="1")]
        refs = _refs(("1", "Smith", 2020))

        cits, stats = recover_citations(text, bibrs, refs)

        assert len(cits) == 1
        assert cits[0].ref_id == "1"
        assert cits[0].marker == "(Smith, 2020)"
        assert stats.grobid_resolved == 1
        assert stats.final == 1


class TestOrphanRescue:
    def test_orphan_with_matching_ref_gets_linked(self):
        text = "AI Scientist v2 (Yamada et al., 2025) introduced..."
        bibrs = [BibrSpan(position=16, marker="(Yamada et al., 2025)",
                          target_ref_id=None)]
        refs = _refs(("99", "Yamada", 2025))

        cits, stats = recover_citations(text, bibrs, refs)

        assert len(cits) == 1
        assert cits[0].ref_id == "99"
        assert stats.grobid_orphan_linked == 1

    def test_orphan_with_no_matching_ref_is_dropped(self):
        text = "We compare to (Unknown et al., 2024) elsewhere."
        bibrs = [BibrSpan(position=14, marker="(Unknown et al., 2024)",
                          target_ref_id=None)]
        refs = _refs(("1", "Smith", 2020))

        cits, stats = recover_citations(text, bibrs, refs)

        assert cits == []
        assert stats.grobid_orphan_dropped == 1


class TestRegexRecovery:
    def test_regex_only_hit_recovered_when_ref_exists(self):
        text = "Sanh et al. (2019) introduced DistilBERT."
        bibrs: list[BibrSpan] = []
        refs = _refs(("18", "Sanh", 2019))

        cits, stats = recover_citations(text, bibrs, refs)

        assert len(cits) == 1
        assert cits[0].ref_id == "18"
        assert stats.regex_added == 1
        assert stats.grobid_resolved == 0

    def test_regex_hit_with_no_matching_ref_is_dropped(self):
        text = "Published in (NeurIPS 2022) at the workshop."
        bibrs: list[BibrSpan] = []
        refs = _refs(("1", "Smith", 2020))

        cits, stats = recover_citations(text, bibrs, refs)

        assert cits == []
        assert stats.regex_dropped >= 1


class TestMultiCiteSplit:
    def test_truncated_first_half_grobid_plus_regex_full_recovers_both(self):
        text = "Recent work (Liang et al., 2024; Zhuang et al., 2025) shows..."
        bibrs = [
            BibrSpan(position=12, marker="(Liang et al., 2024;",
                     target_ref_id="12"),
        ]
        refs = _refs(("12", "Liang", 2024), ("30", "Zhuang", 2025))

        cits, stats = recover_citations(text, bibrs, refs)

        ref_ids = sorted(c.ref_id for c in cits)
        assert ref_ids == ["12", "30"], (
            f"Expected both refs to be recovered, got {ref_ids}"
        )

    def test_two_clean_grobid_bibrs_not_merged_by_regex(self):
        text = "Building on (Smith, 2020; Jones, 2021) for the proof."
        bibrs = [
            BibrSpan(position=13, marker="(Smith, 2020;", target_ref_id="1"),
            BibrSpan(position=27, marker="Jones, 2021)", target_ref_id="2"),
        ]
        refs = _refs(("1", "Smith", 2020), ("2", "Jones", 2021))

        cits, stats = recover_citations(text, bibrs, refs)

        ref_ids = sorted(c.ref_id for c in cits)
        assert ref_ids == ["1", "2"]
        assert stats.grobid_resolved == 2


class TestSplitMultiCiteCollapse:

    def test_grobid_split_multicite_emits_one_citation_per_ref(self):
        body = "Tools like (Foo, 2024, Yamada et al., 2025) help."
        bibrs = [
            BibrSpan(position=12, marker="Foo, 2024", target_ref_id="1"),
            BibrSpan(position=23, marker="Yamada et al., 2025",
                     target_ref_id="2"),
        ]
        refs = _refs(("1", "Foo", 2024), ("2", "Yamada", 2025))

        cits, _ = recover_citations(body, bibrs, refs)

        ref_ids = sorted(c.ref_id for c in cits)
        assert ref_ids == ["1", "2"], (
            f"Expected one Citation per ref, got {ref_ids}"
        )


class TestDedupAuthority:
    def test_resolved_grobid_wins_over_regex_at_same_span(self):
        text = "We build on (Smith, 2020) for analysis."
        bibrs = [BibrSpan(position=13, marker="(Smith, 2020)", target_ref_id="1")]
        refs = _refs(("1", "Smith", 2020))

        cits, stats = recover_citations(text, bibrs, refs)

        assert len(cits) == 1
        assert cits[0].marker == "(Smith, 2020)"
        assert stats.grobid_resolved == 1

    def test_regex_promotes_truncated_orphan_when_longer_marker(self):
        text = "Building on (Smith, 2020) elsewhere."
        bibrs = [BibrSpan(position=13, marker="(Smith,",
                          target_ref_id=None)]
        refs = _refs(("1", "Smith", 2020))

        cits, stats = recover_citations(text, bibrs, refs)

        assert len(cits) == 1
        assert cits[0].ref_id == "1"


class TestCitationOrdering:
    def test_output_sorted_by_position(self):
        text = (
            "First we use (Bravo, 2023) and then later (Alpha, 2020) "
            "is mentioned again."
        )
        bibrs = [
            BibrSpan(position=42, marker="(Alpha, 2020)", target_ref_id="1"),
            BibrSpan(position=13, marker="(Bravo, 2023)", target_ref_id="2"),
        ]
        refs = _refs(("1", "Alpha", 2020), ("2", "Bravo", 2023))

        cits, _ = recover_citations(text, bibrs, refs)

        positions = [c.position for c in cits]
        assert positions == sorted(positions)


class TestRecoveryStats:
    def test_stats_account_for_every_candidate(self):
        text = (
            "First (Smith, 2020) then (Jones, 2021) and finally (Ghost, 1999)."
        )
        bibrs = [
            BibrSpan(position=6, marker="(Smith, 2020)", target_ref_id="1"),
            BibrSpan(position=25, marker="(Jones, 2021)", target_ref_id=None),
            BibrSpan(position=51, marker="(Ghost, 1999)", target_ref_id=None),
        ]
        refs = _refs(("1", "Smith", 2020), ("2", "Jones", 2021))

        cits, stats = recover_citations(text, bibrs, refs)

        assert stats.grobid_resolved == 1
        assert stats.grobid_orphan_linked == 1
        assert stats.grobid_orphan_dropped == 1
        assert stats.final == 2
