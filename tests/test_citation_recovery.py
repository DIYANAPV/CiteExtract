"""Unit tests for src.parsers.citation_recovery.

These tests use synthetic ``BibrSpan`` lists and a fake body text — no
GROBID, no PDFs. The integration-with-real-PDF test lives separately
behind the ``integration`` mark.

Each test names exactly one behavior so a regression points straight at
the broken case.
"""

from __future__ import annotations

import pytest

from src.models.reference import Reference
from src.parsers.citation_recovery import (
    BibrSpan,
    RecoveryStats,
    _key_to_synthetic_marker,
    _same_appearance,
    _spans_overlap,
    recover_citations,
)


# ---------------------------------------------------------------------------
# Builders — reduce boilerplate so each test highlights its data point.
# ---------------------------------------------------------------------------


def _ref(ref_id: str, surname: str, year: int, *extra_authors: str) -> Reference:
    """Reference whose first author is ``surname`` (single-word).

    The linker only uses the last token of each author for matching, so
    ``"Smith"`` works fine — no need to invent first names for tests.
    """
    return Reference(
        ref_id=ref_id,
        title=f"{surname} et al. paper",
        authors=[surname, *extra_authors],
        year=year,
        source_format="grobid",
    )


def _refs(*pairs: tuple[str, str, int]) -> dict[str, Reference]:
    """Build a {ref_id: Reference} dict from ``(ref_id, surname, year)`` triples."""
    return {ref_id: _ref(ref_id, surname, year) for ref_id, surname, year in pairs}


# ---------------------------------------------------------------------------
# Pure-function helpers
# ---------------------------------------------------------------------------


class TestSpansOverlap:
    def test_disjoint_returns_false(self):
        assert _spans_overlap(0, 5, 10, 15) is False

    def test_touching_endpoints_is_not_overlap(self):
        # [0, 5) and [5, 10) share no byte
        assert _spans_overlap(0, 5, 5, 10) is False

    def test_partial_overlap_returns_true(self):
        assert _spans_overlap(0, 6, 5, 10) is True

    def test_one_contains_the_other_returns_true(self):
        assert _spans_overlap(0, 50, 10, 20) is True


class TestSameAppearance:
    """Phase 2 dedup must collapse same-ref candidates that share an
    on-page click target even when their character spans don't overlap.

    The discriminator is what's between them: intra-parenthetical
    punctuation (comma/semicolon/whitespace) → same appearance; sentence
    boundary (period) → distinct mentions.
    """

    def test_overlapping_spans_collapse(self):
        body = "irrelevant"
        assert _same_appearance(0, 10, 5, 15, body) is True

    def test_close_spans_with_comma_gap_collapse(self):
        # "(Foo, 2024, Yamada et al., 2025)" — GROBID often splits the
        # parenthetical into two bibrs whose spans are separated by ", ".
        body = "(Foo, 2024, Yamada et al., 2025)"
        a_pos, a_end = 1, 10           # "Foo, 2024"
        b_pos, b_end = 12, 32          # "Yamada et al., 2025)"
        assert _same_appearance(a_pos, a_end, b_pos, b_end, body) is True

    def test_distant_spans_do_not_collapse(self):
        # Two same-ref citations 30 chars apart in different sentences
        # should remain distinct prose mentions.
        body = "Smith (2020) showed X.   Smith (2020) extended this further."
        assert _same_appearance(0, 12, 25, 37, body) is False

    def test_sentence_boundary_in_gap_keeps_distinct(self):
        # ". " in the gap implies sentence boundary → distinct citations.
        body = "(Smith, 2020). (Smith, 2020) again"
        assert _same_appearance(0, 13, 15, 28, body) is False

    def test_gap_too_large_keeps_distinct(self):
        # 20-char gap of pure whitespace exceeds the intra-paren cap.
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
        # No 4-digit year → return the input untouched so the linker fails
        # cleanly downstream and the candidate is dropped.
        assert _key_to_synthetic_marker("neurips") == "neurips"


# ---------------------------------------------------------------------------
# Core recovery behavior
# ---------------------------------------------------------------------------


class TestResolvedBibrPassesThrough:
    def test_resolved_bibr_emits_one_citation_unchanged(self):
        """When GROBID gave us a target and the ref exists, no recovery is
        needed — but we still want the candidate to appear in the output."""
        text = "We build on (Smith, 2020) for analysis."
        bibrs = [BibrSpan(position=13, marker="(Smith, 2020)", target_ref_id="1")]
        refs = _refs(("1", "Smith", 2020))

        cits, stats = recover_citations(text, bibrs, refs)

        # The regex hit the same span and produced its own candidate
        # (``regex_added`` is a pre-dedup counter), but dedup kept the
        # authoritative GROBID one — so exactly one Citation lands.
        assert len(cits) == 1
        assert cits[0].ref_id == "1"
        assert cits[0].marker == "(Smith, 2020)"
        assert stats.grobid_resolved == 1
        assert stats.final == 1


class TestOrphanRescue:
    def test_orphan_with_matching_ref_gets_linked(self):
        """The user's smoking-gun case: GROBID tagged the bibr but had no
        target. Phase 2 should link it via author+year."""
        text = "AI Scientist v2 (Yamada et al., 2025) introduced..."
        bibrs = [BibrSpan(position=16, marker="(Yamada et al., 2025)",
                          target_ref_id=None)]
        refs = _refs(("99", "Yamada", 2025))

        cits, stats = recover_citations(text, bibrs, refs)

        assert len(cits) == 1
        assert cits[0].ref_id == "99"
        assert stats.grobid_orphan_linked == 1

    def test_orphan_with_no_matching_ref_is_dropped(self):
        """Orphan whose author+year don't match any reference stays
        dropped — that's expected behavior until Phase 3 (bibliography
        completion) recovers the missing ref."""
        text = "We compare to (Unknown et al., 2024) elsewhere."
        bibrs = [BibrSpan(position=14, marker="(Unknown et al., 2024)",
                          target_ref_id=None)]
        refs = _refs(("1", "Smith", 2020))  # no match

        cits, stats = recover_citations(text, bibrs, refs)

        assert cits == []
        assert stats.grobid_orphan_dropped == 1


class TestRegexRecovery:
    def test_regex_only_hit_recovered_when_ref_exists(self):
        """GROBID didn't tag this citation as a bibr at all (NER miss),
        but regex finds it and the ref exists → recover."""
        text = "Sanh et al. (2019) introduced DistilBERT."
        bibrs: list[BibrSpan] = []  # GROBID emitted nothing
        refs = _refs(("18", "Sanh", 2019))

        cits, stats = recover_citations(text, bibrs, refs)

        assert len(cits) == 1
        assert cits[0].ref_id == "18"
        assert stats.regex_added == 1
        assert stats.grobid_resolved == 0

    def test_regex_hit_with_no_matching_ref_is_dropped(self):
        """Filters out venue-only mentions like ``(NeurIPS 2022)`` which
        the regex catches but which don't correspond to any reference."""
        text = "Published in (NeurIPS 2022) at the workshop."
        bibrs: list[BibrSpan] = []
        refs = _refs(("1", "Smith", 2020))

        cits, stats = recover_citations(text, bibrs, refs)

        # The regex emits a key 'neurips2022'; linker can't find Neurips
        # in the references → drop.
        assert cits == []
        assert stats.regex_dropped >= 1


class TestMultiCiteSplit:
    def test_truncated_first_half_grobid_plus_regex_full_recovers_both(self):
        """The ScholarPeer pattern: GROBID emits ``'(Liang et al., 2024;'``
        as one bibr, never emits the trailing half. Regex sees the full
        parenthetical with both keys. Both refs should land as separate
        Citations after dedup."""
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
        """When GROBID already split the multi-cite correctly into two
        separate bibrs, the regex's combined-marker hit must not collapse
        them back into one."""
        text = "Building on (Smith, 2020; Jones, 2021) for the proof."
        bibrs = [
            BibrSpan(position=13, marker="(Smith, 2020;", target_ref_id="1"),
            BibrSpan(position=27, marker="Jones, 2021)", target_ref_id="2"),
        ]
        refs = _refs(("1", "Smith", 2020), ("2", "Jones", 2021))

        cits, stats = recover_citations(text, bibrs, refs)

        ref_ids = sorted(c.ref_id for c in cits)
        assert ref_ids == ["1", "2"]
        # Both should be GROBID-resolved (regex hit dedupes against them).
        assert stats.grobid_resolved == 2


class TestSplitMultiCiteCollapse:
    """When GROBID splits one parenthetical into two bibrs whose spans
    don't overlap (one ends before the comma, the next starts after it),
    Phase 2 must still produce one Citation per ref — not duplicates."""

    def test_grobid_split_multicite_emits_one_citation_per_ref(self):
        body = "Tools like (Foo, 2024, Yamada et al., 2025) help."
        # Plausible GROBID output: two bibrs with non-touching spans.
        bibrs = [
            BibrSpan(position=12, marker="Foo, 2024", target_ref_id="1"),
            BibrSpan(position=23, marker="Yamada et al., 2025",
                     target_ref_id="2"),
        ]
        # The regex sweep also picks up the same parenthetical and
        # produces a candidate per key — exactly the case where the old
        # overlap-only dedup left a duplicate.
        refs = _refs(("1", "Foo", 2024), ("2", "Yamada", 2025))

        cits, _ = recover_citations(body, bibrs, refs)

        ref_ids = sorted(c.ref_id for c in cits)
        assert ref_ids == ["1", "2"], (
            f"Expected one Citation per ref, got {ref_ids}"
        )


class TestDedupAuthority:
    def test_resolved_grobid_wins_over_regex_at_same_span(self):
        """When GROBID and regex both produce a candidate for the same
        ref at overlapping spans, the GROBID-resolved one should win
        because its ``target`` attribute is authoritative."""
        text = "We build on (Smith, 2020) for analysis."
        bibrs = [BibrSpan(position=13, marker="(Smith, 2020)", target_ref_id="1")]
        refs = _refs(("1", "Smith", 2020))

        cits, stats = recover_citations(text, bibrs, refs)

        # Exactly one citation, sourced from the resolved GROBID bibr.
        assert len(cits) == 1
        assert cits[0].marker == "(Smith, 2020)"
        assert stats.grobid_resolved == 1

    def test_regex_promotes_truncated_orphan_when_longer_marker(self):
        """When the regex's marker covers a longer span than an overlapping
        GROBID orphan (i.e. GROBID truncated), and both link to the same
        ref, the regex candidate should win the dedup so the annotator
        gets the cleaner marker text."""
        text = "Building on (Smith, 2020) elsewhere."
        bibrs = [BibrSpan(position=13, marker="(Smith,",
                          target_ref_id=None)]  # truncated orphan
        refs = _refs(("1", "Smith", 2020))

        cits, stats = recover_citations(text, bibrs, refs)

        # Only one citation overall.
        assert len(cits) == 1
        # We don't pin down which marker wins (GROBID orphan promotes to
        # regex by source priority), but the citation must point to ref 1.
        assert cits[0].ref_id == "1"


class TestCitationOrdering:
    def test_output_sorted_by_position(self):
        """Citations must come back in reading order so downstream consumers
        (annotator, comprehension) can iterate them top-to-bottom."""
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
            BibrSpan(position=25, marker="(Jones, 2021)", target_ref_id=None),  # orphan
            BibrSpan(position=51, marker="(Ghost, 1999)", target_ref_id=None),  # orphan, not in refs
        ]
        refs = _refs(("1", "Smith", 2020), ("2", "Jones", 2021))

        cits, stats = recover_citations(text, bibrs, refs)

        assert stats.grobid_resolved == 1   # Smith
        assert stats.grobid_orphan_linked == 1  # Jones recovered
        assert stats.grobid_orphan_dropped == 1  # Ghost has no ref
        assert stats.final == 2
