"""Tests for the classification decision tree (L5)."""

import pytest

from src.classification.classifier import classify_quick, CitationVerdict
from src.models.verdict import ExistenceResult
from src.verification.metadata import MetadataResult, FieldComparison


def _exist(status="FOUND", **kw) -> ExistenceResult:
    defaults = dict(
        ref_id="1", status=status, source="semantic_scholar",
        matched_title="T", matched_authors=["A"], matched_year=2020,
        databases_checked=["semantic_scholar"], flags=[],
    )
    defaults.update(kw)
    return ExistenceResult(**defaults)


def _meta(has_mismatch=False, retracted=False, **kw) -> MetadataResult:
    defaults = dict(
        ref_id="1", comparisons=[], metadata_score=1.0,
        flags=[], is_retracted=retracted, has_metadata_mismatch=has_mismatch,
    )
    defaults.update(kw)
    return MetadataResult(**defaults)


class TestClassifyQuick:
    def test_not_found_is_fabricated(self):
        exist = _exist(
            status="NOT_FOUND", source=None, matched_title=None,
            databases_checked=["semantic_scholar", "openalex"],
        )
        v = classify_quick(exist, None)
        assert v.verdict == "FABRICATED"
        assert v.action == "remove_citation"

    def test_not_found_insufficient_coverage_is_unverifiable(self):
        """NOT_FOUND with <2 databases checked → UNVERIFIABLE, not FABRICATED."""
        exist = _exist(
            status="NOT_FOUND", source=None, matched_title=None,
            databases_checked=["semantic_scholar"],
        )
        v = classify_quick(exist, None)
        assert v.verdict == "UNVERIFIABLE"
        assert v.action == "no_action"
        assert "insufficient_database_coverage" in v.flags

    def test_retracted_is_fabricated(self):
        v = classify_quick(_exist(), _meta(retracted=True))
        assert v.verdict == "FABRICATED"
        assert v.action == "remove_citation"

    def test_metadata_mismatch_is_fabricated(self):
        meta = _meta(
            has_mismatch=True,
            comparisons=[
                FieldComparison(field="title", status="MATCH", similarity=1.0),
                FieldComparison(field="authors", status="MISMATCH", similarity=0.1),
                FieldComparison(field="year", status="MISMATCH"),
            ],
        )
        v = classify_quick(_exist(), meta)
        assert v.verdict == "FABRICATED"
        assert v.action == "remove_citation"
        assert "authors" in v.explanation or "year" in v.explanation

    def test_single_field_mismatch_is_fabricated(self):
        meta = _meta(
            has_mismatch=True,
            comparisons=[
                FieldComparison(field="title", status="MATCH", similarity=1.0),
                FieldComparison(field="year", status="MISMATCH",
                                flag="year_mismatch: 2020 vs 2021"),
            ],
        )
        v = classify_quick(_exist(), meta)
        assert v.verdict == "FABRICATED"
        assert v.action == "remove_citation"

    def test_valid(self):
        v = classify_quick(_exist(), _meta())
        assert v.verdict == "VALID"
        assert v.action == "no_action"

    def test_priority_retracted_over_mismatch(self):
        """Retracted takes priority over metadata mismatch."""
        meta = _meta(has_mismatch=True, retracted=True)
        v = classify_quick(_exist(), meta)
        assert v.verdict == "FABRICATED"

    def test_priority_fabricated_over_all(self):
        """Not found takes priority over everything."""
        exist = _exist(
            status="NOT_FOUND", source=None, matched_title=None,
            databases_checked=["semantic_scholar", "openalex"],
        )
        v = classify_quick(exist, _meta(retracted=True, has_mismatch=True))
        assert v.verdict == "FABRICATED"

    def test_flags_propagated(self):
        exist = _exist(flags=["doi_title_mismatch: ..."])
        meta = _meta(flags=["year_mismatch: 2020 vs 2021"])
        v = classify_quick(exist, meta)
        assert len(v.flags) >= 2

    def test_verdict_has_evidence(self):
        exist = _exist()
        meta = _meta()
        v = classify_quick(exist, meta)
        assert v.existence is not None
        assert v.metadata is not None
