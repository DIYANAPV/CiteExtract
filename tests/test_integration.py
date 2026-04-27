"""End-to-end integration test.

Runs the full pipeline on a small fixture paper (.tex) in rule-based mode.
Hits real external APIs (CrossRef, OpenAlex, Semantic Scholar, PubMed), so
it is slow (~3-10s) and requires network. Skipped when the marker is
deselected via `pytest -m "not integration"`.

Purpose: catch pipeline breakage that unit tests don't see — bad imports,
signature drift, serialization mismatches, bad cache keys, etc.
"""

import os
import shutil
from pathlib import Path

import pytest

from src.pipeline import run_unified

FIXTURE = Path("data/test_inputs/clean_paper.tex")


@pytest.fixture
def clean_caches(tmp_path, monkeypatch):
    """Redirect the report cache to a throwaway dir so the test always
    measures real work instead of a cache hit from a prior run."""
    cache_dir = tmp_path / "reports"
    from src.verification import report_cache
    monkeypatch.setattr(report_cache, "CACHE_DIR", cache_dir)
    yield
    shutil.rmtree(cache_dir, ignore_errors=True)


@pytest.mark.integration
def test_quick_pipeline_end_to_end(clean_caches):
    assert FIXTURE.exists(), f"Fixture missing: {FIXTURE}"

    paper_report, comp_report, parsed = run_unified(
        str(FIXTURE),
        mode="quick",
        run_verification=True,
        run_claim_verification=False,
        run_comprehension=False,
    )

    assert paper_report is not None, "Expected a PaperReport in quick mode"
    assert comp_report is None, "Comprehension report should not be built in quick mode"
    assert parsed is not None

    # clean_paper.tex has 4 references; pipeline must return one verdict per reference.
    assert parsed.references, "Parser returned no references"
    assert len(paper_report.verdicts) == len(parsed.references)

    for v in paper_report.verdicts:
        assert v.verdict in {"VALID", "FABRICATED", "UNVERIFIABLE"}
        assert v.mode == "quick"
        assert v.ref_id, "Verdict missing ref_id"
