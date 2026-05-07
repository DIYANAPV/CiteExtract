
import os
import shutil
from pathlib import Path

import pytest

from citeextract.pipeline import run_unified

FIXTURE = Path("data/test_inputs/clean_paper.tex")


@pytest.fixture
def clean_caches(tmp_path, monkeypatch):
    cache_dir = tmp_path / "reports"
    from citeextract.verification import report_cache
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

    assert parsed.references, "Parser returned no references"
    assert len(paper_report.verdicts) == len(parsed.references)

    for v in paper_report.verdicts:
        assert v.verdict in {"VALID", "FABRICATED", "UNVERIFIABLE"}
        assert v.mode == "quick"
        assert v.ref_id, "Verdict missing ref_id"
