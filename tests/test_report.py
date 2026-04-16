"""Tests for report generation (L6)."""

import json

import pytest

from src.classification.classifier import CitationVerdict
from src.models.parsed_paper import ParsedPaper
from src.models.report import PaperReport, ReportSummary
from src.report.generator import build_report, generate_json, _compute_summary


def _verdict(ref_id: str, verdict: str, flags: list[str] | None = None) -> CitationVerdict:
    return CitationVerdict(
        ref_id=ref_id, verdict=verdict, mode="quick",
        action="no_action", explanation="Test.",
        flags=flags or [],
    )


class TestSummary:
    def test_all_valid(self):
        verdicts = [_verdict("1", "VALID"), _verdict("2", "VALID")]
        s = _compute_summary(verdicts)
        assert s.total_checked == 2
        assert s.integrity_score == 1.0
        assert s.risk_level == "LOW"

    def test_mixed(self):
        verdicts = [
            _verdict("1", "VALID"),
            _verdict("2", "FABRICATED"),
            _verdict("3", "VALID"),
            _verdict("4", "FABRICATED"),
        ]
        s = _compute_summary(verdicts)
        assert s.total_checked == 4
        assert s.by_verdict["FABRICATED"] == 2
        assert s.integrity_score == 0.5
        assert s.risk_level == "CRITICAL"  # 50% is not >50%, so CRITICAL

    def test_all_fabricated(self):
        verdicts = [_verdict("1", "FABRICATED"), _verdict("2", "FABRICATED")]
        s = _compute_summary(verdicts)
        assert s.integrity_score == 0.0
        assert s.risk_level == "CRITICAL"

    def test_flagged_count(self):
        """Non-VALID verdicts count as flagged for review."""
        verdicts = [
            _verdict("1", "FABRICATED", flags=["not_found"]),
            _verdict("2", "VALID"),
        ]
        s = _compute_summary(verdicts)
        assert s.flagged_for_review == 1

    def test_valid_with_flags_not_flagged(self):
        """VALID verdicts don't count as flagged even if they have informational flags."""
        verdicts = [
            _verdict("1", "VALID", flags=["year_close_match"]),
            _verdict("2", "VALID"),
        ]
        s = _compute_summary(verdicts)
        assert s.flagged_for_review == 0

    def test_empty(self):
        s = _compute_summary([])
        assert s.total_checked == 0


class TestBuildReport:
    def test_basic(self):
        parsed = ParsedPaper(
            references=[], citations=[], has_body_text=False,
            input_format="bibtex", metadata={"title": "Test Paper"},
        )
        verdicts = [_verdict("1", "VALID")]
        report = build_report(parsed, verdicts, "quick")
        assert isinstance(report, PaperReport)
        assert report.mode == "quick"
        assert len(report.verdicts) == 1
        assert report.summary.total_checked == 1


class TestJsonReport:
    def test_valid_json(self):
        parsed = ParsedPaper(
            references=[], citations=[], has_body_text=False,
            input_format="bibtex", metadata={},
        )
        verdicts = [_verdict("1", "FABRICATED"), _verdict("2", "VALID")]
        report = build_report(parsed, verdicts, "quick")
        json_str = generate_json(report)
        data = json.loads(json_str)
        assert data["mode"] == "quick"
        assert len(data["verdicts"]) == 2
        assert data["summary"]["total_checked"] == 2
