
import json
from datetime import datetime, timezone
from pathlib import Path

from citeextract import config
from citeextract.classification.classifier import CitationVerdict
from citeextract.models.parsed_paper import ParsedPaper
from citeextract.models.report import PaperReport, ReportSummary


_VALID_VERDICTS = {"VALID"}


def build_report(
    parsed: ParsedPaper,
    verdicts: list[CitationVerdict],
    mode: str,
    input_file: str = "unknown",
) -> PaperReport:
    summary = _compute_summary(verdicts)

    return PaperReport(
        input_file=input_file,
        input_format=parsed.input_format,
        mode=mode,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        total_references=len(parsed.references),
        verdicts=verdicts,
        summary=summary,
        warnings=parsed.warnings,
    )


def _compute_summary(verdicts: list[CitationVerdict]) -> ReportSummary:
    total = len(verdicts)
    if total == 0:
        return ReportSummary()

    by_verdict: dict[str, int] = {}
    claim_breakdown: dict[str, int] = {}
    total_claim_sentences = 0
    flagged = 0
    for v in verdicts:
        meta_v = v.metadata_verdict or v.verdict
        by_verdict[meta_v] = by_verdict.get(meta_v, 0) + 1
        if meta_v != "VALID":
            flagged += 1

        for cv in v.per_sentence_claim_verdicts.values():
            label = cv.verdict
            claim_breakdown[label] = claim_breakdown.get(label, 0) + 1
            total_claim_sentences += 1

    valid_count = sum(by_verdict.get(vv, 0) for vv in _VALID_VERDICTS)
    unverifiable_count = by_verdict.get("UNVERIFIABLE", 0)
    denominator = total - unverifiable_count
    integrity = valid_count / denominator if denominator > 0 else 0.0

    levels = config.risk_levels()
    if integrity > levels["low"]:
        risk = "LOW"
    elif integrity > levels["medium"]:
        risk = "MEDIUM"
    elif integrity > levels["high"]:
        risk = "HIGH"
    else:
        risk = "CRITICAL"

    return ReportSummary(
        total_checked=total,
        by_verdict=by_verdict,
        claim_breakdown=claim_breakdown,
        total_claim_sentences=total_claim_sentences,
        integrity_score=round(integrity, 3),
        risk_level=risk,
        flagged_for_review=flagged,
    )


def generate_json(report: PaperReport) -> str:
    return json.dumps(report.model_dump(), indent=2, default=str)


def save_json(report: PaperReport, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(generate_json(report), encoding="utf-8")
