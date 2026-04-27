"""Report generation (L6) — JSON output.

Builds a PaperReport from verdicts, computes summary stats,
and renders to JSON.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from src import config
from src.classification.classifier import CitationVerdict
from src.models.parsed_paper import ParsedPaper
from src.models.report import PaperReport, ReportSummary


_VALID_VERDICTS = {"VALID"}


def build_report(
    parsed: ParsedPaper,
    verdicts: list[CitationVerdict],
    mode: str,
    input_file: str = "unknown",
) -> PaperReport:
    """Build a PaperReport from parsed paper + classification verdicts."""
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
    """Compute aggregate stats from verdicts.

    ``by_verdict`` is sourced from ``metadata_verdict`` (the "does this paper
    exist?" dimension). ``claim_breakdown`` is computed by counting every
    per-sentence ClaimVerdict across all references. The two are reported
    side by side so claim-uncertainty does not bleed into metadata-uncertainty
    on the headline.
    """
    total = len(verdicts)
    if total == 0:
        return ReportSummary()

    by_verdict: dict[str, int] = {}
    claim_breakdown: dict[str, int] = {}
    total_claim_sentences = 0
    flagged = 0
    for v in verdicts:
        # Metadata dimension. Quick mode mirrors the rolled-up verdict into
        # metadata_verdict, so this also works when no claim agent ran.
        meta_v = v.metadata_verdict or v.verdict
        by_verdict[meta_v] = by_verdict.get(meta_v, 0) + 1
        if meta_v != "VALID":
            flagged += 1

        # Claim dimension — count every per-sentence verdict across all refs.
        for cv in v.per_sentence_claim_verdicts.values():
            label = cv.verdict
            claim_breakdown[label] = claim_breakdown.get(label, 0) + 1
            total_claim_sentences += 1

    valid_count = sum(by_verdict.get(vv, 0) for vv in _VALID_VERDICTS)
    # Exclude UNVERIFIABLE from the denominator — "couldn't check" shouldn't
    # penalize the integrity score.
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


# --- JSON ---

def generate_json(report: PaperReport) -> str:
    """Render report as indented JSON string."""
    return json.dumps(report.model_dump(), indent=2, default=str)


def save_json(report: PaperReport, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(generate_json(report), encoding="utf-8")
