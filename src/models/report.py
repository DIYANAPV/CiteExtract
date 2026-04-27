"""Report data models."""

from pydantic import BaseModel, Field

from src.classification.classifier import CitationVerdict


class ReportSummary(BaseModel):
    """Aggregate statistics for the report.

    ``by_verdict`` counts the **metadata dimension** (VALID/FABRICATED/
    UNVERIFIABLE) — i.e. "does this paper exist and match the reference?".
    ``claim_breakdown`` counts per-citing-sentence claim agent outputs
    (SUPPORTS/CONTRADICTS/NEUTRAL or SUPPORTED/NOT_SUPPORTED depending on
    the configured verdict scheme) — i.e. "does the cited paper support
    the claim?". The two are reported separately so the headline doesn't
    blur claim-uncertainty into metadata-uncertainty.
    """

    total_checked: int = 0
    by_verdict: dict[str, int] = Field(default_factory=dict)
    claim_breakdown: dict[str, int] = Field(default_factory=dict)
    total_claim_sentences: int = 0
    integrity_score: float = Field(default=0.0, description="Fraction of metadata-VALID verdicts")
    risk_level: str = Field(default="LOW", description="LOW, MEDIUM, HIGH, CRITICAL")
    flagged_for_review: int = 0


class PaperReport(BaseModel):
    """Full verification report for a paper."""

    input_file: str
    input_format: str
    mode: str  # "quick" (rule-based) or "agentic"
    timestamp: str
    total_references: int
    verdicts: list[CitationVerdict] = Field(default_factory=list)
    summary: ReportSummary = Field(default_factory=ReportSummary)
    warnings: list[str] = Field(default_factory=list)
