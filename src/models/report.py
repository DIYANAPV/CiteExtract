"""Report data models."""

from pydantic import BaseModel, Field

from src.classification.classifier import CitationVerdict


class ReportSummary(BaseModel):
    """Aggregate statistics for the report."""

    total_checked: int = 0
    by_verdict: dict[str, int] = Field(default_factory=dict)
    integrity_score: float = Field(default=0.0, description="Fraction of VALID* verdicts")
    risk_level: str = Field(default="LOW", description="LOW, MEDIUM, HIGH, CRITICAL")
    flagged_for_review: int = 0


class PaperReport(BaseModel):
    """Full verification report for a paper."""

    input_file: str
    input_format: str
    mode: str  # "quick", "standard", or "agentic"
    timestamp: str
    total_references: int
    verdicts: list[CitationVerdict] = Field(default_factory=list)
    summary: ReportSummary = Field(default_factory=ReportSummary)
    warnings: list[str] = Field(default_factory=list)
