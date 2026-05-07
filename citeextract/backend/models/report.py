
from pydantic import BaseModel, Field

from citeextract.classification.classifier import CitationVerdict


class ReportSummary(BaseModel):

    total_checked: int = 0
    by_verdict: dict[str, int] = Field(default_factory=dict)
    claim_breakdown: dict[str, int] = Field(default_factory=dict)
    total_claim_sentences: int = 0
    integrity_score: float = Field(default=0.0, description="Fraction of metadata-VALID verdicts")
    risk_level: str = Field(default="LOW", description="LOW, MEDIUM, HIGH, CRITICAL")
    flagged_for_review: int = 0


class PaperReport(BaseModel):

    input_file: str
    input_format: str
    mode: str
    timestamp: str
    total_references: int
    verdicts: list[CitationVerdict] = Field(default_factory=list)
    summary: ReportSummary = Field(default_factory=ReportSummary)
    warnings: list[str] = Field(default_factory=list)
