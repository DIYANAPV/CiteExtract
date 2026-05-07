
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from citeextract.models.report import PaperReport


class VerifyResponse(BaseModel):

    n_references: int = Field(..., description="Total references found in the paper.")
    summary: dict = Field(..., description="Count per verdict category.")
    report: PaperReport = Field(..., description="Full per-reference report.")


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str = "0.1.0"


class JobCreated(BaseModel):

    job_id: str = Field(..., description="Opaque ID for this job; use it to poll status.")
    status: Literal["queued", "running", "done", "failed"] = Field(..., description="Initial status.")
    status_url: str = Field(..., description="Relative URL to poll for status and result.")


class JobStatus(BaseModel):

    job_id: str
    status: Literal["queued", "running", "done", "failed"]
    stage: str = Field(default="", description="Human-readable current stage.")
    progress: int = Field(default=0, description="0-100 rough progress hint (stage-level granularity).")
    created_at: float
    updated_at: float
    input_info: dict = Field(default_factory=dict)
    result: Optional[VerifyResponse] = None
    error: Optional[str] = None
