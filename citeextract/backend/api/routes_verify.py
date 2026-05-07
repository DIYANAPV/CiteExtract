
from __future__ import annotations

import logging
import shutil
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from citeextract.api.schemas import VerifyResponse
from citeextract.api.uploads import _build_summary, _persist_upload
from citeextract.pipeline import run_unified


log = logging.getLogger(__name__)


router = APIRouter()


@router.post(
    "/api/v1/verify",
    response_model=VerifyResponse,
    tags=["verification"],
    summary="Verify all citations in a paper.",
)
async def verify(
    file: UploadFile = File(..., description="Paper to verify (.pdf, .tex, .bib, .txt)."),
    mode: Literal["quick", "agentic"] = Form(
        "quick",
        description="'quick' runs L1+L2 only (deterministic, free). 'agentic' adds semantic L3 (LLM, costs a few cents per paper).",
    ),
    retry_failed: bool = Form(
        False,
        description="Clear NOT_FOUND cache and retry references that previously failed existence lookup.",
    ),
) -> VerifyResponse:
    tmp_path, tmpdir, _suffix = _persist_upload(file, prefix="citeextract_api_")

    try:
        report, _comp_report, _parsed = run_unified(
            file_path=str(tmp_path),
            mode=mode,
            run_verification=True,
            run_claim_verification=(mode == "agentic"),
            run_comprehension=False,
            retry_failed=retry_failed,
        )
    except Exception as e:
        log.exception(f"verify() pipeline failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {type(e).__name__}: {str(e)[:200]}",
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if report is None:
        raise HTTPException(
            status_code=500,
            detail="Pipeline returned no report (check server logs).",
        )

    return VerifyResponse(
        n_references=report.total_references,
        summary=_build_summary(report),
        report=report,
    )
