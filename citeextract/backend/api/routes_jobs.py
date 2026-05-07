
from __future__ import annotations

import asyncio
import logging
import shutil
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from citeextract import job_store
from citeextract.api.schemas import JobCreated, JobStatus, VerifyResponse
from citeextract.api.uploads import _build_summary, _persist_upload
from citeextract.pipeline import run_unified


log = logging.getLogger(__name__)


router = APIRouter()


async def _run_job(
    job_id: str,
    file_path: str,
    tmpdir: str,
    mode: str,
    retry_failed: bool,
) -> None:
    try:
        await job_store.update(job_id, status="running", stage="pipeline_starting", progress=5)
        loop = asyncio.get_running_loop()
        report, _comp, _parsed = await loop.run_in_executor(
            None,
            lambda: run_unified(
                file_path=file_path,
                mode=mode,
                run_verification=True,
                run_claim_verification=(mode == "agentic"),
                run_comprehension=False,
                retry_failed=retry_failed,
            ),
        )
        if report is None:
            await job_store.update(
                job_id,
                status="failed",
                error="Pipeline returned no report.",
                stage="failed", progress=100,
            )
            return
        payload = VerifyResponse(
            n_references=report.total_references,
            summary=_build_summary(report),
            report=report,
        ).model_dump()
        await job_store.update(
            job_id, status="done", result=payload,
            stage="complete", progress=100,
        )
    except Exception as e:
        log.exception(f"job {job_id} pipeline failed: {e}")
        await job_store.update(
            job_id, status="failed",
            error=f"{type(e).__name__}: {str(e)[:300]}",
            stage="failed", progress=100,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@router.post(
    "/api/v1/jobs",
    response_model=JobCreated,
    tags=["verification"],
    status_code=202,
    summary="Submit a paper for asynchronous verification.",
)
async def submit_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Paper to verify (.pdf, .tex, .bib, .txt)."),
    mode: Literal["quick", "agentic"] = Form(
        "quick",
        description="'quick' runs L1+L2 only. 'agentic' adds semantic L3 (LLM).",
    ),
    retry_failed: bool = Form(
        False,
        description="Clear NOT_FOUND cache and retry failed existence lookups.",
    ),
) -> JobCreated:
    tmp_path, tmpdir, suffix = _persist_upload(file, prefix="citeextract_job_")
    enqueued = False
    try:
        rec = job_store.create({
            "filename": file.filename or f"paper{suffix}",
            "mode": mode,
            "retry_failed": retry_failed,
        })
        background_tasks.add_task(
            _run_job, rec["job_id"], str(tmp_path), str(tmpdir), mode, retry_failed,
        )
        enqueued = True
        return JobCreated(
            job_id=rec["job_id"],
            status="queued",
            status_url=f"/api/v1/jobs/{rec['job_id']}",
        )
    finally:
        if not enqueued:
            shutil.rmtree(tmpdir, ignore_errors=True)


@router.get(
    "/api/v1/jobs/{job_id}",
    response_model=JobStatus,
    tags=["verification"],
    summary="Fetch the status and (if ready) the result of an async job.",
)
async def get_job(job_id: str) -> JobStatus:
    rec = job_store.get(job_id)
    if rec is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found (may have expired or never existed).",
        )
    return JobStatus(**rec)


@router.delete(
    "/api/v1/jobs/{job_id}",
    tags=["verification"],
    status_code=204,
    summary="Delete a completed or failed job record.",
)
async def delete_job(job_id: str) -> None:
    if not job_store.delete(job_id):
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")


@router.get(
    "/api/v1/jobs",
    tags=["verification"],
    summary="List the most recent jobs (metadata only, no result payloads).",
)
async def list_jobs(limit: int = 50) -> list[dict]:
    return job_store.list_recent(limit=limit)
