"""CheckCite HTTP API — sync + async citation verification.

Designed for integration with agent frameworks (MCP servers, LangChain /
LlamaIndex tool plugins, etc.). Run alongside or instead of the Gradio UI:

    uvicorn src.api:app --host 0.0.0.0 --port 8000

Endpoints (v1):

    POST /api/v1/verify            Synchronous: upload, block until done,
                                   return the full PaperReport.
    POST /api/v1/jobs              Asynchronous: upload, return a job_id
                                   immediately; pipeline runs in the
                                   background. Preferred for long papers
                                   and for clients that can't hold the
                                   HTTP connection open.
    GET  /api/v1/jobs/{job_id}     Poll job status / fetch result when
                                   done. Idempotent — safe after refresh.
    DELETE /api/v1/jobs/{job_id}   Drop a completed or failed job.
    GET  /api/v1/jobs              List recent jobs (metadata only).

    GET  /api/v1/health            Liveness probe.
    GET  /api/v1/manifest          Self-describing tool manifest.
    GET  /api/v1/docs              Interactive Swagger UI.
    GET  /api/v1/openapi.json      Machine-readable OpenAPI spec.

Jobs are persisted to `data/cache/jobs/{job_id}.json` so completed
results survive a server restart.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Literal, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src import job_store
from src.models.report import PaperReport
from src.pipeline import run_unified

log = logging.getLogger(__name__)


# ── App --------------------------------------------------------------

app = FastAPI(
    title="CheckCite API",
    version="0.2.0",
    description=(
        "Citation-verification REST API. Upload a scientific paper "
        "and receive, for every reference, two independent verdicts: a "
        "metadata-driven top-level (VALID / FABRICATED / UNVERIFIABLE) "
        "plus a claim-support dimension (SUPPORTED / CONTRADICTS / NEUTRAL "
        "/ UNVERIFIABLE) grounded in the passage from the cited paper.\n\n"
        "Two flows are offered: a synchronous `POST /api/v1/verify` "
        "that blocks until the pipeline is done (suitable for small "
        "papers and demos), and an asynchronous `POST /api/v1/jobs` "
        "that returns a job id immediately and lets clients poll "
        "`GET /api/v1/jobs/{job_id}` for status — the only option "
        "that survives client disconnects and browser refreshes.\n\n"
        "Designed for integration with agent frameworks and MCP tool servers."
    ),
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
)


# ── Response models --------------------------------------------------


class VerifyResponse(BaseModel):
    """Top-level response payload for /api/v1/verify.

    Wraps the internal `PaperReport` with a minimal summary so that a
    downstream agent can display a one-line result without parsing the
    full verdict list.
    """

    n_references: int = Field(..., description="Total references found in the paper.")
    summary: dict = Field(..., description="Count per verdict category.")
    report: PaperReport = Field(..., description="Full per-reference report.")


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str = "0.1.0"


# ── Helpers ----------------------------------------------------------


_ALLOWED_SUFFIXES = {".pdf", ".tex", ".bib", ".txt"}

# Cap on uploaded file size. Research PDFs run large (figures, supplements);
# 50 MB covers the realistic ceiling without inviting disk-fill DoS from an
# unauthenticated client. Adjust via env if a legitimate workload exceeds it.
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def _suffix_for_upload(upload: UploadFile) -> str:
    """Pick a file extension to persist the upload under.

    Prefers the uploaded filename; falls back to the MIME type if the
    filename is missing or has no usable suffix.
    """
    if upload.filename:
        suf = Path(upload.filename).suffix.lower()
        if suf in _ALLOWED_SUFFIXES:
            return suf
    ct = (upload.content_type or "").lower()
    if "pdf" in ct:
        return ".pdf"
    if "latex" in ct or "x-tex" in ct:
        return ".tex"
    if "bibtex" in ct:
        return ".bib"
    return ".txt"


def _build_summary(report: PaperReport) -> dict:
    counts: dict[str, int] = {}
    for v in report.verdicts:
        counts[v.verdict] = counts.get(v.verdict, 0) + 1
    return counts


def _persist_upload(upload: UploadFile, prefix: str) -> tuple[Path, Path, str]:
    """Validate and stream a FastAPI upload to a fresh temp directory.

    Shared by the sync `verify` endpoint and the async `submit_job`
    endpoint so the two paths cannot drift. Raises 400 if the suffix
    is not accepted, or 413 if the upload exceeds `_MAX_UPLOAD_BYTES`.

    Filenames are stripped to their basename via `Path(...).name` before
    being joined with `tmpdir` so a path-traversal payload like
    "../../etc/passwd" cannot escape the sandbox.

    Returns `(tmp_path, tmpdir, suffix)`; the caller is responsible for
    removing `tmpdir` once the pipeline is done.
    """
    suffix = _suffix_for_upload(upload)
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. "
                   f"Allowed: {sorted(_ALLOWED_SUFFIXES)}.",
        )
    tmpdir = Path(tempfile.mkdtemp(prefix=prefix))
    safe_name = Path(upload.filename or f"paper{suffix}").name or f"paper{suffix}"
    tmp_path = tmpdir / safe_name
    bytes_written = 0
    try:
        with open(tmp_path, "wb") as f:
            while True:
                chunk = upload.file.read(1024 * 1024)  # 1 MiB
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > _MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large (>{_MAX_UPLOAD_BYTES // (1024*1024)} MB).",
                    )
                f.write(chunk)
    except HTTPException:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise
    finally:
        upload.file.close()
    return tmp_path, tmpdir, suffix


# ── Endpoints --------------------------------------------------------


@app.get("/api/v1/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness probe. Returns 200 with a fixed payload."""
    return HealthResponse()


@app.get("/api/v1/manifest", tags=["ops"])
async def manifest() -> JSONResponse:
    """Return the tool manifest (see `TOOL_MANIFEST.json` at repo root).

    Agent frameworks can fetch this to discover the tool's capabilities,
    input schema, and the REST operation(s) they should invoke.
    """
    manifest_path = Path(__file__).resolve().parents[1] / "TOOL_MANIFEST.json"
    if not manifest_path.exists():
        raise HTTPException(
            status_code=500,
            detail="TOOL_MANIFEST.json missing from repository root.",
        )
    return JSONResponse(content=json.loads(manifest_path.read_text(encoding="utf-8")))


@app.post(
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
    """Run the full CheckCite pipeline on an uploaded paper.

    Returns a `PaperReport` with one verdict per reference. Each verdict
    carries the evidence trail that produced it (databases checked,
    metadata comparison, retrieved passages, citing-sentence analysis).

    Raises `400` on unsupported formats, `500` on internal pipeline
    errors.
    """
    tmp_path, tmpdir, _suffix = _persist_upload(file, prefix="checkcite_api_")

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


# ── Async job API ───────────────────────────────────────────────────


class JobCreated(BaseModel):
    """Return payload for POST /api/v1/jobs."""

    job_id: str = Field(..., description="Opaque ID for this job; use it to poll status.")
    status: Literal["queued", "running", "done", "failed"] = Field(..., description="Initial status.")
    status_url: str = Field(..., description="Relative URL to poll for status and result.")


class JobStatus(BaseModel):
    """Return payload for GET /api/v1/jobs/{job_id}.

    `result` is populated only when `status == 'done'`. Polling clients
    can stop as soon as they see a terminal status (`done` or `failed`).
    """

    job_id: str
    status: Literal["queued", "running", "done", "failed"]
    stage: str = Field(default="", description="Human-readable current stage.")
    progress: int = Field(default=0, description="0-100 rough progress hint (stage-level granularity).")
    created_at: float
    updated_at: float
    input_info: dict = Field(default_factory=dict)
    result: Optional[VerifyResponse] = None
    error: Optional[str] = None


@app.on_event("startup")
async def _startup_load_jobs() -> None:
    """Rehydrate the in-memory job index from disk at boot."""
    job_store.load_all_from_disk()


# Hourly prune cadence — long enough that the pruner does no measurable work,
# short enough that orphaned 24-hour-TTL jobs are evicted within an hour of
# their cutoff rather than only when a request happens to arrive.
_PRUNE_INTERVAL_SECONDS = 60 * 60
_prune_task: Optional[asyncio.Task] = None


async def _prune_loop() -> None:
    """Background loop that evicts expired job records on a schedule."""
    while True:
        try:
            await asyncio.sleep(_PRUNE_INTERVAL_SECONDS)
            n = job_store.prune_expired()
            if n:
                log.info(f"job_store: pruned {n} expired job(s)")
        except asyncio.CancelledError:
            raise
        except Exception:
            # Don't let a transient failure kill the loop — just log and retry
            # on the next tick.
            log.exception("prune loop iteration failed")


@app.on_event("startup")
async def _startup_schedule_prune() -> None:
    global _prune_task
    _prune_task = asyncio.create_task(_prune_loop())


@app.on_event("shutdown")
async def _shutdown_cancel_prune() -> None:
    global _prune_task
    if _prune_task is not None:
        _prune_task.cancel()
        try:
            await _prune_task
        except (asyncio.CancelledError, Exception):
            pass
        _prune_task = None


async def _run_job(
    job_id: str,
    file_path: str,
    tmpdir: str,
    mode: str,
    retry_failed: bool,
) -> None:
    """Background coroutine that runs the pipeline for an async job.

    The sync pipeline is pushed into a worker thread via asyncio's
    default executor so it doesn't block the event loop; status
    updates are persisted through job_store.update().
    """
    try:
        await job_store.update(job_id, status="running", stage="pipeline_starting", progress=5)
        # run_unified is synchronous (wraps asyncio.run internally); run in a thread
        # so the API event loop stays free to serve /api/v1/jobs/{id} polls.
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


@app.post(
    "/api/v1/jobs",
    response_model=JobCreated,
    tags=["verification"],
    status_code=202,  # Accepted — work is queued, not complete
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
    """Queue a verification job and return immediately with a job_id.

    Clients poll `GET /api/v1/jobs/{job_id}` to check progress and fetch
    the result when done. Jobs persist across restarts (written to
    `data/cache/jobs/{job_id}.json`) so a refresh, disconnect, or brief
    server outage does not lose the result.
    """
    # Persist the upload for the background worker. tmpdir is cleaned
    # up inside _run_job once the pipeline finishes — but only if the
    # background task is actually scheduled. If anything between persist
    # and add_task raises, we own cleanup here.
    tmp_path, tmpdir, suffix = _persist_upload(file, prefix="checkcite_job_")
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


@app.get(
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


@app.delete(
    "/api/v1/jobs/{job_id}",
    tags=["verification"],
    status_code=204,
    summary="Delete a completed or failed job record.",
)
async def delete_job(job_id: str) -> None:
    if not job_store.delete(job_id):
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")


@app.get(
    "/api/v1/jobs",
    tags=["verification"],
    summary="List the most recent jobs (metadata only, no result payloads).",
)
async def list_jobs(limit: int = 50) -> list[dict]:
    return job_store.list_recent(limit=limit)


# ── Root ------------------------------------------------------------


@app.get("/api/v1", tags=["ops"], include_in_schema=False)
async def root() -> dict:
    """Tiny landing payload so a GET to the API root is not a 404."""
    return {
        "name": "CheckCite API",
        "version": "0.2.0",
        "docs": "/api/v1/docs",
        "openapi": "/api/v1/openapi.json",
        "manifest": "/api/v1/manifest",
        "sync_endpoint": "/api/v1/verify",
        "async_endpoints": {
            "submit": "POST /api/v1/jobs",
            "poll":   "GET /api/v1/jobs/{job_id}",
            "delete": "DELETE /api/v1/jobs/{job_id}",
            "list":   "GET /api/v1/jobs",
        },
    }
