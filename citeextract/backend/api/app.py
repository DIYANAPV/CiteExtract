
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from citeextract import job_store
from citeextract.api import routes_jobs, routes_ops, routes_verify


log = logging.getLogger(__name__)


_PRUNE_INTERVAL_SECONDS = 60 * 60


async def _prune_loop() -> None:
    while True:
        try:
            await asyncio.sleep(_PRUNE_INTERVAL_SECONDS)
            n = job_store.prune_expired()
            if n:
                log.info(f"job_store: pruned {n} expired job(s)")
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("prune loop iteration failed")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    job_store.load_all_from_disk()
    prune_task: Optional[asyncio.Task] = asyncio.create_task(_prune_loop())
    try:
        yield
    finally:
        prune_task.cancel()
        try:
            await prune_task
        except (asyncio.CancelledError, Exception):
            pass


app = FastAPI(
    title="CiteExtract API",
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
    lifespan=lifespan,
)


app.include_router(routes_ops.router)
app.include_router(routes_verify.router)
app.include_router(routes_jobs.router)
