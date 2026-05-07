
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from citeextract.api.schemas import HealthResponse


_MANIFEST_PATH = Path(__file__).parent / "manifest.json"


router = APIRouter()


@router.get("/api/v1/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    return HealthResponse()


@router.get("/api/v1/manifest", tags=["ops"])
async def manifest() -> JSONResponse:
    if not _MANIFEST_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="manifest.json missing from the api package.",
        )
    return JSONResponse(content=json.loads(_MANIFEST_PATH.read_text(encoding="utf-8")))


@router.get("/api/v1", tags=["ops"], include_in_schema=False)
async def root() -> dict:
    return {
        "name": "CiteExtract API",
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
