
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

from citeextract import paths

log = logging.getLogger(__name__)


Status = Literal["queued", "running", "done", "failed"]

_STATES: tuple[Status, ...] = ("queued", "running", "done", "failed")


_TTL_SECONDS = 24 * 60 * 60
_STALE_SECONDS = 60 * 60
_JOBS_DIR = paths.data_dir() / "cache" / "jobs"


_jobs: dict[str, dict[str, Any]] = {}
_locks: dict[str, asyncio.Lock] = {}


def _now() -> float:
    return time.time()


def _path(job_id: str) -> Path:
    return _JOBS_DIR / f"{job_id}.json"


def _lock(job_id: str) -> asyncio.Lock:
    lk = _locks.get(job_id)
    if lk is None:
        lk = asyncio.Lock()
        _locks[job_id] = lk
    return lk


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, default=str, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _read(path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        log.warning(f"job_store: cannot read {path}: {e}")
        return None


def load_all_from_disk() -> int:
    _JOBS_DIR.mkdir(parents=True, exist_ok=True)
    loaded = 0
    pruned = 0
    revived_as_failed = 0
    now_ts = _now()
    prune_cutoff = now_ts - _TTL_SECONDS
    stale_cutoff = now_ts - _STALE_SECONDS

    for fp in _JOBS_DIR.glob("*.json"):
        rec = _read(fp)
        if not rec:
            continue
        created = float(rec.get("created_at", 0))
        if created and created < prune_cutoff:
            try:
                fp.unlink()
                pruned += 1
            except OSError:
                pass
            continue

        updated = float(rec.get("updated_at", created))
        if rec.get("status") in ("queued", "running") and updated < stale_cutoff:
            rec["status"] = "failed"
            rec["stage"] = "server_restart"
            rec["progress"] = 100
            rec["error"] = (
                "Server restarted while this job was still running. "
                "Resubmit to retry."
            )
            rec["updated_at"] = now_ts
            _atomic_write(fp, rec)
            revived_as_failed += 1

        _jobs[rec["job_id"]] = rec
        loaded += 1

    if loaded or pruned or revived_as_failed:
        log.info(
            f"job_store: loaded {loaded} jobs, pruned {pruned} expired, "
            f"marked {revived_as_failed} stale as failed"
        )
    return loaded


def create(input_info: dict[str, Any]) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:16]
    rec = {
        "job_id": job_id,
        "status": "queued",
        "created_at": _now(),
        "updated_at": _now(),
        "input_info": input_info,
        "progress": 0,
        "stage": "queued",
        "result": None,
        "error": None,
    }
    _jobs[job_id] = rec
    _atomic_write(_path(job_id), rec)
    return rec


def get(job_id: str) -> Optional[dict[str, Any]]:
    rec = _jobs.get(job_id)
    if rec is None:
        p = _path(job_id)
        if p.exists():
            rec = _read(p)
            if rec is not None:
                _jobs[job_id] = rec
    return rec


async def update(
    job_id: str,
    *,
    status: Optional[Status] = None,
    stage: Optional[str] = None,
    progress: Optional[int] = None,
    result: Optional[dict[str, Any]] = None,
    error: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    async with _lock(job_id):
        rec = _jobs.get(job_id)
        if rec is None:
            return None
        if status is not None:
            if status not in _STATES:
                raise ValueError(f"invalid status {status!r}")
            rec["status"] = status
        if stage is not None:
            rec["stage"] = stage
        if progress is not None:
            rec["progress"] = int(progress)
        if result is not None:
            rec["result"] = result
        if error is not None:
            rec["error"] = error
        rec["updated_at"] = _now()
        _atomic_write(_path(job_id), rec)
        return rec


def list_recent(limit: int = 50) -> list[dict[str, Any]]:
    out = sorted(_jobs.values(), key=lambda r: r.get("updated_at", 0), reverse=True)[:limit]
    return [{k: v for k, v in r.items() if k != "result"} for r in out]


def delete(job_id: str) -> bool:
    rec = _jobs.pop(job_id, None)
    p = _path(job_id)
    removed = False
    if p.exists():
        try:
            p.unlink()
            removed = True
        except OSError:
            pass
    _locks.pop(job_id, None)
    return removed or rec is not None


def prune_expired() -> int:
    cutoff = _now() - _TTL_SECONDS
    victims = [jid for jid, rec in _jobs.items() if rec.get("created_at", 0) < cutoff]
    for jid in victims:
        delete(jid)
    return len(victims)
