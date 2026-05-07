
from __future__ import annotations

import fcntl
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from citeextract import paths

log = logging.getLogger(__name__)


class BudgetExceededError(RuntimeError):
    pass


def _ledger_dir() -> Path:
    override = os.environ.get("SPEND_LEDGER_DIR")
    if override:
        return Path(override)
    return paths.data_dir() / "cache" / "spend"


def _ledger_path() -> Path:
    d = _ledger_dir()
    d.mkdir(parents=True, exist_ok=True)
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    return d / f"{month}.json"


def _budget_usd() -> float | None:
    raw = os.environ.get("MONTHLY_BUDGET_USD", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        log.warning("MONTHLY_BUDGET_USD=%r is not a number; cap disabled", raw)
        return None
    return value if value > 0 else None


def _read_locked(f) -> dict:
    f.seek(0)
    raw = f.read()
    if not raw.strip():
        return {"total_usd": 0.0, "calls": 0}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("spend ledger corrupt; treating this month as $0")
        return {"total_usd": 0.0, "calls": 0}


def record(cost_usd: float) -> None:
    if cost_usd <= 0:
        return
    try:
        path = _ledger_path()
        with open(path, "a+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                data = _read_locked(f)
                data["total_usd"] = float(data.get("total_usd", 0.0)) + float(cost_usd)
                data["calls"] = int(data.get("calls", 0)) + 1
                f.seek(0)
                f.truncate()
                json.dump(data, f, indent=2)
                f.write("\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except OSError as e:
        log.warning("could not write spend ledger: %s", e)


def current_spend_usd() -> float:
    path = _ledger_path()
    if not path.exists():
        return 0.0
    try:
        with open(path, "r") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                data = _read_locked(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return float(data.get("total_usd", 0.0))
    except OSError as e:
        log.warning("could not read spend ledger: %s", e)
        return 0.0


def check_budget() -> None:
    budget = _budget_usd()
    if budget is None:
        return
    spent = current_spend_usd()
    if spent >= budget:
        raise BudgetExceededError(
            f"Monthly OpenAI budget of ${budget:.2f} reached "
            f"(spent ${spent:.2f}). Service paused until next month "
            f"or until the cap is raised."
        )
