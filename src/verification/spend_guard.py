"""Persistent monthly OpenAI spend ceiling.

Records every LLM call's USD cost in a per-month JSON ledger
(``data/cache/spend/<YYYY-MM>.json``) and raises ``BudgetExceededError``
when the cumulative total reaches ``MONTHLY_BUDGET_USD``.

Set ``MONTHLY_BUDGET_USD=150`` (or your number) in the deploy environment.
Leaving it unset disables the cap entirely — useful for local dev and tests.

Thread- and process-safe within a single host: writes are guarded by an
``fcntl.flock`` exclusive lock. The Cloud Run main service runs with
``min=max=1`` so a single host is the operational case.

Survives instance restarts by living on local disk; if Cloud Run evicts the
instance, the file goes with it. For peer-review traffic this is acceptable —
restarts are rare and the OpenAI dashboard remains the second pair of eyes.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


class BudgetExceededError(RuntimeError):
    """Monthly OpenAI budget exhausted; new analyses must be refused."""


def _ledger_dir() -> Path:
    return Path(os.environ.get("SPEND_LEDGER_DIR", "data/cache/spend"))


def _ledger_path() -> Path:
    d = _ledger_dir()
    d.mkdir(parents=True, exist_ok=True)
    # UTC so the month rollover doesn't depend on the host's timezone.
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
    """Add a single LLM call's USD cost to the current month's ledger.

    Never raises on a transient I/O issue — failure to record is logged
    and the call proceeds. The hard refuse happens via ``check_budget()``
    at request entry points; this function is the recorder side only.
    """
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
    """Return the current month's total USD spend (0.0 if no ledger yet)."""
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
    """Raise ``BudgetExceededError`` if this month's spend has crossed the cap.

    No-op when ``MONTHLY_BUDGET_USD`` is unset, empty, non-numeric, or <= 0.
    """
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
