
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict

from citeextract.verification import spend_guard


class RateLimitExceeded(Exception):
    pass


DAILY_LIMIT = int(os.environ.get("DAILY_ANALYSIS_LIMIT", "40"))
HOURLY_IP_LIMIT = int(os.environ.get("HOURLY_IP_LIMIT", "10"))
HOURLY_IP_BATCH_LIMIT = int(os.environ.get("HOURLY_IP_BATCH_LIMIT", "2"))
BATCH_MAX_PAPERS = int(os.environ.get("BATCH_MAX_PAPERS_PER_BATCH", "30"))

_HOUR_SECONDS = 3600
_IP_PRUNE_INTERVAL_SECONDS = 600

_daily_analyses: dict[str, int] = defaultdict(int)
_daily_lock = threading.Lock()

_ip_requests: dict[str, list[float]] = defaultdict(list)
_ip_batch_requests: dict[str, list[float]] = defaultdict(list)
_ip_lock = threading.Lock()


def _prune_ip_throttle() -> None:
    cutoff = time.time() - _HOUR_SECONDS
    with _ip_lock:
        for store in (_ip_requests, _ip_batch_requests):
            for key, stamps in list(store.items()):
                live = [t for t in stamps if t > cutoff]
                if live:
                    store[key] = live
                else:
                    del store[key]


def _schedule_ip_prune() -> None:
    try:
        _prune_ip_throttle()
    finally:
        t = threading.Timer(_IP_PRUNE_INTERVAL_SECONDS, _schedule_ip_prune)
        t.daemon = True
        t.start()


_schedule_ip_prune()


def client_ip(request) -> str:
    if request is None:
        return "unknown"
    headers = getattr(request, "headers", {}) or {}
    fwd = headers.get("x-forwarded-for") or headers.get("X-Forwarded-For")
    if fwd:
        return fwd.split(",")[0].strip()
    client = getattr(request, "client", None)
    if client and getattr(client, "host", None):
        return client.host
    return "unknown"


def reserve_daily_papers(n: int) -> None:
    today = time.strftime("%Y-%m-%d")
    with _daily_lock:
        if _daily_analyses[today] + n > DAILY_LIMIT:
            remaining = max(0, DAILY_LIMIT - _daily_analyses[today])
            raise RateLimitExceeded(
                f"Daily analysis limit ({DAILY_LIMIT}) would be exceeded. "
                f"Only {remaining} paper(s) remaining today. "
                "This is a research demo, please try again tomorrow."
            )
        _daily_analyses[today] += n
        for k in list(_daily_analyses):
            if k != today:
                del _daily_analyses[k]


def check_monthly_budget() -> None:
    try:
        spend_guard.check_budget()
    except spend_guard.BudgetExceededError as e:
        raise RateLimitExceeded(str(e)) from None


def check_single(request=None) -> None:
    check_monthly_budget()

    ip = client_ip(request)
    now = time.time()
    cutoff = now - _HOUR_SECONDS
    with _ip_lock:
        recent = [t for t in _ip_requests[ip] if t > cutoff]
        if len(recent) >= HOURLY_IP_LIMIT:
            raise RateLimitExceeded(
                f"Hourly limit reached for your IP ({HOURLY_IP_LIMIT} single analyses/hour). "
                "Please try again later."
            )
        reserve_daily_papers(1)

        recent.append(now)
        _ip_requests[ip] = recent
        if len(_ip_requests) > 1000:
            for key, stamps in list(_ip_requests.items()):
                live = [t for t in stamps if t > cutoff]
                if live:
                    _ip_requests[key] = live
                else:
                    del _ip_requests[key]


def check_batch(n_papers: int, request=None) -> None:
    if n_papers <= 0:
        raise RateLimitExceeded("Please upload at least one paper.")
    if n_papers > BATCH_MAX_PAPERS:
        raise RateLimitExceeded(
            f"Too many papers in one batch ({n_papers}). "
            f"Maximum is {BATCH_MAX_PAPERS} per submission."
        )
    check_monthly_budget()

    ip = client_ip(request)
    now = time.time()
    cutoff = now - _HOUR_SECONDS
    with _ip_lock:
        recent = [t for t in _ip_batch_requests[ip] if t > cutoff]
        if len(recent) >= HOURLY_IP_BATCH_LIMIT:
            raise RateLimitExceeded(
                f"Hourly batch limit reached for your IP "
                f"({HOURLY_IP_BATCH_LIMIT} batches/hour). "
                "Please try again later."
            )

        reserve_daily_papers(n_papers)

        recent.append(now)
        _ip_batch_requests[ip] = recent
