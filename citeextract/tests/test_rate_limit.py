
from __future__ import annotations

import time as _time
from collections import defaultdict

import pytest

from citeextract import rate_limit


class _FakeRequest:
    def __init__(self, ip: str = "1.2.3.4", xff: str | None = None):
        self.client = type("C", (), {"host": ip})()
        self.headers = {"x-forwarded-for": xff} if xff else {}


def test_client_ip_returns_unknown_for_None():
    assert rate_limit.client_ip(None) == "unknown"


def test_client_ip_prefers_xff_header():
    r = _FakeRequest(ip="10.0.0.1", xff="203.0.113.5, 10.0.0.1")
    assert rate_limit.client_ip(r) == "203.0.113.5"


def test_client_ip_falls_back_to_client_host():
    r = _FakeRequest(ip="10.0.0.1")
    assert rate_limit.client_ip(r) == "10.0.0.1"


def test_check_batch_zero_raises():
    with pytest.raises(rate_limit.RateLimitExceeded, match="at least one"):
        rate_limit.check_batch(0)


def test_check_batch_too_many_raises():
    with pytest.raises(rate_limit.RateLimitExceeded, match="Maximum"):
        rate_limit.check_batch(rate_limit.BATCH_MAX_PAPERS + 1)


def test_check_single_passes_under_limit(monkeypatch):
    monkeypatch.setattr(rate_limit, "_daily_analyses", defaultdict(int))
    monkeypatch.setattr(rate_limit, "_ip_requests", defaultdict(list))
    rate_limit.check_single(_FakeRequest(ip="9.9.9.9"))


def test_check_single_over_hourly_ip_limit_raises(monkeypatch):
    now = _time.time()
    saturated = defaultdict(list)
    saturated["1.2.3.4"] = [now] * rate_limit.HOURLY_IP_LIMIT
    monkeypatch.setattr(rate_limit, "_daily_analyses", defaultdict(int))
    monkeypatch.setattr(rate_limit, "_ip_requests", saturated)
    with pytest.raises(rate_limit.RateLimitExceeded, match="Hourly limit"):
        rate_limit.check_single(_FakeRequest(ip="1.2.3.4"))
