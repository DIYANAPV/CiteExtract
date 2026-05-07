
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from citeextract.api import app


def test_app_is_fastapi_instance():
    assert isinstance(app, FastAPI)


def test_expected_routes_are_registered():
    methods_by_path: dict[str, set[str]] = {}
    for r in app.routes:
        if not hasattr(r, "methods"):
            continue
        methods_by_path.setdefault(r.path, set()).update(r.methods - {"HEAD"})

    expected = {
        "/api/v1": {"GET"},
        "/api/v1/health": {"GET"},
        "/api/v1/manifest": {"GET"},
        "/api/v1/verify": {"POST"},
        "/api/v1/jobs": {"GET", "POST"},
        "/api/v1/jobs/{job_id}": {"GET", "DELETE"},
    }
    for path, want in expected.items():
        assert path in methods_by_path, f"route missing: {path}"
        assert want.issubset(methods_by_path[path]), (
            f"route {path} missing methods {want - methods_by_path[path]}"
        )


def test_health_endpoint_returns_ok():
    with TestClient(app) as client:
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"


def test_root_endpoint_returns_landing_payload():
    with TestClient(app) as client:
        r = client.get("/api/v1")
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "CiteExtract API"
        assert body["sync_endpoint"] == "/api/v1/verify"
