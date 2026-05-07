
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.testclient import TestClient

from citeextract.api.auth import (
    ACCESS_TOKEN_COOKIE,
    token_gate_middleware,
)


def _make_app(token: str) -> FastAPI:
    app = FastAPI()
    app.middleware("http")(token_gate_middleware(token))

    @app.get("/protected")
    def _p():
        return PlainTextResponse("yes")

    @app.get("/health")
    def _h():
        return PlainTextResponse("ok")

    return app


def test_without_token_returns_403_with_gate_page():
    with TestClient(_make_app("secret")) as c:
        r = c.get("/protected")
        assert r.status_code == 403
        assert "Access required" in r.text


def test_health_path_bypasses_gate():
    with TestClient(_make_app("secret")) as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert r.text == "ok"


def test_query_param_grants_access_and_sets_cookie():
    with TestClient(_make_app("secret")) as c:
        r = c.get("/protected?token=secret")
        assert r.status_code == 200
        assert ACCESS_TOKEN_COOKIE in r.cookies


def test_cookie_grants_access_on_subsequent_request():
    with TestClient(_make_app("secret")) as c:
        c.get("/protected?token=secret")
        r = c.get("/protected")
        assert r.status_code == 200


def test_wrong_token_returns_403():
    with TestClient(_make_app("secret")) as c:
        r = c.get("/protected?token=wrong")
        assert r.status_code == 403
