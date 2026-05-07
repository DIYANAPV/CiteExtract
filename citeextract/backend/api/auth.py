
from __future__ import annotations

import hmac

from fastapi.responses import HTMLResponse


ACCESS_TOKEN_COOKIE = "citeextract_token"
ACCESS_TOKEN_COOKIE_MAX_AGE = 60 * 60 * 24 * 7

_ACCESS_TOKEN_QUERY = "token"

PUBLIC_PATHS = {
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/manifest.json",
    "/favicon.ico",
}

GATE_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>CiteExtract: access required</title>
<style>
body{{font-family:-apple-system,system-ui,sans-serif;background:#f8fafc;color:#334155;
    display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0;}}
.box{{max-width:420px;padding:32px;background:#fff;border-radius:12px;
    box-shadow:0 4px 24px rgba(0,0,0,0.06);text-align:center;}}
h1{{margin:0 0 12px;font-size:22px;}}
p{{margin:8px 0;font-size:16px;line-height:1.5;}}
code{{background:#f1f5f9;padding:2px 6px;border-radius:4px;font-size:15px;}}
</style></head><body>
<div class="box">
<h1>Access required</h1>
<p>This instance of CiteExtract is gated for a specific set of users.</p>
<p>Append the access token to the URL, for example:</p>
<p><code>{url}?token=YOUR_TOKEN</code></p>
</div></body></html>"""


def token_gate_middleware(expected_token: str):

    async def middleware(request, call_next):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        submitted = (
            request.cookies.get(ACCESS_TOKEN_COOKIE)
            or request.query_params.get(_ACCESS_TOKEN_QUERY)
        )
        if not hmac.compare_digest(submitted or "", expected_token):
            base = f"{request.url.scheme}://{request.url.netloc}{request.url.path}"
            return HTMLResponse(GATE_PAGE.format(url=base), status_code=403)

        response = await call_next(request)
        response.set_cookie(
            ACCESS_TOKEN_COOKIE, expected_token,
            max_age=ACCESS_TOKEN_COOKIE_MAX_AGE,
            httponly=True, samesite="lax",
        )
        return response

    return middleware
