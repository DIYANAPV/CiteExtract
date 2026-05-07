
from __future__ import annotations

import logging
import os

import gradio as gr

from citeextract import paths
from citeextract.api.auth import token_gate_middleware
from citeextract_ui.styles import BASE_CSS
from citeextract_ui.theme import build_theme
from citeextract_ui.ui.layout import create_app


logging.basicConfig(level=logging.INFO)


def _build_fastapi_with_health():
    from fastapi import FastAPI
    from fastapi.responses import PlainTextResponse
    from fastapi.staticfiles import StaticFiles

    api = FastAPI()

    expected = os.environ.get("REVIEW_ACCESS_TOKEN", "").strip()
    if expected:
        api.middleware("http")(token_gate_middleware(expected))
        logging.getLogger(__name__).info(
            "REVIEW_ACCESS_TOKEN set — token gate enabled for all non-/health routes"
        )

    @api.get("/health")
    def _health():
        return PlainTextResponse("ok")

    assets_dir = paths.repo_root() / "assets"
    if assets_dir.is_dir():
        api.mount("/cc-static", StaticFiles(directory=str(assets_dir)), name="cc_static")

    demo = create_app()
    fonts_head = (
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?'
        'family=Crimson+Pro:wght@400;500;600;700&display=swap" rel="stylesheet">'
    )
    return gr.mount_gradio_app(
        api, demo, path="/",
        theme=build_theme(),
        css=BASE_CSS,
        footer_links=[],
        head=fonts_head,
    )


def main() -> None:
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(_build_fastapi_with_health(), host="0.0.0.0", port=port)
