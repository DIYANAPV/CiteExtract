
from __future__ import annotations

import gradio as gr

from citeextract_ui.render.helpers import _esc, _strip_html
from citeextract_ui.render.pipeline import (
    PIPELINE_STAGES_DEF,
    _applicable_stages,
)
from citeextract_ui.styles import BASE_CSS
from citeextract_ui.theme import (
    CLAIM_VERDICT_STYLES,
    RISK_COLORS,
    VERDICT_STYLES,
    build_theme,
)


def test_esc_handles_html_and_none():
    assert _esc("<script>") == "&lt;script&gt;"
    assert _esc(None) == ""
    assert _esc("") == ""


def test_strip_html_removes_jats_markup():
    assert _strip_html("<jats:p>hello</jats:p>") == "hello"
    assert _strip_html(None) == ""


def test_base_css_loads_and_is_substantial():
    assert isinstance(BASE_CSS, str)
    assert len(BASE_CSS) > 10_000
    assert ":root" in BASE_CSS


def test_verdict_styles_contains_expected_keys():
    for k in ("VALID", "FABRICATED", "UNVERIFIABLE"):
        assert k in VERDICT_STYLES
        assert "color" in VERDICT_STYLES[k]
    for k in ("SUPPORTED", "CONTRADICTS", "NEUTRAL"):
        assert k in CLAIM_VERDICT_STYLES
    for k in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        assert k in RISK_COLORS


def test_build_theme_returns_gradio_theme():
    assert isinstance(build_theme(), gr.themes.Soft)


def test_applicable_stages_includes_existence_and_finalize():
    stages = _applicable_stages(existence=True, claims=False)
    ids = [sid for sid, _label in stages]
    assert "L1_parse" in ids
    assert "L2_existence" in ids
    assert "__finalize__" in ids
    assert "agentic_metadata_dispatch" not in ids


def test_pipeline_stages_def_shape():
    for sid, label, predicate in PIPELINE_STAGES_DEF:
        assert isinstance(sid, str) and sid
        assert isinstance(label, str) and label
        assert callable(predicate)
