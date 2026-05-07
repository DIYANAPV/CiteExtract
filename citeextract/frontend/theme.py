
from __future__ import annotations

import gradio as gr


VERDICT_STYLES = {
    "VALID":        {"color": "#15803D", "bg": "#F0F7F1", "border": "#A3C6A6", "icon": "&#x2705;", "label": "Valid"},
    "FABRICATED":   {"color": "#991B1B", "bg": "#FDF3F3", "border": "#F07979", "icon": "&#x274C;", "label": "Fabricated"},
    "UNVERIFIABLE": {"color": "#525252", "bg": "#F2F5F7", "border": "#C7D6E1", "icon": "&#x2753;", "label": "Unverifiable"},
}

CLAIM_VERDICT_STYLES = {
    "SUPPORTED":     {"color": "#15803D", "bg": "#F0F7F1", "border": "#A3C6A6", "label": "Supported"},
    "SUPPORTS":      {"color": "#15803D", "bg": "#F0F7F1", "border": "#A3C6A6", "label": "Supports"},
    "CONTRADICTS":   {"color": "#991B1B", "bg": "#FDF3F3", "border": "#F07979", "label": "Contradicts"},
    "NEUTRAL":       {"color": "#B45309", "bg": "#FBF5E8", "border": "#E0B678", "label": "Neutral"},
    "UNVERIFIABLE":  {"color": "#525252", "bg": "#F2F5F7", "border": "#C7D6E1", "label": "Unverifiable"},
    "NOT_SUPPORTED": {"color": "#991B1B", "bg": "#FDF3F3", "border": "#F07979", "label": "Not Supported"},
}

RISK_COLORS = {"LOW": "#15803D", "MEDIUM": "#B45309", "HIGH": "#991B1B", "CRITICAL": "#7F1D1D"}


def build_theme() -> gr.themes.Soft:
    from gradio.themes.utils.colors import Color
    brand_primary = Color(
        name="brand_primary",
        c50="#eef3f8",  c100="#d3e0ec", c200="#a6c1d8", c300="#7aa2c5",
        c400="#4d83b1", c500="#2e6489", c600="#004276", c700="#003862",
        c800="#002d4f", c900="#00223b", c950="#001827",
    )
    brand_neutral = Color(
        name="brand_neutral",
        c50="#f2f5f7",  c100="#ecf2f3", c200="#E1E9EF", c300="#C7D6E1",
        c400="#aec5cb", c500="#98b2bc", c600="#7994a3", c700="#6f8593",
        c800="#5b6d78", c900="#4d5b62", c950="#3a474e",
    )
    return gr.themes.Soft(
        primary_hue=brand_primary,
        secondary_hue=brand_neutral,
        neutral_hue=brand_neutral,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        radius_size=gr.themes.sizes.radius_md,
    )
