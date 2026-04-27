"""CheckCitation Web UI — local Gradio interface for citation verification.

Launch with:
    python app.py

Requires:
    - GROBID running for PDF input: docker run -d -p 8070:8070 grobid/grobid:0.8.2-crf
    - .env with OPENAI_API_KEY for Claim Verification (Agentic mode)
"""

import hmac
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Optional

import gradio as gr
import requests

from src import config
from src.models.report import PaperReport
from src.pipeline import run_unified
from src.verification import spend_guard

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Verdict palette — keep these hex values in sync with the --verdict-* CSS
# variables in BASE_CSS below. Inline styles in rendered HTML can't read CSS
# vars, so the values are duplicated here on purpose.
VERDICT_STYLES = {
    "VALID":        {"color": "#15803D", "bg": "#F0F7F1", "border": "#A3C6A6", "icon": "&#x2705;", "label": "Valid"},
    "FABRICATED":   {"color": "#991B1B", "bg": "#FDF3F3", "border": "#F07979", "icon": "&#x274C;", "label": "Fabricated"},
    "UNVERIFIABLE": {"color": "#525252", "bg": "#F2F5F7", "border": "#C7D6E1", "icon": "&#x2753;", "label": "Unverifiable"},
}

# Claim dimension — both per-sentence (SUPPORTS/CONTRADICTS/NEUTRAL) and
# per-reference (SUPPORTED/CONTRADICTS/NEUTRAL/UNVERIFIABLE) reuse this map.
# The 2-class agent labels (SUPPORTED/NOT_SUPPORTED) also map here for the
# few code paths that read them directly before merger normalisation.
CLAIM_VERDICT_STYLES = {
    # Per-reference rollup + per-sentence 3-class
    "SUPPORTED":     {"color": "#15803D", "bg": "#F0F7F1", "border": "#A3C6A6", "label": "Supported"},
    "SUPPORTS":      {"color": "#15803D", "bg": "#F0F7F1", "border": "#A3C6A6", "label": "Supports"},
    "CONTRADICTS":   {"color": "#991B1B", "bg": "#FDF3F3", "border": "#F07979", "label": "Contradicts"},
    "NEUTRAL":       {"color": "#B45309", "bg": "#FBF5E8", "border": "#E0B678", "label": "Neutral"},
    "UNVERIFIABLE":  {"color": "#525252", "bg": "#F2F5F7", "border": "#C7D6E1", "label": "Unverifiable"},
    # 2-class scheme (raw agent output, before merger normalisation)
    "NOT_SUPPORTED": {"color": "#991B1B", "bg": "#FDF3F3", "border": "#F07979", "label": "Not Supported"},
}

RISK_COLORS = {"LOW": "#15803D", "MEDIUM": "#B45309", "HIGH": "#991B1B", "CRITICAL": "#7F1D1D"}

BASE_CSS = """
/* ── Design tokens — single source of truth ──
   All palette / typography / spacing values live here. Anything elsewhere
   in this stylesheet that still hard-codes a hex value is incremental
   cleanup for a later stage; new rules should reference these vars. */
:root {
  /* Surfaces */
  --bg:           #FAF9F5;            /* warm ivory page background */
  --bg-alt:       #F3F1EA;            /* alternating section bands / hint chips */
  --surface:      #FFFFFF;            /* card / panel background */
  --border:       #E7E9EA;            /* hairline borders */
  --border-muted: rgba(0,0,0,0.08);

  /* Text */
  --text:         #222222;            /* primary body */
  --text-strong:  #0f172a;            /* headings (true near-black for emphasis) */
  --text-muted:   #5B6D78;            /* secondary / captions */

  /* Accent (links, primary CTA) — Distill navy */
  --accent:       #004276;
  --accent-hover: #0066AA;
  --accent-soft:  rgba(0, 66, 118, 0.06);   /* faint accent tint for hovers */

  /* Verdict semantics — independent from brand accent */
  --verdict-ok:        #15803D;       /* SUPPORTED */
  --verdict-ok-bg:     #F0F7F1;
  --verdict-ok-border: #A3C6A6;
  --verdict-warn:        #B45309;     /* NEUTRAL / PARTIAL */
  --verdict-warn-bg:     #FBF5E8;
  --verdict-warn-border: #E0B678;
  --verdict-err:        #991B1B;      /* FABRICATED / CONTRADICTS */
  --verdict-err-bg:     #FDF3F3;
  --verdict-err-border: #F07979;
  --verdict-info:        #525252;     /* UNVERIFIABLE / NEUTRAL */
  --verdict-info-bg:     #F2F5F7;
  --verdict-info-border: #C7D6E1;

  /* Distill-style cream callout (for "Note:" / "Limitation:" boxes) */
  --callout-bg:   hsl(54, 78%, 96%);
  --callout-rule: hsl(54, 60%, 70%);

  /* Type stacks */
  --font-serif: "Crimson Pro", "Source Serif 4", Charter, Georgia, serif;
  --font-sans:  "Inter", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --font-mono:  "JetBrains Mono", Consolas, Monaco, monospace;
}

/* ── Selective serif (Distill / arXiv academic feel) ──
   Apply --font-serif to headings and research-prose surfaces only.
   The universal sans rule below uses !important, so these overrides do too;
   without it the cascade resets every descendant back to Inter. */
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.cc-hero h1,
.cc-feature-title,
.cc-about-section h3,
.cc-about-section p {
    font-family: var(--font-serif) !important;
    letter-spacing: -0.005em;
}

/* ── Hide Gradio chrome ──
   `footer_links=[]` on mount_gradio_app drops the API link / Gradio logo /
   settings gear in Gradio 6+, but older / inner footer elements can still
   leak through. These rules are defensive belt-and-braces. */
footer,
gradio-app footer,
.gradio-container footer,
.api-link, .built-with, .show-api,
.svelte-1lcyrx4 footer { display: none !important; }
[data-testid="settings-button"],
button[aria-label="Settings"],
.icon-button.settings { display: none !important; }

/* ── Layout ── */
.gradio-container {
    /* Narrowed from 1680px — academic / research-tool sites read better at
       a moderate column width. Form widgets still get plenty of room. */
    max-width: 1200px !important;
    width: 100% !important;
    margin: 0 auto !important;
    min-height: 100vh !important;
    padding: 24px 32px 64px !important;
    font-family: var(--font-sans) !important;
    font-size: 15px !important;
    color: var(--text);
}
/* Gradio 6 wraps content in <main class="main"> and <div class="wrap contain">
   each with their own max-width. Knock them out so the UI fills the page. */
.gradio-container > main,
.gradio-container > main > .wrap,
.gradio-container .contain,
.gradio-container .main,
gradio-app { max-width: 100% !important; width: 100% !important; }
/* The page background must override Gradio's own body / container styles,
   which otherwise leak through as a cool slate. Belt-and-braces selector
   list covers Gradio 4-6 internal class shuffles. */
html, body,
gradio-app,
gradio-app > div,
gradio-app .gradio-container,
gradio-app > main,
.gradio-container,
.gradio-container > .main { background: var(--bg) !important; }
/* Propagate Inter to every descendant — Gradio 4+ sets fonts on many
   individual elements with higher specificity than .gradio-container.  */
.gradio-container, .gradio-container *,
.gradio-container button, .gradio-container input,
.gradio-container textarea, .gradio-container select {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* ── Subtle entrance ── */
@keyframes cc-fadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Hero ──
   Hairline border + no shadow (Distill / arXiv-style restraint).
   Centered ~820px column for now — when a hero screenshot lands later,
   widen back to full-width and place the image on the right. */
.cc-hero {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 40px 44px;
    margin: 0 0 32px;
    display: flex; gap: 32px; align-items: center;
}
.cc-hero-text { flex: 1; min-width: 0; }
.cc-hero h1 {
    font-size: 32px; font-weight: 700; color: var(--text-strong);
    letter-spacing: -0.015em; margin: 0 0 14px; line-height: 1.2;
}
.cc-hero h1 .cc-hero-accent {
    /* Pull one phrase of the headline out in accent navy without underline
       gimmicks — keeps the academic register while adding visual rhythm. */
    color: var(--accent);
}
.cc-hero-sub {
    font-size: 15.5px; color: var(--text); margin: 0; font-weight: 400;
    line-height: 1.65;
}

/* ── Feature strip (three cards, below hero) ──
   Each card carries a small monoline icon at the top-left, a pipeline-step
   number on the right, and a single sentence of body. The dotted-grid
   background under the row gives it a subtle texture without needing a
   colored band — borrowed from scite.ai's section-separation trick. */
.cc-features {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin: 0 0 40px;
    padding: 14px;
    background-image: radial-gradient(circle at 1px 1px,
        rgba(0, 66, 118, 0.10) 1px, transparent 0);
    background-size: 14px 14px;
    border-radius: 14px;
}
.cc-feature {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 22px 22px 24px;
    transition: border-color 0.15s ease, transform 0.15s ease;
    position: relative;
}
.cc-feature:hover {
    border-color: var(--accent);
    transform: translateY(-1px);
}
.cc-feature-icon {
    display: inline-flex; align-items: center; justify-content: center;
    color: var(--accent);
    margin-bottom: 14px;
}
.cc-feature-icon svg { width: 32px; height: 32px; }
/* Duotone fill: a low-opacity wash of the same colour as the stroke,
   placed on a copy of the silhouette path. Phosphor-style depth without
   sacrificing the monochrome / academic register. */
.cc-feature-icon svg .duo-fill { opacity: 0.20; }
.cc-feature-step {
    position: absolute; top: 18px; right: 20px;
    font-size: 11px; font-weight: 700; letter-spacing: 0.08em;
    color: var(--text-muted);
    font-family: var(--font-mono);
}
.cc-feature-title {
    font-size: 18px; font-weight: 700; color: var(--text-strong);
    margin: 0 0 10px; letter-spacing: 0; line-height: 1.3;
}
.cc-feature-body {
    font-size: 14px; color: var(--text-muted); line-height: 1.6; margin: 0;
}

/* ── File status chip (post-upload summary + size-limit warning) ── */
.cc-file-status {
    margin: 8px 0 4px; min-height: 0;
    font-size: 13px; line-height: 1.4;
    display: none;          /* hidden until JS populates it */
}
.cc-file-status.cc-has-content { display: block; }
.cc-file-chip {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 12px; border-radius: 999px;
    font-size: 12.5px; font-weight: 500;
    background: #F0F7F1; color: #2F6B33; border: 1px solid #A3C6A6;
}
.cc-file-chip.cc-file-over {
    background: var(--verdict-err-bg); color: var(--verdict-err); border-color: var(--verdict-err-border);
}
.cc-file-chip-name {
    font-weight: 600; max-width: 340px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.cc-file-chip-size { opacity: 0.75; }
.cc-file-chip-warn { font-weight: 600; }

/* ── Discoverability hint under Claim Verification checkbox ── */
.cc-hint {
    font-size: 12.5px; color: #5B6D78; line-height: 1.5;
    padding: 10px 12px; margin: 4px 0 12px;
    background: #F2F5F7; border: 1px solid #E7E9EA;
    border-radius: 10px;
}
.cc-hint b { color: #334155; }

/* ── Drag-and-drop feedback ── */
.cc-drop-overlay {
    position: fixed; inset: 0; z-index: 9998;
    background: rgba(174, 19, 19, 0.06);
    backdrop-filter: blur(1px);
    display: none; align-items: center; justify-content: center;
    pointer-events: none;
    animation: cc-fadeIn 0.15s ease-out both;
}
.cc-drop-overlay.cc-active { display: flex; }
.cc-drop-overlay-msg {
    font-size: 20px; font-weight: 700; color: var(--accent);
    background: var(--surface); padding: 20px 32px; border-radius: 20px;
    border: 2px dashed var(--accent);
    box-shadow: 0 8px 48px rgba(0, 66, 118, 0.18);
}
body.cc-dragging .gradio-container [data-testid="file"],
body.cc-dragging .gradio-container .gr-file {
    border-color: var(--accent) !important;
    background: var(--accent-soft) !important;
}

/* ── About-page sections ──
   Each section is a card with a thin accent left-rail so the page reads as
   a stack of distinct ideas rather than a wall of prose. h3 headings pick
   up the same Distill navy as the hero accent for visual continuity. */
.cc-about-section {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
    border-left: 3px solid var(--accent);
    padding: 30px 38px; margin: 0 0 18px;
}
.cc-about-section p,
.cc-about-section ul,
.cc-about-section ol {
    /* Constrain prose width for readability while keeping the card itself
       full-width so it lines up with everything else on the page. 960px
       matches the docs-site convention used by MDN / GitHub / Notion. */
    max-width: 960px;
}
.cc-about-section h3 {
    font-size: 20px; font-weight: 700; color: var(--accent);
    letter-spacing: -0.01em; margin: 0 0 14px;
}
.cc-about-section p {
    font-size: 16px; color: var(--text); line-height: 1.7;
    margin: 0 0 12px;
}
.cc-about-section ul,
.cc-about-section ol {
    font-size: 16px; color: var(--text); line-height: 1.75;
}
.cc-about-section li { margin: 0 0 6px; }
.cc-about-section li strong { color: var(--text-strong); }
.cc-about-section a {
    color: var(--accent); text-decoration: none; font-weight: 500;
    border-bottom: 1px dashed #F07979; padding-bottom: 1px;
    transition: all 0.15s ease;
}
.cc-about-section a:hover { color: var(--accent-hover); border-bottom-color: var(--accent-hover); border-bottom-style: solid; }

/* ── FAQ accordion (inside About tab) ── */
.cc-faq {
    border-top: 1px solid var(--border);
    padding: 14px 0;
}
.cc-faq:last-of-type { border-bottom: 1px solid var(--border); }
.cc-faq summary {
    cursor: pointer;
    font-size: 15px;
    font-weight: 600;
    color: var(--text-strong);
    list-style: none;
    padding: 0;
    transition: color 0.15s ease;
}
.cc-faq summary::-webkit-details-marker { display: none; }
.cc-faq summary::before {
    content: "+";
    display: inline-block;
    width: 16px;
    color: var(--text-muted);
    font-weight: 400;
    font-family: var(--font-sans);
}
.cc-faq[open] summary::before { content: "−"; }
.cc-faq summary:hover { color: var(--accent); }
.cc-faq p {
    margin: 10px 0 0 16px;
    font-size: 14px;
    color: var(--text);
    line-height: 1.65;
}
.cc-faq code {
    background: var(--bg-alt);
    padding: 1px 5px;
    border-radius: 4px;
    font-family: var(--font-mono);
    font-size: 0.9em;
    border: 1px solid var(--border);
}

/* ── Page footer (academic ISSN-style version pill, not Built-with-Gradio) ── */
.cc-page-footer {
    border-top: 1px solid var(--border);
    margin: 64px 12px 0;
    padding: 20px 0 8px;
    font-size: 12.5px;
    color: var(--text-muted);
    line-height: 1.5;
    font-family: var(--font-sans);
    text-align: center;
}
.cc-page-footer code {
    font-family: var(--font-mono);
    font-size: 0.95em;
    background: var(--bg-alt);
    padding: 1px 5px;
    border-radius: 4px;
    border: 1px solid var(--border);
}
.cc-page-footer .cc-footer-sep {
    margin: 0 8px;
    opacity: 0.5;
}

/* ── BibTeX block + Copy button ── */
.cc-bibtex-wrap {
    position: relative; margin: 12px 0 4px;
}
pre.cc-bibtex {
    background: #F2F5F7; border: 1px solid #E7E9EA; border-radius: 12px;
    padding: 16px 20px; margin: 0; overflow-x: auto;
    font-family: ui-monospace, 'JetBrains Mono', 'SF Mono', Menlo, monospace;
    font-size: 12.5px; line-height: 1.55; color: #1e293b;
    white-space: pre;
}
.cc-bibtex-copy {
    position: absolute; top: 10px; right: 10px;
    font-family: inherit; font-size: 11px; font-weight: 600;
    padding: 4px 10px; border-radius: 8px;
    background: #ffffff; border: 1px solid #C7D6E1; color: #5B6D78;
    cursor: pointer; transition: all 0.15s ease;
}
.cc-bibtex-copy:hover { background: var(--bg-alt); color: var(--accent); border-color: var(--accent); }
.cc-bibtex-copy.cc-copied {
    background: #F0F7F1; color: #2F6B33; border-color: #A3C6A6;
}

/* ── Tab-page header (Batch / About) ── */
.cc-tabpage-header {
    padding: 20px 24px 16px; margin: 0 0 16px;
    border-bottom: 1px solid #E7E9EA;
}
.cc-tabpage-header h2 {
    font-size: 24px; font-weight: 700; color: #0f172a;
    letter-spacing: -0.02em; line-height: 1.2; margin: 0 0 8px;
}
.cc-tabpage-header p {
    font-size: 14px; color: #334155; line-height: 1.6; margin: 0 0 6px;
}
.cc-tabpage-header .cc-tabpage-meta {
    font-size: 13px; color: #5B6D78;
}
.cc-tabpage-header code {
    background: var(--bg-alt); padding: 1px 6px; border-radius: 6px;
    font-size: 12px; color: var(--accent); border: 1px solid var(--border);
    font-family: var(--font-mono);
}

/* ── Sticky context bar on results ── */
.cc-ctxbar {
    position: sticky; top: 0; z-index: 30;
    background: #ffffff; border: 1px solid #E7E9EA; border-radius: 14px;
    padding: 10px 16px; margin: 0 0 12px;
    display: flex; align-items: center; justify-content: space-between;
    gap: 12px; font-size: 13px;
    box-shadow: 0 4px 18px rgba(12, 63, 94, 0.08);
    backdrop-filter: saturate(1.1);
}
.cc-ctxbar-main {
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
    color: #334155;
}
.cc-ctxbar-file {
    font-weight: 700; color: #0f172a; letter-spacing: -0.01em;
    max-width: 420px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.cc-ctxbar-sep { color: #C7D6E1; }
.cc-ctxbar-chip {
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 12px; font-weight: 600;
}
/* Semantic — "this citation is flagged" reads as error, stays red. */
.cc-ctxbar-flagged { background: var(--verdict-err-bg); color: var(--verdict-err); border: 1px solid var(--verdict-err-border); }
.cc-ctxbar-clean   { background: #F0F7F1; color: #2F6B33; border: 1px solid #A3C6A6; }
.cc-ctxbar-btn {
    font-size: 12px; font-weight: 600; color: #5B6D78;
    padding: 6px 12px; border-radius: 10px;
    background: #F2F5F7; border: 1px solid #C7D6E1;
    text-decoration: none; white-space: nowrap;
    transition: all 0.15s ease;
}
.cc-ctxbar-btn:hover { background: var(--bg-alt); color: var(--accent); border-color: var(--border); }

/* ── In-page pipeline progress card ──
   Shown inside the Results region while run_analyze is executing. Each
   stage flips from pending (hollow dot) → active (spinner) → done (check).
   Replaces Gradio's generic horizontal progress bar with something that
   tells the user *what* is happening, not just *that* it is. */
.cc-pipeline {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px 28px;
    margin: 8px 0 0;
    animation: cc-fadeIn 0.25s ease-out both;
}
.cc-pipeline-header {
    display: flex; align-items: baseline; justify-content: space-between;
    gap: 16px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 12px;
    margin-bottom: 18px;
}
.cc-pipeline-title {
    font-size: 15px; font-weight: 600; color: var(--text-strong);
    letter-spacing: -0.005em;
}
.cc-pipeline-elapsed {
    font-size: 13px; color: var(--text-muted);
    font-variant-numeric: tabular-nums;
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
}
.cc-pipeline-stages {
    list-style: none; margin: 0; padding: 0;
    display: flex; flex-direction: column; gap: 2px;
}
.cc-step {
    display: flex; align-items: center; gap: 14px;
    padding: 10px 4px;
    font-size: 14px;
    transition: color 0.18s ease;
}
.cc-step + .cc-step {
    border-top: 1px dashed var(--border-muted);
}
.cc-step-icon {
    flex: 0 0 22px; width: 22px; height: 22px;
    display: inline-flex; align-items: center; justify-content: center;
}
.cc-step-icon-svg { width: 18px; height: 18px; }
.cc-step-dot {
    width: 9px; height: 9px; border-radius: 50%;
    border: 1.5px solid var(--border);
    background: transparent;
}
.cc-step-spinner {
    width: 16px; height: 16px; border-radius: 50%;
    border: 2px solid rgba(0, 66, 118, 0.18);
    border-top-color: var(--accent);
    animation: cc-spin 0.85s linear infinite;
}
@keyframes cc-spin { to { transform: rotate(360deg); } }
.cc-step-label { flex: 1; }
.cc-step-done   { color: var(--text); }
.cc-step-done .cc-step-icon { color: var(--verdict-ok); }
.cc-step-active {
    color: var(--text-strong);
    font-weight: 600;
    background: var(--accent-soft);
    border-radius: 6px;
    padding-left: 10px; padding-right: 10px;
    margin: 0 -6px;
}
.cc-step-pending { color: var(--text-muted); }

/* ── Contextual retry banner (Analyze tab, post-run) ──
   Surfaced only when the run produced FABRICATED / UNVERIFIABLE results
   that could plausibly be transient. Sits between the dashboard and the
   per-reference cards. */
.cc-retry-hint {
    flex: 1; min-width: 0;
    font-size: 13px; color: var(--text-muted);
    line-height: 1.5;
    align-self: center;
}
.cc-retry-btn {
    flex: 0 0 auto !important;
    min-width: 160px !important;
    align-self: center !important;
}

/* ── Resume banner ── */
.cc-resume-banner {
    background: var(--surface); border: 1px solid var(--border);
    border-left: 4px solid var(--accent); border-radius: 12px;
    padding: 20px 24px; margin: 0 0 20px;
    animation: cc-fadeIn 0.3s ease-out both;
}
.cc-resume-title {
    font-size: 16px; font-weight: 700; color: #0f172a;
    letter-spacing: -0.01em; margin: 0 0 12px;
}
.cc-resume-facts {
    font-size: 14px; color: #334155; line-height: 1.7;
    margin: 0 0 16px;
}
.cc-resume-file {
    font-weight: 600; color: #0f172a;
    font-family: 'Inter', system-ui, sans-serif;
}
.cc-resume-actions {
    display: flex; gap: 10px; justify-content: flex-end; flex-wrap: wrap;
}
.cc-resume-btn {
    font-family: inherit; font-size: 13px; font-weight: 600;
    padding: 8px 18px; border-radius: 10px; cursor: pointer;
    border: 1px solid transparent; transition: all 0.15s ease;
}
.cc-resume-btn-primary {
    background: var(--accent); color: #ffffff; border-color: var(--accent);
}
.cc-resume-btn-primary:hover { background: var(--accent-hover); border-color: var(--accent-hover); }
.cc-resume-btn-secondary {
    background: var(--surface); color: var(--text); border-color: var(--border);
}
.cc-resume-btn-secondary:hover { background: var(--bg-alt); border-color: var(--text-muted); }
.cc-resume-hint {
    margin-top: 12px; padding: 10px 14px;
    background: #FBF5E8; border: 1px solid #E0B678; border-radius: 10px;
    font-size: 13px; color: #9A6609; line-height: 1.5;
    animation: cc-fadeIn 0.25s ease-out both;
}

@media (max-width: 900px) {
    .cc-hero { padding: 24px 20px; }
    .cc-features { grid-template-columns: 1fr; }
}

/* ── Section labels ── */
.cc-section-label {
    font-size: 11px; font-weight: 700; color: #5B6D78;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin: 16px 0 10px; padding-bottom: 6px;
    border-bottom: 1px solid #E7E9EA;
}

/* ── Cards ── */
.card {
    border-radius: 12px; padding: 18px 22px; margin-bottom: 10px;
    border: 1px solid var(--border);
    border-left: 4px solid;  /* keeps the verdict-color accent stripe */
    background: var(--surface);
    animation: cc-fadeIn 0.2s ease-out both;
}
/* No card hover — each card has an inline per-verdict bg color (set in
   _render_verdict_card) which overrides any rule we'd put here. The
   verdict-stripe and the badge text are the visual cues; restraint reads
   as more academic anyway. */

/* Claim badge in card summary — secondary pill next to the metadata badge. */
.cc-claim-pill {
    display: inline-block;
    margin-left: 8px;
    padding: 2px 9px;
    border-radius: 999px;
    font-size: 11.5px;
    font-weight: 600;
    letter-spacing: 0.02em;
    vertical-align: middle;
}

/* Per-citation-context panel — each citing sentence + its agent decision +
   its retrieved passages live in their own visually distinct card so a
   reference cited in two paragraphs reads as two separate findings. */
.cc-context-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 18px;
    margin: 10px 0;
}
.cc-context-panel + .cc-context-panel { margin-top: 14px; }
.cc-context-label {
    font-size: 11px; font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 12px;
}

/* Three colored sections inside a citation-context panel.
   Order matters: claim → judgement → evidence. */

/* Section 1 — citing context (what the user wrote in their paper).
   Faint navy tint to read as "input" / "the question". */
.cc-citing-block {
    background: var(--accent-soft);
    border: 1px solid rgba(0, 66, 118, 0.12);
    border-left: 3px solid var(--accent);
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 10px;
}
.cc-citing-text {
    font-size: 14px;
    color: var(--text-strong);
    line-height: 1.6;
}

/* Section 2 — claim agent's decision. The bg/border is set inline by
   _format_claim_verdict using the verdict colour. The wrapper here just
   carries the small label above it. */
.cc-claim-block {
    margin-bottom: 10px;
}
.cc-claim-empty {
    font-size: 13px;
    color: var(--text-muted);
    background: var(--bg-alt);
    border: 1px dashed var(--border);
    border-radius: 6px;
    padding: 10px 14px;
    font-style: italic;
}

/* Section 3 — retrieved passages. Stays on plain white surface. */
.cc-passages-section {
    margin-top: 6px;
}

/* ── Evidence chips (Abstract / Retrieved passages) ──
   Below the claim agent's decision in each citing-context block. Each chip
   is a <details> styled as an inline button — collapsed by default so the
   card stays scannable; expanding shows the long evidence inline. */
.cc-evidence-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 10px;
}
.cc-evidence-chip {
    flex: 1 1 220px;
    min-width: 0;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    transition: border-color 0.15s ease;
}
.cc-evidence-chip:hover { border-color: var(--text-muted); }
.cc-evidence-chip[open] { flex-basis: 100%; border-color: var(--accent); }
.cc-evidence-chip > summary {
    list-style: none;
    cursor: pointer;
    padding: 8px 14px;
    font-size: 13px;
    font-weight: 600;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 8px;
    user-select: none;
}
.cc-evidence-chip > summary::after {
    content: "▸";
    margin-left: auto;
    font-size: 11px;
    color: var(--text-muted);
    transition: transform 0.15s ease;
}
.cc-evidence-chip[open] > summary::after { transform: rotate(90deg); }
.cc-evidence-chip > summary::-webkit-details-marker { display: none; }
.cc-chip-icon { font-size: 14px; opacity: 0.7; }
.cc-chip-body {
    padding: 0 14px 14px;
    font-size: 13.5px;
    line-height: 1.6;
    color: var(--text);
    border-top: 1px dashed var(--border-muted);
    padding-top: 10px;
    margin-top: 2px;
    max-height: 360px;
    overflow-y: auto;
}

/* Small uppercase mini-label used inside each section. */
.cc-section-mini-label {
    font-size: 10.5px;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.card summary { cursor: pointer; font-weight: 600; font-size: 15px; line-height: 1.5; }

/* Better empty state */
.cc-empty {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
    padding: 48px 24px; text-align: center; color: var(--text-muted);
}
.cc-empty-icon { font-size: 40px; margin-bottom: 12px; opacity: 0.4; }
.cc-empty-title { font-size: 17px; font-weight: 600; color: #334155; margin-bottom: 6px; }
.cc-empty-sub { font-size: 13px; color: #5B6D78; line-height: 1.5; max-width: 380px; margin: 0 auto; }

/* ── Skeleton shimmer for loading state ── */
@keyframes cc-shimmer {
    0%   { background-position: -1000px 0; }
    100% { background-position:  1000px 0; }
}
.cc-skel {
    background: linear-gradient(90deg,
        #E7E9EA 0%, #F2F5F7 50%, #E7E9EA 100%);
    background-size: 1000px 100%;
    animation: cc-shimmer 1.6s linear infinite;
    border-radius: 8px;
}
.cc-loading-card {
    background: #fff; border: 1px solid #E7E9EA; border-left: 4px solid #C7D6E1;
    border-radius: 20px; padding: 18px 22px; margin-bottom: 12px;
    box-shadow: 0 2px 24px rgba(12, 63, 94, 0.06);
}
.cc-loading-row { display: flex; gap: 10px; align-items: center; margin-bottom: 10px; }
.cc-loading-stage {
    background: #ffffff; border: 1px solid #E7E9EA; border-radius: 20px;
    padding: 20px 24px; margin-bottom: 14px;
    box-shadow: 0 2px 24px rgba(12, 63, 94, 0.08);
}
.cc-stage-list { display: flex; flex-direction: column; gap: 8px; margin: 14px 0 4px; }
.cc-stage {
    display: flex; align-items: center; gap: 10px;
    font-size: 14px; color: #5B6D78;
}
.cc-stage .cc-stage-dot {
    display: inline-block; width: 10px; height: 10px; border-radius: 50%;
    background: #E7E9EA; position: relative;
}
.cc-stage.cc-active .cc-stage-dot {
    background: var(--accent);
    box-shadow: 0 0 0 6px rgba(0, 66, 118, 0.12);
    animation: cc-pulse 1.3s ease-in-out infinite;
}
.cc-stage.cc-done .cc-stage-dot { background: var(--verdict-ok-border); }
.cc-stage.cc-done { color: var(--verdict-ok); }
.cc-stage.cc-active { color: var(--accent); font-weight: 600; }
@keyframes cc-pulse {
    0%   { box-shadow: 0 0 0 0    rgba(0, 66, 118, 0.35); }
    70%  { box-shadow: 0 0 0 10px rgba(0, 66, 118, 0);    }
    100% { box-shadow: 0 0 0 0    rgba(0, 66, 118, 0);    }
}
.cc-spinner {
    width: 16px; height: 16px; border: 2px solid var(--border);
    border-top-color: var(--accent); border-radius: 50%;
    animation: cc-spin 0.8s linear infinite;
    display: inline-block;
}
@keyframes cc-spin { to { transform: rotate(360deg); } }

/* ── Metadata table ── */
.meta-table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; }
.meta-table th, .meta-table td { text-align: left; padding: 7px 10px; border-bottom: 1px solid #f1f5f9; }
.meta-table th {
    background: #f8fafc; font-weight: 600; color: #475569;
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em;
}
.meta-table tr:hover td { background: #F2F5F7; }
.meta-table td { max-width: 300px; word-break: break-word; }
.status-match    { color: #2F6B33; font-weight: 600; }
.status-close    { color: #9A6609; font-weight: 600; }
.status-mismatch { color: var(--verdict-err); font-weight: 600; }
.status-missing  { color: #C7D6E1; }

/* ── Dashboard ── */
.dashboard {
    background: #F2F5F7; border: 1px solid #E7E9EA; border-radius: 16px;
    padding: 20px 24px; margin-bottom: 14px;
    animation: cc-fadeIn 0.3s ease-out both;
}
.stat-cards { display: flex; gap: 10px; flex-wrap: wrap; margin: 14px 0; }
.stat-card {
    flex: 1; min-width: 100px; text-align: center;
    padding: 14px 10px; border-radius: 12px;
    border: 1px solid #E7E9EA; background: white;
    box-shadow: 0 2px 12px rgba(12, 63, 94, 0.04);
}
.stat-card:hover { box-shadow: 0 2px 18px rgba(12, 63, 94, 0.08); }
.stat-card .num { font-size: 26px; font-weight: 700; line-height: 1.2; }
.stat-card .lbl {
    font-size: 11px; color: #64748b; margin-top: 2px;
    font-weight: 500; text-transform: uppercase; letter-spacing: 0.03em;
}
.progress-bar { height: 6px; border-radius: 6px; background: #e5e7eb; overflow: hidden; margin: 6px 0; }
.progress-fill { height: 100%; border-radius: 6px; transition: width 0.5s ease-out; }

/* ── Claims & passages ── */
.claim-box {
    background: #fff; border: 1px solid #E7E9EA; border-radius: 12px;
    padding: 12px 14px; margin: 6px 0; font-size: 13px; line-height: 1.6;
}
.claim-arrow { text-align: center; font-size: 18px; margin: 4px 0; color: #C7D6E1; }
.claim-section {
    background: #F2F5F7; border: 1px solid #E7E9EA;
    border-radius: 12px; padding: 12px 14px; margin: 8px 0;
}
.passage-card {
    background: #fff; border: 1px solid #E7E9EA; border-radius: 12px;
    padding: 12px 14px; margin: 6px 0;
}
.passage-card:hover { border-color: #AEC5CB; }
.passage-header { font-size: 11px; color: #64748b; margin-bottom: 4px; font-weight: 500; }
.passage-text { font-size: 13px; line-height: 1.65; color: #1e293b; }

/* ── Links & pills ── */
.link-pill {
    display: inline-block; background: #F2F5F7; color: #5B6D78;
    padding: 2px 10px; border-radius: 12px; font-size: 11px;
    text-decoration: none; margin-right: 4px; font-weight: 500;
    border: 1px solid #E1E9EF;
}
.link-pill:hover { background: #E1E9EF; color: #4D5B62; }

/* ── Coverage ── */
.coverage-bar { display: flex; gap: 8px; align-items: center; margin: 5px 0; font-size: 12px; }
.coverage-fill { height: 6px; border-radius: 3px; }
.info-row { font-size: 12px; color: #64748b; margin-top: 8px; }

/* ── Empty state ── */
.cc-empty { text-align: center; padding: 48px 20px; color: #94a3b8; }
.cc-empty-icon { font-size: 36px; margin-bottom: 8px; opacity: 0.35; }
.cc-empty-title { font-size: 14px; font-weight: 600; color: #64748b; margin-bottom: 4px; }
.cc-empty-sub { font-size: 12px; color: #94a3b8; }

/* ── Tab styling ──
   Bottom-border underline (Distill / arXiv style) instead of rounded chips.
   No background tile, no inset shadow — just an accent rule under the
   selected tab. */
.tab-nav,
.tab-container {
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 0 !important;
    gap: 0 !important;
}
.tab-nav button,
.tab-container button {
    font-weight: 500 !important; font-size: 14px !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: var(--text-muted) !important;
    padding: 12px 18px !important;
    border-radius: 0 !important;
    margin-bottom: -1px !important;  /* overlap container's bottom border */
    transition: color 0.15s ease, border-color 0.15s ease !important;
}
.tab-nav button:hover,
.tab-container button:hover {
    background: transparent !important;
    color: var(--text) !important;
}
.tab-nav button.selected,
.tab-container button.selected {
    background: transparent !important;
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    box-shadow: none !important;
}

/* ── File dropzone — override Gradio's generic file input look ── */
.gradio-container .file-preview,
.gradio-container [data-testid="file"],
.gradio-container .gr-file,
.gradio-container div[class*="file_upload"] > div:first-child {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 8px !important;
    padding: 24px !important;
    min-height: 140px !important;
    transition: border-color 0.15s ease, background 0.15s ease !important;
}
.gradio-container .file-preview:hover,
.gradio-container [data-testid="file"]:hover,
.gradio-container .gr-file:hover,
.gradio-container div[class*="file_upload"] > div:first-child:hover {
    border-color: var(--accent) !important;
    background: var(--accent-soft) !important;
}

/* ── Paper / Analysis-options panel row ── */
.cc-panel-row { align-items: stretch !important; gap: 16px !important; }
.cc-panel {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 24px !important;
    /* No shadow — Distill / arXiv-style restraint. */
    display: flex !important;
    flex-direction: column !important;
    min-width: 0 !important;
}
.cc-panel .cc-panel-title {
    /* Sentence-case + thin border: less shouty than the all-caps tracking
       version. Reads as a section label in a research paper, not a
       form-section header in a SaaS dashboard. */
    font-size: 13px; font-weight: 600; color: var(--text-muted);
    margin: 0 0 14px; padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}
/* File dropzone inside the upload panel fills remaining vertical space */
.cc-panel-upload .cc-filedrop { flex: 1 1 auto; }
.cc-panel-upload .cc-filedrop > div { min-height: 180px !important; }
/* Hide Gradio's auto-label on the main file dropzone (we render our own title) */
.cc-panel-upload .cc-filedrop > label,
.cc-panel-upload .cc-filedrop span.svelte-1gfkn6j { display: none !important; }

/* ── Loaded-file state ──
   Once Gradio paints a file, drop the dashed dropzone chrome entirely and
   present the file as a single inline chip with the close (×) button
   vertically centered on the file-name row. Three things have to give:
     1. Gradio's `.auto-margin` on the .cc-filedrop block pushes it to the
        bottom of the panel column — undo with margin-top:0/bottom:auto.
     2. `table.file-preview` ships with `min-height: 140px` which inflates
        the chip to a tall band — clamp to 0.
     3. The Clear (×) button lives in `.icon-button-wrapper.top-panel`,
        positioned absolutely at the top corner of the (now collapsed)
        dropzone — pin it to the file-row's vertical centerline. */
.cc-panel-upload .cc-filedrop:has(.file-preview-holder) {
    flex: 0 0 auto !important;
    margin-top: 0 !important;
    margin-bottom: auto !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    position: relative !important;
}
.cc-panel-upload .cc-filedrop:has(.file-preview-holder) table.file-preview {
    min-height: 0 !important;
    height: auto !important;
    margin: 0 !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
.cc-panel-upload .cc-filedrop:has(.file-preview-holder) .file-preview-holder {
    background: var(--bg-alt) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 12px 48px 12px 16px !important;
    margin: 0 !important;
    height: auto !important;
    min-height: 0 !important;
}
/* Stop Gradio's html/markdown blocks above the dropzone from flex-growing
   to fill the column, which would shove the file chip to the bottom. */
.cc-panel-upload > .block.hide-container,
.cc-panel-upload > .block.padded {
    flex: 0 0 auto !important;
}
.cc-panel-upload .cc-filedrop:has(.file-preview-holder) .icon-button-wrapper.top-panel {
    position: absolute !important;
    top: 50% !important;
    right: 10px !important;
    bottom: auto !important;
    height: auto !important;
    transform: translateY(-50%) !important;
    background: transparent !important;
    padding: 0 !important;
    z-index: 2 !important;
}
.cc-panel-upload .cc-filedrop:has(.file-preview-holder) .icon-button-wrapper.top-panel button {
    width: 28px !important; height: 28px !important;
    border-radius: 6px !important;
    color: var(--text-muted) !important;
    background: transparent !important;
    --bg-color: transparent !important;
    border-color: transparent !important;
    transition: background 0.12s ease, color 0.12s ease;
}
.cc-panel-upload .cc-filedrop:has(.file-preview-holder) .icon-button-wrapper.top-panel button:hover {
    background: rgba(0, 0, 0, 0.06) !important;
    color: var(--verdict-err) !important;
}
/* Push the Analyze button to bottom of the options panel so the two cards
   visually bottom-align even when the file panel is taller. Constrain its
   width — full-width primary buttons read as SaaS, not academic. */
.cc-panel-options > * + * { margin-top: 10px !important; }
.cc-panel-options .cc-analyze-btn {
    margin-top: auto !important;
    align-self: flex-end !important;
    min-width: 140px !important;
    max-width: 200px !important;
    width: auto !important;
}
/* Checkbox inside the options panel: give each row a subtle tile */
.cc-panel-options .gr-form, .cc-panel-options .wrap.svelte-1ixn6qd,
.cc-panel-options [data-testid="checkbox"] { padding: 0 !important; }

/* ── Brand-aligned primary action button ── */
/* Gradio assigns primary buttons various class suffixes across versions; this
   selector list covers the common ones without touching secondary buttons. */
button.primary,
button.lg.primary,
.gr-button-primary,
.gr-button.gr-button-primary {
    background: var(--accent) !important;
    background-image: none !important;
    color: #ffffff !important;
    border: 1px solid var(--accent) !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 18px rgba(0, 66, 118, 0.18) !important;
}
button.primary:hover,
button.lg.primary:hover,
.gr-button-primary:hover,
.gr-button.gr-button-primary:hover {
    background: var(--accent-hover) !important;
    border-color: var(--accent-hover) !important;
    box-shadow: 0 2px 24px rgba(0, 66, 118, 0.24) !important;
}

/* Secondary / outline buttons — academic restraint: hairline border + body text */
button.secondary,
.gr-button-secondary,
.gr-button.gr-button-secondary {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}
button.secondary:hover,
.gr-button-secondary:hover,
.gr-button.gr-button-secondary:hover {
    background: var(--bg-alt) !important;
    border-color: var(--text-muted) !important;
    color: var(--text-strong) !important;
}

/* Primary button: tighter radius for the academic look (Distill / arXiv). */
button.primary,
button.lg.primary,
.gr-button-primary,
.gr-button.gr-button-primary {
    border-radius: 8px !important;
}

/* ── Checkboxes — flat thin border, accent color when checked ── */
.gradio-container input[type="checkbox"] {
    width: 16px !important;
    height: 16px !important;
    accent-color: var(--accent);
    cursor: pointer;
}
.gradio-container [data-testid="checkbox"] label,
.gradio-container [data-testid="checkbox"] span {
    font-size: 14px !important;
    color: var(--text) !important;
    line-height: 1.5 !important;
}

/* ── Toasts (gr.Info / gr.Warning / gr.Error) — slim banner with our palette ──
   Gradio renders these via `.toast-body` / `.toast-message` plus a type
   modifier. The selectors below are intentionally generous so they work
   across Gradio 4 → 6 internal class shuffles. */
.toast-body,
.toast,
[role="status"][class*="toast"] {
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
    box-shadow: none !important;
    font-family: var(--font-sans) !important;
    font-size: 14px !important;
    padding: 10px 14px !important;
}
.toast-body.error, .toast.error, .error[class*="toast"] {
    background: var(--verdict-err-bg) !important;
    border-color: var(--verdict-err-border) !important;
    color: var(--verdict-err) !important;
}
.toast-body.warning, .toast.warning, .warning[class*="toast"] {
    background: var(--verdict-warn-bg) !important;
    border-color: var(--verdict-warn-border) !important;
    color: var(--verdict-warn) !important;
}
.toast-body.info, .toast.info, .info[class*="toast"] {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

"""


# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------

def check_prerequisites(file_path: str, mode: str) -> None:
    if file_path and file_path.lower().endswith(".pdf"):
        grobid_url = config.grobid()["service_url"]
        try:
            requests.get(f"{grobid_url}/api/isalive", timeout=3)
        except Exception:
            raise gr.Error(
                "GROBID is not running. PDF parsing requires GROBID.\n"
                "Run: docker run -d --name grobid -p 8070:8070 grobid/grobid:0.8.2-crf"
            )
    if mode == "agentic":
        if not config.openai_api_key():
            raise gr.Error(
                "Agentic mode requires an OpenAI API key.\n"
                "Add OPENAI_API_KEY to your .env file."
            )


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB per file — aligns with router/api caps
PDF_MAGIC = b"%PDF-"
ALLOWED_SUFFIXES = {".pdf", ".tex", ".bib", ".txt"}

# Hard cap on the .zip itself for batch uploads. Generous enough to hold a
# full batch of 20 MB papers (~30) without surprises, while still stopping a
# truly oversized upload before any extraction work begins. Zip-bomb safety
# is provided by the per-entry uncompressed cap inside ``_expand_zip_to_papers``;
# this constant only guards the on-disk size of the zip itself.
_BATCH_ZIP_MAX_BYTES = 500 * 1024 * 1024  # 500 MB


def _validate_upload(file_path: str, label: str = "file") -> None:
    """Reject files that are too large or lying about their type.

    Raises gr.Error with a user-facing message. Called by every upload entry
    point so a malicious / malformed upload never reaches the pipeline.
    """
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        raise gr.Error(f"Uploaded {label} is missing or not a regular file.")

    suffix = p.suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise gr.Error(
            f"{label.capitalize()} type '{suffix}' not allowed. "
            f"Accepted: {', '.join(sorted(ALLOWED_SUFFIXES))}"
        )

    size = p.stat().st_size
    if size > MAX_UPLOAD_BYTES:
        raise gr.Error(
            f"{label.capitalize()} is {size / 1024 / 1024:.1f} MB "
            f"(limit is {MAX_UPLOAD_BYTES // 1024 // 1024} MB)."
        )
    if size == 0:
        raise gr.Error(f"{label.capitalize()} is empty.")

    if suffix == ".pdf":
        with open(p, "rb") as f:
            head = f.read(len(PDF_MAGIC))
        if head != PDF_MAGIC:
            raise gr.Error(
                f"{label.capitalize()} has a .pdf extension but is not a valid PDF "
                "(missing %PDF- header)."
            )


def _unique_dest(dest_dir: Path, name: str) -> Path:
    """Return a path inside ``dest_dir`` that doesn't yet exist.

    Two uploads can share a basename (e.g. both temp-uploaded as ``paper.pdf``
    by Gradio from different sources, or two ZIP entries at different paths
    that share a leaf). The naive ``dest_dir / name`` would silently
    overwrite the first copy on the second write — pipeline then sees
    fewer reference PDFs than the user uploaded. Disambiguating with
    ``_1``, ``_2`` suffixes preserves all uploads.
    """
    candidate = dest_dir / name
    if not candidate.exists():
        return candidate
    stem = Path(name).stem
    suffix = Path(name).suffix
    i = 1
    while True:
        candidate = dest_dir / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def prepare_ref_pdfs_dir(pdf_paths: Optional[list[str]]) -> Optional[str]:
    if not pdf_paths:
        return None
    for p in pdf_paths:
        _validate_upload(p, label="reference PDF")
    tmp_dir = Path(tempfile.mkdtemp(prefix="checkcitation_refs_"))
    for p in pdf_paths:
        src = Path(p)
        shutil.copy2(str(src), str(_unique_dest(tmp_dir, src.name)))
    return str(tmp_dir)


# ---------------------------------------------------------------------------
# HTML formatting helpers
# ---------------------------------------------------------------------------

def _esc(text: Optional[str], max_len: int = 0) -> str:
    if not text:
        return ""
    s = escape(str(text))
    if max_len and len(s) > max_len:
        s = s[:max_len] + "..."
    return s


_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: Optional[str]) -> str:
    """Strip inline HTML/JATS tags before display.

    Many DB-returned abstracts (Semantic Scholar especially) embed JATS
    markup like ``<jats:p>``, ``<jats:emph>`` or plain HTML ``<em>`` /
    ``<i>``. Passing them through ``_esc`` would render the literal tags
    as visible text. This strips the tags first, then collapses the
    whitespace they leave behind. The result still needs to be passed
    through ``_esc`` for safe HTML embedding.
    """
    if not text:
        return ""
    cleaned = _HTML_TAG_RE.sub(" ", str(text))
    # Decode any leftover entities (&amp;, &lt;, etc.) so they don't get
    # double-encoded on the next escape pass.
    import html as _html
    cleaned = _html.unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


_SAFE_STEM_FALLBACK = "paper"
_SAFE_STEM_MAX = 60


def _safe_stem(file_path: str) -> str:
    """Sanitize a paper's filename stem for use in download filenames.

    Keeps letters, digits, underscore, dot, hyphen; replaces everything
    else with underscore; collapses runs; trims to a sane length so the
    final tempfile name doesn't blow past filesystem limits. Falls back
    to ``"paper"`` when the input is empty or sanitizes to nothing — at
    which point the random tempfile suffix is the only differentiator.
    """
    if not file_path:
        return _SAFE_STEM_FALLBACK
    stem = Path(file_path).stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    if not cleaned:
        return _SAFE_STEM_FALLBACK
    return cleaned[:_SAFE_STEM_MAX]


def build_links(verdict) -> str:
    links = []
    ex = verdict.existence
    if not ex:
        return ""
    if ex.matched_doi:
        links.append(f'<a class="link-pill" href="https://doi.org/{escape(ex.matched_doi)}" target="_blank">DOI</a>')
    if ex.matched_arxiv_id:
        links.append(f'<a class="link-pill" href="https://arxiv.org/abs/{escape(ex.matched_arxiv_id)}" target="_blank">arXiv</a>')
    if ex.oa_url:
        links.append(f'<a class="link-pill" href="{escape(ex.oa_url)}" target="_blank">Open Access</a>')
    return " ".join(links)


def _progress_color(score: float) -> str:
    if score >= 0.9:
        return "#22c55e"
    if score >= 0.7:
        return "#f59e0b"
    return "#ef4444"


def _field_status_icon(status: str) -> str:
    icons = {
        "MATCH": '<span class="status-match">&#x2714;</span>',
        "CLOSE_MATCH": '<span class="status-close">&#x2248;</span>',
        "MISMATCH": '<span class="status-mismatch">&#x2716;</span>',
        "MISSING": '<span class="status-missing">&mdash;</span>',
        "MISSING_REF": '<span class="status-missing">&mdash;</span>',
        "MISSING_BOTH": '<span class="status-missing">&mdash;</span>',
    }
    return icons.get(status, status)


# ---------------------------------------------------------------------------
# Dashboard (verification summary)
# ---------------------------------------------------------------------------

def format_dashboard(report: PaperReport, selected_mode: str = "", elapsed: float = 0) -> str:
    s = report.summary
    score_pct = int(s.integrity_score * 100)
    risk_color = RISK_COLORS.get(s.risk_level, "#6b7280")
    bar_color = _progress_color(s.integrity_score)

    # Metadata-dimension cards: "does this paper exist and match the
    # reference?" Only VALID/FABRICATED/UNVERIFIABLE are metadata outcomes —
    # the claim dimension (SUPPORTED/CONTRADICTS/NEUTRAL/UNVERIFIABLE) is
    # shown per-reference, not aggregated at the page summary.
    by_v = s.by_verdict
    metadata_cards_html = ""
    for key in ["VALID", "FABRICATED", "UNVERIFIABLE"]:
        count = by_v.get(key, 0)
        st = VERDICT_STYLES.get(key, VERDICT_STYLES["VALID"])
        metadata_cards_html += f'''
        <div class="stat-card" style="border-top: 3px solid {st['border']};">
            <div class="num" style="color: {st['color']};">{count}</div>
            <div class="lbl">{st['label']}</div>
        </div>'''

    # Show selected vs resolved mode
    mode_html = f"<b>{_esc(report.mode)}</b>"
    if selected_mode and selected_mode != report.mode:
        mode_html = (
            f"<b>{_esc(report.mode)}</b> "
            f'<span style="font-size:11px;color:#9ca3af;">'
            f"(selected: {_esc(selected_mode)}"
            f"{' &rarr; resolved based on input' if selected_mode == 'auto' else ' &rarr; adjusted based on input'})"
            f"</span>"
        )

    return f'''
    <div class="dashboard">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
            <div>
                <div style="font-size:14px; color:#6b7280;">Integrity Score</div>
                <div style="font-size:36px; font-weight:800; color:{bar_color};">{score_pct}%</div>
            </div>
            <div style="text-align:right;">
                <span style="display:inline-block; background:{risk_color}; color:white; padding:4px 14px; border-radius:20px; font-weight:600; font-size:14px;">
                    {s.risk_level} RISK
                </span>
            </div>
        </div>
        <div class="progress-bar">
            <div class="progress-fill" style="width:{score_pct}%; background:{bar_color};"></div>
        </div>
        <div class="cc-section-label">Metadata</div>
        <div class="stat-cards">{metadata_cards_html}</div>
        <div class="info-row">
            Mode: {mode_html} &nbsp;|&nbsp;
            Format: <b>{_esc(report.input_format)}</b> &nbsp;|&nbsp;
            References: <b>{report.total_references}</b>
            {f"&nbsp;|&nbsp; Time: <b>{elapsed:.1f}s</b>" if elapsed else ""}
        </div>
    </div>'''


# ---------------------------------------------------------------------------
# Metadata comparison table
# ---------------------------------------------------------------------------

def format_metadata_table(verdict) -> str:
    meta = verdict.metadata
    if not meta or not meta.comparisons:
        return ""
    rows = ""
    for c in meta.comparisons:
        icon = _field_status_icon(c.status)
        ref_val = _esc(c.ref_value) or '<span class="status-missing">—</span>'
        db_val = _esc(c.db_value) or '<span class="status-missing">—</span>'
        sim = f" ({c.similarity:.0%})" if c.similarity is not None else ""
        rows += f"<tr><td><b>{_esc(c.field.title())}</b></td><td>{ref_val}</td><td>{db_val}</td><td>{icon}{sim}</td></tr>"
    return f'''
    <table class="meta-table">
        <thead><tr><th>Field</th><th>In Paper</th><th>In Database</th><th>Status</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>'''


# ---------------------------------------------------------------------------
# Synthesized metadata table for agentic mode (no MetadataResult object)
# ---------------------------------------------------------------------------

def format_agentic_metadata_table(verdict, ref) -> str:
    """Build a comparison table from Reference + ExistenceResult when MetadataResult is absent."""
    ex = verdict.existence
    if not ex or not ref:
        return ""
    # Only useful if the agent found something
    has_any = ex.matched_title or ex.matched_authors or ex.matched_year or ex.matched_venue
    if not has_any:
        return ""

    rows = ""

    def _row(field, ref_val, db_val, similarity=None):
        if not ref_val and not db_val:
            return ""
        ref_str = _esc(str(ref_val)) if ref_val else '<span class="status-missing">&mdash;</span>'
        db_str = _esc(str(db_val)) if db_val else '<span class="status-missing">&mdash;</span>'
        sim_str = f" ({similarity:.0%})" if similarity is not None else ""
        if not ref_val or not db_val:
            icon = '<span class="status-missing">&mdash;</span>'
        elif str(ref_val).strip().lower() == str(db_val).strip().lower():
            icon = f'<span class="status-match">&#x2714;</span>{sim_str}'
        elif similarity is not None and similarity >= 0.8:
            icon = f'<span class="status-match">&#x2714;</span>{sim_str}'
        elif similarity is not None and similarity >= 0.5:
            icon = f'<span class="status-close">&#x2248;</span>{sim_str}'
        elif similarity is not None:
            icon = f'<span class="status-mismatch">&#x2716;</span>{sim_str}'
        else:
            # No similarity score — compare as strings
            ref_lower = str(ref_val).strip().lower()
            db_lower = str(db_val).strip().lower()
            if ref_lower == db_lower:
                icon = '<span class="status-match">&#x2714;</span>'
            elif ref_lower in db_lower or db_lower in ref_lower:
                icon = '<span class="status-close">&#x2248;</span>'
            else:
                icon = '<span class="status-mismatch">&#x2716;</span>'
        return f"<tr><td><b>{_esc(field)}</b></td><td>{ref_str}</td><td>{db_str}</td><td>{icon}</td></tr>"

    rows += _row("Title", ref.title, ex.matched_title, ex.title_similarity)
    ref_authors = "; ".join(ref.authors) if ref.authors else None
    db_authors = "; ".join(ex.matched_authors) if ex.matched_authors else None
    rows += _row("Authors", ref_authors, db_authors)
    rows += _row("Year", ref.year, ex.matched_year)
    rows += _row("Venue", ref.venue, ex.matched_venue)
    if ex.matched_doi:
        rows += _row("DOI", ref.doi, ex.matched_doi)

    if not rows:
        return ""

    return f'''
    <table class="meta-table">
        <thead><tr><th>Field</th><th>In Paper</th><th>In Database</th><th>Status</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>'''


# ---------------------------------------------------------------------------
# Passages section for a single reference (used inside unified cards)
# ---------------------------------------------------------------------------


def _format_claim_verdict(cv) -> str:
    """Render a claim verdict card with colored background."""
    if not cv:
        return ""
    st = CLAIM_VERDICT_STYLES.get(cv.verdict, CLAIM_VERDICT_STYLES["NEUTRAL"])
    evidence = ""
    if cv.evidence_quote:
        evidence = f'<div style="margin-top:8px;font-size:13px;color:{st["color"]};font-style:italic;opacity:0.85;">"{_esc(cv.evidence_quote)}"</div>'
    return f'''
    <div style="background:{st['bg']};border:1px solid {st['border']};border-left:4px solid {st['color']};
                border-radius:10px;padding:14px 16px;margin:10px 0;">
        <div style="display:flex;align-items:center;gap:8px;">
            <span style="display:inline-block;background:{st['color']};color:white;
                         padding:3px 12px;border-radius:14px;font-size:12px;font-weight:700;letter-spacing:0.03em;">
                {st['label'].upper()}
            </span>
        </div>
        <div style="margin-top:8px;font-size:13px;color:#374151;line-height:1.6;">
            {_esc(cv.explanation)}
        </div>
        {evidence}
    </div>'''


def _format_passages_for_ref(comp_results: list, abstract: str | None = None) -> str:
    """Render passage retrieval + claim verification results for one reference.

    `abstract` is the cited paper's abstract from the existence lookup. It's
    surfaced as a collapsible chip alongside Retrieved passages so the user
    can sanity-check the agent's decision against the source's own summary.
    """
    if not comp_results:
        return ""
    # Filter trivial citing sentences
    substantive = [r for r in comp_results if r.citing_sentence and len(r.citing_sentence.split()) >= 5]
    if not substantive:
        return ""

    # Source info from first result
    first = substantive[0]
    ft_icon = "&#x2705;" if first.full_text_available else "&#x1F4C4;"
    ft_label = "Full text" if first.full_text_available else "Abstract only"
    source = _esc(first.full_text_source or "unknown")

    # One distinct panel per citing context — agent decision + the passages
    # used to reach it travel together so the user can audit each one
    # independently without bundling unrelated decisions.
    inner = ""
    n = len(substantive)
    for idx, r in enumerate(substantive, 1):
        claim_html = _format_claim_verdict(r.claim_verdict) if r.claim_verdict else ""

        parts = []
        if r.context_before:
            parts.append(_esc(r.context_before))
        parts.append(_esc(r.citing_sentence))
        if r.context_after:
            parts.append(_esc(r.context_after))
        full_context = " ".join(parts)

        # Passages within this context's panel
        passages_inner = ""
        for i, sc in enumerate(r.top_passages, 1):
            score_val = sc.rrf_score if sc.rrf_score is not None else (sc.dense_score if sc.dense_score is not None else sc.bm25_score)
            score_type = "RRF" if sc.rrf_score is not None else ("Dense" if sc.dense_score is not None else "BM25")
            section = f" &middot; Section: {_esc(sc.chunk.section_name)}" if sc.chunk.section_name else ""
            passages_inner += f'''
            <div class="passage-card">
                <div class="passage-header">Passage {i}{section} &middot; {score_type}: {score_val:.3f}</div>
                <div class="passage-text">{_esc(sc.chunk.text)}</div>
            </div>'''
        if not r.top_passages:
            passages_inner = '<div style="font-size:13px;color:#6b7280;padding:8px;">No passages retrieved.</div>'

        label = f"Citing context {idx} of {n}" if n > 1 else "Citing context"
        # Per-citation block, ordered by what the human actually audits:
        #   1. Citing context  — what the user wrote (always visible)
        #   2. Claim agent's decision — verdict-tinted (always visible)
        #   3. Two evidence chips — Abstract + Retrieved passages, both
        #      collapsed by default. Click expands inline.
        # Hiding the long evidence behind chips keeps the card scannable —
        # the user reads the verdict first and only opens the evidence when
        # they want to second-guess it.
        chips = []
        clean_abstract = _strip_html(abstract)
        if clean_abstract:
            chips.append(f'''
            <details class="cc-evidence-chip">
                <summary><span class="cc-chip-icon">&#x1F4C4;</span>Abstract</summary>
                <div class="cc-chip-body">{_esc(clean_abstract)}</div>
            </details>''')
        if passages_inner.strip():
            chips.append(f'''
            <details class="cc-evidence-chip">
                <summary><span class="cc-chip-icon">&#x1F50D;</span>Retrieved passages</summary>
                <div class="cc-chip-body">{passages_inner}</div>
            </details>''')
        chips_block = (
            f'<div class="cc-evidence-chips">{"".join(chips)}</div>'
            if chips else ""
        )

        inner += f'''
        <div class="cc-context-panel">
            <div class="cc-context-label">{label}</div>
            <div class="cc-citing-block">
                <div class="cc-section-mini-label">Citing context</div>
                <div class="cc-citing-text">{full_context}</div>
            </div>
            <div class="cc-claim-block">
                <div class="cc-section-mini-label">Claim agent's decision</div>
                {claim_html if claim_html else '<div class="cc-claim-empty">Claim agent did not run for this citation context.</div>'}
            </div>
            {chips_block}
        </div>'''

    return f'''
    <details style="margin-top:10px;" open>
        <summary style="font-size:13px;color:#6b7280;cursor:pointer;">
            Claim Verification &nbsp;
            <span style="font-size:11px;">Source: {source} &middot; {ft_icon} {ft_label}</span>
        </summary>
        {inner}
    </details>'''


# ---------------------------------------------------------------------------
# Unified reference cards (verification + comprehension combined)
# ---------------------------------------------------------------------------


def format_unified_cards(
    report, comp_report, references: list,
    has_verification: bool = True, has_passages: bool = False,
) -> str:
    ref_map = {r.ref_id: r for r in references}

    # Build comp results grouped by ref_id
    comp_by_ref: dict[str, list] = {}
    if comp_report and has_passages:
        for cr in comp_report.results:
            comp_by_ref.setdefault(cr.ref_id, []).append(cr)

    # If verification ran, iterate over verdicts (sorted by severity)
    if has_verification and report and report.verdicts:
        order = {"FABRICATED": 0, "UNVERIFIABLE": 1, "VALID": 2}
        sorted_verdicts = sorted(report.verdicts, key=lambda v: order.get(v.verdict, 9))

        cards = ""
        for v in sorted_verdicts:
            ref = ref_map.get(v.ref_id)
            ex_abstract = (v.existence.abstract if v.existence else None) if v else None
            passages_html = _format_passages_for_ref(
                comp_by_ref.get(v.ref_id, []), abstract=ex_abstract,
            ) if has_passages else ""
            cards += _render_verdict_card(v, ref, passages_html)
        if not cards:
            return '<div style="color:#6b7280;text-align:center;padding:40px;">No references found.</div>'
        return cards

    # Passage-only mode (no verification) — neutral blue cards per reference
    if has_passages and comp_by_ref:
        cards = ""
        for ref in references:
            results = comp_by_ref.get(ref.ref_id, [])
            if not results:
                continue
            passages_html = _format_passages_for_ref(results)
            if not passages_html:
                continue
            title = _esc(ref.title) if ref.title else "<em>No title</em>"
            authors = _esc("; ".join(ref.authors)) if ref.authors else ""
            year = f" ({ref.year})" if ref.year else ""
            venue = _esc(ref.venue) if ref.venue else ""
            authors_line = (
                f'<div style="font-size:13px;color:#6b7280;margin-top:2px;">{authors}{year}</div>'
                if (authors or year) else ""
            )
            cards += f'''
            <details class="card" style="background:#f8fafc; border-left-color:#3b82f6;" open>
                <summary style="color:#1e40af;">
                    [{_esc(ref.ref_id)}] {title}
                </summary>
                <div style="margin-top:8px;">
                    {authors_line}
                    {"<div style='font-size:13px;color:#6b7280;'>" + venue + "</div>" if venue else ""}
                    {passages_html}
                </div>
            </details>'''
        return cards if cards else '<div style="color:#6b7280;text-align:center;padding:40px;">No passages found.</div>'

    return '<div style="color:#6b7280;text-align:center;padding:40px;">No results.</div>'




def _render_verdict_card(v, ref, passages_html: str = "") -> str:
    """Render a single reference card with verdict + optional passages."""
    st = VERDICT_STYLES.get(v.verdict, VERDICT_STYLES["VALID"])
    # The card auto-expands when there's a metadata problem OR a claim
    # contradiction — both are findings the user should read, not skim past.
    is_problem = v.verdict in ("FABRICATED",) or v.claim_verdict == "CONTRADICTS"
    open_attr = " open" if is_problem else ""

    title = _esc(ref.title) if ref and ref.title else "<em>No title</em>"
    authors = _esc("; ".join(ref.authors)) if ref and ref.authors else ""
    year = f" ({ref.year})" if ref and ref.year else ""
    venue = _esc(ref.venue) if ref and ref.venue else ""

    # Citation format badge (e.g., APA, Vancouver, IEEE)
    fmt = getattr(ref, 'citation_format', None) if ref else None
    fmt_badge = ""
    if fmt:
        fmt_badge = (
            f'<span style="display:inline-block;background:#f0f4ff;color:#4b5563;'
            f'padding:1px 8px;border-radius:10px;font-size:11px;margin-left:6px;'
            f'border:1px solid #d1d5db;">{_esc(fmt.upper())}</span>'
        )

    links_html = build_links(v)

    source_info = ""
    if v.existence and v.existence.status == "FOUND":
        src = _esc(v.existence.source or "")
        sim = f" &middot; similarity: {v.existence.title_similarity:.0%}" if v.existence.title_similarity else ""
        source_info = f'<div style="font-size:12px;color:#6b7280;margin-top:4px;">Found via: <b>{src}</b>{sim}</div>'
    elif v.existence and v.existence.status == "NOT_FOUND":
        dbs = ", ".join(v.existence.databases_checked) if v.existence.databases_checked else "none"
        source_info = f'<div style="font-size:12px;color:#dc2626;margin-top:4px;">Not found in: {_esc(dbs)}</div>'

    action_html = ""
    if v.action == "remove_citation":
        action_html = '<span style="display:inline-block;background:#fef2f2;color:#dc2626;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600;margin-top:6px;">Remove Citation</span>'
    elif v.action == "verify_claim":
        action_html = '<span style="display:inline-block;background:#fffbeb;color:#d97706;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600;margin-top:6px;">Verify Claim</span>'

    meta_table = format_metadata_table(v) or format_agentic_metadata_table(v, ref)
    meta_section = ""
    if meta_table:
        meta_section = f'''
        <details style="margin-top:10px;">
            <summary style="font-size:13px;color:#6b7280;cursor:pointer;">Metadata Comparison</summary>
            {meta_table}
        </details>'''

    ref_id_attr = _esc(v.ref_id)
    # Per-reference claim badge (only when claim was actually checked).
    claim_badge_html = ""
    if v.claim_verdict and v.claim_verdict in CLAIM_VERDICT_STYLES:
        cst = CLAIM_VERDICT_STYLES[v.claim_verdict]
        claim_badge_html = (
            f'<span class="cc-claim-pill" '
            f'style="background:{cst["bg"]};color:{cst["color"]};'
            f'border:1px solid {cst["border"]};">'
            f'Claim: {cst["label"].upper()}'
            f'</span>'
        )

    # Authors line shown in the expanded body, since the title now drives
    # the heading. Keep the year next to authors so the bibliographic
    # context is still complete on a single glance.
    authors_line = ""
    if authors or year:
        authors_line = (
            f'<div style="font-size:13px;color:#6b7280;margin-top:2px;">'
            f'{authors}{year}'
            f'</div>'
        )

    return f'''
    <details class="card" data-cc-ref="{ref_id_attr}" data-cc-verdict="{_esc(v.verdict)}"
             data-cc-claim="{_esc(v.claim_verdict or "")}"
             style="background:{st['bg']}; border-left-color:{st['border']};" {open_attr}>
        <summary>
            <span style="color:{st['color']};">{st['icon']} {st['label'].upper()}</span>
            {claim_badge_html}
            &nbsp;&middot;&nbsp;
            [{ref_id_attr}] {title}
        </summary>
        <div style="margin-top:8px;">
            {authors_line}
            {"<div style='font-size:13px;color:#6b7280;'>" + venue + fmt_badge + "</div>" if venue else (fmt_badge if fmt_badge else "")}
            {links_html}
            {source_info}
            {action_html}
            {meta_section}
            {passages_html}
        </div>
    </details>'''


# ---------------------------------------------------------------------------
# Comprehension formatting
# ---------------------------------------------------------------------------

def format_comprehension_coverage(report) -> str:
    cov = report.coverage
    total = sum(cov.values()) or 1
    ft = cov.get("full_text", 0)
    ab = cov.get("abstract_only", 0)
    nf = cov.get("not_found", 0)

    def bar(count, color, label):
        pct = int(count / total * 100)
        return f'''
        <div class="coverage-bar">
            <div style="width:200px;background:#e5e7eb;border-radius:4px;overflow:hidden;">
                <div class="coverage-fill" style="width:{pct}%;background:{color};"></div>
            </div>
            <span><b>{count}</b> {label}</span>
        </div>'''

    return f'''
    <div class="dashboard">
        <div style="font-size:14px;color:#6b7280;">Coverage</div>
        <div style="font-size:28px;font-weight:800;color:#1f2937;">{report.total_citations} <span style="font-size:14px;font-weight:400;color:#6b7280;">citations analyzed</span></div>
        {bar(ft, "#22c55e", "Full text")}
        {bar(ab, "#f59e0b", "Abstract only")}
        {bar(nf, "#ef4444", "Not found")}
    </div>'''



# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

# --- Rate limiting ---------------------------------------------------------
# Two layers:
#   1. Daily global cap  — protects the OpenAI bill as a whole
#   2. Per-IP hourly cap — stops a single reviewer (or bot) burning through it
# Both are in-memory, fine for a single-container deploy.

_daily_analyses = defaultdict(int)  # date_str -> count of papers (single + batch)
_daily_lock = threading.Lock()
_DAILY_LIMIT = int(os.environ.get("DAILY_ANALYSIS_LIMIT", "40"))

_ip_requests: dict[str, list[float]] = defaultdict(list)  # ip -> single-analysis timestamps
_ip_batch_requests: dict[str, list[float]] = defaultdict(list)  # ip -> batch-submission timestamps
_ip_lock = threading.Lock()
_HOURLY_IP_LIMIT = int(os.environ.get("HOURLY_IP_LIMIT", "10"))
_HOURLY_IP_BATCH_LIMIT = int(os.environ.get("HOURLY_IP_BATCH_LIMIT", "2"))
_BATCH_MAX_PAPERS = int(os.environ.get("BATCH_MAX_PAPERS_PER_BATCH", "30"))
_HOUR_SECONDS = 3600
# Periodic prune cadence for the IP throttle maps. The on-request cleanup
# only fires when an entry crosses 1000 keys, so without this a low-traffic
# instance could accumulate stale state for hours. Re-arming Timer keeps
# memory bounded without a dedicated thread loop.
_IP_PRUNE_INTERVAL_SECONDS = 600  # 10 minutes


def _prune_ip_throttle() -> None:
    """Drop expired entries from both IP throttle maps."""
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
    """Start a self-rearming daemon Timer that prunes the IP maps."""
    try:
        _prune_ip_throttle()
    finally:
        t = threading.Timer(_IP_PRUNE_INTERVAL_SECONDS, _schedule_ip_prune)
        t.daemon = True
        t.start()


# Kick off the pruner once at module load. Daemon=True so it doesn't block
# process exit. The on-request cleanup at line ~1703 stays as a backstop in
# case the timer is somehow stopped.
_schedule_ip_prune()


def _client_ip(request: "gr.Request") -> str:
    """Best-effort client IP. Trusts X-Forwarded-For when running behind a proxy."""
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


def _reserve_daily_papers(n: int) -> None:
    """Reserve n slots in the global daily cap atomically. Raises gr.Error if over."""
    today = time.strftime("%Y-%m-%d")
    with _daily_lock:
        if _daily_analyses[today] + n > _DAILY_LIMIT:
            remaining = max(0, _DAILY_LIMIT - _daily_analyses[today])
            raise gr.Error(
                f"Daily analysis limit ({_DAILY_LIMIT}) would be exceeded. "
                f"Only {remaining} paper(s) remaining today. "
                "This is a research demo, please try again tomorrow."
            )
        _daily_analyses[today] += n
        # Clean old entries
        for k in list(_daily_analyses):
            if k != today:
                del _daily_analyses[k]


def _check_monthly_budget() -> None:
    """Refuse new analyses if the monthly OpenAI spend cap is reached.

    No-op when ``MONTHLY_BUDGET_USD`` is unset (local dev / tests).
    """
    try:
        spend_guard.check_budget()
    except spend_guard.BudgetExceededError as e:
        raise gr.Error(str(e)) from None


def _check_rate_limit(request: Optional["gr.Request"] = None) -> None:
    """Enforce the daily global cap and the per-IP hourly single-paper cap.

    Order matters: the per-IP check runs *before* ``_reserve_daily_papers``
    so a hourly-cap rejection doesn't burn a slot from the global daily
    quota. (The earlier ordering let a single bad client drain everyone
    else's quota by repeatedly tripping its own hourly limit.)
    """
    _check_monthly_budget()

    ip = _client_ip(request)
    now = time.time()
    cutoff = now - _HOUR_SECONDS
    with _ip_lock:
        recent = [t for t in _ip_requests[ip] if t > cutoff]
        if len(recent) >= _HOURLY_IP_LIMIT:
            raise gr.Error(
                f"Hourly limit reached for your IP ({_HOURLY_IP_LIMIT} single analyses/hour). "
                "Please try again later."
            )
        # Reserve the daily slot only after the per-IP check passes.
        # ``_reserve_daily_papers`` raises gr.Error on global cap; in that
        # case we haven't yet appended ``now`` to ``recent``, so the
        # per-IP record stays clean too.
        _reserve_daily_papers(1)

        recent.append(now)
        _ip_requests[ip] = recent
        # Periodically drop empty/stale IP entries to keep memory bounded
        if len(_ip_requests) > 1000:
            for key, stamps in list(_ip_requests.items()):
                live = [t for t in stamps if t > cutoff]
                if live:
                    _ip_requests[key] = live
                else:
                    del _ip_requests[key]


def _check_batch_rate_limit(
    n_papers: int,
    request: Optional["gr.Request"] = None,
) -> None:
    """Enforce the hourly per-IP batch cap, the batch-size cap, and reserve
    n_papers against the global daily cap atomically.

    Separate from `_check_rate_limit` so singles and batches don't consume
    each other's hourly pools — a user can do 10 singles AND 2 batches/hour.
    """
    if n_papers <= 0:
        raise gr.Error("Please upload at least one paper.")
    if n_papers > _BATCH_MAX_PAPERS:
        raise gr.Error(
            f"Too many papers in one batch ({n_papers}). "
            f"Maximum is {_BATCH_MAX_PAPERS} per submission."
        )
    _check_monthly_budget()

    ip = _client_ip(request)
    now = time.time()
    cutoff = now - _HOUR_SECONDS
    with _ip_lock:
        recent = [t for t in _ip_batch_requests[ip] if t > cutoff]
        if len(recent) >= _HOURLY_IP_BATCH_LIMIT:
            raise gr.Error(
                f"Hourly batch limit reached for your IP "
                f"({_HOURLY_IP_BATCH_LIMIT} batches/hour). "
                "Please try again later."
            )

        # Reserve the daily slots BEFORE recording the batch, so a daily-cap
        # rejection doesn't consume the hourly batch slot.
        _reserve_daily_papers(n_papers)

        recent.append(now)
        _ip_batch_requests[ip] = recent


# Pipeline stages shown in the in-page progress UI. Order matters — the list
# is rendered top-to-bottom and stages flip from pending → active → done as
# the timing logger emits matching STAGE events. `applies_when(existence,
# claims)` lets each stage opt out of modes where it doesn't run, so the user
# never sees a step that will never tick.
PIPELINE_STAGES_DEF: list[tuple[str, str, callable]] = [
    ("L1_parse",                   "Parse paper",
        lambda existence, claims: True),
    ("L2_existence",               "Verify references exist",
        lambda existence, claims: existence),
    ("agentic_pre_retrieve",       "Retrieve cited paper passages",
        lambda existence, claims: claims),
    ("agentic_metadata_dispatch",  "Verify metadata (LLM agents)",
        lambda existence, claims: claims),
    ("agentic_claim_dispatch",     "Verify claims (LLM agents)",
        lambda existence, claims: claims),
]

# When the triage routes a reference to NEEDS_BOTH (metadata ambiguous AND
# claim verification needed), the runner emits a single ``agentic_both_dispatch``
# stage event instead of the separate metadata/claim ones. Map it here so the
# UI's two-stage display still animates correctly — without this, the metadata
# and claim spinners hang for the entire dispatch (the actual work happens
# under an untracked stage name) and the UI looks frozen.
_STAGE_ALIASES: dict[str, tuple[str, ...]] = {
    "agentic_both_dispatch": ("agentic_metadata_dispatch", "agentic_claim_dispatch"),
    # When only Claim Verification is selected (Existence unchecked), the
    # pipeline takes the comprehension-only path which emits a single
    # ``L4_comprehension`` event instead of the per-step agentic events.
    # Map it onto all three UI stages so the progress display doesn't hang
    # on retrieve/metadata/claim spinners that will never tick on this path.
    "L4_comprehension": (
        "agentic_pre_retrieve",
        "agentic_metadata_dispatch",
        "agentic_claim_dispatch",
    ),
}

# Final-step pseudo stage — set when the pipeline returns and we're rendering
# results, so the UI shows a "Finalize" tick rather than the last agent step
# spinning forever.
_FINALIZE_STAGE_ID = "__finalize__"

_STAGE_RE = re.compile(r"STAGE (\S+) seconds=")


def _applicable_stages(existence: bool, claims: bool) -> list[tuple[str, str]]:
    stages = [(sid, label) for sid, label, app in PIPELINE_STAGES_DEF
              if app(existence, claims)]
    stages.append((_FINALIZE_STAGE_ID, "Build report"))
    return stages


def _render_pipeline(applicable: list[tuple[str, str]],
                     seen: set[str],
                     elapsed: float | None = None) -> str:
    """Render the in-page progress card. The first stage that hasn't been
    seen yet is marked active; everything before it is done; everything
    after, pending."""
    items = []
    active_marked = False
    for stage_id, label in applicable:
        done = stage_id in seen
        if done:
            cls = "done"
            icon = ('<svg viewBox="0 0 16 16" class="cc-step-icon-svg" '
                    'aria-hidden="true">'
                    '<path d="M3.5 8.5l3 3 6-7" stroke="currentColor" '
                    'stroke-width="2" fill="none" stroke-linecap="round" '
                    'stroke-linejoin="round"/></svg>')
        elif not active_marked:
            cls = "active"
            icon = '<span class="cc-step-spinner" aria-hidden="true"></span>'
            active_marked = True
        else:
            cls = "pending"
            icon = '<span class="cc-step-dot" aria-hidden="true"></span>'
        items.append(
            f'<li class="cc-step cc-step-{cls}">'
            f'<span class="cc-step-icon">{icon}</span>'
            f'<span class="cc-step-label">{label}</span>'
            f'</li>'
        )
    elapsed_html = (
        f'<span class="cc-pipeline-elapsed">{elapsed:.1f}s</span>'
        if elapsed is not None else ''
    )
    return (
        f'<div class="cc-pipeline">'
        f'<div class="cc-pipeline-header">'
        f'<span class="cc-pipeline-title">Analyzing your paper</span>'
        f'{elapsed_html}'
        f'</div>'
        f'<ul class="cc-pipeline-stages">{"".join(items)}</ul>'
        f'</div>'
    )


def run_analyze(file, ref_pdfs, check_existence, check_claims, retry_failed,
                request: gr.Request = None):
    """Generator: yields the analyze tab outputs as the pipeline progresses.

    Each yield emits an 11-tuple matching the click handler's ``outputs``
    list (dashboard, coverage, cards, json, download, bib, annotated_pdf,
    annotated_status, json_acc, dl_row, retry_row). During the run only
    the cards slot changes — it carries the live pipeline-stage card.
    Final yield delivers the full results."""
    if file is None:
        raise gr.Error("Please upload a file.")
    _check_rate_limit(request)
    if not check_existence and not check_claims:
        raise gr.Error("Select at least one analysis option.")

    file_path = file if isinstance(file, str) else file.name
    _validate_upload(file_path, label="paper")

    # Claim verification triggers agentic mode; otherwise rule-based (internal: "quick")
    effective_mode = "agentic" if check_claims else "quick"
    check_prerequisites(file_path, effective_mode)

    ref_dir = prepare_ref_pdfs_dir(ref_pdfs) if check_claims else None

    applicable = _applicable_stages(check_existence, check_claims)
    seen_stages: set[str] = set()

    def _progress_update(elapsed: float | None = None):
        return (
            gr.update(),                                       # dashboard
            gr.update(),                                       # coverage
            _render_pipeline(applicable, seen_stages, elapsed),  # cards
            gr.update(), gr.update(), gr.update(),             # json, dl, bib
            gr.update(), gr.update(),                          # pdf, ann_status
            gr.update(), gr.update(),                          # json_acc, dl_row
            gr.update(visible=False),                          # retry_row
        )

    # Initial state: first stage is "active" (spinning), rest pending.
    # Clear any results from a previous run so the user doesn't see stale
    # dashboard / coverage / downloads while the new analysis is in flight.
    yield (
        "",                                                         # dashboard
        "",                                                         # coverage
        _render_pipeline(applicable, seen_stages, 0.0),             # cards
        "",                                                         # json
        gr.update(value=None, visible=False),                       # download (json file)
        gr.update(value=None, visible=False),                       # bib
        gr.update(value=None, visible=False),                       # annotated pdf
        "",                                                         # annotated status
        gr.update(visible=False),                                   # json accordion
        gr.update(visible=False),                                   # downloads row
        gr.update(visible=False),                                   # retry row
    )

    # Tap the timing logger so we can show stage-by-stage progress without
    # threading a callback through the entire pipeline.
    #
    # ``_StageCap`` is filtered by the worker thread's ident so concurrent
    # ``run_analyze`` calls don't cross-contaminate. Without that filter,
    # both runs' handlers would receive both pipelines' events via the
    # process-wide ``checkcitation.timing`` logger and one user's progress
    # display would tick from the other user's pipeline stages.
    stage_events: list[str] = []
    event_lock = threading.Lock()

    class _StageCap(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            # Set after ``worker.start()`` once we know the thread's ident.
            # Any events that arrive before that are ignored — the worker
            # hasn't started doing pipeline work yet, so they can only be
            # noise from concurrent runs.
            self.allowed_thread: Optional[int] = None

        def emit(self, record: logging.LogRecord) -> None:
            if self.allowed_thread is None or record.thread != self.allowed_thread:
                return
            m = _STAGE_RE.search(record.getMessage())
            if m:
                with event_lock:
                    stage_events.append(m.group(1))

    timing_log = logging.getLogger("checkcitation.timing")
    handler = _StageCap()
    timing_log.addHandler(handler)

    result_holder: dict = {}

    def _worker():
        try:
            result_holder["value"] = run_unified(
                file_path,
                mode=effective_mode,
                run_verification=check_existence,
                run_claim_verification=check_claims,
                run_comprehension=check_claims,
                ref_pdfs_dir=ref_dir,
                retry_failed=retry_failed,
            )
        except BaseException as e:
            result_holder["error"] = e

    start = time.time()
    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    handler.allowed_thread = worker.ident

    try:
        last_yield = 0.0
        applicable_ids = {sid for sid, _ in applicable}
        while worker.is_alive():
            worker.join(timeout=0.4)
            with event_lock:
                current = list(stage_events)
            new_seen = False
            for stage_name in current:
                # Expand stage aliases (e.g. agentic_both_dispatch ticks both
                # the metadata and claim UI stages) so the progress display
                # doesn't hang on stages whose underlying work has finished.
                expanded = _STAGE_ALIASES.get(stage_name, (stage_name,))
                for ui_id in expanded:
                    if ui_id in seen_stages or ui_id not in applicable_ids:
                        continue
                    seen_stages.add(ui_id)
                    new_seen = True
            now = time.time()
            # Yield on every stage flip (instant feedback) and otherwise once a
            # second so the elapsed counter still advances during long stages.
            if new_seen or (now - last_yield) > 1.0:
                yield _progress_update(elapsed=now - start)
                last_yield = now
    finally:
        timing_log.removeHandler(handler)
        if ref_dir:
            # Only delete ``ref_dir`` once the worker is fully done with it.
            # If the user cancelled the run (Gradio closed the generator
            # mid-yield) the worker thread may still be reading reference
            # PDFs; ``rmtree`` while it reads would surface as opaque
            # "file not found" errors deep in the pipeline. Leave the dir
            # to OS tempdir cleanup in that case — it's a few MB at worst.
            if worker.is_alive():
                logging.getLogger(__name__).debug(
                    "Skipping ref_dir cleanup; worker still alive: %s", ref_dir,
                )
            else:
                shutil.rmtree(ref_dir, ignore_errors=True)

    if "error" in result_holder:
        # Clear the in-flight progress card before raising. Without this,
        # the user sees a frozen mid-pipeline spinner alongside the error
        # toast and can't tell that the run actually stopped.
        yield (
            "",                                            # dashboard
            "",                                            # coverage
            "",                                            # cards (clear spinner)
            "",                                            # json
            gr.update(value=None, visible=False),          # download
            gr.update(value=None, visible=False),          # bib
            gr.update(value=None, visible=False),          # annotated pdf
            "",                                            # annotated status
            gr.update(visible=False),                      # json accordion
            gr.update(visible=False),                      # downloads row
            gr.update(visible=False),                      # retry row
        )
        err = result_holder["error"]
        if isinstance(err, gr.Error):
            raise err
        raise gr.Error(f"Analysis failed: {err}")

    paper_report, comp_report, parsed = result_holder["value"]
    elapsed = time.time() - start

    # Mark all pipeline stages done + the synthetic finalize stage active.
    for sid, _ in applicable:
        if sid != _FINALIZE_STAGE_ID:
            seen_stages.add(sid)
    yield _progress_update(elapsed=elapsed)

    # Dashboard (only if verification ran)
    dashboard_html = ""
    if paper_report:
        dashboard_html = format_dashboard(paper_report, selected_mode=effective_mode, elapsed=elapsed)

    # Coverage (only if claim verification ran)
    coverage_html = ""
    if comp_report and check_claims:
        coverage_html = format_comprehension_coverage(comp_report)

    # Unified cards
    cards_html = format_unified_cards(
        paper_report, comp_report, parsed.references,
        has_verification=bool(paper_report),
        has_passages=check_claims,
    )

    # JSON report
    combined = {}
    if paper_report:
        combined["verification"] = paper_report.model_dump()
    if comp_report:
        combined["comprehension"] = comp_report.model_dump()
    report_json = json.dumps(combined, indent=2, default=str)

    paper_stem = _safe_stem(file_path)
    tmp = tempfile.NamedTemporaryFile(
        suffix=".json", prefix=f"{paper_stem}_report_", delete=False, mode="w"
    )
    tmp.write(report_json)
    tmp.close()

    bib_path = _write_problematic_bibtex(
        paper_report, parsed.references if parsed else [],
        source_label=Path(file_path).name,
        paper_stem=paper_stem,
    )

    annotated_pdf_path, annotate_status_html = _try_annotate_pdf(
        file_path, paper_report, parsed, comp_report,
        paper_stem=paper_stem,
    )

    # Show the retry button only when something might be worth retrying.
    # FABRICATED / UNVERIFIABLE can both stem from transient API failures
    # (rate limits, DB outages) — retry clears the NOT_FOUND cache and
    # re-checks them. Already-VALID refs are skipped from the cache anyway,
    # so retry is harmless when nothing failed but also pointless to offer.
    has_retryable = bool(paper_report) and any(
        v.verdict in ("FABRICATED", "UNVERIFIABLE") for v in paper_report.verdicts
    )
    # If this *was* the retry run, hide the button — clicking again would
    # just hit the same APIs that already failed twice.
    show_retry = has_retryable and not retry_failed

    yield (
        dashboard_html, coverage_html, cards_html, report_json,
        gr.update(value=tmp.name, visible=True),
        gr.update(value=bib_path, visible=bool(bib_path)),
        gr.update(value=annotated_pdf_path, visible=bool(annotated_pdf_path)),
        annotate_status_html,
        gr.update(visible=True),                                # JSON accordion
        gr.update(visible=bool(bib_path or annotated_pdf_path)),  # downloads row
        gr.update(visible=show_retry),                          # retry row
    )


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def _format_batch_summary(per_paper: list[dict], elapsed: float, mode: str) -> str:
    """Top-of-page aggregate ribbon for a batch run."""
    ok = [p for p in per_paper if p["status"] == "ok"]
    failed = [p for p in per_paper if p["status"] != "ok"]
    total_refs = sum(p.get("total_refs", 0) for p in ok)

    agg = defaultdict(int)
    for p in ok:
        for k, v in p.get("counts", {}).items():
            agg[k] += v

    def pill(label, count, bg, fg):
        return (
            f'<span style="display:inline-block;background:{bg};color:{fg};'
            f'padding:4px 12px;border-radius:14px;font-size:13px;'
            f'font-weight:600;margin:0 6px 6px 0;">'
            f'{count} {label}</span>'
        )

    # Metadata-dimension pills only. The claim dimension is intentionally
    # not surfaced in the page-level ribbon — claim verdicts cluster on
    # NEUTRAL / UNVERIFIABLE for many references and would skew the
    # at-a-glance summary toward "lots of issues" when most are just
    # tangential citations. Users see claim status per-reference instead.
    pills = "".join([
        pill("fabricated", agg["FABRICATED"], "#fef2f2", "#b91c1c"),
        pill("unverifiable", agg["UNVERIFIABLE"], "#f9fafb", "#4b5563"),
        pill("valid", agg["VALID"], "#f0fdf4", "#15803d"),
    ])

    failed_chip = ""
    if failed:
        failed_chip = pill("failed", len(failed), "#fef2f2", "#991b1b")

    return (
        f'<div class="dashboard" style="padding:16px 20px;">'
        f'<div style="font-size:14px;color:#6b7280;margin-bottom:8px;">'
        f'Batch results · mode: <b>{_esc(mode)}</b> · elapsed: <b>{elapsed:.1f}s</b></div>'
        f'<div style="display:flex;gap:18px;margin-bottom:12px;flex-wrap:wrap;font-size:14px;">'
        f'<div><b>{len(ok)}</b> papers analyzed</div>'
        f'<div><b>{total_refs}</b> total references</div>'
        f'{"<div>" + str(len(failed)) + " failed</div>" if failed else ""}'
        f'</div>'
        f'<div style="margin-top:4px;">{pills}{failed_chip}</div>'
        f'</div>'
    )


def _format_batch_rollup(per_paper: list[dict]) -> str:
    """Sortable-ish per-paper table. One row per paper, counts by verdict type."""
    if not per_paper:
        return ""
    rows = []
    for p in per_paper:
        name = _esc(p["name"])
        if p["status"] != "ok":
            reason = _esc(p.get("error", "unknown error"), 160)
            rows.append(
                f'<tr style="background:#fef2f2;">'
                f'<td style="padding:6px 10px;font-family:monospace;">{name}</td>'
                f'<td colspan="4" style="padding:6px 10px;color:#b91c1c;">Failed: {reason}</td>'
                f'</tr>'
            )
            continue
        c = p.get("counts", {})
        n = p.get("total_refs", 0)

        def cell(value, color="#4b5563"):
            if not value:
                return f'<td style="padding:6px 10px;color:#9ca3af;text-align:right;">0</td>'
            return (
                f'<td style="padding:6px 10px;color:{color};'
                f'font-weight:600;text-align:right;">{value}</td>'
            )

        rows.append(
            f'<tr>'
            f'<td style="padding:6px 10px;font-family:monospace;">{name}</td>'
            f'<td style="padding:6px 10px;text-align:right;color:#6b7280;">{n}</td>'
            f'{cell(c.get("FABRICATED", 0), "#b91c1c")}'
            f'{cell(c.get("UNVERIFIABLE", 0), "#4b5563")}'
            f'{cell(c.get("VALID", 0), "#15803d")}'
            f'</tr>'
        )

    # Metadata-dimension columns only — claim outcomes (Misrep.) live in
    # the per-paper detail view, not the rollup table.
    header = (
        '<tr style="background:#f9fafb;font-size:12px;color:#6b7280;text-align:left;">'
        '<th style="padding:6px 10px;">Paper</th>'
        '<th style="padding:6px 10px;text-align:right;">Refs</th>'
        '<th style="padding:6px 10px;text-align:right;">Fabr.</th>'
        '<th style="padding:6px 10px;text-align:right;">Unver.</th>'
        '<th style="padding:6px 10px;text-align:right;">Valid</th>'
        '</tr>'
    )
    return (
        '<div style="margin-top:16px;">'
        '<div class="cc-section-label">Per-paper rollup</div>'
        f'<table style="width:100%;border-collapse:collapse;font-size:13px;">'
        f'{header}{"".join(rows)}</table>'
        '</div>'
    )


def _format_batch_per_paper(per_paper: list[dict], has_passages: bool) -> str:
    """Expandable accordion with the full card list for each successful paper."""
    chunks = []
    for p in per_paper:
        if p["status"] != "ok":
            continue
        paper_report = p["paper_report"]
        comp_report = p["comp_report"]
        parsed = p["parsed"]
        if paper_report is None or parsed is None:
            continue
        cards = format_unified_cards(
            paper_report, comp_report, parsed.references,
            has_verification=True, has_passages=has_passages,
        )
        chunks.append(
            f'<details style="margin-top:10px;">'
            f'<summary style="font-size:14px;font-weight:600;cursor:pointer;'
            f'padding:8px 12px;background:#f8fafc;border-radius:8px;">'
            f'{_esc(p["name"])} &nbsp;'
            f'<span style="font-weight:400;color:#6b7280;font-size:12px;">'
            f'{p.get("total_refs", 0)} refs</span>'
            f'</summary>'
            f'<div style="padding:10px 4px;">{cards}</div>'
            f'</details>'
        )
    if not chunks:
        return ""
    return (
        '<div style="margin-top:16px;">'
        '<div class="cc-section-label">Per-paper detail</div>'
        + "".join(chunks) +
        '</div>'
    )


def _try_annotate_pdf(
    source_path: str, paper_report, parsed, comp_report=None,
    paper_stem: str = "",
) -> tuple[Optional[str], str]:
    """Produce an annotated copy of the user's PDF, with inline status.

    Returns (output_path_or_None, status_html).
    - Non-PDF uploads: no file, neutral "not available for X inputs" message.
    - Annotation failure: no file, short reason string (placed next to the
      download buttons in the UI, not at the top of the page).
    - Success: (path, "") — empty status so the UI area stays clean.
    """
    def status(msg: str, tone: str = "info") -> str:
        bg = {"info": "#f3f4f6", "warn": "#fef3c7"}.get(tone, "#f3f4f6")
        fg = {"info": "#4b5563", "warn": "#92400e"}.get(tone, "#4b5563")
        return (
            f'<div style="font-size:12px;color:{fg};background:{bg};'
            f'padding:6px 10px;border-radius:6px;display:inline-block;">'
            f'{_esc(msg)}</div>'
        )

    if not source_path or Path(source_path).suffix.lower() != ".pdf":
        return None, status("Annotated PDF export is available for PDF uploads only.")

    if paper_report is None or parsed is None:
        return None, status("No verdicts to annotate.")

    from src.report.pdf_annotator import annotate_pdf

    stem = paper_stem or _safe_stem(source_path)
    out_tmp = tempfile.NamedTemporaryFile(
        suffix=".pdf", prefix=f"{stem}_annotated_",
        delete=False, mode="wb",
    )
    out_tmp.close()
    try:
        stats = annotate_pdf(
            source_path, paper_report, parsed, out_tmp.name,
            comp_report=comp_report,
        )
    except Exception as e:
        logging.getLogger(__name__).warning(f"Annotated PDF failed: {e}")
        Path(out_tmp.name).unlink(missing_ok=True)
        return None, status(
            f"Annotated PDF could not be generated: {e}",
            tone="warn",
        )

    if stats.annotated == 0:
        Path(out_tmp.name).unlink(missing_ok=True)
        return None, status(
            "Annotated PDF skipped: no citation markers could be located on any page.",
            tone="warn",
        )
    return out_tmp.name, ""


_PROBLEM_VERDICTS = {"FABRICATED", "UNVERIFIABLE"}


def _bibtex_escape(text: str) -> str:
    """Minimal BibTeX value escaping: strip braces/newlines that would break the entry."""
    return str(text).replace("{", "").replace("}", "").replace("\n", " ").strip()


def _format_problem_bibtex_entry(v, ref) -> Optional[str]:
    """Emit a single BibTeX entry for a problematic verdict.

    The entry is built from the *original* reference fields (so the
    user can locate it in their bibliography) plus a `note` summarizing
    why it was flagged and what was found (if anything).
    """
    # Include FABRICATED, UNVERIFIABLE *or* anything where the claim
    # dimension contradicted — all three are findings the user needs to
    # action, even when the metadata top-level says VALID.
    if v.verdict not in _PROBLEM_VERDICTS and v.claim_verdict != "CONTRADICTS":
        return None
    if ref is None:
        return None

    # Prefer the original citation key from a .bib source; fall back to ref_id
    key = v.ref_id
    fields: list[tuple[str, str]] = []
    if ref.title:
        fields.append(("title", _bibtex_escape(ref.title)))
    if ref.authors:
        fields.append(("author", _bibtex_escape(" and ".join(ref.authors))))
    if ref.year:
        fields.append(("year", str(ref.year)))
    if ref.venue:
        # @misc is generic; callers can hand-edit entry type if they want
        fields.append(("howpublished", _bibtex_escape(ref.venue)))
    if ref.doi:
        fields.append(("doi", _bibtex_escape(ref.doi)))
    if ref.url:
        fields.append(("url", _bibtex_escape(ref.url)))

    # Build the audit note
    note_parts = [f"CheckCitation verdict: {v.verdict}"]
    if v.explanation:
        note_parts.append(_bibtex_escape(v.explanation))
    ex = v.existence
    if ex:
        if ex.status == "NOT_FOUND" and ex.databases_checked:
            note_parts.append(
                f"Not found in: {', '.join(ex.databases_checked)}"
            )
        elif ex.status == "FOUND" and ex.matched_title:
            note_parts.append(
                f"DB match: '{_bibtex_escape(ex.matched_title)[:120]}' "
                f"via {ex.source or 'unknown'}"
            )
    fields.append(("note", " | ".join(note_parts)))

    body = ",\n  ".join(f'{k} = {{{v}}}' for k, v in fields)
    return f"@misc{{{key},\n  {body}\n}}"


def _build_problematic_bibtex(paper_report, references, source_label: str = "") -> str:
    """Concatenate all problematic entries into one .bib string."""
    if paper_report is None:
        return ""
    ref_map = {r.ref_id: r for r in references}
    entries = []
    for v in paper_report.verdicts:
        entry = _format_problem_bibtex_entry(v, ref_map.get(v.ref_id))
        if entry:
            entries.append(entry)
    if not entries:
        return ""
    header_bits = [
        "% CheckCitation: problematic references",
        "% Exported: " + time.strftime("%Y-%m-%d %H:%M:%S"),
    ]
    if source_label:
        header_bits.append(f"% Source: {source_label}")
    header_bits.append(
        f"% Entries: {len(entries)} "
        "(FABRICATED, UNVERIFIABLE verdicts + references with CONTRADICTS claims)"
    )
    header_bits.append("% Each entry carries a `note = {...}` field with the audit trail.")
    return "\n".join(header_bits) + "\n\n" + "\n\n".join(entries) + "\n"


def _write_problematic_bibtex(
    paper_report, references, source_label: str = "",
    paper_stem: str = "",
) -> Optional[str]:
    """Write problematic .bib to a tempfile and return its path, or None if no problems."""
    bib = _build_problematic_bibtex(paper_report, references, source_label)
    if not bib:
        return None
    stem = paper_stem or (_safe_stem(source_label) if source_label else _SAFE_STEM_FALLBACK)
    tmp = tempfile.NamedTemporaryFile(
        suffix=".bib", prefix=f"{stem}_problems_", delete=False, mode="w",
        encoding="utf-8",
    )
    tmp.write(bib)
    tmp.close()
    return tmp.name


def _write_batch_problem_bibtex(per_paper: list[dict]) -> Optional[str]:
    """Aggregate problematic refs across all successful papers into one .bib."""
    chunks = []
    for p in per_paper:
        if p["status"] != "ok" or p["paper_report"] is None:
            continue
        bib = _build_problematic_bibtex(
            p["paper_report"], p["parsed"].references, source_label=p["name"],
        )
        if bib:
            chunks.append(bib)
    if not chunks:
        return None
    tmp = tempfile.NamedTemporaryFile(
        suffix=".bib", prefix="checkcitation_batch_problems_", delete=False, mode="w",
        encoding="utf-8",
    )
    tmp.write("\n\n".join(chunks))
    tmp.close()
    return tmp.name


def _write_batch_csv(per_paper: list[dict]) -> str:
    """One row per (paper, verdict). Written to a temp file, returned for download."""
    import csv as _csv
    tmp = tempfile.NamedTemporaryFile(
        suffix=".csv", prefix="checkcitation_batch_", delete=False, mode="w",
        newline="", encoding="utf-8",
    )
    writer = _csv.writer(tmp)
    writer.writerow([
        "paper", "ref_id", "verdict", "action", "mode",
        "matched_title", "matched_doi", "source", "explanation",
    ])
    for p in per_paper:
        if p["status"] != "ok" or p["paper_report"] is None:
            continue
        name = p["name"]
        for v in p["paper_report"].verdicts:
            ex = v.existence
            writer.writerow([
                name, v.ref_id, v.verdict, v.action, v.mode,
                (ex.matched_title or "") if ex else "",
                (ex.matched_doi or "") if ex else "",
                (ex.source or "") if ex else "",
                v.explanation,
            ])
    tmp.close()
    return tmp.name


def _write_batch_json(per_paper: list[dict], mode: str, elapsed: float) -> str:
    """Nested JSON of the whole batch."""
    import json as _json
    payload = {
        "mode": mode,
        "elapsed_seconds": round(elapsed, 2),
        "papers": [],
    }
    for p in per_paper:
        entry = {
            "name": p["name"],
            "status": p["status"],
        }
        if p["status"] == "ok":
            entry["total_refs"] = p.get("total_refs", 0)
            entry["counts"] = p.get("counts", {})
            if p["paper_report"]:
                entry["verification"] = p["paper_report"].model_dump(mode="json")
            if p["comp_report"]:
                entry["comprehension"] = p["comp_report"].model_dump(mode="json")
        else:
            entry["error"] = p.get("error", "unknown error")
        payload["papers"].append(entry)

    tmp = tempfile.NamedTemporaryFile(
        suffix=".json", prefix="checkcitation_batch_", delete=False, mode="w",
        encoding="utf-8",
    )
    tmp.write(_json.dumps(payload, indent=2, default=str))
    tmp.close()
    return tmp.name


def _expand_zip_to_papers(zip_path: str) -> list[str]:
    """Extract a .zip into a temp dir and return paths of valid paper files.

    Defenses applied here, in order:

    1. **ZIP file size cap** — refuse uploads above ``_BATCH_ZIP_MAX_BYTES``
       up front so an oversized archive never opens.
    2. **Path traversal** — ``Path(info.filename).name`` flattens absolute
       paths and ``..`` segments to a leaf basename, neutralising
       ``../../etc/passwd``-style entries.
    3. **Filename collision** — ``_unique_dest`` adds numeric suffixes so a
       ZIP with ``a/paper.pdf`` and ``b/paper.pdf`` keeps both copies
       instead of silently overwriting one.
    4. **Per-entry zip-bomb cap** — each entry is streamed through a hard
       ``MAX_UPLOAD_BYTES`` ceiling on uncompressed bytes. Going over
       aborts the whole extraction and removes the temp dir, so a
       small-compressed-to-huge-uncompressed entry can't fill the disk.

    Magic-bytes / suffix-vs-content checks still happen per file via
    ``_validate_upload`` after extraction returns.
    """
    import zipfile

    zip_size = Path(zip_path).stat().st_size
    if zip_size > _BATCH_ZIP_MAX_BYTES:
        raise gr.Error(
            f"ZIP archive is {zip_size / 1024 / 1024:.0f} MB "
            f"(limit is {_BATCH_ZIP_MAX_BYTES // 1024 // 1024} MB)."
        )

    out_dir = Path(tempfile.mkdtemp(prefix="checkcitation_batch_"))
    extracted: list[str] = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                safe_name = Path(info.filename).name
                if not safe_name or safe_name in (".", ".."):
                    continue
                suffix = Path(safe_name).suffix.lower()
                if suffix not in ALLOWED_SUFFIXES:
                    continue
                dest = _unique_dest(out_dir, safe_name)
                # Stream-extract with a per-entry uncompressed ceiling so a
                # zip-bomb (small compressed → many GB uncompressed) can't
                # exhaust /tmp before _validate_upload fires per file.
                bytes_written = 0
                with zf.open(info) as src, open(dest, "wb") as dst:
                    while True:
                        chunk = src.read(64 * 1024)
                        if not chunk:
                            break
                        bytes_written += len(chunk)
                        if bytes_written > MAX_UPLOAD_BYTES:
                            raise gr.Error(
                                f"ZIP entry '{safe_name}' exceeds "
                                f"{MAX_UPLOAD_BYTES // 1024 // 1024} MB "
                                "(possible zip bomb)."
                            )
                        dst.write(chunk)
                extracted.append(str(dest))
    except BaseException:
        # Aborted extraction — remove the partially-populated temp dir so a
        # rejected ZIP doesn't accumulate on disk run-over-run.
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    return extracted


def _collect_batch_inputs(files) -> list[str]:
    """Normalize the batch file input — a list of individual files and/or zips —
    into a flat list of paper paths ready to analyze."""
    if not files:
        return []
    if not isinstance(files, list):
        files = [files]
    paper_paths: list[str] = []
    for f in files:
        p = f if isinstance(f, str) else getattr(f, "name", None)
        if not p:
            continue
        if Path(p).suffix.lower() == ".zip":
            paper_paths.extend(_expand_zip_to_papers(p))
        else:
            paper_paths.append(p)
    return paper_paths


def run_batch(files, check_existence, check_claims, retry_failed,
              request: gr.Request = None, progress=gr.Progress()):
    """Analyze multiple papers in one submission.

    Rate-limit model (see DEPLOY.md):
      - BATCH_MAX_PAPERS_PER_BATCH caps papers per submission
      - HOURLY_IP_BATCH_LIMIT caps batches per IP per hour
      - DAILY_ANALYSIS_LIMIT counts every paper toward the day
    One failed paper does not kill the batch; errors are surfaced per-paper.
    """
    if not files:
        raise gr.Error("Please upload at least one paper (or a .zip of papers).")
    if not check_existence and not check_claims:
        raise gr.Error("Select at least one analysis option.")

    paper_paths = _collect_batch_inputs(files)
    if not paper_paths:
        raise gr.Error(
            "No valid paper files found in the upload "
            f"(accepted: {', '.join(sorted(ALLOWED_SUFFIXES))})."
        )

    _check_batch_rate_limit(len(paper_paths), request)

    effective_mode = "agentic" if check_claims else "quick"

    # Validate every file up front so we fail fast on a bad upload
    for p in paper_paths:
        _validate_upload(p, label="paper")

    # Run each paper sequentially; the report cache already makes repeats free
    per_paper: list[dict] = []
    start = time.time()
    total = len(paper_paths)

    progress(0.02, desc=f"Starting batch of {total} papers...")

    for idx, p in enumerate(paper_paths, start=1):
        name = Path(p).name
        progress(idx / (total + 1), desc=f"Paper {idx}/{total}: {name}")
        try:
            check_prerequisites(p, effective_mode)
            paper_report, comp_report, parsed = run_unified(
                p,
                mode=effective_mode,
                run_verification=check_existence,
                run_claim_verification=check_claims,
                run_comprehension=check_claims,
                retry_failed=retry_failed,
            )
            counts = defaultdict(int)
            if paper_report:
                # Count by metadata dimension to keep the batch ribbon
                # consistent with the single-paper dashboard headline. The
                # rolled-up verdict can drag a metadata-VALID ref into
                # UNVERIFIABLE because of claim-NEUTRAL, which the user
                # rightly flagged as confusing.
                for v in paper_report.verdicts:
                    counts[v.metadata_verdict or v.verdict] += 1
            per_paper.append({
                "name": name,
                "path": p,
                "status": "ok",
                "paper_report": paper_report,
                "comp_report": comp_report,
                "parsed": parsed,
                "counts": dict(counts),
                "total_refs": len(parsed.references) if parsed else 0,
            })
        except BaseException as e:
            logging.getLogger(__name__).warning(f"Batch: {name} failed: {e}")
            per_paper.append({
                "name": name,
                "path": p,
                "status": "error",
                "error": str(e),
            })

    elapsed = time.time() - start
    progress(0.97, desc="Building aggregate report...")

    summary_html = _format_batch_summary(per_paper, elapsed, effective_mode)
    rollup_html = _format_batch_rollup(per_paper)
    per_paper_html = _format_batch_per_paper(per_paper, check_claims)

    # CSV + JSON downloads + problematic-refs BibTeX aggregate
    csv_path = _write_batch_csv(per_paper)
    json_path = _write_batch_json(per_paper, effective_mode, elapsed)
    bib_path = _write_batch_problem_bibtex(per_paper)

    progress(1.0, desc="Done")
    return summary_html, rollup_html, per_paper_html, csv_path, json_path, bib_path


# ---------------------------------------------------------------------------
# About page
# ---------------------------------------------------------------------------

ABOUT_HTML = """
<div class="cc-about">
    <div class="cc-about-section">
        <h3>What it does</h3>
        <p>CheckCite is designed to verify citations in academic writing.
        For each reference, it considers two questions separately: whether
        the cited work exists with the stated bibliographic details, and
        whether it supports the claim made in the text. Each assessment is
        supported by evidence, such as database records for the
        bibliographic check and relevant excerpts from the cited work for
        the claim check.</p>

        <p>Authors can use it to review their manuscripts before
        submission, while reviewers can use it to examine references more
        efficiently. The tool also produces an annotated PDF in which
        citations are labeled with their assessment, and any disputed
        claims are flagged with notes that point to the relevant
        supporting passages.</p>
    </div>

    <div class="cc-about-section">
        <h3>Verdict guide</h3>
        <p><strong>Metadata verdicts</strong> describe the cited paper itself.</p>
        <ul style="margin:8px 0 12px 20px;line-height:1.7;">
            <li><strong>VALID.</strong> The paper exists and its fields match.</li>
            <li><strong>FABRICATED.</strong> At least two databases responded with no match, the metadata mismatches a real paper, or the paper has been retracted. Suggested action: remove the citation.</li>
            <li><strong>UNVERIFIABLE.</strong> Coverage was insufficient to decide. Only one database responded, or all timed out. Flagged for human review.</li>
        </ul>
        <p><strong>Claim verdicts</strong> describe the relationship between the cited paper and the citing sentence.</p>
        <ul style="margin:8px 0 12px 20px;line-height:1.7;">
            <li><strong>SUPPORTS.</strong> The cited paper says what the citing sentence attributes to it.</li>
            <li><strong>CONTRADICTS.</strong> The cited paper says something materially different. Suggested action: re-read the cited paper and revise.</li>
            <li><strong>NEUTRAL.</strong> The cited paper does not address the specific topic the citing sentence claims. The general area may overlap but the specific point is not entailed.</li>
            <li><strong>UNVERIFIABLE.</strong> Claim verification could not run, for example because the full text is paywalled, the agent failed, or there is no substantive citing sentence.</li>
        </ul>
    </div>


    <div class="cc-about-section">
        <h3>Frequently asked questions</h3>

        <details class="cc-faq">
            <summary>Which databases does the tool consult?</summary>
            <p>The tool consults CrossRef, Semantic Scholar, OpenAlex, PubMed, and arXiv. For non-paper citations such as tweets or blog posts, it also checks whether the cited URL resolves.</p>
        </details>

        <details class="cc-faq">
            <summary>What model is used for claim verification?</summary>
            <p>OpenAI's <code>gpt-4o-mini</code>.</p>
        </details>

        <details class="cc-faq">
            <summary>Is my paper sent to OpenAI?</summary>
            <p>No. Only short excerpts are sent to OpenAI: the citing sentence with its surrounding context, and passages retrieved from the cited paper. Your full draft is never uploaded.</p>
        </details>

        <details class="cc-faq">
            <summary>How accurate is it, and how should I read a verdict?</summary>
            <p>The tool's primary output is the <em>evidence trail</em>, not the verdict label. Every verdict shows you what was retrieved — which databases responded for an existence check, and which passages from the cited paper supported or contradicted each claim — and the report exposes all of it so you can read the underlying material and decide whether you agree.</p>
        </details>

    </div>

    <div class="cc-about-section">
        <h3>Cite CheckCite</h3>
        <div class="cc-bibtex-wrap">
            <pre class="cc-bibtex" id="cc-bibtex-cite"><code>@software{checkcite,
  title  = {CheckCite},
  author = {Anonymous},
  year   = {2026},
  note   = {Submitted for double-blind review}
}</code></pre>
            <button type="button" class="cc-bibtex-copy"
                onclick="(function(btn){
                    var pre = document.getElementById('cc-bibtex-cite');
                    if(!pre) return;
                    var text = pre.innerText;
                    var done = function(){
                        btn.classList.add('cc-copied');
                        btn.textContent = 'Copied';
                        setTimeout(function(){ btn.classList.remove('cc-copied'); btn.textContent = 'Copy'; }, 1500);
                    };
                    if (navigator.clipboard && navigator.clipboard.writeText) {
                        navigator.clipboard.writeText(text).then(done).catch(function(){});
                    } else {
                        var ta = document.createElement('textarea'); ta.value = text;
                        document.body.appendChild(ta); ta.select();
                        try { document.execCommand('copy'); done(); } catch(e){}
                        document.body.removeChild(ta);
                    }
                })(this)">Copy</button>
        </div>
    </div>

    <div class="cc-about-section">
        <h3>Status</h3>
        <p>Open-source. Anonymized for double-blind review. Repository link will follow the review period.</p>
    </div>
</div>
"""


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def build_theme() -> gr.themes.Soft:
    """Brand-aligned Gradio theme.

    NOTE (Gradio 6+): theme and css are not honoured on gr.Blocks() anymore;
    they must be passed to mount_gradio_app() (or launch()). This function
    returns the theme object so the mount call can attach it.
    """
    from gradio.themes.utils.colors import Color
    # Distill-navy ramp. c600 anchors at the same value as --accent (#004276).
    # Gradio uses these stops for primary buttons / focus rings / link
    # underlines that our CSS overrides may not catch — keeping them in the
    # navy family stops a TIB-red glint from leaking through.
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


def create_app() -> gr.Blocks:
    with gr.Blocks(title="CheckCite", analytics_enabled=False) as demo:

        gr.HTML('''
            <div class="cc-hero">
                <div class="cc-hero-text">
                    <h1>Verify every citation in your paper, <span class="cc-hero-accent">with evidence</span>.</h1>
                    <p class="cc-hero-sub">
                        CheckCite finds fabricated references, flags metadata mismatches, and checks whether each cited paper actually supports the claim you made for it. It returns the passage that grounds every verdict.
                    </p>
                </div>
            </div>

            <div class="cc-features">
                <div class="cc-feature">
                    <div class="cc-feature-icon">
                        <!-- Duotone: filled silhouette (low-opacity wash) under the stroked outline -->
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                            <path class="duo-fill" fill="currentColor" stroke="none" d="M6 2h8l6 6v6h-3.5a3.5 3.5 0 1 0-2.5 5.95V22H6a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2z"/>
                            <circle class="duo-fill" fill="currentColor" stroke="none" cx="16.5" cy="16.5" r="3"/>
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h7"/>
                            <path d="M14 2v6h6"/>
                            <circle cx="16.5" cy="16.5" r="3"/>
                            <path d="M21 21l-2-2"/>
                        </svg>
                    </div>
                    <div class="cc-feature-step">STEP 1</div>
                    <div class="cc-feature-title">Does the paper exist?</div>
                    <div class="cc-feature-body">We query CrossRef, Semantic Scholar, OpenAlex and PubMed; flag references that resolve nowhere as fabricated.</div>
                </div>
                <div class="cc-feature">
                    <div class="cc-feature-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                            <rect class="duo-fill" fill="currentColor" stroke="none" x="3" y="4" width="18" height="16" rx="2"/>
                            <rect x="3" y="4" width="18" height="16" rx="2"/>
                            <path d="M7.2 9.2l.6.6 1.4-1.4"/>
                            <path d="M12 9h5"/>
                            <path d="M7.2 14.2l.6.6 1.4-1.4"/>
                            <path d="M12 14h5"/>
                        </svg>
                    </div>
                    <div class="cc-feature-step">STEP 2</div>
                    <div class="cc-feature-title">Does the metadata match?</div>
                    <div class="cc-feature-body">Field-level checks across title, authors, venue, year catch chimera citations: a real paper attached to the wrong attribution.</div>
                </div>
                <div class="cc-feature">
                    <div class="cc-feature-icon">
                        <!-- Comment bubble with three dots: the universal "passage / discussion" mark -->
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                            <path class="duo-fill" fill="currentColor" stroke="none" d="M5 4h14a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-7l-4 4v-4H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z"/>
                            <path d="M5 4h14a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-7l-4 4v-4H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z"/>
                            <circle cx="8" cy="10.5" r="1" fill="currentColor"/>
                            <circle cx="12" cy="10.5" r="1" fill="currentColor"/>
                            <circle cx="16" cy="10.5" r="1" fill="currentColor"/>
                        </svg>
                    </div>
                    <div class="cc-feature-step">STEP 3</div>
                    <div class="cc-feature-title">Does it support your claim?</div>
                    <div class="cc-feature-body">Multi-query retrieval pulls passages from the cited paper and an LLM checks semantic support, returning the quote it used.</div>
                </div>
            </div>
        ''')

        with gr.Tabs():
            # ── Tab 1: Analyze ─────────────────────────────────
            with gr.TabItem("Analyze"):
                gr.Markdown(
                    "**Upload a paper** (PDF, LaTeX, BibTeX, or plain text), "
                    "pick what to check, then click **Analyze**."
                )
                with gr.Row(equal_height=True, elem_classes=["cc-panel-row"]):
                    with gr.Column(scale=3, elem_classes=["cc-panel", "cc-panel-upload"]):
                        gr.HTML('<div class="cc-panel-title">Paper</div>')
                        analyze_file = gr.File(
                            label="",
                            show_label=False,
                            file_types=[".pdf", ".tex", ".bib", ".txt"],
                            type="filepath",
                            elem_classes=["cc-filedrop"],
                        )
                        analyze_refs = gr.File(
                            label="Reference PDFs (optional, for non-open-access cited papers)",
                            file_types=[".pdf"],
                            file_count="multiple",
                            type="filepath",
                            visible=False,
                        )
                    with gr.Column(scale=2, elem_classes=["cc-panel", "cc-panel-options"]):
                        gr.HTML('<div class="cc-panel-title">Analysis options</div>')
                        chk_existence = gr.Checkbox(
                            label="Existence & Metadata",
                            value=True,
                        )
                        chk_claims = gr.Checkbox(
                            label="Claim Verification",
                            value=False,
                        )
                        analyze_btn = gr.Button("Analyze", variant="primary", size="lg", elem_classes=["cc-analyze-btn"])

                        chk_claims.change(
                            fn=lambda checked: gr.update(visible=checked),
                            inputs=[chk_claims],
                            outputs=[analyze_refs],
                        )

                # ── Results section ──
                gr.HTML('<div class="cc-section-label" style="margin-top:20px;">Results</div>')
                analyze_dashboard = gr.HTML()
                analyze_coverage = gr.HTML()
                # Contextual retry control. Hidden by default; surfaced only
                # when the run produces FABRICATED / UNVERIFIABLE results that
                # could plausibly be transient (DB timeouts, rate limits).
                # Clicking it re-runs the same inputs with retry_failed=True,
                # which clears NOT_FOUND cache entries before retrying.
                with gr.Row(visible=False) as analyze_retry_row:
                    gr.HTML(
                        '<div class="cc-retry-hint">Some references didn\'t '
                        'resolve. This can happen on transient API errors — '
                        'try again to re-check just the failed ones.</div>'
                    )
                    analyze_retry_btn = gr.Button(
                        "Re-run with retries",
                        variant="secondary", size="sm",
                        elem_classes=["cc-retry-btn"],
                    )
                analyze_cards = gr.HTML(
                    value='''<div class="cc-empty">
                        <div class="cc-empty-icon">&#x1F50D;</div>
                        <div class="cc-empty-title">Ready to analyze</div>
                        <div class="cc-empty-sub">Upload a paper and click Analyze to begin</div>
                    </div>''',
                )

                # Downloads stay hidden until run_analyze yields its final
                # tuple — empty file boxes pre-run look like broken UI.
                with gr.Accordion("JSON Report", open=False, visible=False) as analyze_json_acc:
                    analyze_download = gr.File(label="Download JSON report", interactive=False, visible=False)
                    analyze_json = gr.Code(language="json", label="Report JSON")
                with gr.Row(visible=False) as analyze_dl_row:
                    analyze_bib = gr.File(
                        label="Download problematic refs (.bib)", interactive=False, visible=False,
                    )
                    analyze_annotated_pdf = gr.File(
                        label="Download annotated PDF", interactive=False, visible=False,
                    )
                analyze_annotated_status = gr.HTML()

                # show_progress_on pins the progress bar to the main results
                # area only — without it, Gradio 6 paints a progress bar on
                # every HTML output and you get 3-4 identical bars stacked
                # above the result region.
                _analyze_inputs = [
                    analyze_file, analyze_refs,
                    chk_existence, chk_claims,
                ]
                _analyze_outputs = [
                    analyze_dashboard, analyze_coverage, analyze_cards,
                    analyze_json, analyze_download, analyze_bib,
                    analyze_annotated_pdf, analyze_annotated_status,
                    analyze_json_acc, analyze_dl_row, analyze_retry_row,
                ]

                # Two thin wrappers that pin ``retry_failed`` per button.
                # Earlier versions stored the flag in a ``gr.State`` toggled
                # by a chained ``.then()``, but if the run between the toggle
                # and the reset raised, the State was left at True — the next
                # normal Analyze click silently re-ran as a retry. Splitting
                # into dedicated handlers eliminates that race entirely: the
                # main button never reads any state that could be left dirty
                # by an interrupted retry chain.
                def run_analyze_main(file, ref_pdfs, ce, cc, request: gr.Request = None):
                    yield from run_analyze(
                        file, ref_pdfs, ce, cc,
                        retry_failed=False, request=request,
                    )

                def run_analyze_retry(file, ref_pdfs, ce, cc, request: gr.Request = None):
                    yield from run_analyze(
                        file, ref_pdfs, ce, cc,
                        retry_failed=True, request=request,
                    )

                analyze_btn.click(
                    fn=run_analyze_main,
                    inputs=_analyze_inputs,
                    outputs=_analyze_outputs,
                    show_progress="hidden",
                )

                # Retry button — clears NOT_FOUND cache entries for this
                # paper and re-runs the same analysis. Surfaced contextually
                # by the previous run when there's anything worth retrying.
                analyze_retry_btn.click(
                    fn=run_analyze_retry,
                    inputs=_analyze_inputs,
                    outputs=_analyze_outputs,
                    show_progress="hidden",
                )

            # ── Tab 2: Batch ───────────────────────────────────
            with gr.TabItem("Batch"):
                gr.Markdown(
                    f"**Upload multiple papers** (or a `.zip` of papers) to "
                    f"analyze them in one go. Each paper goes through the same "
                    f"pipeline and you get an aggregate report plus per-paper "
                    f"breakdowns.\n\n"
                    f"**Limits:** up to **{_BATCH_MAX_PAPERS} papers per batch**, "
                    f"**{_HOURLY_IP_BATCH_LIMIT} batches per hour** per IP."
                )
                with gr.Row(equal_height=True, elem_classes=["cc-panel-row"]):
                    with gr.Column(scale=3, elem_classes=["cc-panel", "cc-panel-upload"]):
                        gr.HTML('<div class="cc-panel-title">Papers</div>')
                        batch_files = gr.File(
                            label="",
                            show_label=False,
                            file_types=[".pdf", ".tex", ".bib", ".txt", ".zip"],
                            file_count="multiple",
                            type="filepath",
                            elem_classes=["cc-filedrop"],
                        )
                    with gr.Column(scale=2, elem_classes=["cc-panel", "cc-panel-options"]):
                        gr.HTML('<div class="cc-panel-title">Analysis options</div>')
                        batch_chk_existence = gr.Checkbox(
                            label="Existence & Metadata",
                            value=True,
                        )
                        batch_chk_claims = gr.Checkbox(
                            label="Claim Verification",
                            value=False,
                        )
                        # Hidden state — batch always runs without retry; if a
                        # batch hits transient failures, re-uploading is fine.
                        batch_retry = gr.State(value=False)
                        batch_btn = gr.Button(
                            "Analyze batch", variant="primary", size="lg",
                            elem_classes=["cc-analyze-btn"],
                        )

                batch_summary = gr.HTML()
                batch_rollup = gr.HTML()
                batch_per_paper = gr.HTML()
                with gr.Row(visible=False) as batch_dl_row:
                    batch_csv = gr.File(label="Download CSV (one row per verdict)", interactive=False, visible=False)
                    batch_json = gr.File(label="Download JSON (full batch)", interactive=False, visible=False)
                    batch_bib = gr.File(
                        label="Download problematic refs (.bib)", interactive=False, visible=False,
                    )

                def _run_batch_with_visibility(*args, request: gr.Request = None):
                    # Generator: clear the previous batch's output before
                    # ``run_batch`` starts. Without this, re-running a batch
                    # that errors (rate limit, oversized upload, etc.) leaves
                    # the previous batch's CSV/JSON/.bib download links
                    # visible alongside the error toast — looks like the new
                    # batch produced them.
                    yield (
                        "", "", "",                                       # summary, rollup, per-paper
                        gr.update(value=None, visible=False),             # csv
                        gr.update(value=None, visible=False),             # json
                        gr.update(value=None, visible=False),             # bib
                        gr.update(visible=False),                         # dl row
                    )
                    out = run_batch(*args, request=request)
                    summary_html, rollup_html, per_paper_html, csv_path, json_path, bib_path = out
                    yield (
                        summary_html, rollup_html, per_paper_html,
                        gr.update(value=csv_path, visible=bool(csv_path)),
                        gr.update(value=json_path, visible=bool(json_path)),
                        gr.update(value=bib_path, visible=bool(bib_path)),
                        gr.update(visible=True),
                    )

                batch_btn.click(
                    fn=_run_batch_with_visibility,
                    inputs=[batch_files, batch_chk_existence, batch_chk_claims, batch_retry],
                    outputs=[
                        batch_summary, batch_rollup, batch_per_paper,
                        batch_csv, batch_json, batch_bib, batch_dl_row,
                    ],
                    # Pin progress to the per-paper detail area only, otherwise
                    # Gradio paints a duplicate bar on every HTML output.
                    show_progress_on=batch_per_paper,
                )

            # ── Tab 3: About ──────────────────────────────────
            with gr.TabItem("About"):
                gr.HTML(ABOUT_HTML)

        # Page footer — version + license + anon-review marker, Distill / AISI
        # style. Replaces what would otherwise be Gradio's "Built with Gradio".
        gr.HTML(
            '<footer class="cc-page-footer">'
            '<code>v0.2.0</code>'
            '<span class="cc-footer-sep">·</span>'
            'open-source'
            '<span class="cc-footer-sep">·</span>'
            'anonymous (under review)'
            '</footer>'
        )

    return demo


_ACCESS_TOKEN_COOKIE = "checkcitation_token"
_ACCESS_TOKEN_QUERY = "token"
# 7-day idle window. Refreshed on every authenticated request, so an active
# reviewer stays signed in indefinitely; a stolen cookie is bounded to a week
# of inactivity rather than the previous 30. There is no server-side
# revocation list — bumping REVIEW_ACCESS_TOKEN invalidates all sessions.
_ACCESS_TOKEN_COOKIE_MAX_AGE = 60 * 60 * 24 * 7

_GATE_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>CheckCitation: access required</title>
<style>
body{{font-family:-apple-system,system-ui,sans-serif;background:#f8fafc;color:#334155;
    display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0;}}
.box{{max-width:420px;padding:32px;background:#fff;border-radius:12px;
    box-shadow:0 4px 24px rgba(0,0,0,0.06);text-align:center;}}
h1{{margin:0 0 12px;font-size:20px;}}
p{{margin:8px 0;font-size:14px;line-height:1.5;}}
code{{background:#f1f5f9;padding:2px 6px;border-radius:4px;font-size:13px;}}
</style></head><body>
<div class="box">
<h1>Access required</h1>
<p>This instance of CheckCitation is gated for a specific set of users.</p>
<p>Append the access token to the URL, for example:</p>
<p><code>{url}?token=YOUR_TOKEN</code></p>
</div></body></html>"""


_PUBLIC_PATHS = {
    "/health",           # container healthcheck
    "/docs",             # Swagger UI — API discoverability
    "/redoc",            # alternative API docs
    "/openapi.json",     # machine-readable API spec
}


def _token_gate_middleware(expected_token: str):
    """FastAPI middleware that requires `?token=<expected>` or a matching cookie.

    Bypasses a small allowlist (`/health`, `/docs`, `/redoc`, `/openapi.json`)
    so healthchecks and API-spec discovery work without a token. Actual API
    calls still require the token.
    On successful query-param match, sets a 30-day cookie so reviewers
    don't have to keep pasting the token.
    """
    from fastapi.responses import HTMLResponse

    async def middleware(request, call_next):
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        submitted = (
            request.cookies.get(_ACCESS_TOKEN_COOKIE)
            or request.query_params.get(_ACCESS_TOKEN_QUERY)
        )
        if not hmac.compare_digest(submitted or "", expected_token):
            base = f"{request.url.scheme}://{request.url.netloc}{request.url.path}"
            return HTMLResponse(_GATE_PAGE.format(url=base), status_code=403)

        response = await call_next(request)
        # Refresh cookie on every valid request so the session stays alive
        response.set_cookie(
            _ACCESS_TOKEN_COOKIE, expected_token,
            max_age=_ACCESS_TOKEN_COOKIE_MAX_AGE,
            httponly=True, samesite="lax",
        )
        return response

    return middleware


def _build_fastapi_with_health():
    """Wrap the Gradio Blocks in a FastAPI app so we can expose /health.

    Cloud Run / docker-compose healthchecks hit /health. It returns plain
    text 'ok' and does NOT touch any pipeline or LLM — a 200 here just means
    the process is up and the HTTP server is responding.

    If REVIEW_ACCESS_TOKEN is set in the environment, all routes except
    /health require the token. Unset = gate disabled (dev mode).
    """
    from fastapi import FastAPI
    from fastapi.responses import PlainTextResponse

    api = FastAPI()

    expected = os.environ.get("REVIEW_ACCESS_TOKEN", "").strip()
    if expected:
        api.middleware("http")(_token_gate_middleware(expected))
        logging.getLogger(__name__).info(
            "REVIEW_ACCESS_TOKEN set — token gate enabled for all non-/health routes"
        )

    @api.get("/health")
    def _health():
        return PlainTextResponse("ok")

    demo = create_app()
    # Crimson Pro is loaded via Google Fonts for headings + research-prose
    # surfaces (see --font-serif in BASE_CSS). Loaded with preconnect to
    # mitigate FOUT; display=swap keeps text legible while the font streams.
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
        # Empty list hides all three Gradio chrome elements at once:
        # the "Use via API" link, the Gradio logo, and the settings gear.
        footer_links=[],
        head=fonts_head,
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(_build_fastapi_with_health(), host="0.0.0.0", port=port)
