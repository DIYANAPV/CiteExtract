"""CheckCitation Web UI — local Gradio interface for citation verification.

Launch with:
    python app.py

Requires:
    - GROBID running for PDF input: docker run -d -p 8070:8070 grobid/grobid:0.8.2-crf
    - .env with OPENAI_API_KEY for Claim Verification (Agentic mode)
"""

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
from src.parsers.router import parse_file
from src.pipeline import run_unified

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VERDICT_STYLES = {
    "VALID":          {"color": "#16a34a", "bg": "#f0fdf4", "border": "#22c55e", "icon": "&#x2705;", "label": "Valid"},
    "FABRICATED":     {"color": "#dc2626", "bg": "#fef2f2", "border": "#ef4444", "icon": "&#x274C;", "label": "Fabricated"},
    "MISREPRESENTED": {"color": "#d97706", "bg": "#fffbeb", "border": "#f59e0b", "icon": "&#x26A0;&#xFE0F;", "label": "Misrepresented"},
    "UNVERIFIABLE":   {"color": "#6b7280", "bg": "#f9fafb", "border": "#9ca3af", "icon": "&#x2753;", "label": "Unverifiable"},
}

RISK_COLORS = {"LOW": "#16a34a", "MEDIUM": "#d97706", "HIGH": "#dc2626", "CRITICAL": "#7f1d1d"}

BASE_CSS = """
/* ── Layout ── */
.gradio-container {
    max-width: 1200px !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* ── Subtle entrance ── */
@keyframes cc-fadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Header ── */
.cc-header {
    text-align: center; padding: 24px 0 16px;
    border-bottom: 2px solid #e2e8f0; margin-bottom: 4px;
}
.cc-header h1 {
    font-size: 22px; font-weight: 700; color: #0f172a;
    letter-spacing: -0.01em; margin: 0;
}
.cc-header p {
    font-size: 13px; color: #64748b; margin: 4px 0 0; font-weight: 400;
}

/* ── Section labels ── */
.cc-section-label {
    font-size: 11px; font-weight: 600; color: #475569;
    text-transform: uppercase; letter-spacing: 0.06em;
    margin: 12px 0 6px; padding-bottom: 6px;
    border-bottom: 1px solid #e2e8f0;
}

/* ── Cards ── */
.card {
    border-radius: 8px; padding: 16px 20px; margin-bottom: 10px;
    border-left: 4px solid; background: #fff;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    animation: cc-fadeIn 0.25s ease-out both;
}
.card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.card summary { cursor: pointer; font-weight: 600; font-size: 14px; line-height: 1.5; }

/* ── Metadata table ── */
.meta-table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; }
.meta-table th, .meta-table td { text-align: left; padding: 7px 10px; border-bottom: 1px solid #f1f5f9; }
.meta-table th {
    background: #f8fafc; font-weight: 600; color: #475569;
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em;
}
.meta-table tr:hover td { background: #fafbfd; }
.meta-table td { max-width: 300px; word-break: break-word; }
.status-match    { color: #16a34a; font-weight: 600; }
.status-close    { color: #d97706; font-weight: 600; }
.status-mismatch { color: #dc2626; font-weight: 600; }
.status-missing  { color: #cbd5e1; }

/* ── Dashboard ── */
.dashboard {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 20px 24px; margin-bottom: 14px;
    animation: cc-fadeIn 0.3s ease-out both;
}
.stat-cards { display: flex; gap: 10px; flex-wrap: wrap; margin: 14px 0; }
.stat-card {
    flex: 1; min-width: 100px; text-align: center;
    padding: 14px 10px; border-radius: 8px;
    border: 1px solid #e5e7eb; background: white;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}
.stat-card:hover { box-shadow: 0 2px 6px rgba(0,0,0,0.05); }
.stat-card .num { font-size: 26px; font-weight: 700; line-height: 1.2; }
.stat-card .lbl {
    font-size: 11px; color: #64748b; margin-top: 2px;
    font-weight: 500; text-transform: uppercase; letter-spacing: 0.03em;
}
.progress-bar { height: 6px; border-radius: 6px; background: #e5e7eb; overflow: hidden; margin: 6px 0; }
.progress-fill { height: 100%; border-radius: 6px; transition: width 0.5s ease-out; }

/* ── Claims & passages ── */
.claim-box {
    background: #fff; border: 1px solid #e5e7eb; border-radius: 8px;
    padding: 12px 14px; margin: 6px 0; font-size: 13px; line-height: 1.6;
}
.claim-arrow { text-align: center; font-size: 18px; margin: 4px 0; color: #cbd5e1; }
.claim-section {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 12px 14px; margin: 8px 0;
}
.passage-card {
    background: #fff; border: 1px solid #e5e7eb; border-radius: 8px;
    padding: 12px 14px; margin: 6px 0;
}
.passage-card:hover { border-color: #93c5fd; }
.passage-header { font-size: 11px; color: #64748b; margin-bottom: 4px; font-weight: 500; }
.passage-text { font-size: 13px; line-height: 1.65; color: #1e293b; }

/* ── Links & pills ── */
.link-pill {
    display: inline-block; background: #eff6ff; color: #2563eb;
    padding: 2px 10px; border-radius: 12px; font-size: 11px;
    text-decoration: none; margin-right: 4px; font-weight: 500;
}
.link-pill:hover { background: #dbeafe; }

/* ── Coverage ── */
.coverage-bar { display: flex; gap: 8px; align-items: center; margin: 5px 0; font-size: 12px; }
.coverage-fill { height: 6px; border-radius: 3px; }
.info-row { font-size: 12px; color: #64748b; margin-top: 8px; }

/* ── Empty state ── */
.cc-empty { text-align: center; padding: 48px 20px; color: #94a3b8; }
.cc-empty-icon { font-size: 36px; margin-bottom: 8px; opacity: 0.35; }
.cc-empty-title { font-size: 14px; font-weight: 600; color: #64748b; margin-bottom: 4px; }
.cc-empty-sub { font-size: 12px; color: #94a3b8; }

/* ── Tab styling ── */
.tab-nav button { font-weight: 600 !important; font-size: 13px !important; }

/* ── HITL review controls ── */
.cc-review-bar {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 12px 16px; margin: 0 0 12px; display: flex; align-items: center;
    gap: 14px; font-size: 13px;
}
.cc-review-progress {
    flex: 1; display: flex; align-items: center; gap: 10px;
    color: #475569; font-weight: 500;
}
.cc-review-progress .cc-pb {
    flex: 1; height: 6px; background: #e5e7eb; border-radius: 6px; overflow: hidden;
}
.cc-review-progress .cc-pb-fill {
    height: 100%; background: #3b82f6; border-radius: 6px;
    transition: width 0.25s ease-out;
}
.cc-review-counts .tag {
    display: inline-block; padding: 2px 8px; border-radius: 10px;
    font-size: 11px; font-weight: 600; margin-left: 4px;
}
.cc-review-counts .tag-agree    { background: #dcfce7; color: #15803d; }
.cc-review-counts .tag-disagree { background: #fee2e2; color: #b91c1c; }
.cc-review-counts .tag-info     { background: #fef9c3; color: #a16207; }
.cc-review-btn-reset {
    background: #f1f5f9; border: 1px solid #cbd5e1; color: #475569;
    padding: 4px 10px; border-radius: 6px; font-size: 12px; cursor: pointer;
}
.cc-review-btn-reset:hover { background: #e2e8f0; }
.cc-review-btn-export {
    background: #eff6ff; border: 1px solid #bfdbfe; color: #1d4ed8;
    padding: 4px 10px; border-radius: 6px; font-size: 12px; cursor: pointer;
}
.cc-review-btn-export:hover { background: #dbeafe; }

.cc-review-row {
    display: flex; gap: 6px; margin-top: 12px; padding-top: 10px;
    border-top: 1px dashed #e2e8f0;
}
.cc-review-row .cc-btn {
    font-size: 12px; font-weight: 600; padding: 5px 12px; border-radius: 6px;
    cursor: pointer; border: 1px solid transparent; background: #f8fafc;
    color: #475569; transition: all 0.15s ease;
}
.cc-review-row .cc-btn:hover { background: #f1f5f9; }
.cc-review-row .cc-btn[data-active="agree"],
.cc-review-row .cc-btn.agree:hover {
    background: #dcfce7; color: #15803d; border-color: #86efac;
}
.cc-review-row .cc-btn[data-active="disagree"],
.cc-review-row .cc-btn.disagree:hover {
    background: #fee2e2; color: #b91c1c; border-color: #fca5a5;
}
.cc-review-row .cc-btn[data-active="info"],
.cc-review-row .cc-btn.info:hover {
    background: #fef9c3; color: #a16207; border-color: #fde047;
}
.cc-review-row .cc-btn[data-active] {
    box-shadow: inset 0 0 0 1px currentColor;
}
.cc-review-row .cc-label {
    font-size: 11px; color: #94a3b8; align-self: center; margin-right: auto;
    text-transform: uppercase; letter-spacing: 0.04em; font-weight: 600;
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

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB hard cap per file
PDF_MAGIC = b"%PDF-"
ALLOWED_SUFFIXES = {".pdf", ".tex", ".bib", ".txt"}


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
            f"{label.capitalize()} is {size / 1024 / 1024:.1f} MB — "
            f"limit is {MAX_UPLOAD_BYTES // 1024 // 1024} MB."
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


def prepare_ref_pdfs_dir(pdf_paths: Optional[list[str]]) -> Optional[str]:
    if not pdf_paths:
        return None
    for p in pdf_paths:
        _validate_upload(p, label="reference PDF")
    tmp_dir = tempfile.mkdtemp(prefix="checkcitation_refs_")
    for p in pdf_paths:
        src = Path(p)
        shutil.copy2(str(src), str(Path(tmp_dir) / src.name))
    return tmp_dir


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

    by_v = s.by_verdict
    cards_html = ""
    for key in ["VALID", "FABRICATED", "MISREPRESENTED", "UNVERIFIABLE"]:
        count = by_v.get(key, 0)
        st = VERDICT_STYLES.get(key, VERDICT_STYLES["VALID"])
        cards_html += f'''
        <div class="stat-card" style="border-top: 3px solid {st['border']};">
            <div class="num" style="color: {st['color']};">{count}</div>
            <div class="lbl">{st['label']}</div>
        </div>'''

    warnings_html = ""
    if report.warnings:
        items = "".join(f"<li>{_esc(w)}</li>" for w in report.warnings)
        warnings_html = f'<div style="margin-top:10px;font-size:13px;color:#6b7280;"><b>Notes:</b><ul style="margin:4px 0;">{items}</ul></div>'

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
        <div class="stat-cards">{cards_html}</div>
        <div class="info-row">
            Mode: {mode_html} &nbsp;|&nbsp;
            Format: <b>{_esc(report.input_format)}</b> &nbsp;|&nbsp;
            References: <b>{report.total_references}</b>
            {f"&nbsp;|&nbsp; Time: <b>{elapsed:.1f}s</b>" if elapsed else ""}
        </div>
        {warnings_html}
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

CLAIM_VERDICT_STYLES = {
    # 3-class scheme (agentic.claim_agent.verdict_classes = 3)
    "SUPPORTS":      {"color": "#15803d", "bg": "#dcfce7", "border": "#86efac", "label": "Supports"},
    "CONTRADICTS":   {"color": "#b91c1c", "bg": "#fee2e2", "border": "#fca5a5", "label": "Contradicts"},
    "NEUTRAL":       {"color": "#6b7280", "bg": "#f3f4f6", "border": "#d1d5db", "label": "Neutral"},
    # 2-class scheme (agentic.claim_agent.verdict_classes = 2)
    "SUPPORTED":     {"color": "#15803d", "bg": "#dcfce7", "border": "#86efac", "label": "Supported"},
    "NOT_SUPPORTED": {"color": "#b91c1c", "bg": "#fee2e2", "border": "#fca5a5", "label": "Not Supported"},
}


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


def _format_passages_for_ref(comp_results: list) -> str:
    """Render passage retrieval + claim verification results for one reference."""
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

    inner = ""
    for r in substantive:
        # Claim verdict badge (if LLM analysis was run)
        claim_html = _format_claim_verdict(r.claim_verdict) if r.claim_verdict else ""

        # Build citing context display: combine before + citing + after into
        # one readable block. When context_before or context_after exist, show
        # them so the claim is always visible even if the citing_sentence field
        # is just a trailing author-year fragment from sentence splitting.
        parts = []
        if r.context_before:
            parts.append(_esc(r.context_before))
        parts.append(_esc(r.citing_sentence))
        if r.context_after:
            parts.append(_esc(r.context_after))
        full_context = " ".join(parts)

        inner += f'''
        <div style="margin-top:8px;font-size:12px;color:#6b7280;">Citing context:</div>
        <div class="claim-box" style="border-left:3px solid #3b82f6;">
            <span style="color:#1e293b;">{full_context}</span>
        </div>
        {claim_html}'''
        for i, sc in enumerate(r.top_passages, 1):
            score_val = sc.rrf_score if sc.rrf_score is not None else (sc.dense_score if sc.dense_score is not None else sc.bm25_score)
            score_type = "RRF" if sc.rrf_score is not None else ("Dense" if sc.dense_score is not None else "BM25")
            section = f" &middot; Section: {_esc(sc.chunk.section_name)}" if sc.chunk.section_name else ""
            inner += f'''
            <div class="passage-card">
                <div class="passage-header">Passage {i}{section} &middot; {score_type}: {score_val:.3f}</div>
                <div class="passage-text">{_esc(sc.chunk.text)}</div>
            </div>'''
        if not r.top_passages:
            inner += '<div style="font-size:13px;color:#6b7280;padding:8px;">No passages retrieved.</div>'

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

def _review_ui_bootstrap() -> str:
    """Progress bar + JavaScript that powers per-card HITL review controls.

    Reviews persist in the browser's localStorage keyed by ref_id so users
    can re-analyze without losing state. A Reset button clears reviews for
    the current browser; an Export button downloads a CSV of
    {ref_id, verdict, review} for auditing or user-study data capture.
    """
    return '''
<div class="cc-review-bar" id="cc-review-bar">
    <div class="cc-review-progress">
        <span id="cc-review-progress-label">Reviewed <b>0</b> / <b>0</b></span>
        <div class="cc-pb"><div class="cc-pb-fill" id="cc-pb-fill" style="width:0%;"></div></div>
    </div>
    <div class="cc-review-counts" id="cc-review-counts"></div>
    <button type="button" class="cc-review-btn-export" onclick="ccExportReviews()">Export CSV</button>
    <button type="button" class="cc-review-btn-reset" onclick="ccResetReviews()">Reset</button>
</div>
<script>
(function() {
    if (window.__ccReviewLoaded) { window.ccRefreshReviews(); return; }
    window.__ccReviewLoaded = true;
    const KEY_PREFIX = 'checkcite-review-v1-';

    function getState(refId) {
        try { return localStorage.getItem(KEY_PREFIX + refId) || ''; }
        catch (e) { return ''; }
    }
    function setState(refId, state) {
        try {
            if (state) localStorage.setItem(KEY_PREFIX + refId, state);
            else       localStorage.removeItem(KEY_PREFIX + refId);
        } catch (e) { /* ignore */ }
    }
    function applyCardState(card) {
        const refId = card.getAttribute('data-cc-ref');
        const state = getState(refId);
        card.querySelectorAll('.cc-review-row .cc-btn').forEach(b => b.removeAttribute('data-active'));
        if (!state) return;
        const row = card.querySelector('.cc-review-row');
        if (!row) return;
        const btn = row.querySelector('.cc-btn.' + state);
        if (btn) btn.setAttribute('data-active', state);
    }
    function refresh() {
        const cards = document.querySelectorAll('details.card[data-cc-ref]');
        let total = cards.length, reviewed = 0, agree = 0, disagree = 0, info = 0;
        cards.forEach(card => {
            applyCardState(card);
            const s = getState(card.getAttribute('data-cc-ref'));
            if (s) { reviewed += 1; if (s==='agree') agree++; else if (s==='disagree') disagree++; else if (s==='info') info++; }
        });
        const label = document.getElementById('cc-review-progress-label');
        const fill  = document.getElementById('cc-pb-fill');
        const counts = document.getElementById('cc-review-counts');
        if (label) label.innerHTML = 'Reviewed <b>' + reviewed + '</b> / <b>' + total + '</b>';
        if (fill)  fill.style.width = (total ? (100 * reviewed / total) : 0) + '%';
        if (counts) {
            counts.innerHTML =
                '<span class="tag tag-agree">✓ ' + agree + '</span>' +
                '<span class="tag tag-disagree">✗ ' + disagree + '</span>' +
                '<span class="tag tag-info">? ' + info + '</span>';
        }
    }
    window.ccRefreshReviews = refresh;
    window.ccSetReview = function(refId, state) {
        const cur = getState(refId);
        setState(refId, cur === state ? '' : state);  // toggle off if same
        refresh();
    };
    window.ccResetReviews = function() {
        if (!confirm('Clear all review marks in this browser?')) return;
        try {
            const toRemove = [];
            for (let i = 0; i < localStorage.length; i++) {
                const k = localStorage.key(i);
                if (k && k.indexOf(KEY_PREFIX) === 0) toRemove.push(k);
            }
            toRemove.forEach(k => localStorage.removeItem(k));
        } catch (e) { /* ignore */ }
        refresh();
    };
    window.ccExportReviews = function() {
        const rows = [['ref_id', 'verdict', 'review']];
        document.querySelectorAll('details.card[data-cc-ref]').forEach(card => {
            const refId = card.getAttribute('data-cc-ref');
            const verdict = card.getAttribute('data-cc-verdict') || '';
            const s = getState(refId);
            if (s) rows.push([refId, verdict, s]);
        });
        const csv = rows.map(r => r.map(v => '"' + String(v).replace(/"/g, '""') + '"').join(',')).join('\\n');
        const blob = new Blob([csv], { type: 'text/csv' });
        const url  = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'checkcite-reviews.csv';
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };
    // Re-apply when Gradio replaces the HTML (MutationObserver is safer than onload).
    const target = document.body;
    if (target) {
        const obs = new MutationObserver(() => refresh());
        obs.observe(target, { childList: true, subtree: true });
    }
    refresh();
})();
</script>
'''


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
        order = {"FABRICATED": 0, "MISREPRESENTED": 1, "UNVERIFIABLE": 2, "VALID": 3}
        sorted_verdicts = sorted(report.verdicts, key=lambda v: order.get(v.verdict, 9))

        cards = ""
        for v in sorted_verdicts:
            ref = ref_map.get(v.ref_id)
            passages_html = _format_passages_for_ref(comp_by_ref.get(v.ref_id, [])) if has_passages else ""
            cards += _render_verdict_card(v, ref, passages_html)
        if not cards:
            return '<div style="color:#6b7280;text-align:center;padding:40px;">No references found.</div>'
        return _review_ui_bootstrap() + cards

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
            cards += f'''
            <details class="card" style="background:#f8fafc; border-left-color:#3b82f6;" open>
                <summary style="color:#1e40af;">
                    [{_esc(ref.ref_id)}] {authors}{year}
                </summary>
                <div style="margin-top:8px;">
                    <div style="font-size:15px;font-weight:600;color:#1f2937;">{title}</div>
                    {"<div style='font-size:13px;color:#6b7280;'>" + venue + "</div>" if venue else ""}
                    {passages_html}
                </div>
            </details>'''
        return cards if cards else '<div style="color:#6b7280;text-align:center;padding:40px;">No passages found.</div>'

    return '<div style="color:#6b7280;text-align:center;padding:40px;">No results.</div>'


# Human-readable explanations for the cryptic internal flags that land on a verdict.
# Anything not in this map is shown with its raw code so the user at least sees it.
_FLAG_EXPLANATIONS = {
    "agentic_agent_used":            "LLM agent inspected the evidence for this reference",
    "agent_fallback_to_quick":       "Agent failed; fell back to rule-based classification",
    "agentic_processing_error":      "Unexpected error while running the agent",
    "claim_contradicts":             "The cited paper contradicts what the citing sentence claims",
    "claim_not_supported":           "The cited paper does not support the citing sentence",
    "claim_neutral":                 "Evidence was insufficient to decide — not enough overlap",
    "claim_verification_empty":      "No claim verdicts were returned",
    "claim_verification_unavailable": "Claim verification could not run (no passages)",
    "metadata_unverifiable_claims_skipped": "Metadata couldn't be decided — claims not checked",
    "author_mismatch":               "Authors in the citation differ from the matched record",
    "title_mismatch":                "Title differs from the matched record",
    "year_mismatch":                 "Year differs from the matched record",
    "venue_mismatch":                "Venue differs from the matched record",
    "low_title_similarity":          "Title only loosely matches the database record",
    "missing_existence_result":      "No existence check result was stored for this reference",
    "retracted":                     "This paper has been retracted",
    "preprint_upgraded_to_published": "Matched an arXiv preprint with a later publisher version",
}


def _format_databases_searched(ex) -> str:
    """Pill list showing which scholarly databases were queried and which one matched."""
    if not ex or not ex.databases_checked:
        return ""
    pills = []
    hit_source = (ex.source or "").lower() if ex.status == "FOUND" else ""
    for db in ex.databases_checked:
        is_hit = db.lower() == hit_source
        bg = "#dcfce7" if is_hit else "#f3f4f6"
        fg = "#166534" if is_hit else "#4b5563"
        icon = "&#x2713;" if is_hit else "&#x2022;"
        pills.append(
            f'<span style="display:inline-block;background:{bg};color:{fg};'
            f'padding:2px 10px;border-radius:12px;font-size:12px;margin:2px 4px 2px 0;'
            f'border:1px solid {bg};">{icon} {_esc(db)}</span>'
        )
    label = "Matched in" if hit_source else "Searched (no match)"
    return (
        f'<div style="margin-top:10px;">'
        f'<div style="font-size:12px;color:#6b7280;margin-bottom:4px;">{label}:</div>'
        f'<div>{" ".join(pills)}</div>'
        f'</div>'
    )


def _format_matched_record(ex) -> str:
    """Key-value block for the record that was actually found in a DB."""
    if not ex or ex.status != "FOUND":
        return ""
    rows = []
    if ex.matched_title:
        rows.append(("Title", _esc(ex.matched_title)))
    if ex.matched_authors:
        rows.append(("Authors", _esc("; ".join(ex.matched_authors))))
    if ex.matched_year:
        rows.append(("Year", str(ex.matched_year)))
    if ex.matched_venue:
        rows.append(("Venue", _esc(ex.matched_venue)))
    if ex.matched_doi:
        rows.append(("DOI", f'<a href="https://doi.org/{_esc(ex.matched_doi)}" '
                            f'target="_blank" rel="noopener">{_esc(ex.matched_doi)}</a>'))
    if ex.title_similarity is not None:
        rows.append(("Title similarity", f"{ex.title_similarity:.0%}"))
    if ex.retraction_status:
        rows.append(("Retracted", "Yes"))
    if not rows:
        return ""
    rows_html = "".join(
        f'<tr><td style="padding:3px 12px 3px 0;color:#6b7280;font-size:12px;white-space:nowrap;">'
        f'{k}</td><td style="padding:3px 0;font-size:13px;">{val}</td></tr>'
        for k, val in rows
    )
    return (
        f'<div style="margin-top:12px;">'
        f'<div style="font-size:12px;color:#6b7280;margin-bottom:4px;">Matched record:</div>'
        f'<table style="width:100%;border-collapse:collapse;">{rows_html}</table>'
        f'</div>'
    )


def _format_decision_path(flags: list[str]) -> str:
    """Translate each flag to a plain-English bullet so users see *why* the verdict."""
    if not flags:
        return ""
    items = []
    for f in flags:
        human = _FLAG_EXPLANATIONS.get(f, _esc(f))
        items.append(
            f'<li style="font-size:13px;color:#374151;margin:2px 0;">{human}</li>'
        )
    return (
        f'<div style="margin-top:12px;">'
        f'<div style="font-size:12px;color:#6b7280;margin-bottom:4px;">Decision path:</div>'
        f'<ul style="margin:0;padding-left:18px;">{"".join(items)}</ul>'
        f'</div>'
    )


def _format_evidence_panel(v) -> str:
    """Collapsible 'Evidence trail' block showing DBs searched, matched record, decision path."""
    dbs = _format_databases_searched(v.existence)
    record = _format_matched_record(v.existence)
    decision = _format_decision_path(v.flags)
    if not any([dbs, record, decision]):
        return ""
    body = f'<div style="padding:4px 0 8px;">{dbs}{record}{decision}</div>'
    return (
        f'<details style="margin-top:10px;">'
        f'<summary style="font-size:13px;color:#4b5563;cursor:pointer;'
        f'font-weight:500;">&#x1F50D; Evidence trail</summary>'
        f'{body}'
        f'</details>'
    )


def _render_verdict_card(v, ref, passages_html: str = "") -> str:
    """Render a single reference card with verdict + optional passages."""
    st = VERDICT_STYLES.get(v.verdict, VERDICT_STYLES["VALID"])
    is_problem = v.verdict in ("FABRICATED", "MISREPRESENTED")
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

    explanation = f'<div style="font-size:13px;margin-top:8px;color:#4b5563;line-height:1.5;">{_esc(v.explanation)}</div>'

    meta_table = format_metadata_table(v) or format_agentic_metadata_table(v, ref)
    meta_section = ""
    if meta_table:
        meta_section = f'''
        <details style="margin-top:10px;">
            <summary style="font-size:13px;color:#6b7280;cursor:pointer;">Metadata Comparison</summary>
            {meta_table}
        </details>'''

    evidence_panel = _format_evidence_panel(v)

    ref_id_attr = _esc(v.ref_id)
    review_row = f'''
        <div class="cc-review-row" data-cc-review-ref="{ref_id_attr}">
            <span class="cc-label">Your review</span>
            <button type="button" class="cc-btn agree"
                    onclick="ccSetReview('{ref_id_attr}','agree')">✓ Agree</button>
            <button type="button" class="cc-btn disagree"
                    onclick="ccSetReview('{ref_id_attr}','disagree')">✗ Disagree</button>
            <button type="button" class="cc-btn info"
                    onclick="ccSetReview('{ref_id_attr}','info')">? Need info</button>
        </div>'''

    return f'''
    <details class="card" data-cc-ref="{ref_id_attr}" data-cc-verdict="{_esc(v.verdict)}"
             style="background:{st['bg']}; border-left-color:{st['border']};" {open_attr}>
        <summary>
            <span style="color:{st['color']};">{st['icon']} {st['label'].upper()}</span>
            &nbsp;&mdash;&nbsp;
            [{ref_id_attr}] {authors}{year}
        </summary>
        <div style="margin-top:8px;">
            <div style="font-size:15px;font-weight:600;color:#1f2937;">{title}</div>
            {"<div style='font-size:13px;color:#6b7280;'>" + venue + fmt_badge + "</div>" if venue else (fmt_badge if fmt_badge else "")}
            {links_html}
            {source_info}
            {action_html}
            {explanation}
            {meta_section}
            {evidence_panel}
            {passages_html}
            {review_row}
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
# Parse output formatting
# ---------------------------------------------------------------------------

def format_parse_output(parsed) -> str:
    # Summary
    ref_count = len(parsed.references)
    cit_count = len(parsed.citations)
    fmt = _esc(parsed.input_format)
    body = "Yes" if parsed.has_body_text else "No"

    warnings_html = ""
    if parsed.warnings:
        items = "".join(f"<li>{_esc(w)}</li>" for w in parsed.warnings)
        warnings_html = f'<ul style="margin:4px 0;color:#d97706;">{items}</ul>'

    # References table
    ref_rows = ""
    for r in parsed.references:
        title = _esc(r.title) if r.title else "<em>—</em>"
        authors = _esc("; ".join(r.authors)) if r.authors else "—"
        year = r.year or "—"
        venue = _esc(r.venue) if r.venue else "—"
        cfmt = getattr(r, 'citation_format', None)
        cfmt_str = _esc(cfmt.upper()) if cfmt else "—"
        ref_rows += f"<tr><td><b>{_esc(r.ref_id)}</b></td><td>{title}</td><td>{authors}</td><td>{year}</td><td>{venue}</td><td>{cfmt_str}</td></tr>"

    refs_html = f'''
    <table class="meta-table">
        <thead><tr><th>ID</th><th>Title</th><th>Authors</th><th>Year</th><th>Venue</th><th>Format</th></tr></thead>
        <tbody>{ref_rows}</tbody>
    </table>''' if ref_rows else '<div style="color:#6b7280;">No references extracted.</div>'

    # Citations
    cit_html = ""
    for c in parsed.citations[:50]:
        section = f' <span style="font-size:11px;color:#6b7280;">({_esc(c.section)})</span>' if c.section else ""
        # Combine full context so the claim is always visible
        parts = []
        if c.context_before:
            parts.append(_esc(c.context_before, 150))
        parts.append(f'<b>{_esc(c.citing_sentence, 200)}</b>')
        if c.context_after:
            parts.append(_esc(c.context_after, 150))
        full_context = " ".join(parts)
        cit_html += f'''
        <div style="padding:8px 12px;margin:4px 0;background:#f8fafc;border-radius:6px;font-size:13px;line-height:1.6;">
            <span style="font-weight:600;color:#3b82f6;">[{_esc(c.ref_id)}]</span>{section}<br>
            {full_context}
        </div>'''
    if not cit_html:
        cit_html = '<div style="color:#6b7280;">No in-text citations found (expected for BibTeX-only input).</div>'

    return f'''
    <div class="dashboard">
        <div style="font-size:14px;color:#6b7280;">Parse Results</div>
        <div style="display:flex;gap:20px;margin:10px 0;flex-wrap:wrap;">
            <div><b>Format:</b> {fmt}</div>
            <div><b>References:</b> {ref_count}</div>
            <div><b>Citations:</b> {cit_count}</div>
            <div><b>Body text:</b> {body}</div>
        </div>
        {warnings_html}
    </div>
    <h3 style="margin-top:16px;">References</h3>
    {refs_html}
    <h3 style="margin-top:20px;">Citations</h3>
    {cit_html}'''


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
                f"Only {remaining} paper(s) remaining today — "
                "this is a research demo, please try again tomorrow."
            )
        _daily_analyses[today] += n
        # Clean old entries
        for k in list(_daily_analyses):
            if k != today:
                del _daily_analyses[k]


def _check_rate_limit(request: Optional["gr.Request"] = None) -> None:
    """Enforce the daily global cap and the per-IP hourly single-paper cap."""
    _reserve_daily_papers(1)

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


# Maps timing-log STAGE names to (progress_fraction, user-facing message).
# Progress fractions are approximate — the pipeline does not know total time
# upfront, so we pick values that step monotonically through typical runs.
_STAGE_PROGRESS = {
    "L1_parse":                   (0.10, "Parsed paper. Checking references..."),
    "L1_parse_only":              (0.15, "Loading cached result..."),
    "L2_existence":               (0.35, "References checked. Analyzing..."),
    "L3_L5_quick":                (0.95, "Finalizing..."),
    "agentic_pre_retrieve":       (0.55, "Retrieved cited paper passages. Triaging..."),
    "agentic_triage":             (0.60, "Triaged. Running metadata agents..."),
    "agentic_metadata_dispatch":  (0.72, "Metadata agents done. Running claim agents..."),
    "agentic_claim_dispatch":     (0.88, "Claim agents done. Running combined agents..."),
    "agentic_both_dispatch":      (0.94, "Agents done. Finalizing..."),
    "L5_agentic":                 (0.95, "Agentic pass done. Finalizing..."),
    "L4_comprehension":           (0.92, "Claim verification done. Finalizing..."),
}

_STAGE_RE = re.compile(r"STAGE (\S+) seconds=")


def run_analyze(file, ref_pdfs, check_existence, check_claims, retry_failed,
                request: gr.Request = None, progress=gr.Progress()):
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

    progress(0.02, desc="Preparing...")

    # Tap the timing logger so we can show stage-by-stage progress without
    # threading a callback through the entire pipeline.
    stage_events: list[str] = []
    event_lock = threading.Lock()

    class _StageCap(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
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

    try:
        seen_stages: set[str] = set()
        while worker.is_alive():
            worker.join(timeout=0.4)
            with event_lock:
                current = list(stage_events)
            for stage_name in current:
                if stage_name in seen_stages:
                    continue
                seen_stages.add(stage_name)
                if stage_name in _STAGE_PROGRESS:
                    frac, desc = _STAGE_PROGRESS[stage_name]
                    progress(frac, desc=desc)
    finally:
        timing_log.removeHandler(handler)
        if ref_dir:
            shutil.rmtree(ref_dir, ignore_errors=True)

    if "error" in result_holder:
        err = result_holder["error"]
        if isinstance(err, gr.Error):
            raise err
        raise gr.Error(f"Analysis failed: {err}")

    paper_report, comp_report, parsed = result_holder["value"]
    elapsed = time.time() - start

    progress(0.97, desc="Rendering results...")

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

    tmp = tempfile.NamedTemporaryFile(
        suffix=".json", prefix="checkcitation_report_", delete=False, mode="w"
    )
    tmp.write(report_json)
    tmp.close()

    progress(1.0, desc="Done")
    return dashboard_html, coverage_html, cards_html, report_json, tmp.name


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

    pills = "".join([
        pill("fabricated", agg["FABRICATED"], "#fef2f2", "#b91c1c"),
        pill("misrepresented", agg["MISREPRESENTED"], "#fffbeb", "#b45309"),
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
                f'<td colspan="5" style="padding:6px 10px;color:#b91c1c;">Failed: {reason}</td>'
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
            f'{cell(c.get("MISREPRESENTED", 0), "#b45309")}'
            f'{cell(c.get("UNVERIFIABLE", 0), "#4b5563")}'
            f'{cell(c.get("VALID", 0), "#15803d")}'
            f'</tr>'
        )

    header = (
        '<tr style="background:#f9fafb;font-size:12px;color:#6b7280;text-align:left;">'
        '<th style="padding:6px 10px;">Paper</th>'
        '<th style="padding:6px 10px;text-align:right;">Refs</th>'
        '<th style="padding:6px 10px;text-align:right;">Fabr.</th>'
        '<th style="padding:6px 10px;text-align:right;">Misrep.</th>'
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

    Only files with accepted extensions are returned. Size + magic-bytes
    checks happen later in _validate_upload per file.
    """
    import zipfile
    out_dir = tempfile.mkdtemp(prefix="checkcitation_batch_")
    extracted: list[str] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            # Defensive: reject absolute paths and traversal
            safe_name = Path(info.filename).name
            if not safe_name or Path(info.filename).is_absolute():
                continue
            suffix = Path(safe_name).suffix.lower()
            if suffix not in ALLOWED_SUFFIXES:
                continue
            dest = Path(out_dir) / safe_name
            with zf.open(info) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(str(dest))
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
                for v in paper_report.verdicts:
                    counts[v.verdict] += 1
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

    # CSV + JSON downloads
    csv_path = _write_batch_csv(per_paper)
    json_path = _write_batch_json(per_paper, effective_mode, elapsed)

    progress(1.0, desc="Done")
    return summary_html, rollup_html, per_paper_html, csv_path, json_path


def run_parse(file):
    if file is None:
        raise gr.Error("Please upload a file.")
    file_path = file if isinstance(file, str) else file.name
    _validate_upload(file_path, label="paper")
    check_prerequisites(file_path, "quick")
    parsed = parse_file(file_path)
    return format_parse_output(parsed)


# ---------------------------------------------------------------------------
# About page
# ---------------------------------------------------------------------------

ABOUT_MD = """
# CheckCitation

**Open-source citation hallucination detection for academic papers.**

CheckCitation analyzes a paper's reference list and in-text citations to detect:
- **Fabricated citations** — references that don't exist in any scholarly database
- **Misrepresented citations** — real papers cited but whose content contradicts the claims made

---

## Analysis Options

| Option | What it does | Cost |
|--------|-------------|------|
| **Existence & Metadata (Rule-based)** | Checks references against CrossRef, Semantic Scholar, OpenAlex, PubMed and validates metadata fields. No LLM — rule-based matching only. | Free |
| **Claim Verification (Agentic)** | LLM agents retrieve cited paper text, find relevant passages, and check if the citing sentence accurately represents the cited paper. | ~$0.01–0.02/paper |

Either or both can be checked. Rule-based alone is fastest; adding claim verification triggers the agentic pipeline.

---

## Prerequisites

1. **GROBID** (for PDF input only):
   ```
   docker run -d --name grobid -p 8070:8070 grobid/grobid:0.8.2-crf
   ```

2. **OpenAI API key** (for Claim Verification):
   ```
   # Add to .env file:
   OPENAI_API_KEY=sk-...
   ```

3. **Python dependencies:**
   ```
   pip install -r requirements.txt
   ```

---

## Reference PDFs

You can upload PDFs of cited papers for better claim verification (useful for non-open-access papers). Files are matched to references by:
1. **Filename matches ref_id** — e.g., `ref_1.pdf` matches `[1]`
2. **Filename contains DOI** — e.g., `10.1234_example.pdf` (use `_` instead of `/`)
3. **Title similarity** — filename compared to reference titles (threshold: 70%)

If you don't upload PDFs, the system fetches full text from Semantic Scholar, Unpaywall, and arXiv.

---

## Credits

Built as part of PhD research at **TIB — Leibniz Information Centre for Science and Technology, Hannover**.
"""


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def create_app() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        radius_size=gr.themes.sizes.radius_sm,
    )

    with gr.Blocks(title="CheckCitation", theme=theme, css=BASE_CSS) as demo:

        gr.HTML('''
        <div class="cc-header">
            <h1>CheckCitation</h1>
            <p>Detect fabricated and misrepresented citations in academic papers</p>
        </div>
        ''')

        with gr.Tabs():
            # ── Tab 1: Analyze ─────────────────────────────────
            with gr.TabItem("Analyze"):
                gr.Markdown(
                    "**Upload a paper → pick what to check → click Analyze.** "
                    "Rule-based check is free and fast. Agentic check uses an LLM "
                    "to verify claims against cited paper text (slower, costs a few "
                    "cents per paper). Click **Try sample paper** below if you just "
                    "want to see how it works."
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        analyze_file = gr.File(
                            label="Upload Paper",
                            file_types=[".pdf", ".tex", ".bib", ".txt"],
                            type="filepath",
                        )
                        analyze_refs = gr.File(
                            label="Reference PDFs (optional — for non-open-access cited papers)",
                            file_types=[".pdf"],
                            file_count="multiple",
                            type="filepath",
                            visible=False,
                        )
                    with gr.Column(scale=2):
                        gr.HTML('<div class="cc-section-label">Analysis Options</div>')
                        chk_existence = gr.Checkbox(
                            label="Existence & Metadata (Rule-based)",
                            value=True,
                            info="Fast, no LLM. Checks refs against CrossRef, Semantic Scholar, OpenAlex, PubMed.",
                        )
                        chk_claims = gr.Checkbox(
                            label="Claim Verification (Agentic)",
                            value=False,
                            info="LLM agents retrieve cited paper text and verify claims. Slower, costs OpenAI credits.",
                        )
                        analyze_retry = gr.Checkbox(label="Retry failed references", value=False)
                        analyze_btn = gr.Button("Analyze", variant="primary", size="lg")
                        sample_btn = gr.Button(
                            "Try sample paper",
                            variant="secondary", size="sm",
                        )

                        chk_claims.change(
                            fn=lambda checked: gr.update(visible=checked),
                            inputs=[chk_claims],
                            outputs=[analyze_refs],
                        )

                        sample_btn.click(
                            fn=lambda: str(Path("data/test_inputs/clean_paper.tex").resolve()),
                            inputs=None,
                            outputs=[analyze_file],
                        )

                # ── Results section ──
                gr.HTML('<div class="cc-section-label" style="margin-top:20px;">Results</div>')
                analyze_dashboard = gr.HTML()
                analyze_coverage = gr.HTML()
                analyze_cards = gr.HTML(
                    value='''<div class="cc-empty">
                        <div class="cc-empty-icon">&#x1F50D;</div>
                        <div class="cc-empty-title">Ready to analyze</div>
                        <div class="cc-empty-sub">Upload a paper and click Analyze to begin</div>
                    </div>''',
                )

                with gr.Accordion("JSON Report", open=False):
                    analyze_json = gr.Code(language="json", label="Report JSON")
                analyze_download = gr.File(label="Download Report", interactive=False)

                # Single click handler — Gradio shows its own loading state
                analyze_btn.click(
                    fn=run_analyze,
                    inputs=[
                        analyze_file, analyze_refs,
                        chk_existence, chk_claims,
                        analyze_retry,
                    ],
                    outputs=[
                        analyze_dashboard, analyze_coverage, analyze_cards,
                        analyze_json, analyze_download,
                    ],
                )

            # ── Tab 2: Batch ───────────────────────────────────
            with gr.TabItem("Batch"):
                gr.Markdown(
                    f"**Upload multiple papers** (or a `.zip` of papers) to "
                    f"analyze them in one go. Each paper goes through the same "
                    f"pipeline and you get an aggregate report plus per-paper "
                    f"breakdowns.\n\n"
                    f"**Limits:** up to **{_BATCH_MAX_PAPERS} papers per batch**, "
                    f"**{_HOURLY_IP_BATCH_LIMIT} batches per hour** per IP. "
                    f"All configurable in `.env`."
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        batch_files = gr.File(
                            label="Upload papers (.pdf / .tex / .bib / .txt) or a .zip",
                            file_types=[".pdf", ".tex", ".bib", ".txt", ".zip"],
                            file_count="multiple",
                            type="filepath",
                        )
                    with gr.Column(scale=2):
                        gr.HTML('<div class="cc-section-label">Analysis Options</div>')
                        batch_chk_existence = gr.Checkbox(
                            label="Existence & Metadata (Rule-based)",
                            value=True,
                            info="Fast, no LLM. Checks refs against scholarly databases.",
                        )
                        batch_chk_claims = gr.Checkbox(
                            label="Claim Verification (Agentic)",
                            value=False,
                            info="LLM agents verify claims. Slower, costs OpenAI credits.",
                        )
                        batch_retry = gr.Checkbox(label="Retry failed references", value=False)
                        batch_btn = gr.Button("Analyze batch", variant="primary", size="lg")

                batch_summary = gr.HTML()
                batch_rollup = gr.HTML()
                batch_per_paper = gr.HTML()
                with gr.Row():
                    batch_csv = gr.File(label="Download CSV (one row per verdict)", interactive=False)
                    batch_json = gr.File(label="Download JSON (full batch)", interactive=False)

                batch_btn.click(
                    fn=run_batch,
                    inputs=[batch_files, batch_chk_existence, batch_chk_claims, batch_retry],
                    outputs=[batch_summary, batch_rollup, batch_per_paper, batch_csv, batch_json],
                )

            # ── Tab 3: Parse ───────────────────────────────────
            with gr.TabItem("Parse"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        parse_file_input = gr.File(
                            label="Upload Paper",
                            file_types=[".pdf", ".tex", ".bib", ".txt"],
                            type="filepath",
                        )
                    with gr.Column(scale=2):
                        gr.HTML('''
                        <div style="padding-top:12px;font-size:13px;color:#64748b;line-height:1.6;">
                            Extract references and in-text citations without running
                            verification. Useful for inspecting what the parser sees
                            before a full analysis.
                        </div>
                        ''')
                        parse_btn = gr.Button("Parse", variant="primary")
                parse_output = gr.HTML()

                parse_btn.click(
                    fn=run_parse,
                    inputs=[parse_file_input],
                    outputs=[parse_output],
                )

            # ── Tab 4: About ──────────────────────────────────
            with gr.TabItem("About"):
                gr.Markdown(ABOUT_MD)

    return demo


_ACCESS_TOKEN_COOKIE = "checkcitation_token"
_ACCESS_TOKEN_QUERY = "token"
_ACCESS_TOKEN_COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 days

_GATE_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>CheckCitation — access required</title>
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


def _token_gate_middleware(expected_token: str):
    """FastAPI middleware that requires `?token=<expected>` or a matching cookie.

    Bypasses `/health` so container healthchecks don't need the token.
    On successful query-param match, sets a 30-day cookie so reviewers
    don't have to keep pasting the token.
    """
    from fastapi.responses import HTMLResponse

    async def middleware(request, call_next):
        if request.url.path == "/health":
            return await call_next(request)

        submitted = (
            request.cookies.get(_ACCESS_TOKEN_COOKIE)
            or request.query_params.get(_ACCESS_TOKEN_QUERY)
        )
        if submitted != expected_token:
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
    return gr.mount_gradio_app(api, demo, path="/")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(_build_fastapi_with_health(), host="0.0.0.0", port=port)
