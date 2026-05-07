
from __future__ import annotations

import html as _html
import re
from html import escape
from pathlib import Path
from typing import Optional


_HTML_TAG_RE = re.compile(r"<[^>]+>")

_SAFE_STEM_FALLBACK = "paper"
_SAFE_STEM_MAX = 60


def _esc(text: Optional[str], max_len: int = 0) -> str:
    if not text:
        return ""
    s = escape(str(text))
    if max_len and len(s) > max_len:
        s = s[:max_len] + "..."
    return s


def _strip_html(text: Optional[str]) -> str:
    if not text:
        return ""
    cleaned = _HTML_TAG_RE.sub(" ", str(text))
    cleaned = _html.unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _safe_stem(file_path: str) -> str:
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
