
from __future__ import annotations

from citeextract.models.report import PaperReport
from citeextract_ui.render.helpers import _esc, _progress_color
from citeextract_ui.theme import RISK_COLORS, VERDICT_STYLES


def format_dashboard(report: PaperReport, selected_mode: str = "", elapsed: float = 0) -> str:
    s = report.summary
    score_pct = int(s.integrity_score * 100)
    risk_color = RISK_COLORS.get(s.risk_level, "#6b7280")
    bar_color = _progress_color(s.integrity_score)

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

    mode_html = f"<b>{_esc(report.mode)}</b>"
    if selected_mode and selected_mode != report.mode:
        mode_html = (
            f"<b>{_esc(report.mode)}</b> "
            f'<span style="font-size:13px;color:#9ca3af;">'
            f"(selected: {_esc(selected_mode)}"
            f"{' &rarr; resolved based on input' if selected_mode == 'auto' else ' &rarr; adjusted based on input'})"
            f"</span>"
        )

    return f'''
    <div class="dashboard">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
            <div>
                <div style="font-size:16px; color:#6b7280;">Integrity Score</div>
                <div style="font-size:38px; font-weight:800; color:{bar_color};">{score_pct}%</div>
            </div>
            <div style="text-align:right;">
                <span style="display:inline-block; background:{risk_color}; color:white; padding:4px 14px; border-radius:20px; font-weight:600; font-size:16px;">
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
