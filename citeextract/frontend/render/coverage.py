
from __future__ import annotations


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
        <div style="font-size:16px;color:#6b7280;">Coverage</div>
        <div style="font-size:30px;font-weight:800;color:#1f2937;">{report.total_citations} <span style="font-size:16px;font-weight:400;color:#6b7280;">citations analyzed</span></div>
        {bar(ft, "#22c55e", "Full text")}
        {bar(ab, "#f59e0b", "Abstract only")}
        {bar(nf, "#ef4444", "Not found")}
    </div>'''
