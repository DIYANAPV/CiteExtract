
from __future__ import annotations

from collections import defaultdict

from citeextract_ui.render.cards import format_unified_cards
from citeextract_ui.render.helpers import _esc


def _format_batch_summary(per_paper: list[dict], elapsed: float, mode: str) -> str:
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
            f'padding:4px 12px;border-radius:14px;font-size:15px;'
            f'font-weight:600;margin:0 6px 6px 0;">'
            f'{count} {label}</span>'
        )

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
        f'<div style="font-size:16px;color:#6b7280;margin-bottom:8px;">'
        f'Batch results · mode: <b>{_esc(mode)}</b> · elapsed: <b>{elapsed:.1f}s</b></div>'
        f'<div style="display:flex;gap:18px;margin-bottom:12px;flex-wrap:wrap;font-size:16px;">'
        f'<div><b>{len(ok)}</b> papers analyzed</div>'
        f'<div><b>{total_refs}</b> total references</div>'
        f'{"<div>" + str(len(failed)) + " failed</div>" if failed else ""}'
        f'</div>'
        f'<div style="margin-top:4px;">{pills}{failed_chip}</div>'
        f'</div>'
    )


def _format_batch_rollup(per_paper: list[dict]) -> str:
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

    header = (
        '<tr style="background:#f9fafb;font-size:14px;color:#6b7280;text-align:left;">'
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
        f'<table style="width:100%;border-collapse:collapse;font-size:15px;">'
        f'{header}{"".join(rows)}</table>'
        '</div>'
    )


def _format_batch_per_paper(per_paper: list[dict], has_passages: bool) -> str:
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
            f'<summary style="font-size:16px;font-weight:600;cursor:pointer;'
            f'padding:8px 12px;background:#f8fafc;border-radius:8px;">'
            f'{_esc(p["name"])} &nbsp;'
            f'<span style="font-weight:400;color:#6b7280;font-size:14px;">'
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
