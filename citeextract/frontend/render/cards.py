
from __future__ import annotations

from citeextract_ui.render.helpers import _esc, build_links
from citeextract_ui.render.metadata import format_agentic_metadata_table, format_metadata_table
from citeextract_ui.render.passages import _format_passages_for_ref
from citeextract_ui.theme import CLAIM_VERDICT_STYLES, VERDICT_STYLES


def format_unified_cards(
    report, comp_report, references: list,
    has_verification: bool = True, has_passages: bool = False,
) -> str:
    ref_map = {r.ref_id: r for r in references}

    comp_by_ref: dict[str, list] = {}
    if comp_report and has_passages:
        for cr in comp_report.results:
            comp_by_ref.setdefault(cr.ref_id, []).append(cr)

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
                f'<div style="font-size:15px;color:#6b7280;margin-top:2px;">{authors}{year}</div>'
                if (authors or year) else ""
            )
            cards += f'''
            <details class="card" style="background:#f8fafc; border-left-color:#3b82f6;" open>
                <summary style="color:#1e40af;">
                    [{_esc(ref.ref_id)}] {title}
                </summary>
                <div style="margin-top:8px;">
                    {authors_line}
                    {"<div style='font-size:15px;color:#6b7280;'>" + venue + "</div>" if venue else ""}
                    {passages_html}
                </div>
            </details>'''
        return cards if cards else '<div style="color:#6b7280;text-align:center;padding:40px;">No passages found.</div>'

    return '<div style="color:#6b7280;text-align:center;padding:40px;">No results.</div>'


def _render_verdict_card(v, ref, passages_html: str = "") -> str:
    st = VERDICT_STYLES.get(v.verdict, VERDICT_STYLES["VALID"])
    is_problem = v.verdict in ("FABRICATED",) or v.claim_verdict == "CONTRADICTS"
    open_attr = " open" if is_problem else ""

    if ref and ref.title:
        title = _esc(ref.title)
    elif v.existence and v.existence.matched_title:
        title = f"{_esc(v.existence.matched_title)} <span style=\"font-size:13px;color:#6b7280;font-weight:400;\">(resolved title)</span>"
    else:
        title = "<em>No title</em>"
    authors = _esc("; ".join(ref.authors)) if ref and ref.authors else ""
    year = f" ({ref.year})" if ref and ref.year else ""
    if ref and ref.venue:
        venue = _esc(ref.venue)
    elif v.existence and v.existence.matched_venue:
        venue = _esc(v.existence.matched_venue)
    else:
        venue = ""

    fmt = getattr(ref, 'citation_format', None) if ref else None
    fmt_badge = ""
    if fmt:
        fmt_badge = (
            f'<span style="display:inline-block;background:#f0f4ff;color:#4b5563;'
            f'padding:1px 8px;border-radius:10px;font-size:13px;margin-left:6px;'
            f'border:1px solid #d1d5db;">{_esc(fmt.upper())}</span>'
        )

    links_html = build_links(v)

    source_info = ""
    if v.existence and v.existence.status == "FOUND":
        src = _esc(v.existence.source or "")
        sim = f" &middot; similarity: {v.existence.title_similarity:.0%}" if v.existence.title_similarity else ""
        source_info = f'<div style="font-size:14px;color:#6b7280;margin-top:4px;">Found via: <b>{src}</b>{sim}</div>'
    elif v.existence and v.existence.status == "NOT_FOUND":
        dbs = ", ".join(v.existence.databases_checked) if v.existence.databases_checked else "none"
        source_info = f'<div style="font-size:14px;color:#dc2626;margin-top:4px;">Not found in: {_esc(dbs)}</div>'

    action_html = ""
    if v.action == "remove_citation":
        action_html = '<span style="display:inline-block;background:#fef2f2;color:#dc2626;padding:2px 10px;border-radius:12px;font-size:14px;font-weight:600;margin-top:6px;">Remove Citation</span>'
    elif v.action == "verify_claim":
        action_html = '<span style="display:inline-block;background:#fffbeb;color:#d97706;padding:2px 10px;border-radius:12px;font-size:14px;font-weight:600;margin-top:6px;">Verify Claim</span>'

    meta_table = format_metadata_table(v) or format_agentic_metadata_table(v, ref)
    meta_section = ""
    if meta_table:
        meta_section = f'''
        <details style="margin-top:10px;">
            <summary style="font-size:15px;color:#6b7280;cursor:pointer;">Metadata Comparison</summary>
            {meta_table}
        </details>'''

    ref_id_attr = _esc(v.ref_id)
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

    authors_line = ""
    if authors or year:
        authors_line = (
            f'<div style="font-size:15px;color:#6b7280;margin-top:2px;">'
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
            {"<div style='font-size:15px;color:#6b7280;'>" + venue + fmt_badge + "</div>" if venue else (fmt_badge if fmt_badge else "")}
            {links_html}
            {source_info}
            {action_html}
            {meta_section}
            {passages_html}
        </div>
    </details>'''
