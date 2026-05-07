
from __future__ import annotations

from citeextract_ui.render.helpers import _esc, _strip_html
from citeextract_ui.theme import CLAIM_VERDICT_STYLES


def _format_claim_verdict(cv) -> str:
    if not cv:
        return ""
    st = CLAIM_VERDICT_STYLES.get(cv.verdict, CLAIM_VERDICT_STYLES["NEUTRAL"])
    evidence = ""
    if cv.evidence_quote:
        evidence = f'<div style="margin-top:8px;font-size:15px;color:{st["color"]};font-style:italic;opacity:0.85;">"{_esc(cv.evidence_quote)}"</div>'
    return f'''
    <div style="background:{st['bg']};border:1px solid {st['border']};border-left:4px solid {st['color']};
                border-radius:10px;padding:14px 16px;margin:10px 0;">
        <div style="display:flex;align-items:center;gap:8px;">
            <span style="display:inline-block;background:{st['color']};color:white;
                         padding:3px 12px;border-radius:14px;font-size:14px;font-weight:700;letter-spacing:0.03em;">
                {st['label'].upper()}
            </span>
        </div>
        <div style="margin-top:8px;font-size:15px;color:#374151;line-height:1.6;">
            {_esc(cv.explanation)}
        </div>
        {evidence}
    </div>'''


def _format_passages_for_ref(comp_results: list, abstract: str | None = None) -> str:
    if not comp_results:
        return ""
    substantive = [r for r in comp_results if r.citing_sentence and len(r.citing_sentence.split()) >= 5]
    if not substantive:
        return ""

    first = substantive[0]
    ft_icon = "&#x2705;" if first.full_text_available else "&#x1F4C4;"
    ft_label = "Full text" if first.full_text_available else "Abstract only"
    source = _esc(first.full_text_source or "unknown")

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
            passages_inner = '<div style="font-size:15px;color:#6b7280;padding:8px;">No passages retrieved.</div>'

        label = f"Citing context {idx} of {n}" if n > 1 else "Citing context"
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
        <summary style="font-size:15px;color:#6b7280;cursor:pointer;">
            Claim Verification &nbsp;
            <span style="font-size:13px;">Source: {source} &middot; {ft_icon} {ft_label}</span>
        </summary>
        {inner}
    </details>'''
