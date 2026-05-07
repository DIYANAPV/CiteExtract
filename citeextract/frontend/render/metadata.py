
from __future__ import annotations

from citeextract_ui.render.helpers import _esc, _field_status_icon


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


def format_agentic_metadata_table(verdict, ref) -> str:
    ex = verdict.existence
    if not ex or not ref:
        return ""
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
