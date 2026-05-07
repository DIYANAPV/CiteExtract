
from __future__ import annotations

import re


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

_STAGE_ALIASES: dict[str, tuple[str, ...]] = {
    "agentic_both_dispatch": ("agentic_metadata_dispatch", "agentic_claim_dispatch"),
    "L4_comprehension": (
        "agentic_pre_retrieve",
        "agentic_metadata_dispatch",
        "agentic_claim_dispatch",
    ),
}

_FINALIZE_STAGE_ID = "__finalize__"

_STAGE_RE = re.compile(r"STAGE (\S+) seconds=")
_STAGE_PROGRESS_RE = re.compile(
    r"STAGE (\S+)\.progress\s+seconds=\S+\s+done=(\d+)\s+total=(\d+)"
)


def _applicable_stages(existence: bool, claims: bool) -> list[tuple[str, str]]:
    stages = [(sid, label) for sid, label, app in PIPELINE_STAGES_DEF
              if app(existence, claims)]
    stages.append((_FINALIZE_STAGE_ID, "Build report"))
    return stages


def _render_pipeline(applicable: list[tuple[str, str]],
                     seen: set[str],
                     elapsed: float | None = None,
                     progress: dict[str, tuple[int, int]] | None = None) -> str:
    progress = progress or {}
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
        if cls == "active" and stage_id in progress:
            done_n, total_n = progress[stage_id]
            label_html = (
                f'{label} '
                f'<span class="cc-step-progress">({done_n} / {total_n})</span>'
            )
        else:
            label_html = label
        items.append(
            f'<li class="cc-step cc-step-{cls}">'
            f'<span class="cc-step-icon">{icon}</span>'
            f'<span class="cc-step-label">{label_html}</span>'
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
