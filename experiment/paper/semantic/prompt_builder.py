"""Prompt loader, user-message builder, and JSON-response parser."""

from __future__ import annotations

import json
from pathlib import Path

CONDITIONS = ("title_only", "title_abstract", "title_abstract_passages")

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "semantic_2class.txt"
_ABSTRACT_MAX_CHARS = 1500


def load_system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _trunc(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n].rstrip() + "…"


def build_user_message(record: dict, condition: str) -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")

    citing = (record.get("citing_sentence") or "").strip()
    title = (record.get("cited_paper_title") or "").strip()
    parts: list[str] = [
        f'Citing sentence: "{citing}"',
        f"Cited paper title: {title}",
    ]

    if condition in ("title_abstract", "title_abstract_passages"):
        ab = _trunc(record.get("cited_paper_abstract") or "", _ABSTRACT_MAX_CHARS)
        parts.append(f"Cited paper abstract: {ab or '(none available)'}")

    if condition == "title_abstract_passages":
        passages = record.get("passages") or []
        if passages:
            # Wrap text in untrusted markers so the model's prompt-injection guard fires.
            wrapped = [
                {
                    "section": (p.get("section") or "Unknown").strip(),
                    "text": f"<<<UNTRUSTED_PASSAGE>>>{(p.get('text') or '').strip()}<<<END_UNTRUSTED>>>",
                }
                for p in passages
            ]
            parts.append(
                "Retrieved passages from the cited paper:\n"
                + json.dumps(wrapped, indent=2, ensure_ascii=False)
            )
        else:
            parts.append("Retrieved passages from the cited paper: (none retrieved)")

    return "\n\n".join(parts)


def parse_response(raw: str) -> tuple[str, str, str, str | None]:
    raw = (raw or "").strip()
    if not raw:
        return ("NOT_SUPPORTED", "", "", "empty_response")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Some models emit ```json ... ``` even in JSON mode.
        s = raw
        if s.startswith("```"):
            s = s.strip("`")
            if s.lstrip().lower().startswith("json"):
                s = s.lstrip()[4:].lstrip()
        try:
            data = json.loads(s)
        except Exception:
            return ("NOT_SUPPORTED", "", "", "parse_failure")
    if not isinstance(data, dict):
        return ("NOT_SUPPORTED", "", "", "parse_failure")

    verdict = str(data.get("classification", "")).upper().strip()
    if verdict not in ("SUPPORTED", "NOT_SUPPORTED"):
        return (
            "NOT_SUPPORTED",
            str(data.get("reasoning", "") or "")[:600],
            str(data.get("evidence_quote", "") or "")[:600],
            "bad_verdict_label",
        )
    return (
        verdict,
        str(data.get("reasoning", "") or "")[:600],
        str(data.get("evidence_quote", "") or "")[:600],
        None,
    )


def map_to_gold(verdict: str) -> str:
    return "VALID" if verdict == "SUPPORTED" else "MISREPRESENTED"
