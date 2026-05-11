from __future__ import annotations

import json
from pathlib import Path

CONDITIONS = ("llm_only", "llm_with_search")

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_system_prompt(condition: str) -> str:
    if condition == "llm_only":
        return (_PROMPTS_DIR / "metadata_llm_only.txt").read_text(encoding="utf-8")
    if condition == "llm_with_search":
        return (_PROMPTS_DIR / "metadata_llm_with_search.txt").read_text(encoding="utf-8")
    raise ValueError(f"Unknown condition: {condition}")


def build_user_message(record: dict) -> str:
    title = record.get("title") or "(missing)"
    authors = record.get("authors") or []
    authors_str = ", ".join(authors) if authors else "(missing)"
    year = record.get("year") or "(missing)"
    venue = record.get("venue") or "(missing)"
    doi = record.get("doi") or "(missing)"
    arxiv = record.get("arxiv_id") or "(missing)"
    raw = (record.get("raw_reference_string") or "").strip()
    return (
        "Reference to audit:\n\n"
        f"- Title: {title}\n"
        f"- Authors: {authors_str}\n"
        f"- Year: {year}\n"
        f"- Venue: {venue}\n"
        f"- DOI: {doi}\n"
        f"- arXiv ID: {arxiv}\n\n"
        f'Raw reference string:\n"""\n{raw}\n"""\n'
    )


def parse_response(raw: str) -> tuple[str, str, str, str | None]:
    raw = (raw or "").strip()
    if not raw:
        return ("fabricated", "", "", "empty_response")

    s = raw
    if s.startswith("```"):
        s = s.strip("`")
        if s.lstrip().lower().startswith("json"):
            s = s.lstrip()[4:].lstrip()
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]

    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        if 0 <= start < end:
            try:
                data = json.loads(s[start: end + 1])
            except Exception:
                return ("fabricated", "", "", "parse_failure")
        else:
            return ("fabricated", "", "", "parse_failure")

    if not isinstance(data, dict):
        return ("fabricated", "", "", "parse_failure")

    verdict_raw = str(data.get("classification", "")).strip().lower()
    if verdict_raw not in ("valid", "fabricated"):
        return (
            "fabricated",
            str(data.get("reasoning", "") or "")[:600],
            str(data.get("confidence", "") or "")[:20],
            "bad_verdict_label",
        )
    return (
        verdict_raw,
        str(data.get("reasoning", "") or "")[:600],
        str(data.get("confidence", "") or "")[:20],
        None,
    )
