"""
04_misrep_review.py
===================

Export every MISREP-flagged case from the per-paper JSONLs as a
human-readable Markdown file, one section per case. Lets you skim the
system's verdicts, read the LLM reasoning, and decide whether each case
is genuinely a misrepresentation or a system error.

Filters
-------
* verdict in {NOT_SUPPORTED, CONTRADICTS}
* paper_found = True
* full_text_available = True

Output
------
    results/misrep_review.md   — Markdown, viewable in any editor / GitHub
    results/misrep_review.csv  — Same content as a spreadsheet for offline note-taking

Usage
-----
    python 04_misrep_review.py
    python 04_misrep_review.py --include-neutral    # also include NEUTRAL verdicts
    python 04_misrep_review.py --paper-id <id>      # only one paper
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

# --- repo path setup ------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_PREV_DIR = _THIS_FILE.parent

DEFAULT_PER_PAPER_DIR = _PREV_DIR / "results" / "per_paper"
DEFAULT_MD_PATH = _PREV_DIR / "results" / "misrep_review.md"
DEFAULT_CSV_PATH = _PREV_DIR / "results" / "misrep_review.csv"

MISREP_VERDICTS = {"NOT_SUPPORTED", "CONTRADICTS"}
NEUTRAL_VERDICTS = {"NEUTRAL"}

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger("misrep_review")


def _md_escape(s: str) -> str:
    """Escape characters that would break Markdown rendering of a quoted block."""
    if s is None:
        return ""
    return s.replace("\r", "").strip()


def _md_quote(s: str) -> str:
    """Format multi-line text as a Markdown blockquote."""
    s = _md_escape(s)
    if not s:
        return "_(empty)_"
    lines = s.split("\n")
    return "\n".join("> " + l for l in lines)


def _doi_link(doi: str | None) -> str:
    if not doi:
        return ""
    return f"[{doi}](https://doi.org/{doi})"


def case_to_markdown(idx: int, total: int, r: dict) -> str:
    paper_id = r.get("paper_id", "?")
    ref_id = r.get("ref_id", "?")
    decision = r.get("paper_decision", "")
    cited_title = r.get("cited_title") or "_(no title)_"
    cited_doi = r.get("cited_doi") or ""
    cited_year = r.get("cited_year")
    cited_source = r.get("cited_source") or "?"
    full_text_source = r.get("full_text_source") or "?"
    verdict = r.get("verdict") or "?"
    citing = r.get("citing_sentence") or ""
    ctx_before = r.get("context_before") or ""
    ctx_after = r.get("context_after") or ""
    explanation = r.get("explanation") or ""
    evidence = r.get("evidence_quote") or ""
    top_passage = r.get("top_passage_text") or ""

    parts = [
        f"## Case {idx}/{total} — `{paper_id}` / `{ref_id}` — **{verdict}**",
        "",
        f"- **Citing paper** [`{paper_id}`](https://openreview.net/forum?id={paper_id}) ({decision})",
        f"- **Cited:** {cited_title}",
    ]
    if cited_doi:
        parts.append(f"- **DOI:** {_doi_link(cited_doi)}")
    if cited_year:
        parts.append(f"- **Year:** {cited_year}")
    parts.append(f"- **Resolved by:** {cited_source} · **Full text from:** {full_text_source}")
    parts.append("")

    parts.append("### Citing sentence")
    parts.append(_md_quote(citing))
    if ctx_before or ctx_after:
        parts.append("")
        parts.append("### Surrounding context")
        if ctx_before:
            parts.append("**before:**  ")
            parts.append(_md_quote(ctx_before))
        if ctx_after:
            parts.append("**after:**  ")
            parts.append(_md_quote(ctx_after))

    parts.append("")
    parts.append("### System verdict & evidence")
    parts.append(f"**Verdict:** `{verdict}`")
    parts.append("")
    parts.append("**Explanation (LLM reasoning):**")
    parts.append(_md_quote(explanation))
    parts.append("")
    parts.append("**Evidence quote (passage the model used):**")
    parts.append(_md_quote(evidence))
    if top_passage and top_passage != evidence:
        parts.append("")
        parts.append("**Top retrieved passage (raw):**")
        parts.append(_md_quote(top_passage))

    parts.append("")
    parts.append("### Human review")
    parts.append("- [ ] Verdict correct (MISREP confirmed)")
    parts.append("- [ ] False alarm (citation is faithful)")
    parts.append("- [ ] Unclear (need to read full source)")
    parts.append("")
    parts.append("**Notes:** _(your reasoning here)_")
    parts.append("")
    parts.append("---")
    parts.append("")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--per-paper-dir", type=Path, default=DEFAULT_PER_PAPER_DIR)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--paper-id", help="filter to a single paper")
    parser.add_argument("--include-neutral", action="store_true",
                        help="include NEUTRAL verdicts as well as MISREP")
    args = parser.parse_args()

    if not args.per_paper_dir.exists():
        log.error("per-paper dir does not exist: %s", args.per_paper_dir)
        return 1

    target_verdicts = set(MISREP_VERDICTS)
    if args.include_neutral:
        target_verdicts |= NEUTRAL_VERDICTS

    files = sorted(args.per_paper_dir.glob("*.jsonl"))
    if args.paper_id:
        files = [f for f in files if f.stem == args.paper_id]
    if not files:
        log.error("no per-paper jsonl files found")
        return 1

    cases: list[dict] = []
    for path in files:
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if not r.get("paper_found"):
                    continue
                if not r.get("full_text_available"):
                    continue
                if r.get("verdict") not in target_verdicts:
                    continue
                cases.append(r)
    log.info("found %d cases across %d papers", len(cases), len(files))

    if not cases:
        log.warning("no cases matched the filter")
        return 0

    # --- markdown export ---
    args.md.parent.mkdir(parents=True, exist_ok=True)
    md_parts: list[str] = []
    md_parts.append(f"# Misrep review — {len(cases)} cases\n")
    md_parts.append(
        f"Filter: verdict in {sorted(target_verdicts)} AND paper_found AND full_text_available.\n"
    )
    md_parts.append(
        "Each case below shows the citing sentence, the cited paper, the system's "
        "verdict + reasoning, and the passage it grounded the verdict on. "
        "Tick the appropriate box at the bottom of each case after reading the cited source.\n"
    )
    md_parts.append("---\n")
    for i, r in enumerate(cases, start=1):
        md_parts.append(case_to_markdown(i, len(cases), r))
    args.md.write_text("\n".join(md_parts), encoding="utf-8")
    log.info("wrote %s", args.md)

    # --- csv export ---
    fields = [
        "paper_id", "ref_id", "verdict", "cited_title", "cited_doi", "cited_year",
        "citing_sentence", "explanation", "evidence_quote",
        "human_verdict", "human_notes",
    ]
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in cases:
            writer.writerow({
                "paper_id": r.get("paper_id", ""),
                "ref_id": r.get("ref_id", ""),
                "verdict": r.get("verdict", ""),
                "cited_title": r.get("cited_title", ""),
                "cited_doi": r.get("cited_doi", ""),
                "cited_year": r.get("cited_year", ""),
                "citing_sentence": r.get("citing_sentence", ""),
                "explanation": r.get("explanation", ""),
                "evidence_quote": r.get("evidence_quote", ""),
                "human_verdict": "",
                "human_notes": "",
            })
    log.info("wrote %s", args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
