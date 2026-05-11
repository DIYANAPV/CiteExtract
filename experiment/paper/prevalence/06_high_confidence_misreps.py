"""
06_high_confidence_misreps.py
=============================

Filter misrep cases down to the *high-confidence* set — only those where:

  1. Both LLMs agree (primary=CONTRADICTS/NOT_SUPPORTED AND
     second-opinion=NOT_SUPPORTED/PARTIAL), and
  2. The cited paper resolved by our metadata cascade actually matches the
     bibliography entry [N] in the citing PDF (overlap ≥ 0.4 on long words
     between our `cited_title` and the PDF bib line).

These are the cases we are confident calling "real" misrepresentations. The
output is a single Markdown file + companion CSV with full context for each
case: citing paper, cited paper, citing sentence + surrounding text from the
actual PDF, the PDF's bibliography entry, both LLMs' verdicts and reasoning.

Usage
-----
    python 06_high_confidence_misreps.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path
from typing import Optional

import fitz  # pymupdf

# --- repo path setup ------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_PREV_DIR = _THIS_FILE.parent

DEFAULT_PER_PAPER_DIR = _PREV_DIR / "results" / "per_paper"
DEFAULT_SO_DIR = _PREV_DIR / "results" / "secondopinion"
DEFAULT_PAPERS_DIR = _PREV_DIR / "data" / "papers"
DEFAULT_MD = _PREV_DIR / "results" / "high_confidence_misreps.md"
DEFAULT_CSV = _PREV_DIR / "results" / "high_confidence_misreps.csv"

ALIGNMENT_THRESHOLD = 0.4
MISREP_PRIMARY = {"CONTRADICTS", "NOT_SUPPORTED"}
MISREP_SO = {"NOT_SUPPORTED", "PARTIAL"}

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO, datefmt="%H:%M:%S",
)
log = logging.getLogger("hc_misreps")


def _flat_pdf(pdf_path: Path) -> str:
    if not pdf_path.exists():
        return ""
    doc = fitz.open(str(pdf_path))
    full = "\n".join(p.get_text() for p in doc)
    doc.close()
    return re.sub(r"\s+", " ", full)


def _find_bib_entry(flat: str, ref_id: str) -> Optional[str]:
    """Pull the bibliography line for [ref_id] from the PDF text."""
    ref_start = max(flat.rfind("References "), flat.rfind("REFERENCES "))
    if ref_start == -1:
        ref_start = int(len(flat) * 0.7)
    bib = flat[ref_start:]
    for pat in (
        rf"\[{ref_id}\]\s*([A-Z][^\[]{{15,400}})(?=\s*\[\d|$)",
        rf"(?:^|\.\s|\s){ref_id}\.\s+([A-Z][^\d\n]{{15,400}})",
    ):
        m = re.search(pat, bib)
        if m:
            return m.group(1).strip()[:300]
    return None


def _normalize(s: Optional[str]) -> str:
    if not s:
        return ""
    s = re.sub(r"[^\w\s]", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _title_overlap(a: Optional[str], b: Optional[str]) -> float:
    aw = {w for w in _normalize(a).split() if len(w) > 3}
    bw = {w for w in _normalize(b).split() if len(w) > 3}
    if not aw or not bw:
        return 0.0
    smaller, larger = (aw, bw) if len(aw) <= len(bw) else (bw, aw)
    return len(smaller & larger) / len(smaller)


def _find_citing_window(flat: str, citing_sentence: str, around: int = 350) -> str:
    """Pull a window of citing-paper text around the citing sentence."""
    if not citing_sentence:
        return ""
    needle = citing_sentence[:60].strip()
    idx = flat.find(needle)
    if idx == -1:
        return ""
    return flat[max(0, idx - around): idx + len(citing_sentence) + around]


def load_misrep_instances(per_paper_dir: Path) -> dict[tuple, list[dict]]:
    """Map (paper_id, ref_id) -> list of misrep citation rows."""
    out: dict[tuple, list[dict]] = {}
    for path in sorted(per_paper_dir.glob("*.jsonl")):
        for line in path.open("r"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("verdict") not in MISREP_PRIMARY:
                continue
            if not r.get("paper_found") or not r.get("full_text_available"):
                continue
            out.setdefault((r["paper_id"], r["ref_id"]), []).append(r)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-paper-dir", type=Path, default=DEFAULT_PER_PAPER_DIR)
    parser.add_argument("--so-dir", type=Path, default=DEFAULT_SO_DIR)
    parser.add_argument("--papers-dir", type=Path, default=DEFAULT_PAPERS_DIR)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--threshold", type=float, default=ALIGNMENT_THRESHOLD)
    args = parser.parse_args()

    instances = load_misrep_instances(args.per_paper_dir)
    log.info("loaded %d unique (paper, ref) pairs flagged as misrep", len(instances))

    so_files = sorted(args.so_dir.glob("*.json"))
    log.info("loaded %d second-opinion checkpoints", len(so_files))

    pdf_cache: dict[str, str] = {}

    high_conf: list[dict] = []
    skipped_disagree: list[dict] = []
    skipped_alignment: list[dict] = []

    for sp in so_files:
        so = json.loads(sp.read_text())
        paper_id = so["paper_id"]
        ref_id = so["ref_id"]
        primary_verdict = so.get("primary_verdict") or ""
        so_verdict = (so.get("secondopinion_verdict") or "").upper()
        agrees = primary_verdict in MISREP_PRIMARY and so_verdict in MISREP_SO

        if not agrees:
            skipped_disagree.append({"paper_id": paper_id, "ref_id": ref_id, "so_verdict": so_verdict})
            continue

        # Alignment check: pull PDF bibliography line and compare to our cited_title
        if paper_id not in pdf_cache:
            pdf_cache[paper_id] = _flat_pdf(args.papers_dir / f"{paper_id}.pdf")
        flat = pdf_cache[paper_id]
        bib_line = _find_bib_entry(flat, ref_id)
        cited_title = so.get("cited_title")
        overlap = _title_overlap(bib_line, cited_title)

        if overlap < args.threshold:
            skipped_alignment.append({
                "paper_id": paper_id, "ref_id": ref_id,
                "cited_title": cited_title, "bib_line": bib_line, "overlap": overlap,
            })
            continue

        # Pull all instances (one ref can be cited multiple times — pick the first as canonical
        # but keep a list of citing sentences for reference)
        ref_instances = instances.get((paper_id, ref_id), [])
        citing_sentences = [r["citing_sentence"] for r in ref_instances]
        canonical = ref_instances[0] if ref_instances else None
        citing_window = _find_citing_window(flat, so["citing_sentence"], around=350) if flat else ""

        record = {
            "paper_id": paper_id,
            "ref_id": ref_id,
            "openreview_url": f"https://openreview.net/forum?id={paper_id}",
            "cited_title": cited_title,
            "cited_doi": so.get("cited_doi"),
            "cited_year": canonical.get("cited_year") if canonical else None,
            "pdf_bib_entry": bib_line,
            "alignment_overlap": round(overlap, 2),
            "primary_verdict": primary_verdict,
            "primary_explanation": so.get("primary_explanation", ""),
            "secondopinion_verdict": so_verdict,
            "secondopinion_confidence": so.get("secondopinion_confidence", ""),
            "secondopinion_reasoning": so.get("secondopinion_reasoning", ""),
            "evidence_quote": canonical.get("evidence_quote", "") if canonical else "",
            "citing_sentence": so["citing_sentence"],
            "citing_window": citing_window,
            "n_citation_instances": len(ref_instances),
            "all_citing_sentences": citing_sentences,
        }
        high_conf.append(record)

    log.info(
        "high-confidence misreps: %d  |  agreement-but-bad-alignment: %d  |  no-agreement: %d",
        len(high_conf), len(skipped_alignment), len(skipped_disagree),
    )

    # --- Markdown report ---
    args.md.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# High-confidence citation misrepresentations\n")
    lines.append(
        f"**{len(high_conf)} cases** where two independent LLMs both flagged the citation "
        f"as misrepresenting its source AND the cited paper resolved by our metadata cascade "
        f"matches the bibliography entry in the citing PDF (alignment overlap ≥ {args.threshold}).\n"
    )
    lines.append(
        f"Drawn from a random sample of **45 NeurIPS 2025 accepted papers**.\n"
    )
    lines.append("Filters applied:")
    lines.append(f"- Primary system flagged as `CONTRADICTS` or `NOT_SUPPORTED`.")
    lines.append(f"- Independent gpt-4o second opinion (with full citing context + full cited paper) flagged as `NOT_SUPPORTED` or `PARTIAL`.")
    lines.append(f"- PDF bibliography line for ref `[N]` matches our resolved cited paper.")
    lines.append("")
    lines.append("Triage stats: "
                 f"{len(high_conf)} high-confidence | "
                 f"{len(skipped_alignment)} agreement-but-misaligned (different cited paper resolved) | "
                 f"{len(skipped_disagree)} second-opinion disagreed (need human review)")
    lines.append("\n---\n")

    for i, r in enumerate(high_conf, 1):
        lines.append(f"## Case {i}/{len(high_conf)} — `{r['paper_id']}` / ref `[{r['ref_id']}]`\n")
        lines.append(f"**Citing paper (NeurIPS 2025):** [{r['paper_id']}]({r['openreview_url']})")
        lines.append(f"**Cited paper:** {r['cited_title'] or '_(no title)_'}")
        if r['cited_doi']:
            lines.append(f"  **DOI:** [{r['cited_doi']}](https://doi.org/{r['cited_doi']})")
        if r['cited_year']:
            lines.append(f"  **Year:** {r['cited_year']}")
        lines.append("")
        lines.append("**Bibliography entry [N] from the citing PDF (sanity-check that we resolved the right paper):**")
        lines.append("> " + (r['pdf_bib_entry'] or '_(could not locate)_').replace("\n", "\n> "))
        lines.append(f"  _alignment overlap with our resolved title: {r['alignment_overlap']}_")
        lines.append("")
        lines.append("### Citing sentence")
        lines.append("> " + r['citing_sentence'].strip().replace("\n", "\n> "))
        if r['n_citation_instances'] > 1:
            lines.append(f"")
            lines.append(f"_(This reference is cited in {r['n_citation_instances']} sentences; the verdict above is on this sentence; we list all below.)_")
            for j, s in enumerate(r['all_citing_sentences'], 1):
                if s != r['citing_sentence']:
                    lines.append(f"  - alt sentence {j}: {s.strip()[:200]}…")
        lines.append("")
        if r['citing_window']:
            lines.append("### Surrounding context (from the citing PDF)")
            lines.append("> " + r['citing_window'].strip().replace("\n", "\n> "))
            lines.append("")
        lines.append("### Primary system verdict (gpt-4o-mini, retrieved-passage RAG)")
        lines.append(f"**`{r['primary_verdict']}`**")
        lines.append("")
        lines.append(r['primary_explanation'].strip())
        lines.append("")
        if r['evidence_quote']:
            lines.append("**Passage retrieved from the cited paper used as evidence:**")
            lines.append("> " + r['evidence_quote'].strip().replace("\n", "\n> "))
            lines.append("")
        lines.append("### Second-opinion verdict (gpt-4o, full cited paper + citing context)")
        lines.append(f"**`{r['secondopinion_verdict']}`** ({r['secondopinion_confidence']} confidence)")
        lines.append("")
        lines.append(r['secondopinion_reasoning'].strip())
        lines.append("")
        lines.append("---\n")

    args.md.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s", args.md)

    # --- CSV companion ---
    if high_conf:
        fields = list(high_conf[0].keys())
        # all_citing_sentences is a list — flatten to "; "-joined string
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in high_conf:
                row = dict(r)
                row["all_citing_sentences"] = "; ".join(row["all_citing_sentences"])
                writer.writerow(row)
        log.info("wrote %s", args.csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
