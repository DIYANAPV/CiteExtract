"""
02_aggregate.py
===============

Aggregate the per-paper JSONL checkpoints from ``results/per_paper/`` into a
single dataset and compute the prevalence numbers ready to drop into the
paper.

Outputs
-------
    results/all_instances.csv          — full long-form table, one row per (paper, ref, citing sentence)
    results/prevalence_summary.json    — headline + stratifications + bootstrap CIs

Headline number
---------------
The prevalence of semantic misrepresentation, computed only over citations
where the cited paper was bibliographically resolved AND its full text was
retrievable. Citations without full text are excluded from the numerator
*and* the denominator (they are reported separately as coverage).

Misrepresentation = LLM verdict in {NOT_SUPPORTED, CONTRADICTS}.
Verdicts of NEUTRAL or null are reported separately and *not* counted as
misrepresentation in the headline rate.

Bootstrap is over citations (not over papers) — 10,000 resamples,
percentile-method 95% CI. We also report the per-paper rate distribution as
a sanity check that no single paper dominates.

Usage
-----
    python 02_aggregate.py
    python 02_aggregate.py --per-paper-dir results/per_paper --out-dir results
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, Optional

import numpy as np

# --- repo path setup ------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_PREV_DIR = _THIS_FILE.parent

# --- constants ------------------------------------------------------------
DEFAULT_PER_PAPER_DIR = _PREV_DIR / "results" / "per_paper"
DEFAULT_OUT_DIR = _PREV_DIR / "results"
DEFAULT_INSTANCES_CSV = DEFAULT_OUT_DIR / "all_instances.csv"
DEFAULT_SUMMARY_JSON = DEFAULT_OUT_DIR / "prevalence_summary.json"

MISREP_VERDICTS = {"NOT_SUPPORTED", "CONTRADICTS"}
SUPPORTED_VERDICTS = {"SUPPORTED", "SUPPORTS"}
NEUTRAL_VERDICTS = {"NEUTRAL"}

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger("aggregate")


@dataclass
class CitationRow:
    paper_id: str
    paper_decision: str
    ref_id: str
    citing_sentence: str
    cited_title: Optional[str]
    cited_doi: Optional[str]
    cited_year: Optional[int]
    cited_source: Optional[str]
    paper_found: bool
    full_text_available: bool
    full_text_source: Optional[str]
    verdict: Optional[str]
    explanation: Optional[str]
    evidence_quote: Optional[str]


def load_per_paper_jsonls(per_paper_dir: Path) -> list[CitationRow]:
    rows: list[CitationRow] = []
    if not per_paper_dir.exists():
        log.error("per-paper dir does not exist: %s", per_paper_dir)
        return rows
    files = sorted(per_paper_dir.glob("*.jsonl"))
    log.info("loading %d per-paper jsonl files", len(files))
    for path in files:
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                rows.append(CitationRow(
                    paper_id=d.get("paper_id", path.stem),
                    paper_decision=d.get("paper_decision", ""),
                    ref_id=d.get("ref_id", ""),
                    citing_sentence=d.get("citing_sentence", ""),
                    cited_title=d.get("cited_title"),
                    cited_doi=d.get("cited_doi"),
                    cited_year=d.get("cited_year"),
                    cited_source=d.get("cited_source"),
                    paper_found=bool(d.get("paper_found", False)),
                    full_text_available=bool(d.get("full_text_available", False)),
                    full_text_source=d.get("full_text_source"),
                    verdict=d.get("verdict"),
                    explanation=d.get("explanation"),
                    evidence_quote=d.get("evidence_quote"),
                ))
    log.info("loaded %d total instances", len(rows))
    return rows


def write_instances_csv(rows: Iterable[CitationRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "paper_id", "paper_decision", "ref_id", "citing_sentence",
        "cited_title", "cited_doi", "cited_year", "cited_source",
        "paper_found", "full_text_available", "full_text_source",
        "verdict", "explanation", "evidence_quote",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: getattr(r, k) for k in fields})
    log.info("wrote %s", path)


def is_in_scope(r: CitationRow) -> bool:
    """The denominator definition: bibliographically valid AND full text retrieved."""
    return r.paper_found and r.full_text_available and r.verdict is not None


def is_misrep(r: CitationRow) -> bool:
    return r.verdict in MISREP_VERDICTS


def bootstrap_ci(values: list[int], resamples: int, seed: int) -> tuple[float, float, float]:
    """Bootstrap percentile 95% CI for a binary indicator's mean.

    Returns (point, lo, hi) as percentages (0–100).
    """
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values, dtype=np.uint8)
    n = arr.shape[0]
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        idx = rng.integers(0, n, n)
        means[i] = arr[idx].mean()
    point = float(arr.mean()) * 100.0
    lo = float(np.percentile(means, 2.5)) * 100.0
    hi = float(np.percentile(means, 97.5)) * 100.0
    return point, lo, hi


def stratify_by(rows: list[CitationRow], key_fn) -> dict[str, dict]:
    buckets: dict[str, list[CitationRow]] = defaultdict(list)
    for r in rows:
        buckets[key_fn(r)].append(r)
    out: dict[str, dict] = {}
    for k, bucket in buckets.items():
        in_scope = [r for r in bucket if is_in_scope(r)]
        misrep = [int(is_misrep(r)) for r in in_scope]
        n_total = len(bucket)
        n_in_scope = len(in_scope)
        n_misrep = sum(misrep)
        if n_in_scope >= 30:
            point, lo, hi = bootstrap_ci(misrep, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
        else:
            point = (100.0 * n_misrep / n_in_scope) if n_in_scope else 0.0
            lo, hi = float("nan"), float("nan")
        out[str(k)] = {
            "n_total": n_total,
            "n_in_scope": n_in_scope,
            "n_misrep": n_misrep,
            "rate_pct": round(point, 2),
            "ci_low_pct": round(lo, 2) if lo == lo else None,
            "ci_high_pct": round(hi, 2) if hi == hi else None,
        }
    return out


def year_band(y: Optional[int]) -> str:
    if y is None:
        return "unknown"
    if y >= 2024:
        return "2024-2025"
    if y >= 2020:
        return "2020-2023"
    if y >= 2015:
        return "2015-2019"
    return "<=2014"


def per_paper_stats(rows: list[CitationRow]) -> dict:
    by_paper: dict[str, list[CitationRow]] = defaultdict(list)
    for r in rows:
        by_paper[r.paper_id].append(r)

    per_paper_rates: list[float] = []
    detail: list[dict] = []
    for pid, bucket in by_paper.items():
        in_scope = [r for r in bucket if is_in_scope(r)]
        n = len(in_scope)
        m = sum(1 for r in in_scope if is_misrep(r))
        if n > 0:
            rate = 100.0 * m / n
            per_paper_rates.append(rate)
        else:
            rate = float("nan")
        detail.append({
            "paper_id": pid,
            "n_total": len(bucket),
            "n_in_scope": n,
            "n_misrep": m,
            "rate_pct": round(rate, 2) if rate == rate else None,
        })

    detail.sort(key=lambda d: (d["rate_pct"] is None, -(d["rate_pct"] or 0)))
    summary = {
        "n_papers_with_in_scope_data": len(per_paper_rates),
        "median_rate_pct": round(median(per_paper_rates), 2) if per_paper_rates else None,
        "min_rate_pct": round(min(per_paper_rates), 2) if per_paper_rates else None,
        "max_rate_pct": round(max(per_paper_rates), 2) if per_paper_rates else None,
        "papers": detail,
    }
    return summary


def compute_summary(rows: list[CitationRow]) -> dict:
    total = len(rows)
    paper_found = [r for r in rows if r.paper_found]
    full_text = [r for r in paper_found if r.full_text_available]
    in_scope = [r for r in full_text if r.verdict is not None]

    misrep_indicator = [int(is_misrep(r)) for r in in_scope]
    point, lo, hi = bootstrap_ci(misrep_indicator, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)

    verdict_counts = Counter((r.verdict or "null") for r in in_scope)
    fts_counts = Counter((r.full_text_source or "none") for r in rows)

    log.info(
        "total=%d | paper_found=%d | full_text=%d | in_scope=%d | misrep=%d (%.2f%% [%.2f, %.2f])",
        total, len(paper_found), len(full_text), len(in_scope),
        sum(misrep_indicator), point, lo, hi,
    )

    return {
        "n_total_citations": total,
        "n_paper_found": len(paper_found),
        "n_full_text": len(full_text),
        "n_in_scope": len(in_scope),
        "coverage_pct": {
            "paper_found": round(100.0 * len(paper_found) / total, 2) if total else 0.0,
            "full_text": round(100.0 * len(full_text) / total, 2) if total else 0.0,
            "in_scope": round(100.0 * len(in_scope) / total, 2) if total else 0.0,
        },
        "headline": {
            "n_misrep": sum(misrep_indicator),
            "rate_pct": round(point, 2),
            "ci_low_pct": round(lo, 2),
            "ci_high_pct": round(hi, 2),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        },
        "verdict_breakdown": dict(verdict_counts),
        "full_text_source_breakdown": dict(fts_counts),
        "stratifications": {
            "by_paper_decision": stratify_by(rows, lambda r: r.paper_decision or "unknown"),
            "by_cited_year_band": stratify_by(rows, lambda r: year_band(r.cited_year)),
            "by_full_text_source": stratify_by(rows, lambda r: r.full_text_source or "none"),
        },
        "per_paper": per_paper_stats(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--per-paper-dir", type=Path, default=DEFAULT_PER_PAPER_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--instances-csv", type=Path, default=DEFAULT_INSTANCES_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_per_paper_jsonls(args.per_paper_dir)
    if not rows:
        log.error("no rows loaded; nothing to aggregate")
        return 1

    write_instances_csv(rows, args.instances_csv)

    summary = compute_summary(rows)
    args.summary_json.write_text(json.dumps(summary, indent=2, default=str))
    log.info("wrote %s", args.summary_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
