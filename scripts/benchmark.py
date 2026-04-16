"""Benchmark CheckCitation against NeurIPS and ICLR hallucinated-citation datasets.

Runs citations through quick (rule-based), full (rule-based + LLM semantic),
and agentic modes, then computes accuracy, precision, recall, and F1.

Both datasets contain ONLY hallucinated citations (ground truth = FABRICATED),
so this measures detection rate (recall) directly.  We pair each hallucinated
citation with a synthetic "known-real" probe to also measure false-positive rate.

Usage:
    python scripts/benchmark.py

Settings — edit the variables below:
"""

# ─── Settings ────────────────────────────────────────────────────────────────
NUM_CITATIONS = None                # Per dataset (set to None for all)
MODES = ["quick", "agentic"]       # Any subset of: "quick", "agentic"
DATASETS = ["neurips", "iclr"]     # Any subset of: "neurips", "iclr"
BATCH_SIZE = 5                     # Process this many refs at a time (avoids rate limits)
BATCH_DELAY = 3.0                  # Seconds to wait between batches
OUTPUT_DIR = "data/output/benchmark"
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import csv
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.reference import Reference
from src.models.parsed_paper import ParsedPaper
from src.models.verdict import ExistenceResult
from src.verification.existence import check_all_references
from src.verification.metadata import validate_metadata
from src.classification.classifier import classify_quick, CitationVerdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Dataset paths ───────────────────────────────────────────────────────────

DATASET_CONFIG = {
    "neurips": {
        "path": "data/neurips_hallucinations.csv",
        "citation_col": "Example of Verified Hallucination",
        "comment_col": "Comment",
        "paper_col": "Published Paper",
    },
    "iclr": {
        "path": "data/iclr2026_hallucinations.csv",
        "citation_col": "Example of Verified Hallucination",
        "comment_col": "Comment",
        "paper_col": "Title",
    },
}

# ─── Known-real citations for measuring false positives ──────────────────────
# These are well-known, easily verifiable papers.

KNOWN_REAL_CITATIONS = [
    Reference(
        ref_id="real_0", raw_text="Vaswani et al. Attention is all you need. NeurIPS 2017.",
        title="Attention is all you need",
        authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit",
                 "Llion Jones", "Aidan N. Gomez", "Lukasz Kaiser", "Illia Polosukhin"],
        year=2017, venue="NeurIPS", source_format="text",
    ),
    Reference(
        ref_id="real_1", raw_text="Devlin et al. BERT: Pre-training of deep bidirectional transformers. NAACL 2019.",
        title="BERT: Pre-training of deep bidirectional transformers for language understanding",
        authors=["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
        year=2019, venue="NAACL", source_format="text",
    ),
    Reference(
        ref_id="real_2", raw_text="He et al. Deep residual learning for image recognition. CVPR 2016.",
        title="Deep residual learning for image recognition",
        authors=["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren", "Jian Sun"],
        year=2016, venue="CVPR", source_format="text",
    ),
    Reference(
        ref_id="real_3", raw_text="Brown et al. Language models are few-shot learners. NeurIPS 2020.",
        title="Language models are few-shot learners",
        authors=["Tom Brown", "Benjamin Mann", "Nick Ryder"],
        year=2020, venue="NeurIPS", source_format="text",
    ),
    Reference(
        ref_id="real_4", raw_text="Radford et al. Learning transferable visual models from natural language supervision. ICML 2021.",
        title="Learning transferable visual models from natural language supervision",
        authors=["Alec Radford", "Jong Wook Kim", "Chris Hallacy", "Aditya Ramesh"],
        year=2021, venue="ICML", source_format="text",
    ),
]


# ─── Citation text parser ────────────────────────────────────────────────────

def parse_citation_text(raw: str, dataset: str, idx: int) -> Reference:
    """Parse a free-text citation string into a Reference object."""
    raw = raw.strip()

    arxiv_id = None
    arxiv_match = re.search(r"arXiv[:\s]*(\d{4}\.\d{4,5}(?:v\d+)?)", raw, re.IGNORECASE)
    if arxiv_match:
        arxiv_id = arxiv_match.group(1)
        if "X" in arxiv_id.upper():
            arxiv_id = None

    doi = None
    doi_match = re.search(r"doi:\s*(10\.\d{4,}/[^\s,]+)", raw, re.IGNORECASE)
    if doi_match:
        doi = doi_match.group(1).rstrip(".")

    year = None
    year_matches = re.findall(r"\b(19\d{2}|20\d{2})\b", raw)
    if year_matches:
        year = int(year_matches[-1])

    title = None
    authors = []

    author_title_split = re.match(r"^(.{5,}?)\.\s+([A-Z].*)", raw, re.DOTALL)
    if author_title_split:
        author_str = author_title_split.group(1).strip()
        rest = author_title_split.group(2).strip()
        authors = _parse_author_string(author_str)
        title = _extract_title(rest)
    else:
        title = _extract_title(raw)

    venue = _extract_venue(raw)

    return Reference(
        ref_id=f"{dataset}_{idx}",
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        doi=doi,
        arxiv_id=arxiv_id,
        raw_text=raw[:500],
        source_format="text",
    )


def _parse_author_string(s: str) -> list[str]:
    s = re.sub(r"\s+and\s+et\s+al\.?", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+et\s+al\.?", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+and\s+others", "", s, flags=re.IGNORECASE)
    parts = re.split(r",\s+and\s+|,\s+|\s+and\s+", s)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 1]


def _extract_title(s: str) -> Optional[str]:
    end_patterns = [
        r"\.\s+arXiv\s+preprint",
        r"\.\s+In\s+(?:Proceedings|Proc\.)",
        r"\.\s+(?:IEEE|ACM|AAAI|NeurIPS|ICML|ICLR|CVPR|ECCV|ICCV|EMNLP|ACL|NAACL)",
        r"\.\s+(?:Journal|Transactions|Conference)",
        r",\s+pages?\s+",
        r",\s+pp\.\s+",
        r",\s+\d{4}\.",
        r"\.\s+URL\s+",
        r"\.\s+doi:",
        r",\s+20\d{2}\b",
        r",\s+19\d{2}\b",
    ]
    best_end = len(s)
    for pattern in end_patterns:
        match = re.search(pattern, s, re.IGNORECASE)
        if match and match.start() < best_end:
            best_end = match.start()
    title = s[:best_end].strip().rstrip(".")
    return title if len(title) > 3 else None


def _extract_venue(raw: str) -> Optional[str]:
    in_match = re.search(
        r"In\s+(Proceedings\s+of\s+.*?|.*?(?:Conference|Workshop|Symposium).*?)"
        r"(?:,\s+(?:pp\.|pages|\d{4})|\.\s|$)",
        raw, re.IGNORECASE,
    )
    if in_match:
        return in_match.group(1).strip().rstrip(",.")
    journal_match = re.search(
        r"((?:IEEE|ACM)\s+.*?|(?:Journal|Transactions)\s+.*?)"
        r"(?:,\s+\d+\(|\s+\d+:|\s*,\s+\d{4})",
        raw, re.IGNORECASE,
    )
    if journal_match:
        return journal_match.group(1).strip().rstrip(",.")
    return None


# ─── Error type classification from comments ─────────────────────────────────

def classify_error_type(comment: str) -> str:
    """Classify the GPTZero comment into an error category.

    Returns one of:
        author_error    — paper exists but authors are wrong
        title_error     — paper exists but title is different
        meta_error      — paper exists but venue/year/DOI wrong
        compound_error  — multiple fields wrong
        non_existent    — paper does not exist at all
    """
    c = comment.lower()

    has_author_issue = any(w in c for w in [
        "author", "authors are wrong", "authors are fabricated",
        "authors are not on the paper",
    ])
    has_title_issue = any(w in c for w in [
        "title is", "different title", "title does not match",
        "title is somewhat different",
    ])
    has_meta_issue = any(w in c for w in [
        "year is wrong", "year is off", "venue", "doi", "page numbers",
        "arxiv id", "conference",
    ])
    has_no_match = any(w in c for w in [
        "no match", "doesn't exist", "does not exist", "no author or title match",
        "no title or author match", "nonexistent",
    ])

    issues = sum([has_author_issue, has_title_issue, has_meta_issue])

    if has_no_match and issues == 0:
        return "non_existent"
    if issues >= 2:
        return "compound_error"
    if has_author_issue:
        return "author_error"
    if has_title_issue:
        return "title_error"
    if has_meta_issue:
        return "meta_error"
    if has_no_match:
        return "non_existent"
    return "non_existent"


# ─── Batched processing ──────────────────────────────────────────────────────

async def _run_in_batches(
    references: list[Reference],
    batch_fn,
    label: str,
) -> list[CitationVerdict]:
    """Run a verification function in batches to avoid API rate limits."""
    all_verdicts: list[CitationVerdict] = []
    total = len(references)

    for start in range(0, total, BATCH_SIZE):
        batch = references[start : start + BATCH_SIZE]
        batch_num = start // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        log.info(f"  [{label}] Batch {batch_num}/{total_batches} "
                 f"(refs {start+1}-{start+len(batch)}/{total})")

        verdicts = await batch_fn(batch)
        all_verdicts.extend(verdicts)

        # Delay between batches (skip after last batch)
        if start + BATCH_SIZE < total:
            log.info(f"  Waiting {BATCH_DELAY}s before next batch...")
            await asyncio.sleep(BATCH_DELAY)

    return all_verdicts


# ─── Verification runners ───────────────────────────────────────────────────

async def _quick_batch(references: list[Reference]) -> list[CitationVerdict]:
    """Run rule-based (quick) verification on a single batch."""
    existence_results = await check_all_references(references)
    exist_map = {r.ref_id: r for r in existence_results}

    verdicts = []
    for ref in references:
        exist = exist_map.get(ref.ref_id)
        if exist is None:
            exist = ExistenceResult(
                ref_id=ref.ref_id, status="NOT_FOUND",
                databases_checked=[], flags=["missing_existence_result"],
            )
        meta = None
        if exist.status == "FOUND":
            meta = validate_metadata(ref, exist)
        verdicts.append(classify_quick(exist, meta))
    return verdicts


async def run_quick(references: list[Reference]) -> list[CitationVerdict]:
    return await _run_in_batches(references, _quick_batch, "quick")


async def _full_batch(references: list[Reference]) -> list[CitationVerdict]:
    """Run full mode on a single batch."""
    from src.parsers.llm_reparser import reparse_references_with_llm
    from src.verification.api_clients.llm_client import create_llm_client
    from src import config as app_config

    llm_config = app_config.llm()

    if llm_config:
        try:
            llm_client = create_llm_client(llm_config)
            count_updated, cost = await reparse_references_with_llm(
                references, llm_client
            )
            if count_updated > 0:
                log.info(f"  LLM re-parsed {count_updated} ref(s), cost: ${cost:.4f}")
        except Exception as e:
            log.warning(f"  LLM re-parsing failed: {e}")

    existence_results = await check_all_references(references)
    exist_map = {r.ref_id: r for r in existence_results}

    verdicts = []
    for ref in references:
        exist = exist_map.get(ref.ref_id)
        if exist is None:
            exist = ExistenceResult(
                ref_id=ref.ref_id, status="NOT_FOUND",
                databases_checked=[], flags=["missing_existence_result"],
            )
        meta = None
        if exist.status == "FOUND":
            meta = validate_metadata(ref, exist)
        verdicts.append(classify_quick(exist, meta))
    return verdicts


async def run_full(references: list[Reference]) -> list[CitationVerdict]:
    return await _run_in_batches(references, _full_batch, "full")


async def _agentic_batch(references: list[Reference]) -> list[CitationVerdict]:
    """Run agentic verification on a single batch."""
    from src.verification.agentic.runner import run_agentic_verification
    from src import config as app_config

    agentic_config = app_config.agentic()
    if not agentic_config:
        raise ValueError("No agentic config in config.yaml")

    parsed = ParsedPaper(
        references=references, citations=[], has_body_text=False,
        input_format="text", metadata={}, warnings=[],
    )
    verdicts, cost = await run_agentic_verification(parsed, agentic_config)
    log.info(f"  Agentic batch cost: ${cost:.4f}")
    return verdicts


async def run_agentic(references: list[Reference]) -> list[CitationVerdict]:
    return await _run_in_batches(references, _agentic_batch, "agentic")


MODE_RUNNERS = {
    "quick": run_quick,
    "full": run_full,
    "agentic": run_agentic,
}


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict], mode: str) -> dict:
    """Compute precision, recall, F1 for a single mode.

    Ground truth is binary: FABRICATED (positive) vs VALID (negative).
    System predictions: FABRICATED/UNVERIFIABLE = flagged (positive),
                        VALID = not flagged (negative).
    """
    tp = fp = fn = tn = 0
    for r in results:
        predicted = r[f"{mode}_verdict"]
        gt = r["ground_truth"]

        predicted_positive = predicted in ("FABRICATED", "UNVERIFIABLE", "MISREPRESENTED")
        actual_positive = gt == "FABRICATED"

        if actual_positive and predicted_positive:
            tp += 1
        elif actual_positive and not predicted_positive:
            fn += 1
        elif not actual_positive and predicted_positive:
            fp += 1
        else:
            tn += 1

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


# ─── Load dataset ────────────────────────────────────────────────────────────

def load_dataset(name: str, limit: Optional[int] = None) -> list[dict]:
    """Load a benchmark CSV dataset. Returns list of row dicts."""
    cfg = DATASET_CONFIG[name]
    csv_path = Path(cfg["path"])
    if not csv_path.exists():
        log.error(f"Dataset not found: {csv_path}")
        return []

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get(cfg["citation_col"], "").strip()
            comment = row.get(cfg["comment_col"], "").strip()
            paper = row.get(cfg["paper_col"], "").strip()
            if raw and comment:
                rows.append({
                    "dataset": name,
                    "raw": raw,
                    "comment": comment,
                    "paper": paper,
                })

    if limit is not None:
        rows = rows[:limit]

    log.info(f"Loaded {len(rows)} citations from {name} ({csv_path.name})")
    return rows


# ─── Main ────────────────────────────────────────────────────────────────────

async def main():
    all_results = []

    # ─── Load and parse datasets ────────────────────────────────────────
    hallucinated_refs: list[Reference] = []
    hallucinated_meta: list[dict] = []

    for ds_name in DATASETS:
        rows = load_dataset(ds_name, NUM_CITATIONS)
        for i, row in enumerate(rows):
            ref = parse_citation_text(row["raw"], ds_name, i)
            hallucinated_refs.append(ref)
            hallucinated_meta.append({
                "dataset": ds_name,
                "paper": row["paper"],
                "raw": row["raw"],
                "comment": row["comment"],
                "error_type": classify_error_type(row["comment"]),
                "ground_truth": "FABRICATED",
            })

    # Add known-real references for false-positive measurement
    real_refs = KNOWN_REAL_CITATIONS[:]
    real_meta = [
        {
            "dataset": "known_real",
            "paper": "synthetic probe",
            "raw": r.raw_text,
            "comment": "known real citation",
            "error_type": "none",
            "ground_truth": "VALID",
        }
        for r in real_refs
    ]

    all_refs = hallucinated_refs + real_refs
    all_meta = hallucinated_meta + real_meta

    log.info(f"\nTotal references: {len(all_refs)} "
             f"({len(hallucinated_refs)} hallucinated + {len(real_refs)} real probes)")

    # ─── Parse summary ──────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  PARSED CITATIONS")
    print("=" * 90)
    for i, (ref, meta) in enumerate(zip(all_refs, all_meta)):
        title_str = (ref.title or "(no title)")[:50]
        print(f"  [{meta['dataset']:>10}] {ref.ref_id:<14} "
              f"gt={meta['ground_truth']:<12} err={meta['error_type']:<16} "
              f"{title_str}")
    print()

    # ─── Run each mode ──────────────────────────────────────────────────
    mode_verdicts: dict[str, list[CitationVerdict]] = {}

    for mode in MODES:
        log.info(f"\n{'='*60}")
        log.info(f"  Running {mode.upper()} mode on {len(all_refs)} references")
        log.info(f"{'='*60}")

        runner = MODE_RUNNERS[mode]
        verdicts = await runner(all_refs)
        mode_verdicts[mode] = verdicts

    # ─── Build results ──────────────────────────────────────────────────
    for i, meta in enumerate(all_meta):
        result = {
            "index": i,
            "dataset": meta["dataset"],
            "source_paper": meta["paper"][:80],
            "citation_raw": meta["raw"][:200],
            "comment": meta["comment"][:200],
            "error_type": meta["error_type"],
            "ground_truth": meta["ground_truth"],
            "parsed_title": all_refs[i].title,
            "parsed_authors": all_refs[i].authors,
            "parsed_year": all_refs[i].year,
        }

        for mode in MODES:
            v = mode_verdicts[mode][i]
            result[f"{mode}_verdict"] = v.verdict
            result[f"{mode}_explanation"] = v.explanation[:300] if v.explanation else ""
            is_correct = (
                (v.verdict != "VALID" and meta["ground_truth"] == "FABRICATED") or
                (v.verdict == "VALID" and meta["ground_truth"] == "VALID")
            )
            result[f"{mode}_correct"] = is_correct

        all_results.append(result)

    # ─── Save output ────────────────────────────────────────────────────
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    datasets_str = "+".join(DATASETS)
    modes_str = "+".join(MODES)
    out_path = out_dir / f"benchmark_{datasets_str}_{modes_str}_{timestamp}.json"

    output = {
        "metadata": {
            "timestamp": timestamp,
            "datasets": DATASETS,
            "modes": MODES,
            "num_citations_per_dataset": NUM_CITATIONS,
            "total_hallucinated": len(hallucinated_refs),
            "total_real_probes": len(real_refs),
        },
        "results": all_results,
        "metrics": {},
    }

    # ─── Compute and display metrics ────────────────────────────────────
    for mode in MODES:
        output["metrics"][mode] = {
            "overall": compute_metrics(all_results, mode),
        }
        # Per-dataset breakdown
        for ds_name in DATASETS + ["known_real"]:
            ds_results = [r for r in all_results if r["dataset"] == ds_name]
            if ds_results:
                output["metrics"][mode][ds_name] = compute_metrics(ds_results, mode)

        # Per-error-type breakdown (hallucinated only)
        error_types = set(r["error_type"] for r in all_results if r["error_type"] != "none")
        for et in sorted(error_types):
            et_results = [r for r in all_results if r["error_type"] == et]
            if et_results:
                output["metrics"][mode][f"error_{et}"] = compute_metrics(et_results, mode)

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"\nResults saved to {out_path}")

    # ─── Print summary table ────────────────────────────────────────────
    mode_header = "  ".join(f"{'':>5}{m.upper():^23}" for m in MODES)
    print("\n" + "=" * (50 + 25 * len(MODES)))
    print(f"  BENCHMARK RESULTS — {len(all_results)} citations "
          f"({len(hallucinated_refs)} hallucinated + {len(real_refs)} real)")
    print("=" * (50 + 25 * len(MODES)))

    # Per-citation results
    sub_header = "  ".join(f"{'Verdict':<13} {'':>4}" for _ in MODES)
    print(f"\n{'#':<4} {'Dataset':<12} {'GT':<12} {'Error Type':<16} {sub_header} {'Title'}")
    print("-" * (50 + 25 * len(MODES)))

    for r in all_results:
        cols = []
        for mode in MODES:
            mark = "OK" if r[f"{mode}_correct"] else "MISS"
            cols.append(f"{r[f'{mode}_verdict']:<13} {mark:<4}")
        mode_str = "  ".join(cols)
        title = (r["parsed_title"] or "(no title)")[:30]
        print(f"{r['index']:<4} {r['dataset']:<12} {r['ground_truth']:<12} "
              f"{r['error_type']:<16} {mode_str} {title}")

    # Metrics summary
    print("\n" + "-" * (50 + 25 * len(MODES)))
    print("\n  METRICS SUMMARY")
    print("-" * 80)

    metric_header = "".join(f"{m.upper():>15}" for m in MODES)
    print(f"  {'Category':<25}{metric_header}")
    print("  " + "-" * (25 + 15 * len(MODES)))

    categories = ["overall"] + DATASETS + ["known_real"]
    error_types = sorted(set(r["error_type"] for r in all_results if r["error_type"] != "none"))
    categories += [f"error_{et}" for et in error_types]

    for cat in categories:
        vals = []
        for mode in MODES:
            m = output["metrics"][mode].get(cat)
            if m:
                vals.append(f"F1={m['f1']:.3f}")
            else:
                vals.append("—")
        val_str = "".join(f"{v:>15}" for v in vals)
        print(f"  {cat:<25}{val_str}")

    # Detailed metrics per mode
    for mode in MODES:
        m = output["metrics"][mode]["overall"]
        print(f"\n  {mode.upper()} — Acc={m['accuracy']:.3f}  "
              f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  "
              f"(TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']})")

    print(f"\n  Detailed results: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
