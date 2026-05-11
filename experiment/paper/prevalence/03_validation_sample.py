"""
03_validation_sample.py
=======================

Sample N citation instances from ``results/all_instances.csv`` for hand
annotation and write them to ``results/validation_sample.csv`` in a format
ready for two annotators to fill in.

Sampling
--------
Stratified random:
* 50 instances where the system labelled the citation as misrepresenting
  the cited source (NOT_SUPPORTED / CONTRADICTS)
* 50 instances where the system labelled the citation as faithful
  (SUPPORTED / SUPPORTS)

Only in-scope instances are eligible (``paper_found=True`` AND
``full_text_available=True``).

Annotators read the cited source and fill ``annotator_1``, ``annotator_2``,
``final_label``, ``notes``. The validation script then computes precision
and recall of the system relative to the human consensus.

Usage
-----
    python 03_validation_sample.py
    python 03_validation_sample.py --n-misrep 30 --n-supported 30
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from pathlib import Path

# --- repo path setup ------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_PREV_DIR = _THIS_FILE.parent

# --- constants ------------------------------------------------------------
DEFAULT_INSTANCES_CSV = _PREV_DIR / "results" / "all_instances.csv"
DEFAULT_OUT_CSV = _PREV_DIR / "results" / "validation_sample.csv"
DEFAULT_N_MISREP = 50
DEFAULT_N_SUPPORTED = 50
DEFAULT_SEED = 42

MISREP_VERDICTS = {"NOT_SUPPORTED", "CONTRADICTS"}
SUPPORTED_VERDICTS = {"SUPPORTED", "SUPPORTS"}

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger("validation_sample")


def in_scope(row: dict) -> bool:
    return (
        str(row.get("paper_found", "")).lower() == "true"
        and str(row.get("full_text_available", "")).lower() == "true"
        and (row.get("verdict") or "").strip() != ""
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--instances-csv", type=Path, default=DEFAULT_INSTANCES_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--n-misrep", type=int, default=DEFAULT_N_MISREP)
    parser.add_argument("--n-supported", type=int, default=DEFAULT_N_SUPPORTED)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if not args.instances_csv.exists():
        log.error("instances csv not found: %s — run 02_aggregate.py first", args.instances_csv)
        return 1

    rng = random.Random(args.seed)

    misrep_pool: list[dict] = []
    supported_pool: list[dict] = []

    with args.instances_csv.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not in_scope(row):
                continue
            v = (row.get("verdict") or "").strip()
            if v in MISREP_VERDICTS:
                misrep_pool.append(row)
            elif v in SUPPORTED_VERDICTS:
                supported_pool.append(row)

    log.info(
        "in-scope pools — misrep: %d, supported: %d",
        len(misrep_pool), len(supported_pool),
    )

    n_m = min(args.n_misrep, len(misrep_pool))
    n_s = min(args.n_supported, len(supported_pool))
    if n_m < args.n_misrep:
        log.warning("only %d misrep instances available (asked for %d)", n_m, args.n_misrep)
    if n_s < args.n_supported:
        log.warning("only %d supported instances available (asked for %d)", n_s, args.n_supported)

    sampled_misrep = rng.sample(misrep_pool, n_m)
    sampled_supported = rng.sample(supported_pool, n_s)

    sampled = sampled_misrep + sampled_supported
    rng.shuffle(sampled)  # blind annotators to the system label position
    log.info("sampled %d total instances (%d misrep + %d supported)",
             len(sampled), n_m, n_s)

    fields = [
        "paper_id",
        "ref_id",
        "citing_sentence",
        "cited_title",
        "cited_doi",
        "cited_year",
        "evidence_quote",
        "system_explanation",
        "system_verdict",
        "annotator_1",
        "annotator_2",
        "final_label",
        "notes",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sampled:
            writer.writerow({
                "paper_id": row.get("paper_id", ""),
                "ref_id": row.get("ref_id", ""),
                "citing_sentence": row.get("citing_sentence", ""),
                "cited_title": row.get("cited_title", ""),
                "cited_doi": row.get("cited_doi", ""),
                "cited_year": row.get("cited_year", ""),
                "evidence_quote": row.get("evidence_quote", ""),
                "system_explanation": row.get("explanation", ""),
                "system_verdict": row.get("verdict", ""),
                "annotator_1": "",
                "annotator_2": "",
                "final_label": "",
                "notes": "",
            })

    log.info("wrote %s", args.out)
    log.info(
        "annotators fill annotator_1 / annotator_2 with FAITHFUL or MISREP "
        "(or UNCLEAR), then final_label after consensus."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
