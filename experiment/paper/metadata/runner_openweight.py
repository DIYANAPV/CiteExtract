"""Open-weight (HuggingFace) metadata runner.

Counterpart to ``runner.py`` (which targets the OpenAI API). Uses the shared
``TransformersClient`` from ``experiment/paper/_shared/openweight_client.py`` to
execute greedy decode locally on a GPU box, and writes results to the same
metadata results layout (``results/metadata/csv/``, ``results/metadata/partial/``,
``results/metadata/summaries/``) so ``make_table.py`` and ``analysis/significance.py``
pick them up uniformly.

Usage
-----
    python -m experiment.paper.metadata.runner_openweight --model Qwen/Qwen3-8B --full
    python -m experiment.paper.metadata.runner_openweight --model meta-llama/Llama-3.1-8B-Instruct --smoke
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PAPER_ROOT = _THIS_FILE.parent.parent
_REPO_ROOT = _PAPER_ROOT.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.paper._shared.openweight_client import (  # noqa: E402
    PARSE_RETRY_SUFFIX,
    TransformersClient,
    append_partial,
    is_qwen3,
    load_completed_ids,
    parse_two_class,
    read_partial_as_rows,
    safe_model_name,
    stratified_sample,
    update_summary,
    write_csv,
)

PROMPTS_DIR = _PAPER_ROOT / "prompts"
DATA_PATH = _PAPER_ROOT / "data" / "benchmark_metadata.jsonl"
RESULTS_DIR = _PAPER_ROOT / "results" / "metadata"
CSV_DIR = RESULTS_DIR / "csv"
PARTIAL_DIR = RESULTS_DIR / "partial"
SUMMARIES_DIR = RESULTS_DIR / "summaries"

CSV_COLUMNS = [
    "instance_id", "source", "gold_label", "predicted_verdict", "raw_verdict",
    "explanation", "confidence", "raw_response",
    "prompt_tokens", "completion_tokens", "cost_usd", "latency_seconds",
    "n_search_invocations", "search_queries",
    "triage_route", "metadata_agent_called", "db_source_matched",
    "error",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("metadata_openweight")


@dataclass
class MetadataRow:
    instance_id: int
    source: str
    gold_label: str
    predicted_verdict: str
    raw_verdict: str
    explanation: str
    confidence: str
    raw_response: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_seconds: float
    n_search_invocations: int = 0
    search_queries: str = ""
    triage_route: str = ""
    metadata_agent_called: bool = False
    db_source_matched: str = ""
    error: str | None = None


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def _build_user_message(record: dict, *, qwen3: bool) -> str:
    title = record.get("title") or "(missing)"
    authors = record.get("authors") or []
    authors_str = ", ".join(authors) if authors else "(missing)"
    parts: list[str] = []
    if qwen3:
        parts.append("/no_think")
    parts.append(
        "Reference to audit:\n\n"
        f"- Title: {title}\n"
        f"- Authors: {authors_str}\n"
        f"- Year: {record.get('year') or '(missing)'}\n"
        f"- Venue: {record.get('venue') or '(missing)'}\n"
        f"- DOI: {record.get('doi') or '(missing)'}\n"
        f"- arXiv ID: {record.get('arxiv_id') or '(missing)'}\n\n"
        f'Raw reference string:\n"""\n{(record.get("raw_reference_string") or "").strip()}\n"""\n'
    )
    return "\n\n".join(parts)


def _parse(raw: str) -> tuple[str, str, str, str, str | None]:
    verdict, reason, conf, err = parse_two_class(
        raw, ("valid", "fabricated"), evidence_key="confidence",
    )
    if err is not None and verdict == "":
        return "fabricated", "", "", "", err
    return verdict, verdict, reason, conf, err


def _csv_path(model: str) -> Path:
    return CSV_DIR / f"metadata_{safe_model_name(model)}_llm_only.csv"


def _partial_path(model: str) -> Path:
    return PARTIAL_DIR / f"metadata_partial_{safe_model_name(model)}_llm_only.jsonl"


def load_records() -> list[dict]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)
    return [json.loads(l) for l in open(DATA_PATH, encoding="utf-8")]


async def run_cell(client: TransformersClient, records: list[dict], *, resume: bool) -> list[MetadataRow]:
    csv_path = _csv_path(client.model_id)
    partial_path = _partial_path(client.model_id)
    completed = load_completed_ids(partial_path) if resume else set()
    todo = [r for r in records if r["instance_id"] not in completed]
    if not todo:
        log.info(f"[{client.model_id} | metadata | llm_only] all done; regenerating CSV from partial")
        rows = read_partial_as_rows(partial_path,
                                    {r["instance_id"] for r in records}, MetadataRow,
                                    metadata_defaults=True)
        write_csv(csv_path, rows, CSV_COLUMNS)
        return rows

    if completed:
        log.info(f"[{client.model_id} | metadata | llm_only] resume: {len(completed)} in partial")

    system = _load_prompt("metadata_llm_only.txt")
    qwen3 = is_qwen3(client.model_id)
    t0 = time.perf_counter()

    for i, rec in enumerate(todo, start=1):
        user = _build_user_message(rec, qwen3=qwen3)
        res = await client.call(system, user, max_tokens=512)
        verdict, raw_v, reason, conf, perr = _parse(res.response_text)
        if perr == "parse_failure" and res.error is None:
            retry = await client.call(system, user + PARSE_RETRY_SUFFIX, max_tokens=512)
            if retry.error is None:
                v2, rv2, r2, c2, err2 = _parse(retry.response_text)
                if err2 is None:
                    verdict, raw_v, reason, conf, perr = v2, rv2, r2, c2, None
            tot_in = res.prompt_tokens + retry.prompt_tokens
            tot_out = res.completion_tokens + retry.completion_tokens
            tot_lat = res.latency_seconds + retry.latency_seconds
            raw_resp = retry.response_text if perr is None else res.response_text
        else:
            tot_in = res.prompt_tokens
            tot_out = res.completion_tokens
            tot_lat = res.latency_seconds
            raw_resp = res.response_text
        row = MetadataRow(
            instance_id=rec["instance_id"], source=rec["source"],
            gold_label=rec["gold_label"],
            predicted_verdict=verdict, raw_verdict=raw_v,
            explanation=reason, confidence=conf,
            raw_response=(raw_resp or "")[:4000],
            prompt_tokens=tot_in, completion_tokens=tot_out,
            cost_usd=0.0,
            latency_seconds=round(tot_lat, 3),
            error=res.error or perr,
        )
        append_partial(partial_path, row)
        if i % 5 == 0 or i == len(todo):
            el = time.perf_counter() - t0
            rate = i / el if el else 0
            log.info(f"[{client.model_id} | metadata | llm_only] {i}/{len(todo)} ({rate:.2f}/s)")

    rows = read_partial_as_rows(partial_path,
                                {r["instance_id"] for r in records}, MetadataRow,
                                metadata_defaults=True)
    write_csv(csv_path, rows, CSV_COLUMNS)
    log.info(f"[{client.model_id} | metadata | llm_only] cell done in "
             f"{time.perf_counter() - t0:.1f}s; CSV: {csv_path}")
    return rows


def cell_summary(rows: list[MetadataRow]) -> dict:
    if not rows:
        return {"n": 0}
    correct = sum(1 for r in rows if r.predicted_verdict == r.gold_label)
    tp = sum(1 for r in rows if r.gold_label == "fabricated"
             and r.predicted_verdict == "fabricated")
    fp = sum(1 for r in rows if r.gold_label == "valid"
             and r.predicted_verdict == "fabricated")
    fn = sum(1 for r in rows if r.gold_label == "fabricated"
             and r.predicted_verdict == "valid")
    p = tp / (tp + fp) if tp + fp else 0.0
    r_ = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r_ / (p + r_) if p + r_ else 0.0
    err = sum(1 for r in rows if r.error)
    lat = sum(r.latency_seconds for r in rows)
    return {"n": len(rows), "accuracy": correct / len(rows),
            "fab_precision": p, "fab_recall": r_, "fab_f1": f1,
            "errors": err, "total_latency_seconds": round(lat, 1)}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True,
                    help="HuggingFace model id, e.g. Qwen/Qwen3-8B")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true",
                   help="10 instances, deterministic stratified sample")
    g.add_argument("--full", action="store_true", help="all instances")
    g.add_argument("--instances", type=int, help="ad-hoc cap (first N by id)")
    ap.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp16"),
                    help="torch dtype for model weights (default: auto)")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore partial logs and re-run everything")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


async def main_async() -> int:
    args = parse_args()
    client = TransformersClient(model_id=args.model, dtype=args.dtype)
    grand_t0 = time.perf_counter()
    summaries: dict = {}

    try:
        records = load_records()
        if args.smoke:
            sample = stratified_sample(
                records, n=10, seed=args.seed,
                label_key="gold_label", label_values=("valid", "fabricated"),
            )
            log.info(f"smoke: {len(sample)} instances; "
                     f"valid={sum(1 for r in sample if r['gold_label']=='valid')} "
                     f"fab={sum(1 for r in sample if r['gold_label']=='fabricated')}")
        elif args.full:
            sample = sorted(records, key=lambda r: r["instance_id"])
        else:
            sample = sorted(records, key=lambda r: r["instance_id"])[: args.instances]

        log.info(f"=== cell: {client.model_id} × metadata × llm_only (n={len(sample)}) ===")
        rows = await run_cell(client, sample, resume=not args.no_resume)
        summaries[f"{client.model_id}__metadata__llm_only"] = cell_summary(rows)
    finally:
        client.aclose()

    el = time.perf_counter() - grand_t0
    log.info(f"all cells done in {el:.1f}s")
    for k, s in summaries.items():
        if s.get("n", 0) == 0:
            log.info(f"  {k:<60} (empty)")
            continue
        log.info(f"  {k:<60} acc={s['accuracy']*100:.2f}%  "
                 f"P={s['fab_precision']:.3f} R={s['fab_recall']:.3f} "
                 f"F1={s['fab_f1']:.3f} err={s['errors']}")

    side = SUMMARIES_DIR / (
        "metadata_smoke_summary.json" if args.smoke else "metadata_full_summary.json"
    )
    update_summary(side, summaries)
    log.info(f"summary: {side}")
    return 0


def main() -> None:
    sys.exit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
