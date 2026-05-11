
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

_THIS_FILE = Path(__file__).resolve()
_PAPER_ROOT = _THIS_FILE.parent.parent
_REPO_ROOT = _PAPER_ROOT.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# `citeextract/config/` is a YAML data directory at the repo root that Python
# would otherwise treat as a namespace package, shadowing `citeextract.config`
# (a real module at `citeextract/backend/config.py`). The editable install's
# finder maps `citeextract` -> `citeextract/backend` correctly, but it is
# *appended* to sys.meta_path so PathFinder wins. Promote it to position 0.
try:
    import __editable___citeextract_0_1_0_finder as _ef  # type: ignore
    if _ef._EditableFinder in sys.meta_path:
        sys.meta_path.remove(_ef._EditableFinder)
    sys.meta_path.insert(0, _ef._EditableFinder)
except ImportError:
    pass

try:
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env", override=False)
except Exception:
    pass

import httpx

from experiment.paper.metadata.llm_clients import (
    build_client,
)
from experiment.paper.metadata.prompt_builder import (
    build_user_message, load_system_prompt, parse_response,
)
from experiment.paper.metadata.run_production import (
    ProductionResult, run_one_production,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("metadata_table")

DATA_PATH = _PAPER_ROOT / "data" / "benchmark_metadata.jsonl"
RESULTS_DIR = _PAPER_ROOT / "results"

DEFAULT_MODELS = ("gpt-4o-mini", "gpt-4o")
DEFAULT_CELLS: list[tuple[str, str]] = [
    ("gpt-4o-mini", "llm_only"),
    ("gpt-4o-mini", "llm_with_search"),
    ("gpt-4o", "llm_only"),
    ("gpt-4o", "llm_with_search"),
    ("gpt-5-min", "llm_only"),
    ("gpt-5-min", "llm_with_search"),
    ("gpt-5-med", "llm_only"),
    ("gpt-5-med", "llm_with_search"),
    ("gpt-5.5-min", "llm_only"),
    ("gpt-5.5-min", "llm_with_search"),
    ("gpt-5.5-med", "llm_only"),
    ("gpt-5.5-med", "llm_with_search"),
    ("our_system", "production"),
    ("our_system", "production_gpt55med"),
    ("our_system", "production_gpt5mini"),
]

PRODUCTION_VARIANTS: dict[str, dict] = {
    "production":          {"agent_model": "gpt-4o-mini", "pricing": (0.15, 0.60)},
    "production_gpt55med": {"agent_model": "gpt-5.5",     "pricing": (5.00, 30.00)},
    "production_gpt5mini": {"agent_model": "gpt-5-mini",  "pricing": (0.25,  2.00)},
}

CSV_COLUMNS = [
    "instance_id", "source", "gold_label", "predicted_verdict", "raw_verdict",
    "explanation", "confidence", "raw_response",
    "prompt_tokens", "completion_tokens", "cost_usd", "latency_seconds",
    "n_search_invocations", "search_queries",
    "triage_route", "metadata_agent_called", "db_source_matched",
    "error",
]


@dataclass
class Row:
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


def load_records() -> list[dict]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)
    rows: list[dict] = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def stratified_smoke_sample(records: list[dict], n: int = 10, seed: int = 42) -> list[dict]:
    import random
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {"valid": [], "fabricated": []}
    for r in records:
        by_label.setdefault(r["gold_label"], []).append(r)

    by_source: dict[str, list[dict]] = {}
    for r in records:
        by_source.setdefault(r["source"], []).append(r)

    picked: list[dict] = []
    seen: set[int] = set()

    for src in sorted(by_source):
        cand = rng.choice(by_source[src])
        picked.append(cand); seen.add(cand["instance_id"])

    target_valid = n // 2
    target_fab = n - target_valid
    have_valid = sum(1 for r in picked if r["gold_label"] == "valid")
    have_fab = sum(1 for r in picked if r["gold_label"] == "fabricated")

    pool_valid = [r for r in by_label["valid"] if r["instance_id"] not in seen]
    pool_fab = [r for r in by_label["fabricated"] if r["instance_id"] not in seen]
    rng.shuffle(pool_valid); rng.shuffle(pool_fab)

    while len(picked) < n:
        if have_valid < target_valid and pool_valid:
            r = pool_valid.pop(); picked.append(r); seen.add(r["instance_id"]); have_valid += 1
        elif have_fab < target_fab and pool_fab:
            r = pool_fab.pop(); picked.append(r); seen.add(r["instance_id"]); have_fab += 1
        elif pool_valid:
            r = pool_valid.pop(); picked.append(r); seen.add(r["instance_id"])
        elif pool_fab:
            r = pool_fab.pop(); picked.append(r); seen.add(r["instance_id"])
        else:
            break

    picked.sort(key=lambda r: r["instance_id"])
    return picked


def _csv_path(model: str, condition: str) -> Path:
    return RESULTS_DIR / f"metadata_{model}_{condition}.csv"


def _partial_path(model: str, condition: str) -> Path:
    return RESULTS_DIR / f"metadata_partial_{model}_{condition}.jsonl"


def load_completed_ids(p: Path) -> set[int]:
    if not p.exists():
        return set()
    seen: set[int] = set()
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                seen.add(int(json.loads(line)["instance_id"]))
            except Exception:
                continue
    return seen


def write_partial(p: Path, row: Row) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def write_csv(p: Path, rows: Iterable[Row]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def _read_partial_as_rows(p: Path, records: list[dict]) -> list[Row]:
    keep_ids = {r["instance_id"] for r in records}
    rows: list[Row] = []
    if not p.exists():
        return rows
    with open(p, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if int(d["instance_id"]) not in keep_ids:
                continue
            d.setdefault("n_search_invocations", 0)
            d.setdefault("search_queries", "")
            d.setdefault("triage_route", "")
            d.setdefault("metadata_agent_called", False)
            d.setdefault("db_source_matched", "")
            rows.append(Row(**d))
    rows.sort(key=lambda r: r.instance_id)
    return rows


_PARSE_RETRY_SUFFIX = (
    "\n\nReturn ONLY a single JSON object matching the schema in the system "
    "prompt. No prose, no fences, no commentary."
)


async def run_llm_cell(
    model: str,
    condition: str,
    records: list[dict],
    *,
    concurrency: int,
    resume: bool,
) -> list[Row]:
    csv_path = _csv_path(model, condition)
    partial_path = _partial_path(model, condition)

    completed = load_completed_ids(partial_path) if resume else set()
    todo = [r for r in records if r["instance_id"] not in completed]
    if not todo:
        log.info(f"[{model} | {condition}] all done ({len(records)}); regenerating CSV from partial")
        rows = _read_partial_as_rows(partial_path, records)
        write_csv(csv_path, rows)
        return rows

    if completed:
        log.info(f"[{model} | {condition}] resume: {len(completed)} already in partial")

    client = build_client(model, condition)
    system = load_system_prompt(condition)
    sem = asyncio.Semaphore(concurrency)
    t0 = time.perf_counter()
    done = 0

    async def _one(rec: dict) -> Row:
        async with sem:
            user = build_user_message(rec)
            res = await client.call(system, user, max_tokens=512)
            cost = res.cost_usd(model)

            if res.error is not None:
                row = Row(
                    instance_id=rec["instance_id"], source=rec["source"],
                    gold_label=rec["gold_label"],
                    predicted_verdict="fabricated", raw_verdict="",
                    explanation="", confidence="",
                    raw_response="",
                    prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens,
                    cost_usd=round(cost, 6), latency_seconds=round(res.latency_seconds, 3),
                    n_search_invocations=res.n_search_invocations,
                    search_queries=json.dumps(res.search_queries, ensure_ascii=False),
                    error=res.error,
                )
                write_partial(partial_path, row)
                return row

            verdict, reason, conf, perr = parse_response(res.response_text)

            if perr == "parse_failure":
                retry = await client.call(system, user + _PARSE_RETRY_SUFFIX, max_tokens=512)
                cost += retry.cost_usd(model)
                if retry.error is None:
                    v2, r2, c2, perr2 = parse_response(retry.response_text)
                    if perr2 is None:
                        verdict, reason, conf, perr = v2, r2, c2, None
                tot_in = res.prompt_tokens + retry.prompt_tokens
                tot_out = res.completion_tokens + retry.completion_tokens
                tot_lat = res.latency_seconds + retry.latency_seconds
                tot_search = res.n_search_invocations + retry.n_search_invocations
                queries = res.search_queries + retry.search_queries
                raw = retry.response_text if perr is None else res.response_text
            else:
                tot_in = res.prompt_tokens
                tot_out = res.completion_tokens
                tot_lat = res.latency_seconds
                tot_search = res.n_search_invocations
                queries = res.search_queries
                raw = res.response_text

            row = Row(
                instance_id=rec["instance_id"], source=rec["source"],
                gold_label=rec["gold_label"],
                predicted_verdict=verdict, raw_verdict=verdict,
                explanation=reason, confidence=conf,
                raw_response=raw[:4000],
                prompt_tokens=tot_in, completion_tokens=tot_out,
                cost_usd=round(cost, 6), latency_seconds=round(tot_lat, 3),
                n_search_invocations=tot_search,
                search_queries=json.dumps(queries, ensure_ascii=False),
                error=perr,
            )
            write_partial(partial_path, row)
            return row

    cell_rows: list[Row] = []
    tasks = [asyncio.create_task(_one(r)) for r in todo]
    for fut in asyncio.as_completed(tasks):
        try:
            row = await fut
            cell_rows.append(row)
        except Exception as e:
            log.error(f"[{model} | {condition}] task crashed: {e}")
        done += 1
        if done % 5 == 0 or done == len(todo):
            el = time.perf_counter() - t0
            rate = done / el if el else 0
            log.info(f"[{model} | {condition}] {done}/{len(todo)} ({rate:.1f}/s)")

    rows = _read_partial_as_rows(partial_path, records)
    write_csv(csv_path, rows)
    el = time.perf_counter() - t0
    log.info(f"[{model} | {condition}] cell done in {el:.1f}s; CSV: {csv_path}")
    return rows


async def run_production_cell(
    records: list[dict],
    *,
    concurrency: int,
    resume: bool,
    variant: str = "production",
) -> list[Row]:
    cfg = PRODUCTION_VARIANTS[variant]
    agent_model: str = cfg["agent_model"]
    in_rate, out_rate = cfg["pricing"]
    cell_id = f"our_system | {variant} ({agent_model})"

    csv_path = _csv_path("our_system", variant)
    partial_path = _partial_path("our_system", variant)

    completed = load_completed_ids(partial_path) if resume else set()
    todo = [r for r in records if r["instance_id"] not in completed]
    if not todo:
        log.info(f"[{cell_id}] all done; regenerating CSV from partial")
        rows = _read_partial_as_rows(partial_path, records)
        write_csv(csv_path, rows)
        return rows

    if completed:
        log.info(f"[{cell_id}] resume: {len(completed)} already in partial")

    from openai import AsyncOpenAI
    from citeextract.verification.cache import APICache
    from citeextract.verification.agentic.metadata_agent import MetadataAgent
    from citeextract.verification.agentic.tools import ToolExecutor
    from citeextract.verification.api_clients.llm_client import CostTracker

    cache = APICache()
    sem = asyncio.Semaphore(concurrency)
    t0 = time.perf_counter()
    done = 0

    async def _drive(http_client):
        openai_client = AsyncOpenAI()
        tool_exec = ToolExecutor(http_client, cache)

        async def _one(rec: dict) -> Row:
            async with sem:
                def _agent_factory():
                    return MetadataAgent(
                        openai_client=openai_client,
                        tool_executor=tool_exec,
                        model=agent_model,
                        temperature=0.0,
                        max_tool_rounds=3,
                        cost_tracker=CostTracker(),
                    )

                pres: ProductionResult = await run_one_production(
                    rec,
                    http_client=http_client,
                    cache=cache,
                    metadata_agent_factory=_agent_factory,
                )
                # Recompute cost using this variant's pricing — CostTracker
                # uses config.toml's global rates (gpt-4o-mini), so the
                # pres.cost_usd would otherwise be wrong for gpt-5/5.5/-mini.
                cost_recomputed = (
                    pres.prompt_tokens * in_rate / 1_000_000
                    + pres.completion_tokens * out_rate / 1_000_000
                )
                row = Row(
                    instance_id=rec["instance_id"], source=rec["source"],
                    gold_label=rec["gold_label"],
                    predicted_verdict=pres.predicted_verdict,
                    raw_verdict=pres.raw_verdict,
                    explanation=pres.explanation, confidence="",
                    raw_response="",
                    prompt_tokens=pres.prompt_tokens,
                    completion_tokens=pres.completion_tokens,
                    cost_usd=round(cost_recomputed, 6),
                    latency_seconds=round(pres.latency_seconds, 3),
                    triage_route=pres.triage_route,
                    metadata_agent_called=pres.metadata_agent_called,
                    db_source_matched=pres.db_source_matched or "",
                    error=pres.error,
                )
                write_partial(partial_path, row)
                return row

        cell_rows: list[Row] = []
        tasks = [asyncio.create_task(_one(r)) for r in todo]
        nonlocal done
        for fut in asyncio.as_completed(tasks):
            try:
                row = await fut
                cell_rows.append(row)
            except Exception as e:
                log.error(f"[{cell_id}] task crashed: {e}")
            done += 1
            if done % 5 == 0 or done == len(todo):
                el = time.perf_counter() - t0
                rate = done / el if el else 0
                log.info(f"[{cell_id}] {done}/{len(todo)} ({rate:.1f}/s)")
        return cell_rows

    async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(60.0)) as http_client:
        await _drive(http_client)

    rows = _read_partial_as_rows(partial_path, records)
    write_csv(csv_path, rows)
    el = time.perf_counter() - t0
    log.info(f"[{cell_id}] cell done in {el:.1f}s; CSV: {csv_path}")
    return rows


def cell_summary(rows: list[Row]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    correct = sum(1 for r in rows if r.predicted_verdict == r.gold_label)
    tp_f = sum(1 for r in rows if r.gold_label == "fabricated" and r.predicted_verdict == "fabricated")
    fp_f = sum(1 for r in rows if r.gold_label == "valid" and r.predicted_verdict == "fabricated")
    fn_f = sum(1 for r in rows if r.gold_label == "fabricated" and r.predicted_verdict == "valid")
    p = tp_f / (tp_f + fp_f) if tp_f + fp_f else 0.0
    rcl = tp_f / (tp_f + fn_f) if tp_f + fn_f else 0.0
    f1 = 2 * p * rcl / (p + rcl) if p + rcl else 0.0
    cost = sum(r.cost_usd for r in rows)
    lat = sum(r.latency_seconds for r in rows)
    err = sum(1 for r in rows if r.error)
    n_search_total = sum(r.n_search_invocations for r in rows)
    n_search_avg = n_search_total / n if n else 0.0
    return {
        "n": n,
        "accuracy": correct / n,
        "fab_precision": p, "fab_recall": rcl, "fab_f1": f1,
        "errors": err,
        "total_cost_usd": round(cost, 4),
        "total_latency_seconds": round(lat, 1),
        "avg_searches_per_ref": round(n_search_avg, 3),
    }


def print_mini_table(summaries: dict[str, dict]) -> None:
    log.info("Cell summary:")
    for cell, s in summaries.items():
        if s.get("n", 0) == 0:
            log.info(f"  {cell:<40} (empty)")
            continue
        log.info(
            f"  {cell:<40}  acc={s['accuracy']*100:5.2f}%  "
            f"P={s['fab_precision']:.3f}  R={s['fab_recall']:.3f}  F1={s['fab_f1']:.3f}  "
            f"err={s['errors']}  search/ref={s['avg_searches_per_ref']}  "
            f"cost=${s['total_cost_usd']}"
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--full", action="store_true")
    g.add_argument("--instances", type=int)
    ap.add_argument("--cells", nargs="+", default=None,
                    help="subset of cells, format model:condition; e.g. gpt-4o-mini:llm_only")
    ap.add_argument("--openai-concurrency", type=int, default=8)
    ap.add_argument("--production-concurrency", type=int, default=4,
                    help="lower than OpenAI because S2 is rate-limited 1 req/s")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


async def main_async() -> int:
    args = parse_args()

    cells: list[tuple[str, str]] = list(DEFAULT_CELLS)
    if args.cells:
        cells = []
        for spec in args.cells:
            m, c = spec.split(":", 1)
            cells.append((m, c))

    records = load_records()
    if args.smoke:
        sample = stratified_smoke_sample(records, n=10, seed=args.seed)
        log.info(f"smoke: {len(sample)} instances; "
                 f"valid={sum(1 for r in sample if r['gold_label']=='valid')} "
                 f"fab={sum(1 for r in sample if r['gold_label']=='fabricated')}; "
                 f"sources={sorted({r['source'] for r in sample})}")
    elif args.full:
        sample = sorted(records, key=lambda r: r["instance_id"])
    else:
        sample = sorted(records, key=lambda r: r["instance_id"])[: args.instances]

    summaries: dict[str, dict] = {}
    grand_t0 = time.perf_counter()

    for model, condition in cells:
        cell_id = f"{model} × {condition}"
        log.info(f"=== cell: {cell_id} (n={len(sample)}) ===")
        if model == "our_system":
            rows = await run_production_cell(
                sample,
                concurrency=args.production_concurrency,
                resume=not args.no_resume,
                variant=condition,
            )
        else:
            rows = await run_llm_cell(
                model, condition, sample,
                concurrency=args.openai_concurrency,
                resume=not args.no_resume,
            )
        summaries[f"{model}__{condition}"] = cell_summary(rows)

    el = time.perf_counter() - grand_t0
    log.info(f"all cells done in {el:.1f}s")
    print_mini_table(summaries)

    side = RESULTS_DIR / ("metadata_smoke_summary.json" if args.smoke else "metadata_full_summary.json")
    side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text(json.dumps(summaries, indent=2, default=str), encoding="utf-8")
    log.info(f"summary: {side}")
    return 0


def main() -> None:
    sys.exit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
