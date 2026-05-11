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

try:
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env", override=False)
except Exception:
    pass

from experiment.paper.semantic.llm_clients import (
    PRICING, LLMCallResult, build_client, resolve_model,
)
from experiment.paper.semantic.prompt_builder import (
    CONDITIONS, build_user_message, load_system_prompt, map_to_gold,
    parse_response,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("semantic_table")

DATA_PATH = _PAPER_ROOT / "data" / "benchmark_semantic.jsonl"
RESULTS_DIR = _PAPER_ROOT / "results"

DEFAULT_MODELS = (
    "gpt-4o-mini", "gpt-4o",
    "gpt-5-min", "gpt-5-med",
    "gpt-5.5-min", "gpt-5.5-med",
)

CSV_COLUMNS = [
    "instance_id", "source", "gold_label", "predicted_verdict", "raw_verdict",
    "explanation", "evidence_quote", "raw_response",
    "prompt_tokens", "completion_tokens", "cost_usd", "latency_seconds", "error",
]


@dataclass
class Row:
    instance_id: int
    source: str
    gold_label: str
    predicted_verdict: str
    raw_verdict: str
    explanation: str
    evidence_quote: str
    raw_response: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_seconds: float
    error: str | None


def load_evaluable() -> list[dict]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)
    out: list[dict] = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("evaluable"):
                out.append(d)
    return out


def stratified_smoke_sample(records: list[dict], n: int = 10, seed: int = 42) -> list[dict]:
    import random
    rng = random.Random(seed)
    by_source: dict[str, list[dict]] = {}
    for r in records:
        by_source.setdefault(r["source"], []).append(r)

    picked: list[dict] = []
    seen_ids: set[int] = set()
    for src in sorted(by_source):
        bucket = by_source[src]
        v = [r for r in bucket if r["label"] == "VALID"]
        m = [r for r in bucket if r["label"] == "MISREPRESENTED"]
        choice = rng.choice(v) if v else rng.choice(m) if m else None
        if choice is not None and choice["idx"] not in seen_ids:
            picked.append(choice)
            seen_ids.add(choice["idx"])

    target_valid = round(n * 556 / 741)
    valid_count = sum(1 for r in picked if r["label"] == "VALID")
    pool_valid = [r for r in records if r["label"] == "VALID" and r["idx"] not in seen_ids]
    pool_misrep = [r for r in records if r["label"] == "MISREPRESENTED" and r["idx"] not in seen_ids]
    rng.shuffle(pool_valid)
    rng.shuffle(pool_misrep)

    while len(picked) < n:
        if valid_count < target_valid and pool_valid:
            r = pool_valid.pop(); picked.append(r); seen_ids.add(r["idx"]); valid_count += 1
        elif pool_misrep:
            r = pool_misrep.pop(); picked.append(r); seen_ids.add(r["idx"])
        elif pool_valid:
            r = pool_valid.pop(); picked.append(r); seen_ids.add(r["idx"]); valid_count += 1
        else:
            break

    picked.sort(key=lambda r: r["idx"])
    return picked


def _csv_path(model: str, condition: str) -> Path:
    return RESULTS_DIR / f"semantic_{model.replace('/', '_')}_{condition}.csv"


def _partial_path(model: str, condition: str) -> Path:
    return RESULTS_DIR / f"semantic_partial_{model.replace('/', '_')}_{condition}.jsonl"


def load_completed_ids(partial_path: Path) -> set[int]:
    if not partial_path.exists():
        return set()
    seen: set[int] = set()
    with open(partial_path, encoding="utf-8") as f:
        for line in f:
            try:
                seen.add(int(json.loads(line)["instance_id"]))
            except Exception:
                continue
    return seen


def write_partial(partial_path: Path, row: Row) -> None:
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    with open(partial_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def write_csv(csv_path: Path, rows: Iterable[Row]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


_PARSE_RETRY_SUFFIX = (
    "\n\nReturn ONLY a single JSON object matching the schema in the "
    "system prompt. No prose, no fences, no commentary."
)


async def verify_one(
    client, system: str, user: str, model: str
) -> tuple[Row, dict]:
    res: LLMCallResult = await client.call(system, user, max_tokens=512)
    cost = res.cost_usd(model)

    if res.error is not None:
        return (
            Row(
                instance_id=-1, source="", gold_label="",
                predicted_verdict="MISREPRESENTED", raw_verdict="",
                explanation="", evidence_quote="",
                raw_response="", prompt_tokens=res.prompt_tokens,
                completion_tokens=res.completion_tokens, cost_usd=cost,
                latency_seconds=round(res.latency_seconds, 3),
                error=res.error,
            ),
            {"raw": "", "verdict": "", "reason": "", "ev": "", "err": res.error},
        )

    raw = res.response_text
    verdict, reason, evidence, perr = parse_response(raw)

    if perr == "parse_failure":
        retry = await client.call(system, user + _PARSE_RETRY_SUFFIX, max_tokens=512)
        cost += retry.cost_usd(model)
        if retry.error is None:
            v2, r2, e2, perr2 = parse_response(retry.response_text)
            if perr2 is None:
                verdict, reason, evidence, perr = v2, r2, e2, None
                raw = retry.response_text
        in_tok = res.prompt_tokens + retry.prompt_tokens
        out_tok = res.completion_tokens + retry.completion_tokens
        lat = res.latency_seconds + retry.latency_seconds
    else:
        in_tok, out_tok, lat = res.prompt_tokens, res.completion_tokens, res.latency_seconds

    return (
        Row(
            instance_id=-1, source="", gold_label="",
            predicted_verdict=map_to_gold(verdict), raw_verdict=verdict,
            explanation=reason, evidence_quote=evidence,
            raw_response=raw[:4000],
            prompt_tokens=in_tok, completion_tokens=out_tok,
            cost_usd=round(cost, 6),
            latency_seconds=round(lat, 3),
            error=perr,
        ),
        {"raw": raw, "verdict": verdict, "reason": reason, "ev": evidence, "err": perr},
    )


def _read_partial_as_rows(partial_path: Path, records: list[dict]) -> list[Row]:
    keep_ids = {r["idx"] for r in records}
    rows: list[Row] = []
    if not partial_path.exists():
        return rows
    with open(partial_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if int(d["instance_id"]) not in keep_ids:
                continue
            rows.append(Row(**d))
    rows.sort(key=lambda r: r.instance_id)
    return rows


async def run_cell(
    model: str,
    condition: str,
    records: list[dict],
    *,
    concurrency: int,
    resume: bool = True,
) -> list[Row]:
    csv_path = _csv_path(model, condition)
    partial_path = _partial_path(model, condition)

    completed = load_completed_ids(partial_path) if resume else set()
    if completed:
        log.info(f"[{model} | {condition}] resume: {len(completed)} already in partial log")

    todo = [r for r in records if r["idx"] not in completed]
    if not todo:
        log.info(f"[{model} | {condition}] all {len(records)} already done; reading partial → CSV")
        rows = _read_partial_as_rows(partial_path, records)
        write_csv(csv_path, rows)
        return rows

    client = build_client(model)
    system = load_system_prompt()
    sem = asyncio.Semaphore(concurrency)
    t_cell = time.perf_counter()
    cell_rows: list[Row] = []

    async def _one(rec: dict) -> Row:
        async with sem:
            user = build_user_message(rec, condition)
            row, _ = await verify_one(client, system, user, model)
            row.instance_id = rec["idx"]
            row.source = rec["source"]
            row.gold_label = rec["label"]
            write_partial(partial_path, row)
            return row

    done = 0
    tasks = [asyncio.create_task(_one(r)) for r in todo]
    for fut in asyncio.as_completed(tasks):
        try:
            cell_rows.append(await fut)
        except Exception as e:
            log.error(f"[{model} | {condition}] task crashed: {e}")
        done += 1
        if done % 5 == 0 or done == len(todo):
            el = time.perf_counter() - t_cell
            rate = done / el if el else 0
            log.info(f"[{model} | {condition}] {done}/{len(todo)} ({rate:.1f}/s)")

    rows = _read_partial_as_rows(partial_path, records)
    write_csv(csv_path, rows)
    log.info(f"[{model} | {condition}] cell done in {time.perf_counter() - t_cell:.1f}s; CSV: {csv_path}")
    return rows


def cell_summary(rows: list[Row]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    correct = sum(1 for r in rows if r.predicted_verdict == r.gold_label)
    tp_v = sum(1 for r in rows if r.gold_label == "VALID" and r.predicted_verdict == "VALID")
    fp_v = sum(1 for r in rows if r.gold_label == "MISREPRESENTED" and r.predicted_verdict == "VALID")
    fn_v = sum(1 for r in rows if r.gold_label == "VALID" and r.predicted_verdict == "MISREPRESENTED")
    tp_m = sum(1 for r in rows if r.gold_label == "MISREPRESENTED" and r.predicted_verdict == "MISREPRESENTED")

    def _f1(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    vp, vr, vf = _f1(tp_v, fp_v, fn_v)
    mp, mr, mf = _f1(tp_m, fn_v, fp_v)
    return {
        "n": n,
        "accuracy": correct / n,
        "macro_f1": (vf + mf) / 2,
        "valid":  {"precision": vp, "recall": vr, "f1": vf, "support": tp_v + fn_v},
        "misrep": {"precision": mp, "recall": mr, "f1": mf, "support": tp_m + fp_v},
        "errors": sum(1 for r in rows if r.error),
        "total_cost_usd": round(sum(r.cost_usd for r in rows), 4),
        "total_latency_seconds": round(sum(r.latency_seconds for r in rows), 1),
    }


def print_mini_table(summaries: dict[tuple[str, str], dict], models: tuple[str, ...]) -> None:
    from io import StringIO
    buf = StringIO()
    print(file=buf)
    print(f"{'model':<22} | " + " | ".join(f"{c:^28}" for c in CONDITIONS), file=buf)
    print("-" * (24 + 31 * len(CONDITIONS)), file=buf)
    for m in models:
        cells = []
        for c in CONDITIONS:
            s = summaries.get((m, c), {})
            if not s:
                cells.append(f"{'(missing)':^28}")
                continue
            acc = s.get("accuracy", 0.0) * 100
            n = s.get("n", 0)
            errs = s.get("errors", 0)
            cells.append(f"acc={acc:5.2f}% n={n:>3} err={errs}".center(28))
        print(f"{m:<22} | " + " | ".join(cells), file=buf)
    log.info("Smoke summary:\n" + buf.getvalue())


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true", help="10-instance smoke test")
    g.add_argument("--full", action="store_true", help="all 741 evaluable instances")
    g.add_argument("--instances", type=int, help="ad-hoc cap (first N evaluable, sorted by idx)")
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--openai-concurrency", type=int, default=12)
    ap.add_argument("--gemini-concurrency", type=int, default=8)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


async def main_async() -> int:
    args = parse_args()
    for m in args.models:
        api_model, _ = resolve_model(m)
        if api_model not in PRICING:
            log.error(f"Unknown model {m!r}; allowed: {list(PRICING)} (with optional -min/-med suffix)")
            return 2

    records = load_evaluable()
    if args.smoke:
        sample = stratified_smoke_sample(records, n=10, seed=args.seed)
        log.info(f"smoke: {len(sample)} instances; "
                 f"VALID={sum(1 for r in sample if r['label']=='VALID')} "
                 f"MISREP={sum(1 for r in sample if r['label']=='MISREPRESENTED')}; "
                 f"sources={sorted({r['source'] for r in sample})}")
    elif args.full:
        sample = records
    else:
        sample = sorted(records, key=lambda r: r["idx"])[: args.instances]

    summaries: dict[tuple[str, str], dict] = {}
    grand_t0 = time.perf_counter()

    for model in args.models:
        for condition in args.conditions:
            conc = (
                args.gemini_concurrency if model.startswith("gemini-")
                else args.openai_concurrency
            )
            log.info(f"=== cell: {model} × {condition} (n={len(sample)}, concurrency={conc}) ===")
            rows = await run_cell(
                model, condition, sample,
                concurrency=conc, resume=not args.no_resume,
            )
            summaries[(model, condition)] = cell_summary(rows)

    log.info(f"all cells done in {time.perf_counter() - grand_t0:.1f}s")
    print_mini_table(summaries, tuple(args.models))

    side = RESULTS_DIR / ("smoke_summary.json" if args.smoke else "full_summary.json")
    side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text(json.dumps(
        {f"{m}__{c}": s for (m, c), s in summaries.items()},
        indent=2, default=str,
    ), encoding="utf-8")
    log.info(f"summary: {side}")
    return 0


def main() -> None:
    sys.exit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
