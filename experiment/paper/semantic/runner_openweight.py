"""Open-weight (HuggingFace) semantic runner. Counterpart to ``runner.py`` (OpenAI API).

Usage
-----
    python -m experiment.paper.semantic.runner_openweight --model Qwen/Qwen3-8B --full
    python -m experiment.paper.semantic.runner_openweight --model Qwen/Qwen3-8B --smoke
    python -m experiment.paper.semantic.runner_openweight --model Qwen/Qwen3-8B \
        --conditions title_only title_abstract title_abstract_passages --full
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
    trunc,
    update_summary,
    write_csv,
)

PROMPTS_DIR = _PAPER_ROOT / "prompts"
DATA_PATH = _PAPER_ROOT / "data" / "benchmark_semantic.jsonl"
RESULTS_DIR = _PAPER_ROOT / "results" / "semantic"
CSV_DIR = RESULTS_DIR / "csv"
PARTIAL_DIR = RESULTS_DIR / "partial"
SUMMARIES_DIR = RESULTS_DIR / "summaries"

CSV_COLUMNS = [
    "instance_id", "source", "gold_label", "predicted_verdict", "raw_verdict",
    "explanation", "evidence_quote", "raw_response",
    "prompt_tokens", "completion_tokens", "cost_usd", "latency_seconds", "error",
]

CONDITIONS = ("title_only", "title_abstract", "title_abstract_passages")
ABSTRACT_MAX_CHARS = 1500

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("semantic_openweight")


@dataclass
class SemanticRow:
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
    error: str | None = None


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def _build_user_message(record: dict, condition: str, *, qwen3: bool) -> str:
    citing = (record.get("citing_sentence") or "").strip()
    title = (record.get("cited_paper_title") or "").strip()
    parts: list[str] = []
    if qwen3:
        parts.append("/no_think")
    parts.append(f'Citing sentence: "{citing}"')
    parts.append(f"Cited paper title: {title}")
    if condition in ("title_abstract", "title_abstract_passages"):
        ab = trunc(record.get("cited_paper_abstract") or "", ABSTRACT_MAX_CHARS)
        parts.append(f"Cited paper abstract: {ab or '(none available)'}")
    if condition == "title_abstract_passages":
        passages = record.get("passages") or []
        if passages:
            wrapped = []
            for p in passages:
                section = (p.get("section") or "Unknown").strip()
                text = (p.get("text") or "").strip()
                wrapped.append({
                    "section": section,
                    "text": f"<<<UNTRUSTED_PASSAGE>>>{text}<<<END_UNTRUSTED>>>",
                })
            parts.append(
                "Retrieved passages from the cited paper:\n"
                + json.dumps(wrapped, indent=2, ensure_ascii=False)
            )
        else:
            parts.append("Retrieved passages from the cited paper: (none retrieved)")
    return "\n\n".join(parts)


def _parse(raw: str) -> tuple[str, str, str, str, str | None]:
    verdict, reason, evidence, err = parse_two_class(
        raw, ("SUPPORTED", "NOT_SUPPORTED"), evidence_key="evidence_quote",
    )
    if err is not None and verdict == "":
        return "MISREPRESENTED", "", "", "", err
    mapped = "VALID" if verdict == "SUPPORTED" else "MISREPRESENTED"
    return mapped, verdict, reason, evidence, err


def _csv_path(model: str, condition: str) -> Path:
    return CSV_DIR / f"semantic_{safe_model_name(model)}_{condition}.csv"


def _partial_path(model: str, condition: str) -> Path:
    return PARTIAL_DIR / f"semantic_partial_{safe_model_name(model)}_{condition}.jsonl"


def load_records() -> list[dict]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)
    out: list[dict] = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("evaluable"):
                out.append(d)
    return out


async def run_cell(client: TransformersClient, condition: str,
                   records: list[dict], *, resume: bool) -> list[SemanticRow]:
    csv_path = _csv_path(client.model_id, condition)
    partial_path = _partial_path(client.model_id, condition)
    completed = load_completed_ids(partial_path) if resume else set()
    todo = [r for r in records if r["idx"] not in completed]
    if not todo:
        log.info(f"[{client.model_id} | semantic | {condition}] all done; regenerating CSV from partial")
        rows = read_partial_as_rows(partial_path, {r["idx"] for r in records}, SemanticRow)
        write_csv(csv_path, rows, CSV_COLUMNS)
        return rows

    if completed:
        log.info(f"[{client.model_id} | semantic | {condition}] resume: {len(completed)} in partial")

    system = _load_prompt("semantic_2class.txt")
    qwen3 = is_qwen3(client.model_id)
    t0 = time.perf_counter()

    for i, rec in enumerate(todo, start=1):
        user = _build_user_message(rec, condition, qwen3=qwen3)
        res = await client.call(system, user, max_tokens=512)
        mapped, raw_v, reason, evidence, perr = _parse(res.response_text)
        if perr == "parse_failure" and res.error is None:
            retry = await client.call(system, user + PARSE_RETRY_SUFFIX, max_tokens=512)
            if retry.error is None:
                m2, rv2, r2, e2, err2 = _parse(retry.response_text)
                if err2 is None:
                    mapped, raw_v, reason, evidence, perr = m2, rv2, r2, e2, None
            tot_in = res.prompt_tokens + retry.prompt_tokens
            tot_out = res.completion_tokens + retry.completion_tokens
            tot_lat = res.latency_seconds + retry.latency_seconds
            raw_resp = retry.response_text if perr is None else res.response_text
        else:
            tot_in = res.prompt_tokens
            tot_out = res.completion_tokens
            tot_lat = res.latency_seconds
            raw_resp = res.response_text
        row = SemanticRow(
            instance_id=rec["idx"], source=rec["source"], gold_label=rec["label"],
            predicted_verdict=mapped, raw_verdict=raw_v,
            explanation=reason, evidence_quote=evidence,
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
            log.info(f"[{client.model_id} | semantic | {condition}] {i}/{len(todo)} ({rate:.2f}/s)")

    rows = read_partial_as_rows(partial_path, {r["idx"] for r in records}, SemanticRow)
    write_csv(csv_path, rows, CSV_COLUMNS)
    log.info(f"[{client.model_id} | semantic | {condition}] cell done in "
             f"{time.perf_counter() - t0:.1f}s; CSV: {csv_path}")
    return rows


def cell_summary(rows: list[SemanticRow]) -> dict:
    if not rows:
        return {"n": 0}
    correct = sum(1 for r in rows if r.predicted_verdict == r.gold_label)
    err = sum(1 for r in rows if r.error)
    lat = sum(r.latency_seconds for r in rows)
    return {"n": len(rows), "accuracy": correct / len(rows), "errors": err,
            "total_latency_seconds": round(lat, 1)}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True,
                    help="HuggingFace model id, e.g. Qwen/Qwen3-8B")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true",
                   help="10 instances, deterministic stratified sample")
    g.add_argument("--full", action="store_true", help="all instances")
    g.add_argument("--instances", type=int, help="ad-hoc cap (first N by id)")
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="subset of: title_only title_abstract title_abstract_passages")
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
                label_key="label", label_values=("VALID", "MISREPRESENTED"),
            )
            log.info(f"smoke: {len(sample)} instances; "
                     f"VALID={sum(1 for r in sample if r['label']=='VALID')} "
                     f"MISREP={sum(1 for r in sample if r['label']=='MISREPRESENTED')}")
        elif args.full:
            sample = sorted(records, key=lambda r: r["idx"])
        else:
            sample = sorted(records, key=lambda r: r["idx"])[: args.instances]

        conds = tuple(args.conditions) if args.conditions else CONDITIONS
        for c in conds:
            if c not in CONDITIONS:
                log.error(f"unknown semantic condition: {c}; allowed {CONDITIONS}")
                return 2
            log.info(f"=== cell: {client.model_id} × semantic × {c} (n={len(sample)}) ===")
            rows = await run_cell(client, c, sample, resume=not args.no_resume)
            summaries[f"{client.model_id}__semantic__{c}"] = cell_summary(rows)
    finally:
        client.aclose()

    el = time.perf_counter() - grand_t0
    log.info(f"all cells done in {el:.1f}s")
    for k, s in summaries.items():
        if s.get("n", 0) == 0:
            log.info(f"  {k:<60} (empty)")
            continue
        log.info(f"  {k:<60} acc={s['accuracy']*100:.2f}%  "
                 f"err={s['errors']}  lat={s['total_latency_seconds']}s")

    side = SUMMARIES_DIR / (
        "semantic_smoke_summary.json" if args.smoke else "semantic_full_summary.json"
    )
    update_summary(side, summaries)
    log.info(f"summary: {side}")
    return 0


def main() -> None:
    sys.exit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
