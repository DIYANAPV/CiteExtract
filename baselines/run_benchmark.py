"""Run multiple evaluation pipelines on the pre-collected benchmark data.

Reads benchmark_data/benchmark_enriched.jsonl (produced by collect_benchmark_data.py)
and runs each pipeline config on every evaluable instance. No paper search or
full text fetch — only LLM calls.

Pipeline configs (ablation study):
  P1: title_only        — LLM gets citing_sentence + title only
  P2: title_abstract    — LLM gets citing_sentence + title + abstract
  P3: passages_simple   — LLM gets citing_sentence + title + abstract + passages (simple prompt)
  P4: claim_agent       — Full ClaimAgent with tools + context-aware prompt

Usage:
    python baselines/run_benchmark.py

Settings — edit the variables below:
"""

# ── Settings ─────────────────────────────────────────────────────────────────
SAMPLE_SIZE = None                      # None = all evaluable, or int for testing
PIPELINES = ["P1", "P2", "P3", "P4"]   # Which pipelines to run
MODEL = "gpt-4o-mini"                   # LLM model for all pipelines
BATCH_SIZE = 3                          # Instances per batch (4 LLM calls each)
BATCH_DELAY = 2.0                       # Seconds between batches
DATA_PATH = "benchmark_data/benchmark_enriched.jsonl"
OUTPUT_DIR = "eval_results"
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import csv
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from openai import AsyncOpenAI
import httpx

from src import config as app_config
from src.verification.agentic.claim_agent import ClaimAgent, build_claim_user_message
from src.verification.agentic.tools import ToolExecutor
from src.verification.api_clients.llm_client import CostTracker
from src.verification.cache import APICache
from src.verification.triage import ContextQuality

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Verdict mapping ──────────────────────────────────────────────────────────

VERDICT_TO_LABEL = {
    "SUPPORTS": "VALID",
    "CONTRADICTS": "MISREPRESENTED",
    "NEUTRAL": "VALID",
}


# ── Structured output schema (shared by P1-P3) ──────────────────────────────

SIMPLE_VERDICT_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "citation_verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["SUPPORTS", "CONTRADICTS", "NEUTRAL"],
                    "description": (
                        "SUPPORTS if the citation accurately represents the paper. "
                        "CONTRADICTS if the citation misrepresents the paper. "
                        "NEUTRAL if there is not enough information to determine."
                    ),
                },
                "explanation": {
                    "type": "string",
                    "description": "Brief reasoning for this verdict.",
                },
            },
            "required": ["verdict", "explanation"],
            "additionalProperties": False,
        },
    },
}

SIMPLE_SYSTEM_PROMPT = """\
You are a citation verification expert. Given a citing sentence from a \
scientific paper and information about the cited paper, determine whether \
the citing sentence accurately represents what the cited paper says.

Respond with:
- SUPPORTS — the citing sentence accurately represents the cited paper
- CONTRADICTS — the citing sentence misrepresents the cited paper \
  (exaggerates, contradicts, cherry-picks, or attributes something not in the paper)
- NEUTRAL — there is not enough information to determine accuracy"""


# ── Load enriched data ───────────────────────────────────────────────────────

def load_enriched_data(data_path: Path, limit: int | None = None) -> list[dict]:
    """Load benchmark_enriched.jsonl, return only evaluable instances."""
    records = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("evaluable"):
                records.append(record)

    if limit is not None:
        records = records[:limit]

    log.info(f"Loaded {len(records)} evaluable instances from {data_path.name}")
    return records


# ── Pipeline runners ─────────────────────────────────────────────────────────

async def run_p1_title_only(
    record: dict,
    openai_client: AsyncOpenAI,
    cost_tracker: CostTracker,
) -> dict:
    """P1: LLM gets citing_sentence + paper title only."""
    citing = record["citing_sentence"]
    title = record.get("search_result", {}).get("found_title", "") or record["cited_paper_title"]

    user_msg = (
        f"## Citing Sentence\n{citing}\n\n"
        f"## Cited Paper\n- **Title**: {title}\n\n"
        f"Does this citing sentence accurately represent the cited paper?"
    )

    return await _simple_llm_call(user_msg, openai_client, cost_tracker)


async def run_p2_title_abstract(
    record: dict,
    openai_client: AsyncOpenAI,
    cost_tracker: CostTracker,
) -> dict:
    """P2: LLM gets citing_sentence + title + abstract."""
    citing = record["citing_sentence"]
    sr = record.get("search_result", {}) or {}
    title = sr.get("found_title", "") or record["cited_paper_title"]
    abstract = sr.get("found_abstract", "") or record.get("cited_paper_abstract", "")

    parts = [
        f"## Citing Sentence\n{citing}",
        f"\n## Cited Paper\n- **Title**: {title}",
    ]
    if abstract:
        parts.append(f"\n### Abstract\n{abstract[:2000]}")
    parts.append("\nDoes this citing sentence accurately represent the cited paper?")

    return await _simple_llm_call("\n".join(parts), openai_client, cost_tracker)


async def run_p3_passages_simple(
    record: dict,
    openai_client: AsyncOpenAI,
    cost_tracker: CostTracker,
) -> dict:
    """P3: LLM gets citing_sentence + title + abstract + passages (simple prompt)."""
    citing = record["citing_sentence"]
    sr = record.get("search_result", {}) or {}
    title = sr.get("found_title", "") or record["cited_paper_title"]
    abstract = sr.get("found_abstract", "") or record.get("cited_paper_abstract", "")
    passages = record.get("passages", [])

    parts = [
        f"## Citing Sentence\n{citing}",
        f"\n## Cited Paper\n- **Title**: {title}",
    ]
    if abstract:
        parts.append(f"\n### Abstract\n{abstract[:1500]}")

    if passages:
        parts.append(f"\n### Retrieved Passages from the Cited Paper ({len(passages)} found)")
        for i, p in enumerate(passages, 1):
            section = f" (Section: {p.get('section', 'unknown')})" if p.get("section") else ""
            parts.append(f"\n**Passage {i}**{section}\n{p.get('text', '')}")
    else:
        parts.append("\n*No passages retrieved from the cited paper.*")

    parts.append(
        "\nBased on the abstract and retrieved passages, does this citing sentence "
        "accurately represent what the cited paper says?"
    )

    return await _simple_llm_call("\n".join(parts), openai_client, cost_tracker)


async def run_p4_claim_agent(
    record: dict,
    claim_agent: ClaimAgent,
) -> dict:
    """P4: Full ClaimAgent with tools + context-aware prompt."""
    citing = record["citing_sentence"]
    sr = record.get("search_result", {}) or {}
    ft = record.get("fulltext", {}) or {}
    passages = record.get("passages", [])

    title = sr.get("found_title", "") or record["cited_paper_title"]
    abstract = sr.get("found_abstract", "") or record.get("cited_paper_abstract", "")
    doi = sr.get("found_doi") or None
    arxiv_id = sr.get("found_arxiv_id") or None
    full_text_available = ft.get("available", False)
    text_source = ft.get("source", "")

    # Build passage dicts with score
    passage_dicts = []
    for p in passages:
        score = p.get("rrf_score") or p.get("dense_score") or p.get("bm25_score") or 0
        passage_dicts.append({
            "text": p.get("text", ""),
            "section": p.get("section"),
            "score": score,
        })

    context_quality = ContextQuality(
        has_before=False,
        has_after=False,
        total_sentences=1,
    )

    citing_contexts = [{
        "citing_sentence": citing,
        "context_before": "",
        "context_after": "",
        "context_quality": context_quality,
        "passages": passage_dicts,
        "marker": "",
    }]

    user_message = build_claim_user_message(
        citing_contexts=citing_contexts,
        paper_title=title,
        paper_abstract=abstract if abstract else None,
        full_text_available=full_text_available,
        full_text_source=text_source,
        paper_doi=doi,
        arxiv_id=arxiv_id,
    )

    try:
        verdicts = await claim_agent.verify_claims(user_message, expected_count=1)
        if verdicts:
            cv = verdicts[0]
            return {
                "verdict": cv.verdict,
                "explanation": cv.explanation[:500],
                "evidence_quote": cv.evidence_quote[:500],
            }
    except Exception as e:
        log.warning(f"P4 ClaimAgent error: {e}")

    return {"verdict": "NEUTRAL", "explanation": "ClaimAgent failed", "evidence_quote": ""}


async def _simple_llm_call(
    user_message: str,
    openai_client: AsyncOpenAI,
    cost_tracker: CostTracker,
) -> dict:
    """Make a single structured LLM call for P1/P2/P3."""
    try:
        response = await openai_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SIMPLE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=512,
            timeout=60,
            response_format=SIMPLE_VERDICT_SCHEMA,
        )

        if response.usage:
            cost_tracker.add(response.usage.prompt_tokens, response.usage.completion_tokens)

        raw = response.choices[0].message.content or ""
        data = json.loads(raw)
        return {
            "verdict": data.get("verdict", "NEUTRAL"),
            "explanation": data.get("explanation", "")[:500],
        }
    except Exception as e:
        log.warning(f"Simple LLM call error: {e}")
        return {"verdict": "NEUTRAL", "explanation": f"LLM error: {str(e)[:200]}"}


# ── Evaluate single instance across all pipelines ────────────────────────────

async def evaluate_one(
    record: dict,
    openai_client: AsyncOpenAI,
    claim_agent: ClaimAgent | None,
    cost_trackers: dict[str, CostTracker],
) -> dict:
    """Run all enabled pipelines on a single instance."""
    result = {
        "idx": record["idx"],
        "source": record["source"],
        "citing_sentence": record["citing_sentence"][:200],
        "cited_paper_title": record["cited_paper_title"][:200],
        "label": record["label"],
        "original_label": record["original_label"],
        "found_title": (record.get("search_result") or {}).get("found_title", ""),
        "found_doi": (record.get("search_result") or {}).get("found_doi", ""),
        "text_source": (record.get("fulltext") or {}).get("source", ""),
    }

    # P1: Title only
    if "P1" in PIPELINES:
        r = await run_p1_title_only(record, openai_client, cost_trackers["P1"])
        predicted = VERDICT_TO_LABEL.get(r["verdict"], "VALID")
        result["p1_verdict"] = r["verdict"]
        result["p1_predicted"] = predicted
        result["p1_explanation"] = r["explanation"]
        result["p1_correct"] = str(predicted == record["label"])

    # P2: Title + Abstract
    if "P2" in PIPELINES:
        r = await run_p2_title_abstract(record, openai_client, cost_trackers["P2"])
        predicted = VERDICT_TO_LABEL.get(r["verdict"], "VALID")
        result["p2_verdict"] = r["verdict"]
        result["p2_predicted"] = predicted
        result["p2_explanation"] = r["explanation"]
        result["p2_correct"] = str(predicted == record["label"])

    # P3: Passages (simple prompt)
    if "P3" in PIPELINES:
        r = await run_p3_passages_simple(record, openai_client, cost_trackers["P3"])
        predicted = VERDICT_TO_LABEL.get(r["verdict"], "VALID")
        result["p3_verdict"] = r["verdict"]
        result["p3_predicted"] = predicted
        result["p3_explanation"] = r["explanation"]
        result["p3_correct"] = str(predicted == record["label"])

    # P4: Full ClaimAgent
    if "P4" in PIPELINES and claim_agent is not None:
        r = await run_p4_claim_agent(record, claim_agent)
        predicted = VERDICT_TO_LABEL.get(r["verdict"], "VALID")
        result["p4_verdict"] = r["verdict"]
        result["p4_predicted"] = predicted
        result["p4_explanation"] = r["explanation"]
        result["p4_evidence_quote"] = r.get("evidence_quote", "")
        result["p4_correct"] = str(predicted == record["label"])

    return result


# ── Metrics computation ──────────────────────────────────────────────────────

def compute_pipeline_metrics(results: list[dict], pipeline_id: str) -> dict:
    """Compute accuracy, macro F1, per-class, per-source for one pipeline."""
    prefix = pipeline_id.lower()
    pred_key = f"{prefix}_predicted"
    correct_key = f"{prefix}_correct"

    evaluated = [r for r in results if pred_key in r and r[pred_key]]
    if not evaluated:
        return {"accuracy": 0, "macro_f1": 0, "per_class": {}, "per_source": {}, "n": 0}

    labels = ["VALID", "MISREPRESENTED"]
    cm = {gt: {pred: 0 for pred in labels} for gt in labels}
    for r in evaluated:
        gt = r.get("label", "")
        pred = r.get(pred_key, "")
        if gt in cm and pred in cm[gt]:
            cm[gt][pred] += 1

    per_class = {}
    f1_scores = []
    for cls in labels:
        tp = cm[cls][cls]
        fp = sum(cm[other][cls] for other in labels if other != cls)
        fn = sum(cm[cls][other] for other in labels if other != cls)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        per_class[cls] = {
            "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(f1, 4), "support": sum(cm[cls].values()),
        }
        f1_scores.append(f1)

    correct = sum(1 for r in evaluated if r.get(correct_key) == "True")
    accuracy = correct / len(evaluated)
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

    # Per source
    per_source = {}
    groups = defaultdict(list)
    for r in evaluated:
        groups[r.get("source", "unknown")].append(r)
    for src, group in sorted(groups.items()):
        src_correct = sum(1 for r in group if r.get(correct_key) == "True")
        per_source[src] = {
            "n": len(group),
            "accuracy": round(src_correct / len(group), 4),
        }

    return {
        "n": len(evaluated),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "per_source": per_source,
        "confusion_matrix": cm,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

async def main():
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    script_dir = Path(__file__).resolve().parent
    data_path = script_dir / DATA_PATH
    out_dir = script_dir / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        log.error(
            f"Data file not found: {data_path}\n"
            f"Run collect_benchmark_data.py first."
        )
        return

    records = load_enriched_data(data_path, SAMPLE_SIZE)
    if not records:
        log.error("No evaluable instances found.")
        return

    # API key
    api_key = app_config.openai_api_key()
    if not api_key:
        log.error("OPENAI_API_KEY not set in .env")
        return

    # Shared resources
    openai_client = AsyncOpenAI(api_key=api_key)
    cost_trackers = {p: CostTracker() for p in PIPELINES}

    # ClaimAgent for P4
    claim_agent = None
    if "P4" in PIPELINES:
        hybrid_cfg = app_config.hybrid() or {}
        claim_cfg = hybrid_cfg.get("claim_agent", {})
        cache = APICache()
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as http_client:
            tool_executor = ToolExecutor(http_client, cache)
            claim_agent = ClaimAgent(
                openai_client=openai_client,
                tool_executor=tool_executor,
                model=MODEL,
                temperature=hybrid_cfg.get("temperature", 0.1),
                max_tool_rounds=claim_cfg.get("max_tool_rounds", 2),
                max_tokens=claim_cfg.get("max_tokens", 2048),
                timeout=hybrid_cfg.get("timeout", 60),
                cost_tracker=cost_trackers["P4"],
            )

            all_results = await _run_all(records, openai_client, claim_agent, cost_trackers)
        await cache.close()
    else:
        all_results = await _run_all(records, openai_client, None, cost_trackers)

    elapsed = round(time.time() - start_time, 1)

    # ── Compute metrics ──────────────────────────────────────────────────
    # Subset stats
    subset_stats = {
        "evaluable": len(records),
        "valid": sum(1 for r in records if r["label"] == "VALID"),
        "misrepresented": sum(1 for r in records if r["label"] == "MISREPRESENTED"),
    }

    pipeline_metrics = {}
    for p in PIPELINES:
        pipeline_metrics[p] = compute_pipeline_metrics(all_results, p)

    # Comparison table (ready for paper)
    comparison_table = []
    pipeline_names = {
        "P1": "Title only",
        "P2": "Title + Abstract",
        "P3": "Passages (simple prompt)",
        "P4": "ClaimAgent (full pipeline)",
    }
    for p in PIPELINES:
        m = pipeline_metrics[p]
        row = {
            "pipeline": p,
            "method": pipeline_names.get(p, p),
            "model": MODEL,
            "n": m["n"],
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "valid_f1": m.get("per_class", {}).get("VALID", {}).get("f1", 0),
            "misrep_f1": m.get("per_class", {}).get("MISREPRESENTED", {}).get("f1", 0),
            "cost_usd": round(cost_trackers[p].estimated_cost_usd, 4),
        }
        comparison_table.append(row)

    metrics_output = {
        "subset_stats": subset_stats,
        "pipelines": pipeline_metrics,
        "comparison_table": comparison_table,
        "cost_per_pipeline": {
            p: round(cost_trackers[p].estimated_cost_usd, 4) for p in PIPELINES
        },
        "total_cost_usd": round(sum(t.estimated_cost_usd for t in cost_trackers.values()), 4),
        "model": MODEL,
        "dense_model": app_config.comprehension().get("dense_model", ""),
        "elapsed_seconds": elapsed,
        "timestamp": timestamp,
    }

    # ── Save outputs ─────────────────────────────────────────────────────
    n_eval = len(all_results)
    pipelines_tag = "+".join(PIPELINES)

    csv_name = f"eval_{pipelines_tag}_{n_eval}_{timestamp}.csv"
    json_name = f"metrics_{pipelines_tag}_{n_eval}_{timestamp}.json"

    # CSV
    csv_path = out_dir / csv_name
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)
        log.info(f"Results CSV: {csv_path}")

    # JSON
    json_path = out_dir / json_name
    with open(json_path, "w") as f:
        json.dump(metrics_output, f, indent=2, default=str)
    log.info(f"Metrics JSON: {json_path}")

    # ── Print comparison table ───────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print(f"  BENCHMARK COMPARISON — {MODEL} — {n_eval} instances")
    print(f"{'=' * 90}")
    print(f"\n  {'Pipeline':<8} {'Method':<30} {'Acc':>8} {'M-F1':>8} "
          f"{'V-F1':>8} {'MR-F1':>8} {'Cost':>8}")
    print(f"  {'-'*8:<8} {'-'*30:<30} {'-'*8:>8} {'-'*8:>8} "
          f"{'-'*8:>8} {'-'*8:>8} {'-'*8:>8}")

    for row in comparison_table:
        print(f"  {row['pipeline']:<8} {row['method']:<30} "
              f"{row['accuracy']:>8.4f} {row['macro_f1']:>8.4f} "
              f"{row['valid_f1']:>8.4f} {row['misrep_f1']:>8.4f} "
              f"${row['cost_usd']:>7.4f}")

    # Per-source breakdown for each pipeline
    for p in PIPELINES:
        m = pipeline_metrics[p]
        if m.get("per_source"):
            print(f"\n  {p} per source:")
            for src, sm in sorted(m["per_source"].items()):
                print(f"    {src:15s}: n={sm['n']:4d}, acc={sm['accuracy']:.4f}")

    print(f"\n  Total cost: ${metrics_output['total_cost_usd']:.4f}")
    print(f"  Elapsed: {elapsed}s")
    print(f"\n  CSV: {csv_path}")
    print(f"  JSON: {json_path}")


async def _run_all(
    records: list[dict],
    openai_client: AsyncOpenAI,
    claim_agent: ClaimAgent | None,
    cost_trackers: dict[str, CostTracker],
) -> list[dict]:
    """Process all records in batches."""
    all_results = []
    total_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num, start in enumerate(range(0, len(records), BATCH_SIZE), 1):
        batch = records[start : start + BATCH_SIZE]
        log.info(
            f"Batch {batch_num}/{total_batches} "
            f"({len(batch)} instances, pipelines: {PIPELINES})"
        )

        tasks = [
            evaluate_one(rec, openai_client, claim_agent, cost_trackers)
            for rec in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for rec, res in zip(batch, results):
            if isinstance(res, Exception):
                log.error(f"[{rec['idx']}] Exception: {res}")
                all_results.append({
                    "idx": rec["idx"],
                    "source": rec["source"],
                    "label": rec["label"],
                })
            else:
                all_results.append(res)

        if start + BATCH_SIZE < len(records):
            await asyncio.sleep(BATCH_DELAY)

    return all_results


if __name__ == "__main__":
    asyncio.run(main())
