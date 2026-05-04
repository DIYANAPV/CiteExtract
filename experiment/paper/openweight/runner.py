"""Open-weight model benchmark runner — semantic + metadata.

Self-contained, single-process runner using HuggingFace ``transformers``.
Designed for a SLURM/GPU compute node. Reads JSONLs from
``experiment/paper/data/`` and writes per-cell CSVs to
``experiment/paper/results/``, matching the OpenAI runners' schemas so
the aggregators pick them up uniformly.

CLI:
    python -m experiment.paper.openweight.runner --model Qwen/Qwen3-8B --task semantic --smoke
    python -m experiment.paper.openweight.runner --model Qwen/Qwen3-8B --task semantic --full
    python -m experiment.paper.openweight.runner --model Qwen/Qwen3-8B --task metadata --full
    python -m experiment.paper.openweight.runner --model meta-llama/Llama-3.1-8B-Instruct --task semantic --full
    # Llama is gated: huggingface-cli login first.

Generation is greedy (deterministic). Qwen 3 thinking mode is suppressed
three ways: ``enable_thinking=False`` chat-template kwarg, a ``/no_think``
user-message prefix, and a ``<think>...</think>`` parser scrub.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PAPER_ROOT = THIS_FILE.parent.parent
REPO_ROOT = PAPER_ROOT.parent.parent

PROMPTS_DIR = PAPER_ROOT / "prompts"
DATA_DIR = PAPER_ROOT / "data"
RESULTS_DIR = PAPER_ROOT / "results"

SEMANTIC_DATA = DATA_DIR / "benchmark_semantic.jsonl"
METADATA_DATA = DATA_DIR / "benchmark_metadata.jsonl"

CSV_COLUMNS_SEMANTIC = [
    "instance_id", "source", "gold_label", "predicted_verdict", "raw_verdict",
    "explanation", "evidence_quote", "raw_response",
    "prompt_tokens", "completion_tokens", "cost_usd", "latency_seconds", "error",
]
CSV_COLUMNS_METADATA = [
    "instance_id", "source", "gold_label", "predicted_verdict", "raw_verdict",
    "explanation", "confidence", "raw_response",
    "prompt_tokens", "completion_tokens", "cost_usd", "latency_seconds",
    "n_search_invocations", "search_queries",
    "triage_route", "metadata_agent_called", "db_source_matched",
    "error",
]

SEMANTIC_CONDITIONS = ("title_only", "title_abstract", "title_abstract_passages")
METADATA_CONDITIONS = ("llm_only",)
ABSTRACT_MAX_CHARS = 1500

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("openweight")


# ── data classes ──────────────────────────────────────────────────────


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


@dataclass
class CallResult:
    response_text: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    error: str | None = None


class TransformersClient:
    """Loads a HuggingFace model and runs greedy generation in-process."""

    def __init__(self, model_id: str, dtype: str = "auto"):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "transformers / torch are required. Install with:\n"
                "  pip install -r experiment/paper/openweight/requirements.txt"
            ) from e

        self.model_id = model_id
        self.is_qwen3 = "qwen3" in model_id.lower()
        self.is_qwen = "qwen" in model_id.lower()

        log.info(f"loading tokenizer {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        torch_dtype = "auto"
        if dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif dtype == "fp16":
            torch_dtype = torch.float16
        elif dtype != "auto":
            raise ValueError(f"unsupported dtype: {dtype} (use auto / bf16 / fp16)")

        log.info(f"loading model weights ({dtype}); first run downloads from HuggingFace")
        t0 = time.perf_counter()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        log.info(f"model loaded in {time.perf_counter() - t0:.1f}s; "
                 f"device={next(self.model.parameters()).device}; "
                 f"dtype={next(self.model.parameters()).dtype}")

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self._torch = torch

    def _format_prompt(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self.is_qwen3:
            kwargs["enable_thinking"] = False
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            # Older tokenizers reject enable_thinking; drop it and retry.
            kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _generate_sync(self, system: str, user: str, max_tokens: int) -> CallResult:
        prompt = self._format_prompt(system, user)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_tok = int(inputs.input_ids.shape[1])
        t0 = time.perf_counter()
        try:
            with self._torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,        # greedy → deterministic
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        except Exception as e:
            return CallResult("", prompt_tok, 0, time.perf_counter() - t0,
                              error=f"generate_error:{type(e).__name__}:{str(e)[:120]}")
        latency = time.perf_counter() - t0
        out_ids = output[0][prompt_tok:]
        text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
        out_tok = int(out_ids.shape[0])
        return CallResult(text, prompt_tok, out_tok, latency)

    async def call(self, system: str, user: str, max_tokens: int = 512,
                   _want_json: bool = True) -> CallResult:
        # ``_want_json`` matches the OpenAI/Gemini client signatures for
        # drop-in compatibility but has no effect — transformers has no
        # native JSON mode. The prompt itself enforces JSON output, and
        # the parser tolerates fences / leaked <think> blocks.
        return await asyncio.to_thread(self._generate_sync, system, user, max_tokens)

    def aclose(self) -> None:
        try:
            del self.model
            del self.tokenizer
            self._torch.cuda.empty_cache()
        except Exception:
            pass


def _is_qwen3(model_id: str) -> bool:
    return "qwen3" in model_id.lower()


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def _trunc(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n].rstrip() + "…"


def build_user_message_semantic(record: dict, condition: str, *, qwen3: bool) -> str:
    citing = (record.get("citing_sentence") or "").strip()
    title = (record.get("cited_paper_title") or "").strip()
    parts: list[str] = []
    if qwen3:
        # Belt-and-suspenders alongside the chat template's enable_thinking=False.
        parts.append("/no_think")
    parts.append(f'Citing sentence: "{citing}"')
    parts.append(f"Cited paper title: {title}")
    if condition in ("title_abstract", "title_abstract_passages"):
        ab = _trunc(record.get("cited_paper_abstract") or "", ABSTRACT_MAX_CHARS)
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


def build_user_message_metadata(record: dict, *, qwen3: bool) -> str:
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


def parse_semantic(raw: str) -> tuple[str, str, str, str, str | None]:
    """Returns (mapped_verdict, raw_verdict, reasoning, evidence, error)."""
    verdict, reason, evidence, err = _parse_two_class(
        raw, ("SUPPORTED", "NOT_SUPPORTED"), evidence_key="evidence_quote",
    )
    if err is not None and verdict == "":
        return "MISREPRESENTED", "", "", "", err
    mapped = "VALID" if verdict == "SUPPORTED" else "MISREPRESENTED"
    return mapped, verdict, reason, evidence, err


def parse_metadata(raw: str) -> tuple[str, str, str, str, str | None]:
    """Returns (verdict, raw_verdict, reasoning, confidence, error)."""
    verdict, reason, conf, err = _parse_two_class(
        raw, ("valid", "fabricated"), evidence_key="confidence",
    )
    if err is not None and verdict == "":
        return "fabricated", "", "", "", err
    return verdict, verdict, reason, conf, err


def _parse_two_class(
    raw: str, allowed: tuple[str, str], *, evidence_key: str,
) -> tuple[str, str, str, str | None]:
    raw = (raw or "").strip()
    if not raw:
        return ("", "", "", "empty_response")
    s = raw
    # Strip Qwen 3 <think>...</think> blocks that leak past enable_thinking=False.
    if "<think>" in s:
        end = s.find("</think>")
        if end >= 0:
            s = s[end + len("</think>"):].strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lstrip().lower().startswith("json"):
            s = s.lstrip()[4:].lstrip()
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        if 0 <= start < end:
            try:
                data = json.loads(s[start: end + 1])
            except Exception:
                return ("", "", "", "parse_failure")
        else:
            return ("", "", "", "parse_failure")
    if not isinstance(data, dict):
        return ("", "", "", "parse_failure")

    cls = str(data.get("classification", "")).strip()
    cls_norm = cls.upper() if allowed[0].isupper() else cls.lower()
    if cls_norm not in allowed:
        return (
            "",
            str(data.get("reasoning", "") or "")[:600],
            str(data.get(evidence_key, "") or "")[:600],
            "bad_verdict_label",
        )
    return (
        cls_norm,
        str(data.get("reasoning", "") or "")[:600],
        str(data.get(evidence_key, "") or "")[:600],
        None,
    )


def load_semantic_records() -> list[dict]:
    if not SEMANTIC_DATA.exists():
        raise FileNotFoundError(SEMANTIC_DATA)
    out: list[dict] = []
    with open(SEMANTIC_DATA, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("evaluable"):
                out.append(d)
    return out


def load_metadata_records() -> list[dict]:
    if not METADATA_DATA.exists():
        raise FileNotFoundError(METADATA_DATA)
    return [json.loads(l) for l in open(METADATA_DATA, encoding="utf-8")]


def stratified_sample(records: list[dict], n: int, seed: int,
                      label_key: str, label_values: tuple[str, str]) -> list[dict]:
    """Balanced sample with at least one row per source. Deterministic."""
    rng = random.Random(seed)
    by_src: dict[str, list[dict]] = {}
    for r in records:
        by_src.setdefault(r["source"], []).append(r)
    picked: list[dict] = []
    seen: set = set()
    for src in sorted(by_src):
        cand = rng.choice(by_src[src])
        key = cand.get("instance_id", cand.get("idx"))
        picked.append(cand); seen.add(key)
    rest_a = [r for r in records if (r.get("instance_id") or r.get("idx")) not in seen
              and r[label_key] == label_values[0]]
    rest_b = [r for r in records if (r.get("instance_id") or r.get("idx")) not in seen
              and r[label_key] == label_values[1]]
    rng.shuffle(rest_a); rng.shuffle(rest_b)
    target_a = n // 2
    while len(picked) < n:
        cur_a = sum(1 for r in picked if r[label_key] == label_values[0])
        if cur_a < target_a and rest_a:
            r = rest_a.pop(); picked.append(r); seen.add(r.get("instance_id") or r.get("idx"))
        elif rest_b:
            r = rest_b.pop(); picked.append(r); seen.add(r.get("instance_id") or r.get("idx"))
        elif rest_a:
            r = rest_a.pop(); picked.append(r); seen.add(r.get("instance_id") or r.get("idx"))
        else:
            break
    picked.sort(key=lambda r: r.get("instance_id") or r.get("idx"))
    return picked


def _safe_model_name(model: str) -> str:
    # "Qwen/Qwen3-8B" → "Qwen3-8B"
    base = model.rsplit("/", 1)[-1]
    return base.replace(":", "-").replace("/", "_")


def _csv_path(task: str, model: str, condition: str) -> Path:
    return RESULTS_DIR / f"{task}_{_safe_model_name(model)}_{condition}.csv"


def _partial_path(task: str, model: str, condition: str) -> Path:
    return RESULTS_DIR / f"{task}_partial_{_safe_model_name(model)}_{condition}.jsonl"


def _load_completed_ids(p: Path) -> set[int]:
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


def _append_partial(p: Path, row) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def _read_partial_as_rows(p: Path, kept_ids: set[int], row_cls):
    rows = []
    if not p.exists():
        return rows
    with open(p, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if int(d["instance_id"]) not in kept_ids:
                continue
            if row_cls is MetadataRow:
                d.setdefault("n_search_invocations", 0)
                d.setdefault("search_queries", "")
                d.setdefault("triage_route", "")
                d.setdefault("metadata_agent_called", False)
                d.setdefault("db_source_matched", "")
            rows.append(row_cls(**d))
    rows.sort(key=lambda r: r.instance_id)
    return rows


def _write_csv(p: Path, rows, columns: list[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


_PARSE_RETRY_SUFFIX = (
    "\n\nReturn ONLY a single JSON object matching the schema in the system "
    "prompt. No prose, no fences, no commentary."
)


async def run_semantic_cell(client: TransformersClient, condition: str,
                             records: list[dict], *, resume: bool) -> list[SemanticRow]:
    csv_path = _csv_path("semantic", client.model_id, condition)
    partial_path = _partial_path("semantic", client.model_id, condition)
    completed = _load_completed_ids(partial_path) if resume else set()
    todo = [r for r in records if r["idx"] not in completed]
    if not todo:
        log.info(f"[{client.model_id} | semantic | {condition}] all done; regenerating CSV from partial")
        rows = _read_partial_as_rows(partial_path, {r["idx"] for r in records}, SemanticRow)
        _write_csv(csv_path, rows, CSV_COLUMNS_SEMANTIC)
        return rows

    if completed:
        log.info(f"[{client.model_id} | semantic | {condition}] resume: {len(completed)} in partial")

    system = _load_prompt("semantic_2class.txt")
    qwen3 = _is_qwen3(client.model_id)
    t0 = time.perf_counter()

    # Sequential: generation is GPU-bound; one model instance can't usefully parallelize.
    for i, rec in enumerate(todo, start=1):
        user = build_user_message_semantic(rec, condition, qwen3=qwen3)
        res = await client.call(system, user, max_tokens=512)
        mapped, raw_v, reason, evidence, perr = parse_semantic(res.response_text)
        if perr == "parse_failure" and res.error is None:
            retry = await client.call(system, user + _PARSE_RETRY_SUFFIX, max_tokens=512)
            if retry.error is None:
                m2, rv2, r2, e2, err2 = parse_semantic(retry.response_text)
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
        _append_partial(partial_path, row)
        if i % 5 == 0 or i == len(todo):
            el = time.perf_counter() - t0
            rate = i / el if el else 0
            log.info(f"[{client.model_id} | semantic | {condition}] {i}/{len(todo)} ({rate:.2f}/s)")

    rows = _read_partial_as_rows(partial_path, {r["idx"] for r in records}, SemanticRow)
    _write_csv(csv_path, rows, CSV_COLUMNS_SEMANTIC)
    log.info(f"[{client.model_id} | semantic | {condition}] cell done in "
             f"{time.perf_counter() - t0:.1f}s; CSV: {csv_path}")
    return rows


async def run_metadata_cell(client: TransformersClient, records: list[dict],
                             *, resume: bool) -> list[MetadataRow]:
    condition = "llm_only"
    csv_path = _csv_path("metadata", client.model_id, condition)
    partial_path = _partial_path("metadata", client.model_id, condition)
    completed = _load_completed_ids(partial_path) if resume else set()
    todo = [r for r in records if r["instance_id"] not in completed]
    if not todo:
        log.info(f"[{client.model_id} | metadata | {condition}] all done; regenerating CSV from partial")
        rows = _read_partial_as_rows(partial_path,
                                     {r["instance_id"] for r in records}, MetadataRow)
        _write_csv(csv_path, rows, CSV_COLUMNS_METADATA)
        return rows

    if completed:
        log.info(f"[{client.model_id} | metadata | {condition}] resume: {len(completed)} in partial")

    system = _load_prompt("metadata_llm_only.txt")
    qwen3 = _is_qwen3(client.model_id)
    t0 = time.perf_counter()

    for i, rec in enumerate(todo, start=1):
        user = build_user_message_metadata(rec, qwen3=qwen3)
        res = await client.call(system, user, max_tokens=512)
        verdict, raw_v, reason, conf, perr = parse_metadata(res.response_text)
        if perr == "parse_failure" and res.error is None:
            retry = await client.call(system, user + _PARSE_RETRY_SUFFIX, max_tokens=512)
            if retry.error is None:
                v2, rv2, r2, c2, err2 = parse_metadata(retry.response_text)
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
        _append_partial(partial_path, row)
        if i % 5 == 0 or i == len(todo):
            el = time.perf_counter() - t0
            rate = i / el if el else 0
            log.info(f"[{client.model_id} | metadata | {condition}] {i}/{len(todo)} ({rate:.2f}/s)")

    rows = _read_partial_as_rows(partial_path,
                                 {r["instance_id"] for r in records}, MetadataRow)
    _write_csv(csv_path, rows, CSV_COLUMNS_METADATA)
    log.info(f"[{client.model_id} | metadata | {condition}] cell done in "
             f"{time.perf_counter() - t0:.1f}s; CSV: {csv_path}")
    return rows


def cell_summary_semantic(rows: list[SemanticRow]) -> dict:
    if not rows:
        return {"n": 0}
    correct = sum(1 for r in rows if r.predicted_verdict == r.gold_label)
    err = sum(1 for r in rows if r.error)
    lat = sum(r.latency_seconds for r in rows)
    return {"n": len(rows), "accuracy": correct / len(rows), "errors": err,
            "total_latency_seconds": round(lat, 1)}


def cell_summary_metadata(rows: list[MetadataRow]) -> dict:
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
                    help="HuggingFace model id, e.g. Qwen/Qwen3-8B or meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--task", required=True, choices=("semantic", "metadata"))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true",
                   help="10 instances, deterministic stratified sample")
    g.add_argument("--full", action="store_true", help="all instances")
    g.add_argument("--instances", type=int, help="ad-hoc cap (first N by id)")
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="(semantic only) subset of: title_only title_abstract title_abstract_passages")
    ap.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp16"),
                    help="torch dtype for model weights (default: auto, picks model's native)")
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
        if args.task == "semantic":
            records = load_semantic_records()
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

            conds = tuple(args.conditions) if args.conditions else SEMANTIC_CONDITIONS
            for c in conds:
                if c not in SEMANTIC_CONDITIONS:
                    log.error(f"unknown semantic condition: {c}; allowed {SEMANTIC_CONDITIONS}")
                    return 2
                log.info(f"=== cell: {client.model_id} × semantic × {c} (n={len(sample)}) ===")
                rows = await run_semantic_cell(
                    client, c, sample, resume=not args.no_resume,
                )
                summaries[f"{client.model_id}__semantic__{c}"] = cell_summary_semantic(rows)

        else:  # metadata
            records = load_metadata_records()
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
            rows = await run_metadata_cell(client, sample, resume=not args.no_resume)
            summaries[f"{client.model_id}__metadata__llm_only"] = cell_summary_metadata(rows)
    finally:
        client.aclose()

    el = time.perf_counter() - grand_t0
    log.info(f"all cells done in {el:.1f}s")
    for k, s in summaries.items():
        if s.get("n", 0) == 0:
            log.info(f"  {k:<60} (empty)")
            continue
        if "fab_f1" in s:
            log.info(f"  {k:<60} acc={s['accuracy']*100:.2f}%  "
                     f"P={s['fab_precision']:.3f} R={s['fab_recall']:.3f} "
                     f"F1={s['fab_f1']:.3f} err={s['errors']}")
        else:
            log.info(f"  {k:<60} acc={s['accuracy']*100:.2f}%  "
                     f"err={s['errors']}  lat={s['total_latency_seconds']}s")

    side = RESULTS_DIR / (
        "openweight_smoke_summary.json" if args.smoke else "openweight_full_summary.json"
    )
    side.parent.mkdir(parents=True, exist_ok=True)
    if side.exists():
        try:
            existing = json.loads(side.read_text())
        except Exception:
            existing = {}
    else:
        existing = {}
    existing.update(summaries)
    side.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
    log.info(f"summary: {side}")
    return 0


def main() -> None:
    sys.exit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
