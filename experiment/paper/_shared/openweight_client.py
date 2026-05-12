"""Shared infrastructure for running open-weight HuggingFace models against the
metadata/semantic benchmarks.

Provides:
  - ``TransformersClient``  — async wrapper around ``transformers`` greedy decode
  - ``CallResult``          — uniform response container
  - ``stratified_sample``   — deterministic per-source sampler used by --smoke
  - JSONL/CSV helpers       — partial-checkpoint append + read + final write
  - ``parse_two_class``     — robust two-label JSON parser (handles ``<think>``,
                              code fences, and embedded objects)
  - ``PARSE_RETRY_SUFFIX``  — retry instruction appended on a parse failure

Imported by ``experiment/paper/{metadata,semantic}/runner_openweight.py``.
The closed-weight (OpenAI) runners do not need this module.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

log = logging.getLogger("openweight")


@dataclass
class CallResult:
    response_text: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    error: str | None = None


class TransformersClient:
    """Greedy-decode HuggingFace causal-LM client.

    Loads weights once on construction; ``call`` runs generation off the asyncio
    thread so it co-operates with the runner's existing async loop. Qwen3 models
    get the ``enable_thinking=False`` chat-template flag (otherwise they emit
    ``<think>`` blocks the parser would have to strip).
    """

    def __init__(self, model_id: str, dtype: str = "auto"):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "transformers / torch are required. Install with:\n"
                "  pip install -r experiment/paper/_shared/requirements_openweight.txt"
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
                    do_sample=False,
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
        return await asyncio.to_thread(self._generate_sync, system, user, max_tokens)

    def aclose(self) -> None:
        try:
            del self.model
            del self.tokenizer
            self._torch.cuda.empty_cache()
        except Exception:
            pass


def is_qwen3(model_id: str) -> bool:
    return "qwen3" in model_id.lower()


def safe_model_name(model: str) -> str:
    base = model.rsplit("/", 1)[-1]
    return base.replace(":", "-").replace("/", "_")


def trunc(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n].rstrip() + "…"


PARSE_RETRY_SUFFIX = (
    "\n\nReturn ONLY a single JSON object matching the schema in the system "
    "prompt. No prose, no fences, no commentary."
)


def parse_two_class(
    raw: str, allowed: tuple[str, str], *, evidence_key: str,
) -> tuple[str, str, str, str | None]:
    """Parse a two-label JSON verdict, tolerating ``<think>`` blocks, ```` ``` ```` fences,
    and JSON embedded in surrounding prose.

    Returns ``(verdict, reasoning, evidence_or_confidence, error)``. ``error`` is
    one of ``empty_response``, ``parse_failure``, ``bad_verdict_label``, or None.
    """
    raw = (raw or "").strip()
    if not raw:
        return ("", "", "", "empty_response")
    s = raw
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


def stratified_sample(records: list[dict], n: int, seed: int,
                      label_key: str, label_values: tuple[str, str]) -> list[dict]:
    """One pick per source, then fill toward a balanced label split."""
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


def append_partial(p: Path, row) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def read_partial_as_rows(p: Path, kept_ids: set[int], row_cls,
                         metadata_defaults: bool = False):
    """Replay a partial-results JSONL into typed rows, filtering by ``kept_ids``.

    ``metadata_defaults=True`` backfills missing search-related fields with
    sensible defaults — older partials predate those columns.
    """
    rows = []
    if not p.exists():
        return rows
    with open(p, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if int(d["instance_id"]) not in kept_ids:
                continue
            if metadata_defaults:
                d.setdefault("n_search_invocations", 0)
                d.setdefault("search_queries", "")
                d.setdefault("triage_route", "")
                d.setdefault("metadata_agent_called", False)
                d.setdefault("db_source_matched", "")
            rows.append(row_cls(**d))
    rows.sort(key=lambda r: r.instance_id)
    return rows


def write_csv(p: Path, rows: Iterable, columns: list[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def update_summary(summary_path: Path, new_entries: dict) -> None:
    """Merge ``new_entries`` into the JSON file at ``summary_path``."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text())
        except Exception:
            existing = {}
    else:
        existing = {}
    existing.update(new_entries)
    summary_path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
