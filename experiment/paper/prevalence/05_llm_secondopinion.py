"""For every MISREP case (CONTRADICTS) from the prevalence run, send a stronger"""

from __future__ import annotations

import argparse
import asyncio
import csv
import glob
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

_THIS_FILE = Path(__file__).resolve()
_PREV_DIR = _THIS_FILE.parent
_REPO_ROOT = _PREV_DIR.parent.parent.parent

try:
    from dotenv import load_dotenv
    for _candidate in (_REPO_ROOT / "citeextract" / ".env", _REPO_ROOT / ".env"):
        if _candidate.exists():
            load_dotenv(_candidate, override=False)
            break
except Exception:
    pass

import fitz
import httpx
from openai import AsyncOpenAI

from citeextract.models.reference import Reference
from citeextract.models.verdict import ExistenceResult
from citeextract.verification.api_clients.fulltext import get_full_text
from citeextract.verification.cache import APICache
from citeextract.verification.existence import check_existence

DEFAULT_PER_PAPER_DIR = _PREV_DIR / "results" / "per_paper"
DEFAULT_PAPERS_DIR = _PREV_DIR / "data" / "papers"
DEFAULT_OUT_DIR = _PREV_DIR / "results" / "secondopinion"
DEFAULT_CSV = _PREV_DIR / "results" / "misrep_secondopinion.csv"
DEFAULT_MD = _PREV_DIR / "results" / "misrep_secondopinion.md"

DEFAULT_MODEL = "gpt-4o"
CITING_WINDOW_CHARS = 4000
MAX_CITED_CHARS = 350_000
TEMPERATURE = 0.0

MISREP_VERDICTS = {"NOT_SUPPORTED", "CONTRADICTS"}

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger("secondopinion")


@dataclass
class SecondOpinionResult:
    paper_id: str
    ref_id: str
    citing_sentence: str
    cited_title: Optional[str]
    cited_doi: Optional[str]
    primary_verdict: str
    primary_explanation: str
    secondopinion_verdict: str
    secondopinion_reasoning: str
    secondopinion_confidence: str
    agrees: bool
    cited_text_chars: int
    citing_context_chars: int
    cited_text_source: str
    model: str
    cost_usd: float
    error: Optional[str]


def load_misrep_cases(per_paper_dir: Path) -> list[dict]:
    cases: list[dict] = []
    for path in sorted(per_paper_dir.glob("*.jsonl")):
        for line in path.open("r"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("paper_found") or not r.get("full_text_available"):
                continue
            if r.get("verdict") in MISREP_VERDICTS:
                cases.append(r)
    return cases


def extract_citing_context(pdf_path: Path, citing_sentence: str, window: int) -> str:
    if not pdf_path.exists():
        return ""
    doc = fitz.open(str(pdf_path))
    full_text = "\n".join(page.get_text() for page in doc)
    doc.close()
    if not full_text or not citing_sentence:
        return full_text[: window * 2]
    needle = re.sub(r"\s+", " ", citing_sentence[:100]).strip()
    haystack = re.sub(r"\s+", " ", full_text)
    idx = haystack.find(needle)
    if idx == -1:
        mid = len(haystack) // 2
        return haystack[max(0, mid - window // 2): mid + window // 2]
    start = max(0, idx - window // 2)
    end = min(len(haystack), idx + len(needle) + window // 2)
    return haystack[start:end]


async def fetch_cited_full_text(
    ref_id: str,
    cited_doi: Optional[str],
    cited_title: Optional[str],
    client: httpx.AsyncClient,
    cache: APICache,
) -> tuple[str, str]:
    if not cited_title and not cited_doi:
        return "", "not_found"
    ref = Reference(
        ref_id=ref_id,
        title=cited_title or "",
        doi=cited_doi,
        authors=[],
        source_format="text",
    )
    try:
        exist = await check_existence(ref, client, cache)
    except Exception as exc:
        log.warning("existence check failed: %s", exc)
        return "", "not_found"
    if not exist or exist.status != "FOUND":
        return "", "not_found"
    try:
        ft = await get_full_text(exist, client, cache)
    except Exception as exc:
        log.warning("full-text fetch failed: %s", exc)
        return "", "not_found"
    if ft.full_text:
        return ft.full_text, ft.source or "full_text"
    if ft.abstract:
        return ft.abstract, "abstract_only"
    return "", "not_found"


PROMPT_TEMPLATE = """\
You are an independent reviewer checking whether a scientific citation is \
faithfully supported by the cited paper.

You will be given:
1. A passage from the CITING paper (the paper that makes the claim).
2. The CITING SENTENCE — the specific sentence containing the citation.
3. The full text of the CITED paper.

Your job is to determine whether the cited paper actually supports the \
specific claim that the citing sentence makes about it.

Pay attention to:
- What the citing sentence specifically attributes to the cited paper.
- Whether the cited paper actually does, says, or proposes that thing.
- Whether the citing sentence categorizes the cited paper correctly \
(e.g., is it really an "X-based method" if it's actually presenting itself \
as a "Y-based method"?).
- Whether the cited paper's findings or claims actually substantiate the \
specific point being made.

Respond strictly in JSON with these keys:
- verdict: one of "SUPPORTED" (cited paper supports the claim faithfully), \
"NOT_SUPPORTED" (cited paper does not support or actually contradicts the claim), \
"PARTIAL" (partially supports / overstates), \
"UNCLEAR" (insufficient evidence to decide either way).
- reasoning: 2-4 sentences explaining your verdict, quoting at least one specific \
phrase from the cited paper.
- confidence: "high" / "medium" / "low".

CITING PAPER CONTEXT (excerpt around the citation):
=====
{citing_context}
=====

CITING SENTENCE (the specific claim about the cited reference):
=====
{citing_sentence}
=====

CITED PAPER (full text, possibly truncated):
=====
{cited_text}
=====
"""

PRICING = {
    "gpt-4o":         {"in": 2.50,  "out": 10.00},
    "gpt-4o-mini":    {"in": 0.15,  "out": 0.60},
    "gpt-5":          {"in": 5.00,  "out": 15.00},
    "gpt-5-mini":     {"in": 0.40,  "out": 1.60},
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p = PRICING.get(model, {"in": 2.50, "out": 10.00})
    return prompt_tokens / 1_000_000 * p["in"] + completion_tokens / 1_000_000 * p["out"]


async def call_llm(
    openai_client: AsyncOpenAI,
    model: str,
    prompt: str,
) -> tuple[dict, float, dict]:
    resp = await openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"verdict": "UNCLEAR", "reasoning": f"raw response: {content[:300]}", "confidence": "low"}
    usage = resp.usage.model_dump() if resp.usage else {}
    cost = estimate_cost(model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
    return parsed, cost, usage


async def process_case(
    case: dict,
    args: argparse.Namespace,
    httpx_client: httpx.AsyncClient,
    cache: APICache,
    openai_client: AsyncOpenAI,
) -> SecondOpinionResult:
    paper_id = case["paper_id"]
    ref_id = case["ref_id"]
    citing_sentence = case["citing_sentence"]
    cited_doi = case.get("cited_doi")
    cited_title = case.get("cited_title")
    primary_verdict = case.get("verdict") or "?"
    primary_explanation = case.get("explanation") or ""

    pdf_path = args.papers_dir / f"{paper_id}.pdf"
    citing_context = extract_citing_context(pdf_path, citing_sentence, args.window)

    cited_text, cited_source = await fetch_cited_full_text(
        ref_id, cited_doi, cited_title, httpx_client, cache,
    )
    if not cited_text:
        return SecondOpinionResult(
            paper_id=paper_id, ref_id=ref_id,
            citing_sentence=citing_sentence,
            cited_title=cited_title, cited_doi=cited_doi,
            primary_verdict=primary_verdict,
            primary_explanation=primary_explanation,
            secondopinion_verdict="UNCLEAR",
            secondopinion_reasoning="Could not retrieve cited paper full text.",
            secondopinion_confidence="low",
            agrees=False,
            cited_text_chars=0,
            citing_context_chars=len(citing_context),
            cited_text_source=cited_source,
            model=args.model,
            cost_usd=0.0,
            error="no_cited_text",
        )

    cited_text = cited_text[: args.max_cited_chars]
    prompt = PROMPT_TEMPLATE.format(
        citing_context=citing_context,
        citing_sentence=citing_sentence,
        cited_text=cited_text,
    )

    try:
        parsed, cost, _usage = await call_llm(openai_client, args.model, prompt)
        verdict = (parsed.get("verdict") or "UNCLEAR").upper()
        reasoning = parsed.get("reasoning") or ""
        confidence = (parsed.get("confidence") or "low").lower()
        err = None
    except Exception as exc:
        verdict, reasoning, confidence, cost, err = "UNCLEAR", str(exc), "low", 0.0, repr(exc)

    agrees = verdict in ("NOT_SUPPORTED", "PARTIAL")

    return SecondOpinionResult(
        paper_id=paper_id, ref_id=ref_id,
        citing_sentence=citing_sentence,
        cited_title=cited_title, cited_doi=cited_doi,
        primary_verdict=primary_verdict,
        primary_explanation=primary_explanation,
        secondopinion_verdict=verdict,
        secondopinion_reasoning=reasoning,
        secondopinion_confidence=confidence,
        agrees=agrees,
        cited_text_chars=len(cited_text),
        citing_context_chars=len(citing_context),
        cited_text_source=cited_source,
        model=args.model,
        cost_usd=round(cost, 6),
        error=err,
    )


def write_per_case_checkpoint(out_dir: Path, r: SecondOpinionResult) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{r.paper_id}__{r.ref_id}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(r), indent=2, ensure_ascii=False))
    tmp.replace(path)
    return path


def write_csv(results: list[SecondOpinionResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(results[0]).keys()) if results else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def write_markdown(results: list[SecondOpinionResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(results)
    n_agree = sum(1 for r in results if r.agrees)
    n_disagree = sum(1 for r in results if not r.agrees and not r.error)
    n_err = sum(1 for r in results if r.error)
    lines = [f"# Second-opinion review — {n} misrep cases\n"]
    lines.append(
        f"Primary system flagged these as MISREP. A second LLM ({results[0].model if results else '?'}) "
        f"with full citing context + full cited paper agreed with the primary system on "
        f"**{n_agree}/{n - n_err}** cases ({(100 * n_agree / max(1, n - n_err)):.1f}%). "
        f"{n_disagree} cases the second LLM judged SUPPORTED (potential false positives). "
        f"{n_err} could not be evaluated (no full text available)."
    )
    lines.append("\n---\n")
    for i, r in enumerate(results, start=1):
        lines.append(f"## Case {i}/{n} — `{r.paper_id}` / `{r.ref_id}`")
        lines.append("")
        lines.append(f"- Cited: {r.cited_title or '?'}")
        if r.cited_doi:
            lines.append(f"- DOI: [{r.cited_doi}](https://doi.org/{r.cited_doi})")
        lines.append(f"- Cited text source: `{r.cited_text_source}` ({r.cited_text_chars} chars)")
        lines.append("")
        lines.append("### Citing sentence")
        lines.append("> " + (r.citing_sentence or "").strip().replace("\n", "\n> "))
        lines.append("")
        lines.append("### Primary system (gpt-4o-mini, retrieval-only) verdict")
        lines.append(f"**`{r.primary_verdict}`** — {r.primary_explanation.strip()}")
        lines.append("")
        lines.append(f"### Second-opinion ({r.model}, full cited paper) verdict")
        lines.append(f"**`{r.secondopinion_verdict}`** ({r.secondopinion_confidence} confidence)")
        lines.append("")
        lines.append(r.secondopinion_reasoning.strip())
        lines.append("")
        if r.error:
            lines.append(f"_Error: {r.error}_\n")
        lines.append(
            "**Agree with primary?** " + ("✅ yes" if r.agrees else ("❌ no" if not r.error else "—"))
        )
        lines.append("")
        lines.append("---")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


async def main_async(args: argparse.Namespace) -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY is not set")
        return 1

    cases = load_misrep_cases(args.per_paper_dir)
    log.info("loaded %d misrep cases", len(cases))
    if args.limit:
        cases = cases[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cache = APICache()
    httpx_client = httpx.AsyncClient(follow_redirects=True, timeout=60)
    openai_client = AsyncOpenAI()

    results: list[SecondOpinionResult] = []
    total_cost = 0.0
    try:
        for i, case in enumerate(cases, start=1):
            paper_id = case["paper_id"]
            ref_id = case["ref_id"]
            ckpt = args.out_dir / f"{paper_id}__{ref_id}.json"
            if ckpt.exists() and not args.force:
                log.info("[%d/%d] skip — checkpoint exists: %s", i, len(cases), ckpt.name)
                results.append(SecondOpinionResult(**json.loads(ckpt.read_text())))
                continue
            t0 = time.time()
            r = await process_case(case, args, httpx_client, cache, openai_client)
            write_per_case_checkpoint(args.out_dir, r)
            results.append(r)
            total_cost += r.cost_usd
            log.info(
                "[%d/%d] %s/%s primary=%s SO=%s agree=%s cost=$%.4f t=%.1fs",
                i, len(cases), paper_id, ref_id,
                r.primary_verdict, r.secondopinion_verdict, r.agrees,
                r.cost_usd, time.time() - t0,
            )
    finally:
        await httpx_client.aclose()
        await cache.close()

    if results:
        write_csv(results, args.csv)
        write_markdown(results, args.md)
        n_agree = sum(1 for r in results if r.agrees)
        n_err = sum(1 for r in results if r.error)
        log.info(
            "DONE — %d cases, agreement: %d/%d (%.1f%%), errors: %d, total cost: $%.2f",
            len(results), n_agree, len(results) - n_err,
            100 * n_agree / max(1, len(results) - n_err), n_err, total_cost,
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-paper-dir", type=Path, default=DEFAULT_PER_PAPER_DIR)
    parser.add_argument("--papers-dir", type=Path, default=DEFAULT_PAPERS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model (default: gpt-4o)")
    parser.add_argument("--limit", type=int, help="only process first N cases (debug)")
    parser.add_argument("--force", action="store_true", help="re-run cases with existing checkpoints")
    parser.add_argument("--window", type=int, default=CITING_WINDOW_CHARS)
    parser.add_argument("--max-cited-chars", type=int, default=MAX_CITED_CHARS)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
