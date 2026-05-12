"""Run the CheckCitation pipeline on each paper in ``data/corpus_manifest.csv``"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

os.environ["SERPAPI_KEY"] = ""
os.environ["CITEEXTRACT_SKIP_OPENALEX"] = "1"
os.environ["CITEEXTRACT_SKIP_ARXIV"] = "1"
os.environ["CITEEXTRACT_SKIP_OPENREVIEW"] = "1"

_THIS_FILE = Path(__file__).resolve()
_PREV_DIR = _THIS_FILE.parent
_REPO_ROOT = _PREV_DIR.parent.parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env", override=False)
except Exception:
    pass

from citeextract.models.comprehension import ComprehensionResult
from citeextract.pipeline import run_unified_pipeline

DEFAULT_MANIFEST = _PREV_DIR / "data" / "corpus_manifest.csv"
DEFAULT_RESULTS_DIR = _PREV_DIR / "results"
DEFAULT_PER_PAPER_DIR = DEFAULT_RESULTS_DIR / "per_paper"
DEFAULT_SUMMARY = DEFAULT_RESULTS_DIR / "run_summary.json"

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_prevalence")


@dataclass
class CitationRecord:

    paper_id: str
    paper_decision: str
    ref_id: str
    citing_sentence: str
    context_before: str
    context_after: str
    cited_title: Optional[str]
    cited_doi: Optional[str]
    cited_year: Optional[int]
    cited_source: Optional[str]
    paper_found: bool
    full_text_available: bool
    full_text_source: Optional[str]
    verdict: Optional[str]
    explanation: Optional[str]
    evidence_quote: Optional[str]
    top_passage_text: Optional[str]
    top_passage_score: Optional[float]
    metadata_verdict: Optional[str]
    metadata_flags: Optional[list[str]]
    metadata_explanation: Optional[str]
    top_level_verdict: Optional[str]
    top_level_flags: Optional[list[str]]
    retraction_status: Optional[bool]


def load_manifest(path: Path) -> list[dict]:
    with path.open("r") as f:
        return list(csv.DictReader(f))


def comp_result_to_record(
    paper_id: str,
    paper_decision: str,
    r: ComprehensionResult,
    verdict_map: dict,
) -> CitationRecord:
    md = r.paper_metadata or {}
    cv = r.claim_verdict
    top = r.top_passages[0] if r.top_passages else None
    pv = verdict_map.get(r.ref_id)
    retr = None
    if pv is not None and getattr(pv, "existence", None) is not None:
        retr = getattr(pv.existence, "retraction_status", None)
    return CitationRecord(
        paper_id=paper_id,
        paper_decision=paper_decision,
        ref_id=r.ref_id,
        citing_sentence=r.citing_sentence,
        context_before=r.context_before or "",
        context_after=r.context_after or "",
        cited_title=md.get("title"),
        cited_doi=md.get("doi"),
        cited_year=md.get("year"),
        cited_source=md.get("source"),
        paper_found=r.paper_found,
        full_text_available=r.full_text_available,
        full_text_source=r.full_text_source,
        verdict=cv.verdict if cv else None,
        explanation=cv.explanation if cv else None,
        evidence_quote=cv.evidence_quote if cv else None,
        top_passage_text=top.chunk.text if top else None,
        top_passage_score=top.bm25_score if top else None,
        metadata_verdict=getattr(pv, "metadata_verdict", None) if pv else None,
        metadata_flags=list(getattr(pv, "metadata_flags", []) or []) if pv else None,
        metadata_explanation=getattr(pv, "metadata_explanation", None) if pv else None,
        top_level_verdict=getattr(pv, "verdict", None) if pv else None,
        top_level_flags=list(getattr(pv, "flags", []) or []) if pv else None,
        retraction_status=retr,
    )


def atomic_write_jsonl(path: Path, records: list[CitationRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    tmp.replace(path)


def extract_cost(paper_report) -> float:
    if paper_report is None:
        return 0.0
    summary = getattr(paper_report, "summary", None)
    if summary is None:
        return 0.0
    return float(getattr(summary, "total_cost_usd", 0.0) or 0.0)


async def process_paper(
    paper_row: dict,
    per_paper_dir: Path,
    force: bool,
) -> dict:
    paper_id = paper_row["paper_id"]
    pdf_rel = paper_row["pdf_path"]
    pdf_abs = _REPO_ROOT / pdf_rel
    out_path = per_paper_dir / f"{paper_id}.jsonl"

    if out_path.exists() and not force:
        n = sum(1 for _ in out_path.open())
        log.info("[%s] skip — checkpoint exists (%d records)", paper_id, n)
        return {"paper_id": paper_id, "status": "skipped", "n_citations": n}

    if not pdf_abs.exists():
        log.warning("[%s] pdf missing: %s", paper_id, pdf_abs)
        return {"paper_id": paper_id, "status": "missing_pdf", "pdf_path": pdf_rel}

    t0 = time.time()
    try:
        paper_report, comp_report, _parsed = await run_unified_pipeline(
            file_path=str(pdf_abs),
            mode="agentic",
            run_verification=True,
            run_comprehension=True,
        )
    except Exception as exc:
        log.exception("[%s] pipeline failed", paper_id)
        return {
            "paper_id": paper_id,
            "status": "failed",
            "error": repr(exc),
            "elapsed_s": round(time.time() - t0, 1),
        }

    if comp_report is None:
        log.error("[%s] no comp_report returned", paper_id)
        return {"paper_id": paper_id, "status": "no_report"}

    decision = paper_row.get("decision", "")
    verdict_map = {v.ref_id: v for v in (paper_report.verdicts or [])} if paper_report else {}
    records = [
        comp_result_to_record(paper_id, decision, r, verdict_map)
        for r in comp_report.results
    ]
    atomic_write_jsonl(out_path, records)

    coverage = comp_report.coverage or {}
    n_full = coverage.get("full_text", 0)
    n_abs = coverage.get("abstract_only", 0)
    n_none = coverage.get("not_found", 0)
    n_records = len(records)
    n_supported = sum(1 for r in records if r.verdict in ("SUPPORTED", "SUPPORTS"))
    n_misrep = sum(1 for r in records if r.verdict in ("NOT_SUPPORTED", "CONTRADICTS"))
    n_neutral = sum(1 for r in records if r.verdict == "NEUTRAL")
    n_no_verdict = sum(1 for r in records if r.verdict is None)
    cost = extract_cost(paper_report)

    elapsed = time.time() - t0
    log.info(
        "[%s] done in %.1fs | refs=%d full=%d abs=%d none=%d | "
        "sup=%d misrep=%d neutral=%d no_verdict=%d | $%.4f",
        paper_id, elapsed, n_records, n_full, n_abs, n_none,
        n_supported, n_misrep, n_neutral, n_no_verdict, cost,
    )

    return {
        "paper_id": paper_id,
        "status": "ok",
        "n_citations": n_records,
        "coverage": {"full_text": n_full, "abstract_only": n_abs, "not_found": n_none},
        "verdicts": {
            "supported": n_supported,
            "misrep": n_misrep,
            "neutral": n_neutral,
            "none": n_no_verdict,
        },
        "cost_usd": cost,
        "elapsed_s": round(elapsed, 1),
    }


def write_summary(path: Path, summaries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_papers": len(summaries),
        "n_ok": sum(1 for s in summaries if s["status"] == "ok"),
        "n_skipped": sum(1 for s in summaries if s["status"] == "skipped"),
        "n_failed": sum(1 for s in summaries if s["status"] == "failed"),
        "total_cost_usd": round(
            sum(s.get("cost_usd", 0.0) or 0.0 for s in summaries), 6,
        ),
        "results": summaries,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


async def main_async(args: argparse.Namespace) -> int:
    if not args.manifest.exists():
        log.error("manifest not found: %s", args.manifest)
        return 1

    manifest = load_manifest(args.manifest)
    if args.paper_id:
        manifest = [r for r in manifest if r["paper_id"] == args.paper_id]
        if not manifest:
            log.error("paper_id %s not in manifest", args.paper_id)
            return 1
    if args.limit:
        manifest = manifest[: args.limit]

    log.info("processing %d papers", len(manifest))
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.per_paper_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    for i, row in enumerate(manifest, start=1):
        log.info("[%d/%d] paper_id=%s", i, len(manifest), row["paper_id"])
        summary = await process_paper(row, args.per_paper_dir, force=args.force)
        summaries.append(summary)
        write_summary(args.summary, summaries)

    log.info("done — summary at %s", args.summary)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--per-paper-dir", type=Path, default=DEFAULT_PER_PAPER_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--paper-id", help="run only this single paper (for debugging)")
    parser.add_argument("--limit", type=int, help="run only the first N papers from the manifest")
    parser.add_argument("--force", action="store_true", help="re-run papers even if a checkpoint exists")
    args = parser.parse_args()

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
