"""Collect and persist all data needed for benchmark evaluation.

Searches for each cited paper, fetches full text, retrieves passages,
and saves everything to disk as JSONL. This data is then consumed by
run_benchmark.py (and any future external baseline scripts) without
any additional API calls for paper retrieval.

Supports RESUME: if benchmark_enriched.jsonl already exists, skips
rows that have already been processed.

Usage:
    python baselines/collect_benchmark_data.py

Settings — edit the variables below:
"""

# ── Settings ─────────────────────────────────────────────────────────────────
SAMPLE_SIZE = None                  # None = all 1451, or int for quick test
BATCH_SIZE = 5                      # Concurrent paper lookups per batch
BATCH_DELAY = 3.0                   # Seconds between batches (rate limiting)
RESUME = True                       # Skip already-processed rows
BENCHMARK_CSV = "benchmark_b_unified.csv"
OUTPUT_DIR = "benchmark_data"
OUTPUT_FILE = "benchmark_enriched.jsonl"
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import csv
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import httpx

from src import config as app_config
from src.models.verdict import ExistenceResult
from src.verification.api_clients import semantic_scholar, openalex
from src.verification.api_clients.fulltext import get_full_text
from src.verification.cache import APICache
from src.verification.comprehension import (
    build_retrieval_query,
    chunk_text,
    retrieve_passages_bm25,
    retrieve_passages_hybrid,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Load benchmark ───────────────────────────────────────────────────────────

def load_benchmark(csv_path: Path, limit: int | None = None) -> list[dict]:
    """Load benchmark CSV. Returns list of row dicts."""
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if limit is not None:
        rows = rows[:limit]
    log.info(f"Loaded {len(rows)} instances from {csv_path.name}")
    return rows


def load_already_processed(output_path: Path) -> set[int]:
    """Load indices of already-processed rows from existing JSONL."""
    processed = set()
    if not output_path.exists():
        return processed
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                processed.add(record["idx"])
            except (json.JSONDecodeError, KeyError):
                continue
    log.info(f"Resume: {len(processed)} rows already processed")
    return processed


# ── Paper search ─────────────────────────────────────────────────────────────

async def find_paper(
    title: str,
    client: httpx.AsyncClient,
) -> dict | None:
    """Search for a paper by title. Cascade: S2 → OpenAlex."""
    result = await semantic_scholar.search_by_title(title, client)
    if result:
        result["_search_source"] = "semantic_scholar"
        return result

    result = await openalex.search_by_title(title, client)
    if result:
        result["_search_source"] = "openalex"
        return result

    return None


def build_existence_result(paper: dict, ref_id: str, benchmark_abstract: str) -> ExistenceResult:
    """Build ExistenceResult from search result for fulltext fetcher."""
    oa_url = None
    oa_pdf = paper.get("openAccessPdf")
    if isinstance(oa_pdf, dict):
        oa_url = oa_pdf.get("url")
    elif isinstance(oa_pdf, str):
        oa_url = oa_pdf
    if not oa_url:
        oa_url = paper.get("oa_url")

    abstract = paper.get("abstract") or benchmark_abstract or None

    return ExistenceResult(
        ref_id=ref_id,
        status="FOUND",
        source=paper.get("_search_source", "semantic_scholar"),
        matched_title=paper.get("title", ""),
        matched_authors=paper.get("authors", []),
        matched_year=paper.get("year"),
        matched_venue=paper.get("venue", ""),
        matched_doi=paper.get("doi"),
        matched_arxiv_id=paper.get("arxiv_id"),
        oa_url=oa_url,
        abstract=abstract,
        databases_checked=[paper.get("_search_source", "unknown")],
    )


# ── Process single instance ──────────────────────────────────────────────────

# ── Paper cache (de-duplicate by title) ──────────────────────────────────────
# Sarol has 517 instances but only 20 unique papers. Without this cache we'd
# search + fetch the same paper ~26 times. The cache stores the search result,
# full text, and pre-chunked chunks per unique title.

_paper_cache: dict[str, dict] = {}  # normalized_title -> {search_result, fulltext, chunks}


def _normalize_title_key(title: str) -> str:
    """Normalize title for cache key (lowercase, strip punctuation)."""
    import re
    return re.sub(r"[^\w\s]", "", title.lower()).strip()


async def _get_or_fetch_paper(
    cited_title: str,
    cited_abstract: str,
    idx: int,
    http_client: httpx.AsyncClient,
    cache: APICache,
) -> dict | None:
    """Search and fetch a paper, using cache if already seen.

    Returns dict with keys: search_result, fulltext, chunks — or None if not found.
    """
    key = _normalize_title_key(cited_title)
    if key in _paper_cache:
        log.debug(f"[{idx}] Paper cache hit: {cited_title[:60]}")
        return _paper_cache[key]

    # Search
    paper = await find_paper(cited_title, http_client)
    if paper is None:
        _paper_cache[key] = None
        return None

    found_abstract = paper.get("abstract") or cited_abstract or ""

    search_result = {
        "found": True,
        "search_source": paper.get("_search_source", ""),
        "found_title": paper.get("title", ""),
        "found_doi": paper.get("doi") or "",
        "found_arxiv_id": paper.get("arxiv_id") or "",
        "found_abstract": found_abstract,
        "oa_url": "",
    }
    oa_pdf = paper.get("openAccessPdf")
    if isinstance(oa_pdf, dict):
        search_result["oa_url"] = oa_pdf.get("url", "")
    elif isinstance(oa_pdf, str):
        search_result["oa_url"] = oa_pdf
    if not search_result["oa_url"]:
        search_result["oa_url"] = paper.get("oa_url", "")

    # Fetch full text
    exist = build_existence_result(paper, f"bench_{idx}", cited_abstract)
    try:
        ft = await get_full_text(exist, http_client, cache)
    except Exception as e:
        log.warning(f"[{idx}] Full text fetch failed: {e}")
        entry = {"search_result": search_result, "fulltext": None, "chunks": None}
        _paper_cache[key] = entry
        return entry

    full_text = ft.full_text or ""
    abstract_text = ft.abstract or found_abstract or ""
    sections = ft.sections or []

    fulltext_data = {
        "available": bool(full_text),
        "source": ft.source,
        "text": full_text,
        "sections": sections,
        "abstract": abstract_text,
        "char_count": len(full_text),
    }

    # Pre-chunk (reused across all citing sentences for this paper)
    chunks = None
    text_for_chunking = full_text or abstract_text
    if text_for_chunking.strip():
        sections_for_chunking = sections if sections else None
        chunks = chunk_text(text_for_chunking, sections=sections_for_chunking)

    entry = {"search_result": search_result, "fulltext": fulltext_data, "chunks": chunks}
    _paper_cache[key] = entry
    return entry


async def process_one(
    idx: int,
    row: dict,
    http_client: httpx.AsyncClient,
    cache: APICache,
    comp_cfg: dict,
) -> dict:
    """Process a single benchmark instance: search, fetch, chunk, retrieve."""
    citing_sentence = row.get("citing_sentence", "").strip()
    cited_title = row.get("cited_paper_title", "").strip()
    cited_abstract = row.get("cited_paper_abstract", "").strip()

    record = {
        "idx": idx,
        "source": row.get("source", ""),
        "citing_sentence": citing_sentence,
        "cited_paper_title": cited_title,
        "cited_paper_abstract": cited_abstract,
        "label": row.get("label", ""),
        "original_label": row.get("original_label", ""),
        "search_result": None,
        "fulltext": None,
        "passages": [],
        "skip_reason": None,
        "evaluable": False,
    }

    # ── Step 1+2: Search + fetch (cached per unique title) ───────────────
    if not cited_title:
        record["skip_reason"] = "no_title"
        return record

    paper_data = await _get_or_fetch_paper(
        cited_title, cited_abstract, idx, http_client, cache,
    )
    if paper_data is None:
        record["skip_reason"] = "paper_not_found"
        return record

    record["search_result"] = paper_data["search_result"]

    if paper_data["fulltext"] is None:
        record["skip_reason"] = "fulltext_error"
        return record

    record["fulltext"] = paper_data["fulltext"]
    ft = paper_data["fulltext"]
    text_source = ft["source"]

    if text_source == "not_found" or (not ft["text"] and not ft["abstract"]):
        record["skip_reason"] = "no_text_available"
        return record

    if not ft["text"] and text_source == "abstract_only":
        record["skip_reason"] = "abstract_only"
        record["evaluable"] = False
        return record

    # ── Step 3: Retrieve passages (per citing sentence — not cached) ─────
    chunks = paper_data["chunks"]
    passage_list = []
    if chunks and citing_sentence:
        top_k = comp_cfg.get("top_k", 3)
        dense_model = comp_cfg.get("dense_model")
        query = build_retrieval_query(citing_sentence)

        try:
            if dense_model:
                scored = retrieve_passages_hybrid(
                    query, chunks, top_k=top_k,
                    model_name=dense_model,
                    bm25_candidates=comp_cfg.get("bm25_candidates", 10),
                    dense_candidates=comp_cfg.get("dense_candidates", 10),
                    rrf_k=comp_cfg.get("rrf_k", 60),
                )
            else:
                scored = retrieve_passages_bm25(query, chunks, top_k=top_k)

            for sc in scored:
                passage_list.append({
                    "text": sc.chunk.text,
                    "section": sc.chunk.section_name,
                    "bm25_score": round(sc.bm25_score, 4),
                    "dense_score": round(sc.dense_score, 4) if sc.dense_score is not None else None,
                    "rrf_score": round(sc.rrf_score, 4) if sc.rrf_score is not None else None,
                })
        except Exception as e:
            log.warning(f"[{idx}] Passage retrieval failed: {e}")

    record["passages"] = passage_list
    record["evaluable"] = True
    return record


# ── Batch processing ─────────────────────────────────────────────────────────

async def process_batch(
    batch: list[tuple[int, dict]],
    http_client: httpx.AsyncClient,
    cache: APICache,
    comp_cfg: dict,
) -> list[dict]:
    """Process a batch of instances concurrently."""
    tasks = [
        process_one(idx, row, http_client, cache, comp_cfg)
        for idx, row in batch
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed = []
    for (idx, row), result in zip(batch, results):
        if isinstance(result, Exception):
            log.error(f"[{idx}] Exception: {result}")
            processed.append({
                "idx": idx,
                "source": row.get("source", ""),
                "citing_sentence": row.get("citing_sentence", ""),
                "cited_paper_title": row.get("cited_paper_title", ""),
                "cited_paper_abstract": row.get("cited_paper_abstract", ""),
                "label": row.get("label", ""),
                "original_label": row.get("original_label", ""),
                "search_result": None,
                "fulltext": None,
                "passages": [],
                "skip_reason": f"exception: {str(result)[:200]}",
                "evaluable": False,
            })
        else:
            processed.append(result)
    return processed


# ── Main ─────────────────────────────────────────────────────────────────────

async def main():
    start_time = time.time()

    script_dir = Path(__file__).resolve().parent
    csv_path = script_dir / BENCHMARK_CSV
    out_dir = script_dir / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / OUTPUT_FILE

    # Load benchmark
    rows = load_benchmark(csv_path, SAMPLE_SIZE)
    if not rows:
        log.error("No instances loaded.")
        return

    # Resume support
    already_processed = set()
    if RESUME:
        already_processed = load_already_processed(output_path)

    # Filter to unprocessed rows
    indexed_rows = [
        (idx, row) for idx, row in enumerate(rows)
        if idx not in already_processed
    ]
    log.info(f"{len(indexed_rows)} rows to process ({len(already_processed)} already done)")

    if not indexed_rows:
        log.info("All rows already processed. Nothing to do.")
        _print_summary(output_path)
        return

    # Config
    comp_cfg = app_config.comprehension()
    cache = APICache()

    async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as http_client:
        total_batches = (len(indexed_rows) + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_num, start in enumerate(range(0, len(indexed_rows), BATCH_SIZE), 1):
            batch = indexed_rows[start : start + BATCH_SIZE]
            indices = [idx for idx, _ in batch]
            log.info(
                f"Batch {batch_num}/{total_batches} — "
                f"indices {indices[0]}-{indices[-1]} "
                f"({len(batch)} instances)"
            )

            results = await process_batch(batch, http_client, cache, comp_cfg)

            # Append results to JSONL immediately (crash-safe)
            with open(output_path, "a", encoding="utf-8") as f:
                for record in results:
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

            # Log progress
            evaluable = sum(1 for r in results if r.get("evaluable"))
            skipped = sum(1 for r in results if r.get("skip_reason"))
            log.info(f"  → {evaluable} evaluable, {skipped} skipped")

            # Rate limit delay
            if start + BATCH_SIZE < len(indexed_rows):
                await asyncio.sleep(BATCH_DELAY)

    await cache.close()

    elapsed = round(time.time() - start_time, 1)
    log.info(f"\nCollection complete in {elapsed}s")
    log.info(f"Output: {output_path}")

    _print_summary(output_path)


def _print_summary(output_path: Path):
    """Print summary statistics of the collected data."""
    records = []
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total = len(records)
    evaluable = sum(1 for r in records if r.get("evaluable"))
    skipped = sum(1 for r in records if r.get("skip_reason"))

    skip_reasons = Counter(r.get("skip_reason", "") for r in records if r.get("skip_reason"))
    sources_eval = Counter(r.get("source", "") for r in records if r.get("evaluable"))
    labels_eval = Counter(r.get("label", "") for r in records if r.get("evaluable"))
    text_sources = Counter(
        r.get("fulltext", {}).get("source", "") for r in records
        if r.get("evaluable") and r.get("fulltext")
    )

    print(f"\n{'=' * 70}")
    print(f"  BENCHMARK DATA COLLECTION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total records:    {total}")
    print(f"  Evaluable:        {evaluable} ({evaluable/total*100:.1f}%)")
    print(f"  Skipped:          {skipped}")

    if skip_reasons:
        print(f"\n  Skip reasons:")
        for reason, count in skip_reasons.most_common():
            print(f"    {reason}: {count}")

    if labels_eval:
        print(f"\n  Evaluable label distribution:")
        for label, count in sorted(labels_eval.items()):
            print(f"    {label}: {count} ({count/evaluable*100:.1f}%)")

    if sources_eval:
        print(f"\n  Evaluable by dataset source:")
        for src, count in sorted(sources_eval.items()):
            print(f"    {src}: {count}")

    if text_sources:
        print(f"\n  Full text sources:")
        for src, count in text_sources.most_common():
            print(f"    {src}: {count}")

    # Estimate file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n  File size: {size_mb:.1f} MB")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
