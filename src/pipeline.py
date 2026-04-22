"""Main pipeline orchestrator — wires L1 through L6.

Quick mode (rule-based): L1 → L2 → L3 → L5 → L6 (existence + metadata, no LLM)
Agentic mode:            L1 → L2 → L3 → Triage → focused agents where needed
Comprehension mode:      L1 → L2 → fulltext → chunk → BM25 retrieve → report
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from src import config
from src.classification.classifier import CitationVerdict, classify_quick
from src.models.parsed_paper import ParsedPaper
from src.models.report import PaperReport
from src.models.verdict import ExistenceResult
from src.parsers.router import parse_file
from src.report.generator import build_report, save_json
from src.utils.timing import stage
from src.verification.cache import APICache
from src.verification.existence import check_all_references
from src.verification.matching import WEB_SOURCES
from src.verification.metadata import MetadataResult, validate_metadata

log = logging.getLogger(__name__)


def _build_metadata_map(
    references: list,
    exist_map: dict[str, ExistenceResult],
) -> dict[str, MetadataResult]:
    """Build metadata validation results for all FOUND references."""
    metadata_map: dict[str, MetadataResult] = {}
    for ref in references:
        exist = exist_map.get(ref.ref_id)
        if exist and exist.status == "FOUND" and exist.source not in WEB_SOURCES:
            metadata_map[ref.ref_id] = validate_metadata(ref, exist)
    return metadata_map


def _run_quick_verification(
    references: list,
    exist_map: dict[str, ExistenceResult],
    mode: str = "quick",
) -> list[CitationVerdict]:
    """Shared L3+L5: metadata validation + classification using a pre-computed exist_map."""
    metadata_map = _build_metadata_map(references, exist_map)

    verdicts: list[CitationVerdict] = []
    for ref in references:
        exist = exist_map.get(ref.ref_id)
        if exist is None:
            log.warning(f"No existence result for {ref.ref_id}, marking NOT_FOUND")
            exist = ExistenceResult(
                ref_id=ref.ref_id, status="NOT_FOUND",
                databases_checked=[], flags=["missing_existence_result"],
            )
        meta = metadata_map.get(ref.ref_id)
        verdict = classify_quick(exist, meta)
        verdict.mode = mode
        verdicts.append(verdict)
    return verdicts


async def run_pipeline(
    file_path: str,
    mode: str = "quick",
    retry_failed: bool = False,
) -> PaperReport:
    """Run the verification pipeline (existence + metadata).

    Delegates to run_unified_pipeline for consistency.
    """
    report, _, _ = await run_unified_pipeline(
        file_path, mode=mode, run_verification=True, retry_failed=retry_failed,
    )
    return report




async def _run_agentic(
    parsed: ParsedPaper,
    exist_map: dict[str, ExistenceResult],
    file_path: str,
    ref_pdfs_dir: Optional[str] = None,
) -> tuple[PaperReport, dict[str, dict[str, list]], dict]:
    """Run agentic verification: rule-based triage + focused agents.

    Uses shared L2 exist_map and L3 metadata. Clear-cut cases are resolved
    by rules. Ambiguous cases get dispatched to MetadataAgent / ClaimAgent.

    Returns the ``PaperReport`` along with the already-retrieved passages and
    fulltexts so the caller can synthesize a ``ComprehensionReport`` without
    running the comprehension pipeline a second time.
    """
    from src.verification.agentic.runner import run_agentic_verification

    agentic_cfg = config.agentic()
    if not agentic_cfg:
        raise ValueError(
            "No agentic config found in config.yaml. "
            "Add an 'agentic' section with at least a model name."
        )

    # L3: Metadata validation for FOUND refs
    metadata_map = _build_metadata_map(parsed.references, exist_map)

    verdicts, llm_cost, passages_by_ref, fulltext_by_ref = await run_agentic_verification(
        parsed, exist_map, metadata_map, agentic_cfg, ref_pdfs_dir=ref_pdfs_dir,
    )

    report = build_report(parsed, verdicts, "agentic", input_file=file_path)

    if llm_cost > 0:
        report.warnings.append(f"LLM cost (agentic): ${llm_cost:.4f}")

    # Count how many references used agents vs rule-based
    agent_count = sum(1 for v in verdicts if "agentic_agent_used" in v.flags)
    clear_count = len(verdicts) - agent_count
    if clear_count > 0:
        report.warnings.append(
            f"Agentic mode: {clear_count}/{len(verdicts)} references resolved "
            f"by rule-based triage (no LLM needed)."
        )

    return report, passages_by_ref, fulltext_by_ref



def run_and_save(
    file_path: str,
    mode: str = "quick",
    output_dir: str = "data/output",
    retry_failed: bool = False,
) -> PaperReport:
    """Sync wrapper: run pipeline and save JSON report."""
    report = asyncio.run(run_pipeline(file_path, mode, retry_failed=retry_failed))

    p = Path(file_path)
    name = f"{p.stem}_{p.suffix.lstrip('.')}" if p.suffix else p.stem
    save_json(report, f"{output_dir}/{name}_{report.mode}_report.json")

    return report


# ---------------------------------------------------------------------------
# Comprehension pipeline
# ---------------------------------------------------------------------------


async def run_comprehension_pipeline(
    file_path: str,
    ref_pdfs_dir: Optional[str] = None,
    retry_failed: bool = False,
) -> "ComprehensionReport":
    """Run the comprehension support pipeline.

    L1 → L2 → full text retrieval → chunking → BM25 passage retrieval.

    For each (citing_sentence, reference) pair, retrieves the most relevant
    passage from the cited paper's full text.

    Args:
        file_path: Path to input file (.pdf, .tex, .bib, .txt).
        ref_pdfs_dir: Optional directory containing PDFs of cited papers.
            Files should be named by DOI, title, or ref_id.

    Returns:
        ComprehensionReport with passage retrieval results and coverage stats.
    """
    import json as _json
    from collections import defaultdict
    from datetime import datetime, timezone

    import httpx

    from src.models.comprehension import ComprehensionReport, ComprehensionResult, ScoredChunk
    from src.verification.api_clients.fulltext import get_full_text
    from src.verification.comprehension import (
        build_retrieval_query,
        chunk_text, retrieve_passages_bm25, retrieve_passages_hybrid,
    )
    from src.verification.filters import is_substantive_citation

    comp_cfg = config.comprehension()
    top_k = comp_cfg["top_k"]
    dense_model = comp_cfg.get("dense_model")
    use_hybrid = bool(dense_model)

    # L1: Parse
    parsed = parse_file(file_path)

    # L2: Existence check (reuse existing)
    existence_results = await check_all_references(parsed.references, retry_failed=retry_failed)
    exist_map = {r.ref_id: r for r in existence_results}

    # Group citations by reference (filter trivial ones like "[1, 3].")
    cit_by_ref: dict[str, list] = defaultdict(list)
    for cit in parsed.citations:
        if is_substantive_citation(cit):
            cit_by_ref[cit.ref_id].append(cit)

    # Resolve user-uploaded PDFs directory
    user_pdf_map: dict[str, str] = {}
    if ref_pdfs_dir:
        user_pdf_map = _scan_ref_pdfs(ref_pdfs_dir, parsed.references)

    # Process each (citation, reference) pair
    cache = APICache()
    results: list[ComprehensionResult] = []
    coverage = {"full_text": 0, "abstract_only": 0, "not_found": 0}

    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            for ref in parsed.references:
                exist = exist_map.get(ref.ref_id)
                if exist is None:
                    continue

                # Get full text of cited paper
                user_pdf = user_pdf_map.get(ref.ref_id)
                ft = await get_full_text(exist, client, cache, user_pdf_path=user_pdf)

                # Track coverage
                if ft.full_text:
                    coverage["full_text"] += 1
                elif ft.abstract:
                    coverage["abstract_only"] += 1
                else:
                    coverage["not_found"] += 1

                # Determine text to chunk
                text_to_chunk = ft.full_text or ft.abstract or ""
                if not text_to_chunk.strip():
                    # No text at all — create empty results for each citation
                    for cit in cit_by_ref.get(ref.ref_id, []):
                        results.append(ComprehensionResult(
                            ref_id=ref.ref_id,
                            citing_sentence=cit.citing_sentence,
                            context_before=cit.context_before or "",
                            context_after=cit.context_after or "",
                            paper_found=exist.status == "FOUND",
                            full_text_available=False,
                            full_text_source=ft.source,
                            top_passages=[],
                            paper_metadata=_paper_metadata(exist),
                        ))
                    continue

                # Chunk and retrieve
                sections = ft.sections if ft.sections else None
                chunks = chunk_text(text_to_chunk, sections=sections)

                # If only abstract available and very few chunks, return directly
                # (retrieval scoring is meaningless on 1-2 chunks)
                is_abstract_only = ft.source == "abstract_only" or not ft.full_text

                for cit in cit_by_ref.get(ref.ref_id, []):
                    if is_abstract_only and len(chunks) <= 2:
                        # Return all chunks directly without retrieval scoring
                        direct_passages = [ScoredChunk(chunk=c, bm25_score=1.0) for c in chunks]
                        results.append(ComprehensionResult(
                            ref_id=ref.ref_id,
                            citing_sentence=cit.citing_sentence,
                            context_before=cit.context_before or "",
                            context_after=cit.context_after or "",
                            paper_found=True,
                            full_text_available=False,
                            full_text_source=ft.source,
                            top_passages=direct_passages,
                            paper_metadata=_paper_metadata(exist),
                        ))
                        continue
                    # Normal retrieval path — use full context for better recall
                    # Strip citation markers from retrieval query for cleaner matching
                    query = build_retrieval_query(
                        cit.citing_sentence,
                        cit.context_before or "",
                        cit.context_after or "",
                        markers=[cit.marker] if cit.marker else None,
                    )
                    if use_hybrid:
                        passages = retrieve_passages_hybrid(
                            query, chunks, top_k=top_k,
                            model_name=dense_model,
                            bm25_candidates=comp_cfg.get("bm25_candidates", 10),
                            dense_candidates=comp_cfg.get("dense_candidates", 10),
                            rrf_k=comp_cfg.get("rrf_k", 60),
                        )
                    else:
                        passages = retrieve_passages_bm25(
                            query, chunks, top_k=top_k
                        )
                    results.append(ComprehensionResult(
                        ref_id=ref.ref_id,
                        citing_sentence=cit.citing_sentence,
                        context_before=cit.context_before or "",
                        context_after=cit.context_after or "",
                        paper_found=True,
                        full_text_available=bool(ft.full_text),
                        full_text_source=ft.source,
                        top_passages=passages,
                        paper_metadata=_paper_metadata(exist),
                    ))

                # References with no citations still get tracked
                if ref.ref_id not in cit_by_ref:
                    results.append(ComprehensionResult(
                        ref_id=ref.ref_id,
                        citing_sentence="",
                        paper_found=exist.status == "FOUND",
                        full_text_available=bool(ft.full_text),
                        full_text_source=ft.source,
                        top_passages=[],
                        paper_metadata=_paper_metadata(exist),
                    ))
    finally:
        await cache.close()

    return ComprehensionReport(
        input_file=file_path,
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_citations=len(results),
        results=results,
        coverage=coverage,
    )


def _paper_metadata(exist: ExistenceResult) -> dict:
    """Extract paper metadata from an ExistenceResult."""
    return {
        "title": exist.matched_title or "",
        "authors": exist.matched_authors,
        "year": exist.matched_year,
        "doi": exist.matched_doi,
        "source": exist.source,
    }


def _build_comprehension_from_passages(
    parsed: ParsedPaper,
    exist_map: dict[str, ExistenceResult],
    passages_by_ref: dict[str, dict[str, list]],
    fulltext_by_ref: dict,
    file_path: str,
) -> "ComprehensionReport":
    """Build a ComprehensionReport from data the agentic pipeline already has.

    Agentic's ``_pre_retrieve_passages`` fetches full text, chunks it, and
    retrieves passages for every substantive citing sentence. This function
    repackages that data as a ComprehensionReport so the UI can render passages
    without us re-running the comprehension pipeline (which would duplicate
    every fetch + retrieval for a 2x cost on wall time and API calls).

    Claim-level verdicts are intentionally left ``None`` here — in agentic
    mode the claim judgment lives on the ``CitationVerdict`` (flags +
    explanation) rather than on a per-sentence ClaimVerdict, and the UI
    sources that from the verdict card, not from the comp_report.
    """
    from datetime import datetime, timezone
    from src.models.comprehension import ComprehensionReport, ComprehensionResult
    from src.verification.filters import is_substantive_citation

    results: list[ComprehensionResult] = []
    coverage = {"full_text": 0, "abstract_only": 0, "not_found": 0}

    seen_refs: set[str] = set()

    for cit in parsed.citations:
        if not is_substantive_citation(cit):
            continue
        exist = exist_map.get(cit.ref_id)
        ft = fulltext_by_ref.get(cit.ref_id)
        passages = passages_by_ref.get(cit.ref_id, {}).get(cit.citing_sentence, [])

        if cit.ref_id not in seen_refs:
            seen_refs.add(cit.ref_id)
            if ft and ft.full_text:
                coverage["full_text"] += 1
            elif ft and ft.abstract:
                coverage["abstract_only"] += 1
            else:
                coverage["not_found"] += 1

        results.append(ComprehensionResult(
            ref_id=cit.ref_id,
            citing_sentence=cit.citing_sentence,
            context_before=cit.context_before or "",
            context_after=cit.context_after or "",
            paper_found=bool(exist and exist.status == "FOUND"),
            full_text_available=bool(ft and ft.full_text),
            full_text_source=ft.source if ft else None,
            top_passages=passages,
            claim_verdict=None,
            paper_metadata=_paper_metadata(exist) if exist else {},
        ))

    return ComprehensionReport(
        input_file=file_path,
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_citations=len(results),
        results=results,
        coverage=coverage,
    )


def _scan_ref_pdfs(
    ref_pdfs_dir: str, references: list,
) -> dict[str, str]:
    """Scan a directory for user-uploaded reference PDFs.

    Tries to match PDF filenames to references by DOI, title, or ref_id.
    Returns {ref_id: pdf_path} mapping.
    """
    from src.verification.matching import normalize_title, title_similarity

    pdf_dir = Path(ref_pdfs_dir)
    if not pdf_dir.is_dir():
        log.warning(f"ref-pdfs directory not found: {ref_pdfs_dir}")
        return {}

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        return {}

    result: dict[str, str] = {}
    for ref in references:
        for pdf in pdf_files:
            stem = pdf.stem.lower().replace("_", " ").replace("-", " ")

            # Match by ref_id
            if stem == ref.ref_id.lower():
                result[ref.ref_id] = str(pdf)
                break

            # Match by DOI (replace / with _)
            if ref.doi and ref.doi.lower().replace("/", "_") in stem:
                result[ref.ref_id] = str(pdf)
                break

            # Match by title similarity
            if ref.title:
                sim = title_similarity(ref.title, stem)
                if sim >= 0.70:
                    result[ref.ref_id] = str(pdf)
                    break

    if result:
        log.info(f"Matched {len(result)} user PDFs from {ref_pdfs_dir}")
    return result


def run_comprehension_and_save(
    file_path: str,
    ref_pdfs_dir: Optional[str] = None,
    output_dir: str = "data/output",
) -> "ComprehensionReport":
    """Sync wrapper: run comprehension pipeline and save JSON report."""
    import json as _json
    from src.models.comprehension import ComprehensionReport

    report = asyncio.run(run_comprehension_pipeline(file_path, ref_pdfs_dir))

    p = Path(file_path)
    name = f"{p.stem}_{p.suffix.lstrip('.')}" if p.suffix else p.stem
    out_path = Path(output_dir) / f"{name}_comprehension.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        _json.dumps(report.model_dump(), indent=2, default=str),
        encoding="utf-8",
    )

    return report


# ---------------------------------------------------------------------------
# Unified pipeline (shared L1+L2 for UI)
# ---------------------------------------------------------------------------


async def _run_comprehension_from_exist_map(
    parsed: ParsedPaper,
    exist_map: dict[str, ExistenceResult],
    file_path: str,
    ref_pdfs_dir: Optional[str] = None,
    verify_claims: bool = False,
) -> "ComprehensionReport":
    """Run comprehension using a pre-computed L2 exist_map.

    Args:
        parsed: L1 parse output.
        exist_map: Pre-computed existence results from L2.
        file_path: Input file path (for report metadata).
        ref_pdfs_dir: Optional directory with user-uploaded reference PDFs.
        verify_claims: If True, run LLM claim verification on retrieved passages.
    """
    from collections import defaultdict
    from datetime import datetime, timezone

    import httpx

    from src.models.comprehension import (
        ClaimVerdict, ComprehensionReport, ComprehensionResult, ScoredChunk,
    )
    from src.verification.api_clients.fulltext import get_full_text
    from src.verification.comprehension import (
        build_retrieval_query,
        chunk_text, retrieve_passages_bm25, retrieve_passages_hybrid,
        verify_claim_with_passages,
    )
    from src.verification.filters import is_substantive_citation

    comp_cfg = config.comprehension()
    top_k = comp_cfg["top_k"]
    dense_model = comp_cfg.get("dense_model")
    use_hybrid = bool(dense_model)

    # Set up LLM client for claim verification
    llm_client = None
    llm_cost_tracker = None
    if verify_claims:
        from src.verification.api_clients.llm_client import create_llm_client
        llm_cfg = config.claim_verification()
        try:
            llm_client = create_llm_client(llm_cfg)
            llm_cost_tracker = llm_client.cost_tracker
        except ValueError as e:
            log.warning(f"LLM client init failed: {e}. Skipping claim verification.")

    cit_by_ref: dict[str, list] = defaultdict(list)
    for cit in parsed.citations:
        if is_substantive_citation(cit):
            cit_by_ref[cit.ref_id].append(cit)

    user_pdf_map: dict[str, str] = {}
    if ref_pdfs_dir:
        user_pdf_map = _scan_ref_pdfs(ref_pdfs_dir, parsed.references)

    cache = APICache()
    results: list[ComprehensionResult] = []
    coverage = {"full_text": 0, "abstract_only": 0, "not_found": 0}

    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            for ref in parsed.references:
                exist = exist_map.get(ref.ref_id)
                if exist is None:
                    continue

                user_pdf = user_pdf_map.get(ref.ref_id)
                ft = await get_full_text(exist, client, cache, user_pdf_path=user_pdf)

                if ft.full_text:
                    coverage["full_text"] += 1
                elif ft.abstract:
                    coverage["abstract_only"] += 1
                else:
                    coverage["not_found"] += 1

                text_to_chunk = ft.full_text or ft.abstract or ""
                if not text_to_chunk.strip():
                    for cit in cit_by_ref.get(ref.ref_id, []):
                        results.append(ComprehensionResult(
                            ref_id=ref.ref_id,
                            citing_sentence=cit.citing_sentence,
                            context_before=cit.context_before or "",
                            context_after=cit.context_after or "",
                            paper_found=exist.status == "FOUND",
                            full_text_available=False,
                            full_text_source=ft.source,
                            top_passages=[],
                            paper_metadata=_paper_metadata(exist),
                        ))
                    continue

                sections = ft.sections if ft.sections else None
                chunks = chunk_text(text_to_chunk, sections=sections)
                is_abstract_only = ft.source == "abstract_only" or not ft.full_text

                for cit in cit_by_ref.get(ref.ref_id, []):
                    ctx_before = cit.context_before or ""
                    ctx_after = cit.context_after or ""

                    if is_abstract_only and len(chunks) <= 2:
                        direct_passages = [ScoredChunk(chunk=c, bm25_score=1.0) for c in chunks]
                        # Run claim verification on direct passages if enabled
                        claim = None
                        if llm_client and direct_passages:
                            cv = await verify_claim_with_passages(
                                cit.citing_sentence, direct_passages, llm_client,
                                ctx_before, ctx_after,
                            )
                            claim = ClaimVerdict(**cv)
                        results.append(ComprehensionResult(
                            ref_id=ref.ref_id,
                            citing_sentence=cit.citing_sentence,
                            context_before=ctx_before,
                            context_after=ctx_after,
                            paper_found=True,
                            full_text_available=False,
                            full_text_source=ft.source,
                            top_passages=direct_passages,
                            claim_verdict=claim,
                            paper_metadata=_paper_metadata(exist),
                        ))
                        continue
                    # Use full context for better retrieval recall
                    # Strip citation markers from retrieval query for cleaner matching
                    query = build_retrieval_query(
                        cit.citing_sentence, ctx_before, ctx_after,
                        markers=[cit.marker] if cit.marker else None,
                    )
                    if use_hybrid:
                        passages = retrieve_passages_hybrid(
                            query, chunks, top_k=top_k,
                            model_name=dense_model,
                            bm25_candidates=comp_cfg.get("bm25_candidates", 10),
                            dense_candidates=comp_cfg.get("dense_candidates", 10),
                            rrf_k=comp_cfg.get("rrf_k", 60),
                        )
                    else:
                        passages = retrieve_passages_bm25(
                            query, chunks, top_k=top_k
                        )
                    # Run claim verification on retrieved passages if enabled
                    claim = None
                    if llm_client and passages:
                        cv = await verify_claim_with_passages(
                            cit.citing_sentence, passages, llm_client,
                            ctx_before, ctx_after,
                        )
                        claim = ClaimVerdict(**cv)
                    results.append(ComprehensionResult(
                        ref_id=ref.ref_id,
                        citing_sentence=cit.citing_sentence,
                        paper_found=True,
                        full_text_available=bool(ft.full_text),
                        full_text_source=ft.source,
                        top_passages=passages,
                        claim_verdict=claim,
                        paper_metadata=_paper_metadata(exist),
                    ))

                if ref.ref_id not in cit_by_ref:
                    results.append(ComprehensionResult(
                        ref_id=ref.ref_id,
                        citing_sentence="",
                        paper_found=exist.status == "FOUND",
                        full_text_available=bool(ft.full_text),
                        full_text_source=ft.source,
                        top_passages=[],
                        paper_metadata=_paper_metadata(exist),
                    ))
    finally:
        await cache.close()

    report = ComprehensionReport(
        input_file=file_path,
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_citations=len(results),
        results=results,
        coverage=coverage,
    )

    # Attach LLM cost info
    if llm_cost_tracker and llm_cost_tracker.total_calls > 0:
        report.warnings = [f"Claim verification LLM cost: ${llm_cost_tracker.estimated_cost_usd:.4f}"]

    return report


async def run_unified_pipeline(
    file_path: str,
    mode: str = "quick",
    run_verification: bool = True,
    run_claim_verification: bool = False,
    run_comprehension: bool = False,
    ref_pdfs_dir: Optional[str] = None,
    retry_failed: bool = False,
    force_refresh: bool = False,
) -> tuple:
    """Unified pipeline: shared L1+L2 for verification and comprehension.

    Caches full reports on disk keyed by (file, mode, options, config). Pass
    ``force_refresh=True`` or ``retry_failed=True`` to bypass the cache.

    Returns:
        (paper_report or None, comprehension_report or None, parsed)
    """
    from src.models.comprehension import ComprehensionReport
    from src.verification import report_cache

    # Resolve effective mode before hashing so the cache key matches post-resolution
    effective_mode = mode
    if not run_claim_verification and mode != "agentic":
        effective_mode = "quick"

    use_cache = not force_refresh and not retry_failed
    cache_key: Optional[str] = None
    if use_cache:
        cache_key = report_cache.make_key(
            file_path, effective_mode,
            run_verification, run_claim_verification, run_comprehension,
            ref_pdfs_dir,
        )
        cached = report_cache.load(cache_key)
        if cached is not None:
            paper_cached, comp_cached = cached
            log.info(
                f"report_cache: HIT {cache_key} "
                f"(mode={effective_mode}, file={Path(file_path).name})"
            )
            # Still need parsed for callers that use it; parse_file is itself cached
            with stage("pipeline_total_cached",
                       mode=effective_mode, file=Path(file_path).name):
                with stage("L1_parse_only"):
                    parsed = parse_file(file_path)
            return paper_cached, comp_cached, parsed

    with stage("pipeline_total", mode=effective_mode, file=Path(file_path).name):
        # L1: Parse (once)
        with stage("L1_parse"):
            parsed = parse_file(file_path)

        # Mode resolution: if no claim verification requested and not agentic,
        # fall back to quick mode
        if not run_claim_verification and mode != "agentic":
            mode = "quick"

        paper_report: Optional[PaperReport] = None
        comp_report: Optional[ComprehensionReport] = None

        # Shared L2 for all modes
        with stage("L2_existence", refs=len(parsed.references)):
            existence_results = await check_all_references(
                parsed.references, retry_failed=retry_failed
            )
            exist_map = {r.ref_id: r for r in existence_results}

        # Agentic branch: triage + focused agents for ambiguous cases
        if mode == "agentic":
            passages_by_ref: dict = {}
            fulltext_by_ref: dict = {}
            if run_verification:
                with stage("L5_agentic", refs=len(parsed.references)):
                    paper_report, passages_by_ref, fulltext_by_ref = await _run_agentic(
                        parsed, exist_map, file_path, ref_pdfs_dir=ref_pdfs_dir,
                    )
            if run_comprehension:
                # Agentic already retrieved passages + fulltexts during
                # _pre_retrieve_passages. Repackage them as a ComprehensionReport
                # instead of re-running the whole comprehension pipeline.
                if run_verification:
                    with stage("L4_from_agentic", refs=len(parsed.references)):
                        comp_report = _build_comprehension_from_passages(
                            parsed, exist_map,
                            passages_by_ref, fulltext_by_ref, file_path,
                        )
                else:
                    # Comprehension requested without agentic verification —
                    # fall back to the full comprehension pipeline.
                    with stage("L4_comprehension", refs=len(parsed.references),
                               verify_claims=run_claim_verification):
                        comp_report = await _run_comprehension_from_exist_map(
                            parsed, exist_map, file_path, ref_pdfs_dir,
                            verify_claims=run_claim_verification,
                        )
            if use_cache and cache_key is not None:
                report_cache.save(cache_key, paper_report, comp_report)
            return paper_report, comp_report, parsed

        # Quick/Standard branch: L3 → L5
        if run_verification:
            with stage("L3_L5_quick", refs=len(parsed.references), mode=mode):
                verdicts = _run_quick_verification(parsed.references, exist_map, mode)
                paper_report = build_report(parsed, verdicts, mode, input_file=file_path)

        # Comprehension + claim verification (reuse exist_map)
        if run_comprehension or run_claim_verification:
            with stage("L4_comprehension", refs=len(parsed.references),
                       verify_claims=run_claim_verification):
                comp_report = await _run_comprehension_from_exist_map(
                    parsed, exist_map, file_path, ref_pdfs_dir,
                    verify_claims=run_claim_verification,
                )

        if use_cache and cache_key is not None:
            report_cache.save(cache_key, paper_report, comp_report)
        return paper_report, comp_report, parsed


def run_unified(
    file_path: str,
    mode: str = "quick",
    run_verification: bool = True,
    run_claim_verification: bool = False,
    run_comprehension: bool = False,
    ref_pdfs_dir: Optional[str] = None,
    retry_failed: bool = False,
    force_refresh: bool = False,
) -> tuple:
    """Sync wrapper for run_unified_pipeline."""
    return asyncio.run(run_unified_pipeline(
        file_path, mode, run_verification, run_claim_verification,
        run_comprehension, ref_pdfs_dir, retry_failed, force_refresh,
    ))
