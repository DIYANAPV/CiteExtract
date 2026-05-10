
import asyncio
import logging
from pathlib import Path
from typing import Optional

from citeextract import config, paths
from citeextract.classification.classifier import CitationVerdict, classify_quick
from citeextract.models.parsed_paper import ParsedPaper
from citeextract.models.report import PaperReport
from citeextract.models.verdict import ExistenceResult
from citeextract.parsers.router import parse_file
from citeextract.report.generator import build_report, save_json
from citeextract.utils.timing import collect_stages, stage
from citeextract.verification.agentic.runner import (
    FullTextByRef,
    PassagesByRef,
    PrefetchedChunks,
)
from citeextract.verification.cache import APICache
from citeextract.verification.existence import check_all_references
from citeextract.verification.matching import WEB_SOURCES
from citeextract.verification.metadata import MetadataResult, validate_metadata

log = logging.getLogger(__name__)


def _build_metadata_map(
    references: list,
    exist_map: dict[str, ExistenceResult],
) -> dict[str, MetadataResult]:
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
    report, _, _ = await run_unified_pipeline(
        file_path, mode=mode, run_verification=True, retry_failed=retry_failed,
    )
    return report


async def _run_agentic(
    parsed: ParsedPaper,
    exist_map: dict[str, ExistenceResult],
    file_path: str,
    ref_pdfs_dir: Optional[str] = None,
    prefetched_chunks: Optional[PrefetchedChunks] = None,
    metadata_map: Optional[dict[str, MetadataResult]] = None,
    cache: Optional[APICache] = None,
) -> tuple[PaperReport, PassagesByRef, FullTextByRef]:
    from citeextract.verification.agentic.runner import run_agentic_verification

    agentic_cfg = config.agentic()
    if not agentic_cfg:
        raise ValueError(
            "No agentic config found in config.yaml. "
            "Add an 'agentic' section with at least a model name."
        )

    if metadata_map is None:
        metadata_map = _build_metadata_map(parsed.references, exist_map)

    verdicts, llm_cost, passages_by_ref, fulltext_by_ref = await run_agentic_verification(
        parsed, exist_map, metadata_map, agentic_cfg,
        ref_pdfs_dir=ref_pdfs_dir,
        prefetched_chunks=prefetched_chunks,
        cache=cache,
    )

    for v in verdicts:
        if v.existence and not v.existence.abstract:
            ft = fulltext_by_ref.get(v.ref_id)
            if ft and ft.abstract:
                v.existence.abstract = ft.abstract

    report = build_report(parsed, verdicts, "agentic", input_file=file_path)
    report.summary.total_cost_usd = round(llm_cost, 6)

    if llm_cost > 0:
        report.warnings.append(f"LLM cost (agentic): ${llm_cost:.4f}")

    agent_count = sum(1 for v in verdicts if "agentic_agent_used" in v.flags)
    clear_count = len(verdicts) - agent_count
    if clear_count > 0:
        report.warnings.append(
            f"Agentic mode: {clear_count}/{len(verdicts)} references resolved "
            f"by rule-based triage (no LLM needed)."
        )

    from citeextract.verification.api_clients.fulltext import grobid_was_probed_unavailable
    if grobid_was_probed_unavailable():
        report.warnings.append(
            "GROBID is not running — all full-text fetches degraded to abstract-only. "
            "Start GROBID for full-text coverage: "
            "`docker start grobid` or `docker run -d --name grobid -p 8070:8070 grobid/grobid:0.8.2-crf`"
        )

    return report, passages_by_ref, fulltext_by_ref


def run_and_save(
    file_path: str,
    mode: str = "quick",
    output_dir: str = str(paths.data_dir() / "output"),
    retry_failed: bool = False,
) -> PaperReport:
    report = asyncio.run(run_pipeline(file_path, mode, retry_failed=retry_failed))

    p = Path(file_path)
    name = f"{p.stem}_{p.suffix.lstrip('.')}" if p.suffix else p.stem
    save_json(report, f"{output_dir}/{name}_{report.mode}_report.json")

    return report


async def run_comprehension_pipeline(
    file_path: str,
    ref_pdfs_dir: Optional[str] = None,
    retry_failed: bool = False,
) -> "ComprehensionReport":
    from collections import defaultdict
    from datetime import datetime, timezone

    import httpx

    from citeextract.models.comprehension import ComprehensionReport, ComprehensionResult, ScoredChunk
    from citeextract.verification.api_clients.fulltext import get_full_text
    from citeextract.verification.comprehension import (
        build_retrieval_query,
        chunk_text, retrieve_passages_bm25, retrieve_passages_hybrid,
    )
    from citeextract.verification.filters import is_substantive_citation

    comp_cfg = config.comprehension()
    top_k = comp_cfg["top_k"]
    dense_model = comp_cfg.get("dense_model")
    use_hybrid = bool(dense_model)

    parsed = parse_file(file_path)

    existence_results = await check_all_references(parsed.references, retry_failed=retry_failed)
    exist_map = {r.ref_id: r for r in existence_results}

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

                text_to_chunk = ft.full_text or ""
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
                            fetch_attempts=list(ft.attempts),
                        ))
                    continue

                sections = ft.sections if ft.sections else None
                chunks = chunk_text(text_to_chunk, sections=sections)

                is_abstract_only = ft.source == "abstract_only" or not ft.full_text

                for cit in cit_by_ref.get(ref.ref_id, []):
                    if is_abstract_only and len(chunks) <= 2:
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
                            fetch_attempts=list(ft.attempts),
                        ))
                        continue
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
                        fetch_attempts=list(ft.attempts),
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
                        fetch_attempts=list(ft.attempts),
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
    passages_by_ref: PassagesByRef,
    fulltext_by_ref: FullTextByRef,
    verdicts: list[CitationVerdict],
    file_path: str,
) -> "ComprehensionReport":
    from datetime import datetime, timezone
    from citeextract.models.comprehension import ComprehensionReport, ComprehensionResult
    from citeextract.verification.filters import is_substantive_citation

    claim_lookup: dict[str, dict] = {
        v.ref_id: v.per_sentence_claim_verdicts for v in verdicts
    }

    results: list[ComprehensionResult] = []
    coverage = {"full_text": 0, "abstract_only": 0, "not_found": 0}

    seen_refs: set[str] = set()

    for cit in parsed.citations:
        if not is_substantive_citation(cit):
            continue
        exist = exist_map.get(cit.ref_id)
        ft = fulltext_by_ref.get(cit.ref_id)
        passages = passages_by_ref.get(cit.ref_id, {}).get(cit.citing_sentence, [])
        sentence_claim = claim_lookup.get(cit.ref_id, {}).get(cit.citing_sentence)

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
            claim_verdict=sentence_claim,
            paper_metadata=_paper_metadata(exist) if exist else {},
            fetch_attempts=list(ft.attempts) if ft else [],
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
    from citeextract.verification.matching import title_similarity

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

            if stem == ref.ref_id.lower():
                result[ref.ref_id] = str(pdf)
                break

            if ref.doi and ref.doi.lower().replace("/", "_") in stem:
                result[ref.ref_id] = str(pdf)
                break

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
    output_dir: str = str(paths.data_dir() / "output"),
) -> "ComprehensionReport":
    import json as _json

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


async def _run_comprehension_from_exist_map(
    parsed: ParsedPaper,
    exist_map: dict[str, ExistenceResult],
    file_path: str,
    ref_pdfs_dir: Optional[str] = None,
    verify_claims: bool = False,
    cache: Optional[APICache] = None,
) -> "ComprehensionReport":
    from collections import defaultdict
    from datetime import datetime, timezone

    import httpx

    from citeextract.models.comprehension import (
        ClaimVerdict, ComprehensionReport, ComprehensionResult, ScoredChunk,
    )
    from citeextract.verification.api_clients.fulltext import get_full_text
    from citeextract.verification.comprehension import (
        build_retrieval_query,
        chunk_text, retrieve_passages_bm25, retrieve_passages_hybrid,
        verify_claim_with_passages,
    )
    from citeextract.verification.filters import is_substantive_citation

    comp_cfg = config.comprehension()
    top_k = comp_cfg["top_k"]
    dense_model = comp_cfg.get("dense_model")
    use_hybrid = bool(dense_model)

    llm_client = None
    llm_cost_tracker = None
    if verify_claims:
        from citeextract.verification.api_clients.llm_client import create_llm_client
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

    owns_cache = cache is None
    if cache is None:
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
                            fetch_attempts=list(ft.attempts),
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
                            fetch_attempts=list(ft.attempts),
                        ))
                        continue
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
                        fetch_attempts=list(ft.attempts),
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
                        fetch_attempts=list(ft.attempts),
                    ))
    finally:
        if owns_cache:
            await cache.close()

    report = ComprehensionReport(
        input_file=file_path,
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_citations=len(results),
        results=results,
        coverage=coverage,
    )

    if llm_cost_tracker and llm_cost_tracker.total_calls > 0:
        report.warnings = [f"Claim verification LLM cost: ${llm_cost_tracker.estimated_cost_usd:.4f}"]

    from citeextract.verification.api_clients.fulltext import grobid_was_probed_unavailable
    if grobid_was_probed_unavailable():
        report.warnings.append(
            "GROBID is not running — all full-text fetches degraded to abstract-only. "
            "Start GROBID for full-text coverage: "
            "`docker start grobid` or `docker run -d --name grobid -p 8070:8070 grobid/grobid:0.8.2-crf`"
        )

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
    from citeextract.models.comprehension import ComprehensionReport
    from citeextract.verification import report_cache

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
            with stage("pipeline_total_cached",
                       mode=effective_mode, file=Path(file_path).name):
                with stage("L1_parse_only"):
                    parsed = parse_file(file_path)
            return paper_cached, comp_cached, parsed

    with collect_stages() as run_stats_bucket, stage(
        "pipeline_total", mode=effective_mode, file=Path(file_path).name,
    ):
        with stage("L1_parse"):
            parsed = parse_file(file_path)

        if not run_claim_verification and mode != "agentic":
            mode = "quick"

        paper_report: Optional[PaperReport] = None
        comp_report: Optional[ComprehensionReport] = None

        with stage("L2_existence", refs=len(parsed.references)):
            existence_results = await check_all_references(
                parsed.references, retry_failed=retry_failed
            )
            exist_map = {r.ref_id: r for r in existence_results}

        shared_cache = APICache()
        try:
            if mode == "agentic":
                passages_by_ref: PassagesByRef = {}
                fulltext_by_ref: FullTextByRef = {}
                if run_verification:
                    from collections import defaultdict
                    from citeextract.verification.agentic.runner import fetch_and_chunk_for_refs

                    citation_groups: dict[str, list] = defaultdict(list)
                    for _cit in parsed.citations:
                        citation_groups[_cit.ref_id].append(_cit)

                    agentic_cfg_for_fetch = config.agentic()
                    if not agentic_cfg_for_fetch:
                        raise ValueError(
                            "No agentic config found in config.yaml. "
                            "Add an 'agentic' section with at least a model name."
                        )
                    fetch_concurrency = int(
                        agentic_cfg_for_fetch.get("max_concurrent_agents", 15)
                    )

                    fetch_task = asyncio.create_task(
                        fetch_and_chunk_for_refs(
                            parsed, exist_map, dict(citation_groups),
                            max_concurrent=fetch_concurrency,
                            ref_pdfs_dir=ref_pdfs_dir,
                            cache=shared_cache,
                        )
                    )

                    with stage("L3_metadata", refs=len(parsed.references)):
                        metadata_map = _build_metadata_map(parsed.references, exist_map)

                    with stage("L5_agentic", refs=len(parsed.references)):
                        prefetched_chunks = await fetch_task
                        paper_report, passages_by_ref, fulltext_by_ref = await _run_agentic(
                            parsed, exist_map, file_path,
                            ref_pdfs_dir=ref_pdfs_dir,
                            prefetched_chunks=prefetched_chunks,
                            metadata_map=metadata_map,
                            cache=shared_cache,
                        )
                if run_comprehension:
                    if run_verification:
                        with stage("L4_from_agentic", refs=len(parsed.references)):
                            verdicts_for_comp = (
                                paper_report.verdicts if paper_report else []
                            )
                            comp_report = _build_comprehension_from_passages(
                                parsed, exist_map,
                                passages_by_ref, fulltext_by_ref,
                                verdicts_for_comp, file_path,
                            )
                    else:
                        with stage("L4_comprehension", refs=len(parsed.references),
                                   verify_claims=run_claim_verification):
                            comp_report = await _run_comprehension_from_exist_map(
                                parsed, exist_map, file_path, ref_pdfs_dir,
                                verify_claims=run_claim_verification,
                                cache=shared_cache,
                            )
                if paper_report is not None:
                    paper_report.run_stats = dict(run_stats_bucket)
                if use_cache and cache_key is not None:
                    report_cache.save(cache_key, paper_report, comp_report)
                return paper_report, comp_report, parsed

            if run_verification:
                with stage("L3_L5_quick", refs=len(parsed.references), mode=mode):
                    verdicts = _run_quick_verification(parsed.references, exist_map, mode)
                    paper_report = build_report(parsed, verdicts, mode, input_file=file_path)

            if run_comprehension or run_claim_verification:
                with stage("L4_comprehension", refs=len(parsed.references),
                           verify_claims=run_claim_verification):
                    comp_report = await _run_comprehension_from_exist_map(
                        parsed, exist_map, file_path, ref_pdfs_dir,
                        verify_claims=run_claim_verification,
                        cache=shared_cache,
                    )

            if paper_report is not None:
                paper_report.run_stats = dict(run_stats_bucket)
            if use_cache and cache_key is not None:
                report_cache.save(cache_key, paper_report, comp_report)
            return paper_report, comp_report, parsed
        finally:
            await shared_cache.close()


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
    return asyncio.run(run_unified_pipeline(
        file_path, mode, run_verification, run_claim_verification,
        run_comprehension, ref_pdfs_dir, retry_failed, force_refresh,
    ))
