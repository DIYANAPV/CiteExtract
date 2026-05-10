
import asyncio
import logging
from collections import defaultdict
from typing import Optional

import httpx
from openai import AsyncOpenAI

from citeextract import config as app_config

from citeextract.classification.classifier import CitationVerdict
from citeextract.models.citation import Citation
from citeextract.models.comprehension import ClaimVerdict, Chunk, FullTextResult, ScoredChunk
from citeextract.models.parsed_paper import ParsedPaper
from citeextract.models.verdict import ExistenceResult
from citeextract.verification.api_clients.llm_client import CostTracker
from citeextract.utils.timing import stage
from citeextract.verification.cache import APICache
from citeextract.verification.metadata import MetadataResult
from citeextract.verification.agentic.tools import ToolExecutor

log = logging.getLogger(__name__)


PrefetchedChunks = dict[str, tuple[FullTextResult, list[Chunk]]]

PassagesByRef = dict[str, dict[str, list[ScoredChunk]]]

FullTextByRef = dict[str, FullTextResult]


_AGENT_TASK_TIMEOUT_S = 240.0


def _group_citations_by_ref(citations: list[Citation]) -> dict[str, list[Citation]]:
    groups: dict[str, list[Citation]] = defaultdict(list)
    for cit in citations:
        groups[cit.ref_id].append(cit)
    return dict(groups)


async def run_agentic_verification(
    parsed: ParsedPaper,
    exist_map: dict[str, ExistenceResult],
    metadata_map: dict[str, MetadataResult],
    agentic_config: dict,
    ref_pdfs_dir: Optional[str] = None,
    prefetched_chunks: Optional[PrefetchedChunks] = None,
    cache: Optional[APICache] = None,
) -> tuple[list[CitationVerdict], float, PassagesByRef, FullTextByRef]:
    from citeextract.verification.agentic.claim_agent import ClaimAgent
    from citeextract.verification.agentic.metadata_agent import MetadataAgent
    from citeextract.verification.agentic.verdict_merger import (
        fallback_to_quick,
        resolve_clear_route,
    )
    from citeextract.verification.triage import TriageRoute, triage_all

    api_key = app_config.openai_api_key()
    if not api_key:
        raise ValueError(
            "OpenAI API key required for agentic mode. "
            "Set OPENAI_API_KEY in your .env file."
        )

    citation_groups = _group_citations_by_ref(parsed.citations)

    cost_tracker = CostTracker()

    owns_cache = cache is None
    if cache is None:
        cache = APICache()

    with stage("agentic_triage", refs=len(parsed.references)):
        triage_config = agentic_config.get("triage", {})
        triage_results = triage_all(
            exist_map=exist_map,
            metadata_map=metadata_map,
            citations_by_ref=citation_groups,
            passages_by_ref={},
            fulltext_by_ref={},
            triage_config=triage_config,
        )

    verdicts: dict[str, CitationVerdict] = {}
    needs_metadata: list = []
    needs_claim: list = []
    needs_both: list = []

    for tr in triage_results:
        clear = resolve_clear_route(tr)
        if clear is not None:
            verdicts[tr.ref_id] = clear
        elif tr.route == TriageRoute.NEEDS_METADATA:
            needs_metadata.append(tr)
        elif tr.route == TriageRoute.NEEDS_CLAIM:
            needs_claim.append(tr)
        elif tr.route == TriageRoute.NEEDS_BOTH:
            needs_both.append(tr)

    clear_count = len(verdicts)
    agent_count = len(needs_metadata) + len(needs_claim) + len(needs_both)
    log.info(
        f"Agentic triage: {clear_count} clear-cut, "
        f"{agent_count} need agents "
        f"(meta={len(needs_metadata)}, claim={len(needs_claim)}, both={len(needs_both)})"
    )

    if agent_count == 0:
        with stage("agentic_pre_retrieve", refs=len(parsed.references)):
            passages_by_ref, fulltext_by_ref = await _pre_retrieve_passages(
                parsed, exist_map, citation_groups, agentic_config,
                ref_pdfs_dir=ref_pdfs_dir,
                prefetched_chunks=prefetched_chunks,
                cache=cache,
            )
        if owns_cache:
            await cache.close()
        ordered = [verdicts[ref.ref_id] for ref in parsed.references if ref.ref_id in verdicts]
        return ordered, cost_tracker.estimated_cost_usd, passages_by_ref, fulltext_by_ref

    openai_client = AsyncOpenAI(api_key=api_key)
    max_concurrent = agentic_config.get("max_concurrent_agents", 15)
    semaphore = asyncio.Semaphore(max_concurrent)

    meta_cfg = agentic_config.get("metadata_agent", {})
    claim_cfg = agentic_config.get("claim_agent", {})

    async with httpx.AsyncClient(follow_redirects=True) as http_client:
        tool_executor = ToolExecutor(http_client, cache)

        if parsed.body_text:
            from citeextract.verification.comprehension import chunk_text
            current_paper_chunks = chunk_text(parsed.body_text)
            tool_executor.set_current_paper_chunks(current_paper_chunks)

        meta_agent = MetadataAgent(
            openai_client=openai_client,
            tool_executor=tool_executor,
            model=agentic_config.get("model", "gpt-5-mini"),
            temperature=agentic_config.get("temperature", 0.0),
            max_tool_rounds=meta_cfg.get("max_tool_rounds", 3),
            max_tokens=meta_cfg.get("max_tokens", 1024),
            timeout=agentic_config.get("timeout", 60),
            cost_tracker=cost_tracker,
        )

        claim_agent = ClaimAgent(
            openai_client=openai_client,
            tool_executor=tool_executor,
            model=agentic_config.get("model", "gpt-5-mini"),
            temperature=agentic_config.get("temperature", 0.0),
            max_tool_rounds=claim_cfg.get("max_tool_rounds", 2),
            max_tokens=claim_cfg.get("max_tokens", 1024),
            timeout=agentic_config.get("timeout", 60),
            cost_tracker=cost_tracker,
            verdict_classes=int(claim_cfg.get("verdict_classes", 3)),
            prompt_variant=str(claim_cfg.get("prompt_variant", "v3")),
        )

        async def _run_metadata(tr):
            async with semaphore:
                try:
                    return await asyncio.wait_for(
                        _dispatch_metadata(tr, meta_agent, parsed),
                        timeout=_AGENT_TASK_TIMEOUT_S,
                    )
                except asyncio.TimeoutError:
                    log.warning(
                        "MetadataAgent for %s exceeded %.0fs task timeout; "
                        "falling back to quick verdict",
                        tr.ref_id, _AGENT_TASK_TIMEOUT_S,
                    )
                    return fallback_to_quick(tr)

        async def _do_pre_retrieve():
            with stage("agentic_pre_retrieve", refs=len(parsed.references)):
                return await _pre_retrieve_passages(
                    parsed, exist_map, citation_groups, agentic_config,
                    ref_pdfs_dir=ref_pdfs_dir,
                    prefetched_chunks=prefetched_chunks,
                    cache=cache,
                )

        async def _do_metadata_dispatch():
            if not needs_metadata:
                return {}
            with stage("agentic_metadata_dispatch", count=len(needs_metadata),
                       concurrency=max_concurrent):
                tasks = {tr.ref_id: _run_metadata(tr) for tr in needs_metadata}
                results = await asyncio.gather(
                    *tasks.values(), return_exceptions=True,
                )
            return dict(zip(tasks.keys(), results))

        (passages_by_ref, fulltext_by_ref), meta_results = await asyncio.gather(
            _do_pre_retrieve(),
            _do_metadata_dispatch(),
        )

        for ref_id, result in meta_results.items():
            tr = next(t for t in needs_metadata if t.ref_id == ref_id)
            if isinstance(result, Exception):
                log.error(f"MetadataAgent failed for {ref_id}: {result}")
                verdicts[ref_id] = fallback_to_quick(tr)
            else:
                verdicts[ref_id] = result

        _attach_passages_to_triage(needs_claim + needs_both, passages_by_ref, fulltext_by_ref)

        async def _run_claim(tr):
            async with semaphore:
                try:
                    return await asyncio.wait_for(
                        _dispatch_claim(tr, claim_agent, parsed),
                        timeout=_AGENT_TASK_TIMEOUT_S,
                    )
                except asyncio.TimeoutError:
                    log.warning(
                        "ClaimAgent for %s exceeded %.0fs task timeout; "
                        "falling back to quick verdict",
                        tr.ref_id, _AGENT_TASK_TIMEOUT_S,
                    )
                    return fallback_to_quick(tr)

        claim_tasks = {tr.ref_id: _run_claim(tr) for tr in needs_claim}
        if claim_tasks:
            with stage("agentic_claim_dispatch", count=len(claim_tasks),
                       concurrency=max_concurrent):
                claim_results = await asyncio.gather(
                    *claim_tasks.values(), return_exceptions=True,
                )
            for ref_id, result in zip(claim_tasks.keys(), claim_results):
                tr = next(t for t in needs_claim if t.ref_id == ref_id)
                if isinstance(result, Exception):
                    log.error(f"ClaimAgent failed for {ref_id}: {result}")
                    verdicts[ref_id] = fallback_to_quick(tr)
                else:
                    verdicts[ref_id] = result

        async def _run_both(tr):
            async with semaphore:
                try:
                    return await asyncio.wait_for(
                        _dispatch_both(tr, meta_agent, claim_agent, parsed),
                        timeout=_AGENT_TASK_TIMEOUT_S,
                    )
                except asyncio.TimeoutError:
                    log.warning(
                        "Both-agents for %s exceeded %.0fs task timeout; "
                        "falling back to quick verdict",
                        tr.ref_id, _AGENT_TASK_TIMEOUT_S,
                    )
                    return fallback_to_quick(tr)

        both_tasks = {tr.ref_id: _run_both(tr) for tr in needs_both}
        if both_tasks:
            with stage("agentic_both_dispatch", count=len(both_tasks),
                       concurrency=max_concurrent):
                both_results = await asyncio.gather(
                    *both_tasks.values(), return_exceptions=True,
                )
            for ref_id, result in zip(both_tasks.keys(), both_results):
                tr = next(t for t in needs_both if t.ref_id == ref_id)
                if isinstance(result, Exception):
                    log.error(f"BothAgent failed for {ref_id}: {result}")
                    verdicts[ref_id] = fallback_to_quick(tr)
                else:
                    verdicts[ref_id] = result

    if owns_cache:
        await cache.close()

    ordered = []
    for ref in parsed.references:
        if ref.ref_id in verdicts:
            ordered.append(verdicts[ref.ref_id])
        else:
            log.warning(f"Missing verdict for {ref.ref_id}, creating fallback")
            ordered.append(CitationVerdict(
                ref_id=ref.ref_id,
                verdict="UNVERIFIABLE",
                mode="agentic",
                action="no_action",
                explanation="Reference was not processed by agentic pipeline.",
                flags=["agentic_processing_error"],
                existence=exist_map.get(ref.ref_id),
            ))
    return ordered, cost_tracker.estimated_cost_usd, passages_by_ref, fulltext_by_ref


_NO_PASSAGES_REASON = "Metadata matches, claims present but no passages pre-retrieved"
_PASSAGES_REASON = "Metadata matches, substantive citing claims to verify"


def _attach_passages_to_triage(
    triage_results: list,
    passages_by_ref: PassagesByRef,
    fulltext_by_ref: FullTextByRef,
) -> None:
    for tr in triage_results:
        passages = passages_by_ref.get(tr.ref_id)
        if passages:
            tr.pre_retrieved_passages = passages
            if tr.triage_reason == _NO_PASSAGES_REASON:
                tr.triage_reason = _PASSAGES_REASON
        ft = fulltext_by_ref.get(tr.ref_id)
        if ft is not None:
            tr.full_text_result = ft


async def _dispatch_metadata(tr, meta_agent, parsed) -> CitationVerdict:
    from citeextract.verification.agentic.verdict_merger import merge_metadata_verdict

    ref = next(r for r in parsed.references if r.ref_id == tr.ref_id)
    user_msg = _build_metadata_msg(tr, ref, tr.existence)

    result = await meta_agent.investigate(user_msg)
    agent_verdict = result.get("verdict", "UNVERIFIABLE")

    if agent_verdict != "VALID" and ref.title and app_config.serpapi_key():
        gs_result = await _google_scholar_reinvestigate(
            ref, meta_agent, result, tr,
        )
        if gs_result is not None:
            return gs_result

    return merge_metadata_verdict(
        triage=tr,
        agent_verdict=agent_verdict,
        agent_explanation=result.get("explanation", ""),
        agent_flags=result.get("flags", []),
        field_discrepancies=result.get("field_discrepancies", []),
    )


def _build_citing_contexts(tr) -> list[dict]:
    citing_contexts = []
    for cit in tr.substantive_citations:
        passages_raw = tr.pre_retrieved_passages.get(cit.citing_sentence, [])
        passages = []
        for p in passages_raw:
            if isinstance(p, ScoredChunk):
                score = p.rrf_score or p.dense_score or p.bm25_score
                passages.append({
                    "text": p.chunk.text,
                    "section": p.chunk.section_name,
                    "score": score,
                })
            elif isinstance(p, dict):
                passages.append(p)

        ctx = {
            "citing_sentence": cit.citing_sentence,
            "context_before": cit.context_before,
            "context_after": cit.context_after,
            "context_quality": tr.context_quality.get(cit.citing_sentence),
            "passages": passages,
            "marker": cit.marker,
        }
        citing_contexts.append(ctx)
    return citing_contexts


def _align_claim_verdicts(
    tr, claim_verdicts: Optional[list[ClaimVerdict]],
) -> dict[str, ClaimVerdict]:
    if not claim_verdicts:
        return {}
    citations = tr.substantive_citations
    if len(claim_verdicts) != len(citations):
        log.warning(
            f"claim verdict count mismatch for {tr.ref_id}: "
            f"{len(claim_verdicts)} verdicts, {len(citations)} citations"
        )
    return {
        cit.citing_sentence: cv
        for cit, cv in zip(citations, claim_verdicts)
    }


def _build_claim_msg(tr, ref, exist) -> str:
    from citeextract.verification.agentic.claim_agent import build_claim_user_message

    return build_claim_user_message(
        citing_contexts=_build_citing_contexts(tr),
        paper_title=exist.matched_title or ref.title or "(unknown)",
        paper_abstract=exist.abstract if exist else None,
        full_text_available=bool(tr.full_text_result and tr.full_text_result.full_text),
        full_text_source=tr.full_text_result.source if tr.full_text_result else None,
        paper_doi=exist.matched_doi if exist else ref.doi,
        arxiv_id=exist.matched_arxiv_id if exist else ref.arxiv_id,
    )


def _build_metadata_msg(tr, ref, exist) -> str:
    from citeextract.verification.agentic.metadata_agent import build_metadata_user_message
    from citeextract.verification.author_comparison import compare_authors
    from citeextract.verification.title_comparison import compare_titles

    title_comp = None
    if ref.title and exist and exist.matched_title:
        title_comp = compare_titles(ref.title, exist.matched_title).to_dict()

    author_comp = None
    if ref.authors and exist and exist.matched_authors:
        author_comp = compare_authors(
            ref.authors, exist.matched_authors, ref.citation_format,
        ).to_dict()

    return build_metadata_user_message(
        ref_title=ref.title,
        ref_authors=ref.authors,
        ref_year=ref.year,
        ref_venue=ref.venue,
        ref_doi=ref.doi,
        ref_raw_text=ref.raw_text,
        citation_format=ref.citation_format,
        db_title=exist.matched_title if exist else None,
        db_authors=exist.matched_authors if exist else [],
        db_year=exist.matched_year if exist else None,
        db_venue=exist.matched_venue if exist else None,
        db_doi=exist.matched_doi if exist else None,
        db_source=exist.source if exist else None,
        title_comparison=title_comp,
        author_comparison=author_comp,
        triage_reason=tr.triage_reason,
    )


async def _dispatch_claim(tr, claim_agent, parsed) -> CitationVerdict:
    from citeextract.verification.agentic.verdict_merger import merge_claim_verdicts

    ref = next(r for r in parsed.references if r.ref_id == tr.ref_id)
    user_msg = _build_claim_msg(tr, ref, tr.existence)
    n_contexts = len(tr.substantive_citations)
    claim_verdicts = await claim_agent.verify_claims(user_msg, expected_count=n_contexts)
    verdict = merge_claim_verdicts(triage=tr, claim_verdicts=claim_verdicts)
    verdict.per_sentence_claim_verdicts = _align_claim_verdicts(tr, claim_verdicts)
    return verdict


async def _dispatch_both(tr, meta_agent, claim_agent, parsed) -> CitationVerdict:
    from citeextract.verification.agentic.verdict_merger import merge_both_verdicts

    ref = next(r for r in parsed.references if r.ref_id == tr.ref_id)
    exist = tr.existence

    meta_msg = _build_metadata_msg(tr, ref, exist)

    async def _run_meta():
        return await meta_agent.investigate(meta_msg)

    async def _run_claim():
        if not tr.substantive_citations:
            return None
        user_msg = _build_claim_msg(tr, ref, exist)
        n_contexts = len(tr.substantive_citations)
        return await claim_agent.verify_claims(user_msg, expected_count=n_contexts)

    meta_raw, claim_raw = await asyncio.gather(
        _run_meta(), _run_claim(), return_exceptions=True,
    )

    if isinstance(meta_raw, BaseException):
        raise meta_raw
    meta_result = meta_raw

    claim_error: Optional[str] = None
    claim_verdicts: Optional[list] = None
    if isinstance(claim_raw, BaseException):
        log.warning(f"Claim agent failed for NEEDS_BOTH ref {tr.ref_id}: {claim_raw}")
        claim_error = str(claim_raw)[:200]
    else:
        claim_verdicts = claim_raw

    meta_verdict = meta_result.get("verdict", "UNVERIFIABLE")

    if meta_verdict != "VALID" and ref.title and app_config.serpapi_key():
        gs_verdict = await _google_scholar_reinvestigate_meta(
            ref, meta_agent, meta_result,
        )
        if gs_verdict is not None:
            meta_result = gs_verdict
            meta_verdict = meta_result.get("verdict", meta_verdict)

    verdict = merge_both_verdicts(
        triage=tr,
        metadata_verdict=meta_verdict,
        metadata_explanation=meta_result.get("explanation", ""),
        metadata_flags=meta_result.get("flags", []),
        field_discrepancies=meta_result.get("field_discrepancies", []),
        claim_verdicts=claim_verdicts,
        claim_error=claim_error,
    )
    verdict.per_sentence_claim_verdicts = _align_claim_verdicts(tr, claim_verdicts)
    return verdict


async def _google_scholar_reinvestigate(
    ref, meta_agent, first_result: dict, tr,
) -> Optional[CitationVerdict]:
    from citeextract.verification.agentic.verdict_merger import merge_metadata_verdict

    first_verdict = first_result.get("verdict", "UNVERIFIABLE")
    first_explanation = first_result.get("explanation", "")

    followup = (
        f"Your initial verdict was {first_verdict}: {first_explanation}\n\n"
        f"Before finalizing, search Google Scholar for this paper using "
        f"search_google_scholar(\"{ref.title}\"). Google Scholar often "
        f"shows the version people actually cite (e.g. arXiv preprint "
        f"vs published journal version).\n\n"
        f"Compare what Google Scholar returns against the reference metadata. "
        f"Then give your final verdict — the same or different from before."
    )

    try:
        result = await meta_agent.investigate_followup(followup)
    except Exception as e:
        log.warning(f"Google Scholar reinvestigation failed for {ref.ref_id}: {e}")
        return None

    new_verdict = result.get("verdict", first_verdict)

    if new_verdict == first_verdict:
        return None

    return merge_metadata_verdict(
        triage=tr,
        agent_verdict=new_verdict,
        agent_explanation=result.get("explanation", ""),
        agent_flags=result.get("flags", []) + ["google_scholar_reinvestigated"],
        field_discrepancies=result.get("field_discrepancies", []),
    )


async def _google_scholar_reinvestigate_meta(
    ref, meta_agent, first_result: dict,
) -> Optional[dict]:
    first_verdict = first_result.get("verdict", "UNVERIFIABLE")
    first_explanation = first_result.get("explanation", "")

    followup = (
        f"Your initial verdict was {first_verdict}: {first_explanation}\n\n"
        f"Before finalizing, search Google Scholar for this paper using "
        f"search_google_scholar(\"{ref.title}\"). Google Scholar often "
        f"shows the version people actually cite (e.g. arXiv preprint "
        f"vs published journal version).\n\n"
        f"Compare what Google Scholar returns against the reference metadata. "
        f"Then give your final verdict — the same or different from before."
    )

    try:
        result = await meta_agent.investigate_followup(followup)
    except Exception as e:
        log.warning(f"Google Scholar meta reinvestigation failed for {ref.ref_id}: {e}")
        return None

    new_verdict = result.get("verdict", first_verdict)
    if new_verdict == first_verdict:
        return None

    enriched = dict(result)
    enriched["flags"] = list(result.get("flags", [])) + ["google_scholar_reinvestigated"]
    return enriched


def _identify_refs_needing_passages(
    parsed: ParsedPaper,
    exist_map: dict[str, ExistenceResult],
    citation_groups: dict[str, list[Citation]],
) -> list[tuple]:
    from citeextract.verification.filters import is_substantive_citation

    selected: list[tuple] = []
    for ref in parsed.references:
        exist = exist_map.get(ref.ref_id)
        if not exist or exist.status != "FOUND":
            continue
        citations = citation_groups.get(ref.ref_id, [])
        substantive = [c for c in citations if is_substantive_citation(c)]
        if not substantive:
            continue
        selected.append((ref, exist, substantive))
    return selected


async def fetch_and_chunk_for_refs(
    parsed: ParsedPaper,
    exist_map: dict[str, ExistenceResult],
    citation_groups: dict[str, list[Citation]],
    max_concurrent: int = 15,
    ref_pdfs_dir: Optional[str] = None,
    cache: Optional[APICache] = None,
) -> PrefetchedChunks:
    from citeextract.verification.api_clients.fulltext import get_full_text
    from citeextract.verification.comprehension import chunk_text

    refs_needing = _identify_refs_needing_passages(parsed, exist_map, citation_groups)
    if not refs_needing:
        return {}

    user_pdf_map: dict[str, str] = {}
    if ref_pdfs_dir:
        try:
            from citeextract.pipeline import _scan_ref_pdfs
            user_pdf_map = _scan_ref_pdfs(ref_pdfs_dir, parsed.references)
        except Exception as e:
            log.warning(f"fetch_and_chunk: scanning ref_pdfs_dir failed: {e}")

    owns_cache = cache is None
    if cache is None:
        cache = APICache()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _one(ref, exist, client):
        async with semaphore:
            try:
                user_pdf = user_pdf_map.get(ref.ref_id)
                ft = await get_full_text(exist, client, cache, user_pdf_path=user_pdf)
                text = ft.full_text or ""
                if not text.strip():
                    return ref.ref_id, ft, []
                sections = ft.sections if ft.sections else None
                chunks = chunk_text(text, sections=sections)
                return ref.ref_id, ft, chunks
            except (httpx.RequestError, httpx.HTTPError, OSError, ValueError) as e:
                log.warning(f"Fetch/chunk failed for {ref.ref_id}: {e}")
                return ref.ref_id, None, []

    log.info(f"fetch_and_chunk: {len(refs_needing)} refs, max_workers={max_concurrent}")
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            results = await asyncio.gather(
                *[_one(ref, exist, client) for ref, exist, _ in refs_needing],
                return_exceptions=True,
            )
    finally:
        if owns_cache:
            await cache.close()

    out: PrefetchedChunks = {}
    for result in results:
        if isinstance(result, Exception):
            log.warning(f"fetch_and_chunk task raised: {result}")
            continue
        ref_id, ft, chunks = result
        if ft is not None:
            out[ref_id] = (ft, chunks)
    return out


async def _pre_retrieve_passages(
    parsed: ParsedPaper,
    exist_map: dict[str, ExistenceResult],
    citation_groups: dict[str, list[Citation]],
    config: dict,
    ref_pdfs_dir: Optional[str] = None,
    prefetched_chunks: Optional[PrefetchedChunks] = None,
    cache: Optional[APICache] = None,
) -> tuple[PassagesByRef, FullTextByRef]:
    import time

    from citeextract import config as app_cfg
    from citeextract.utils.timing import stage
    from citeextract.verification.comprehension import (
        build_retrieval_index,
        build_retrieval_query,
        retrieve_with_index,
    )

    comp_cfg = app_cfg.comprehension()
    top_k = comp_cfg.get("top_k", 3)
    dense_model = comp_cfg.get("dense_model")

    passages_by_ref: PassagesByRef = {}
    fulltext_by_ref: FullTextByRef = {}

    refs_needing = _identify_refs_needing_passages(parsed, exist_map, citation_groups)
    if not refs_needing:
        return passages_by_ref, fulltext_by_ref

    max_concurrent = int(config.get("max_concurrent_agents", 15)) if config else 15

    if prefetched_chunks is None:
        with stage("pre_retrieve.fetch_chunk", refs=len(refs_needing)):
            prefetched_chunks = await fetch_and_chunk_for_refs(
                parsed, exist_map, citation_groups,
                max_concurrent=max_concurrent,
                ref_pdfs_dir=ref_pdfs_dir,
                cache=cache,
            )
    else:
        log.info(
            f"pre_retrieve: using {len(prefetched_chunks)} pre-fetched "
            f"fulltexts (fetch ran in parallel with L3)"
        )

    log.info(
        f"Pre-retrieving passages for {len(refs_needing)} references "
        f"(concurrency={max_concurrent})"
    )

    semaphore = asyncio.Semaphore(max_concurrent)

    bm25_candidates = comp_cfg.get("bm25_candidates", 10)
    dense_candidates = comp_cfg.get("dense_candidates", 10)
    rrf_k = comp_cfg.get("rrf_k", 60)
    rerank_pool = comp_cfg.get("rerank_pool", 10)

    index_build_total = 0.0
    retrieve_total = 0.0

    async def _retrieve_one(ref, substantive):
        nonlocal index_build_total, retrieve_total
        prefetched = prefetched_chunks.get(ref.ref_id)
        if prefetched is None:
            return ref.ref_id, None, None
        ft, chunks = prefetched
        if not chunks:
            return ref.ref_id, ft, None

        async with semaphore:
            try:
                t_index_start = time.perf_counter()
                index = build_retrieval_index(
                    chunks, model_name=dense_model if dense_model else None,
                )
                ref_index_secs = time.perf_counter() - t_index_start
            except (ValueError, KeyError, RuntimeError, OSError) as e:
                log.warning(
                    f"Index build failed for {ref.ref_id}: "
                    f"{type(e).__name__}: {e}"
                )
                return ref.ref_id, ft, None

            t_retrieve_start = time.perf_counter()
            ref_passages: dict[str, list] = {}

            for cit in substantive:
                query = build_retrieval_query(
                    cit.citing_sentence, cit.context_before, cit.context_after,
                    markers=[cit.marker] if cit.marker else None,
                )
                try:
                    scored = retrieve_with_index(
                        query, index,
                        top_k=top_k,
                        bm25_candidates=bm25_candidates,
                        dense_candidates=dense_candidates,
                        rrf_k=rrf_k,
                        rerank_pool=rerank_pool,
                    )
                    ref_passages[cit.citing_sentence] = scored
                except (ValueError, KeyError, RuntimeError) as e:
                    log.warning(
                        f"Retrieval for citation in {ref.ref_id} failed "
                        f"({type(e).__name__}: {e}); recording empty passages "
                        "for this citation only."
                    )
                    ref_passages[cit.citing_sentence] = []

            ref_retrieve_secs = time.perf_counter() - t_retrieve_start
            index_build_total += ref_index_secs
            retrieve_total += ref_retrieve_secs
            log.debug(
                "pre_retrieve.ref ref_id=%s chunks=%d citations=%d "
                "index=%.3fs retrieve=%.3fs",
                ref.ref_id, len(chunks), len(substantive),
                ref_index_secs, ref_retrieve_secs,
            )

            return ref.ref_id, ft, ref_passages

    _PER_REF_TIMEOUT_S = 180.0

    timing_log_progress = logging.getLogger("citeextract.timing")

    async def _retrieve_one_bounded(ref, substantive):
        try:
            return await asyncio.wait_for(
                _retrieve_one(ref, substantive),
                timeout=_PER_REF_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            log.warning(
                "pre_retrieve: ref_id=%s exceeded %.0fs timeout; "
                "recording empty passages and continuing.",
                ref.ref_id, _PER_REF_TIMEOUT_S,
            )
            return ref.ref_id, None, None

    with stage("pre_retrieve.retrieve_block", refs=len(refs_needing)):
        tasks = [
            _retrieve_one_bounded(ref, substantive)
            for ref, _, substantive in refs_needing
        ]
        total_refs = len(tasks)
        for done_count, fut in enumerate(asyncio.as_completed(tasks), 1):
            try:
                result = await fut
            except Exception as e:
                log.warning(f"Passage retrieval task raised: {e}")
                result = None
            if result is not None:
                ref_id, ft, ref_passages = result
                if ft is not None:
                    fulltext_by_ref[ref_id] = ft
                if ref_passages is not None:
                    passages_by_ref[ref_id] = ref_passages
            timing_log_progress.info(
                "STAGE agentic_pre_retrieve.progress seconds=0 "
                "done=%d total=%d", done_count, total_refs,
            )

    timing_log = logging.getLogger("citeextract.timing")
    timing_log.info(
        "STAGE pre_retrieve.index_build_total seconds=%.3f refs=%d",
        index_build_total, len(refs_needing),
    )
    timing_log.info(
        "STAGE pre_retrieve.retrieve_total seconds=%.3f refs=%d",
        retrieve_total, len(refs_needing),
    )

    return passages_by_ref, fulltext_by_ref
