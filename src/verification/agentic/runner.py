"""Concurrent orchestrator — fans out agent verification across all references.

Creates shared resources (HTTP client, cache, OpenAI client) and runs
agents in parallel with a concurrency limiter.

Uses rule-based triage first (L2 existence + L3 metadata), then dispatches
focused MetadataAgent and ClaimAgent only for references that need deeper
investigation. Clear-cut cases (exact matches, obvious fabrications) are
resolved without LLM calls.
"""

import asyncio
import logging
from collections import defaultdict
from typing import Optional

import httpx
from openai import AsyncOpenAI

from src import config as app_config

from src.classification.classifier import CitationVerdict
from src.models.citation import Citation
from src.models.comprehension import ScoredChunk
from src.models.parsed_paper import ParsedPaper
from src.models.verdict import ExistenceResult
from src.verification.api_clients.llm_client import CostTracker
from src.verification.cache import APICache
from src.verification.metadata import MetadataResult
from src.verification.agentic.tools import ToolExecutor

log = logging.getLogger(__name__)


def _group_citations_by_ref(citations: list[Citation]) -> dict[str, list[Citation]]:
    """Group citations by their reference ID."""
    groups: dict[str, list[Citation]] = defaultdict(list)
    for cit in citations:
        groups[cit.ref_id].append(cit)
    return dict(groups)


async def run_agentic_verification(
    parsed: ParsedPaper,
    exist_map: dict[str, ExistenceResult],
    metadata_map: dict[str, MetadataResult],
    agentic_config: dict,
) -> tuple[list[CitationVerdict], float]:
    """Run agentic verification: rule-based triage + focused agents where needed.

    Clear-cut cases (exact matches, obvious fabrications) are resolved
    without LLM calls. Ambiguous cases get dispatched to focused
    MetadataAgent and/or ClaimAgent.

    Args:
        parsed: L1 parse output.
        exist_map: Pre-computed L2 existence results (ref_id → ExistenceResult).
        metadata_map: Pre-computed L3 metadata results (ref_id → MetadataResult).
        agentic_config: Config from config.yaml 'agentic' section.

    Returns:
        (list of CitationVerdicts, total LLM cost in USD)
    """
    from src.verification.agentic.claim_agent import (
        ClaimAgent,
        build_claim_user_message,
    )
    from src.verification.agentic.metadata_agent import (
        MetadataAgent,
        build_metadata_user_message,
    )
    from src.verification.agentic.verdict_merger import (
        fallback_to_quick,
        merge_both_verdicts,
        merge_claim_verdicts,
        merge_metadata_verdict,
        resolve_clear_route,
    )
    from src.verification.triage import TriageRoute, triage_all

    # --- Resolve API key ---
    api_key = app_config.openai_api_key()
    if not api_key:
        raise ValueError(
            "OpenAI API key required for agentic mode. "
            "Set OPENAI_API_KEY in your .env file."
        )

    # --- Group citations by ref ---
    citation_groups = _group_citations_by_ref(parsed.citations)

    # --- Pre-retrieve passages for FOUND refs with substantive citations ---
    passages_by_ref, fulltext_by_ref = await _pre_retrieve_passages(
        parsed, exist_map, citation_groups, agentic_config,
    )

    # --- Triage all references ---
    triage_config = agentic_config.get("triage", {})
    triage_results = triage_all(
        exist_map=exist_map,
        metadata_map=metadata_map,
        citations_by_ref=citation_groups,
        passages_by_ref=passages_by_ref,
        fulltext_by_ref=fulltext_by_ref,
        triage_config=triage_config,
    )

    # --- Resolve clear-cut cases ---
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

    # --- Skip agents if nothing needs them ---
    if agent_count == 0:
        ordered = [verdicts[ref.ref_id] for ref in parsed.references if ref.ref_id in verdicts]
        return ordered, 0.0

    # --- Set up shared resources ---
    openai_client = AsyncOpenAI(api_key=api_key)
    cost_tracker = CostTracker()
    cache = APICache()
    max_concurrent = agentic_config.get("max_concurrent_agents", 5)
    semaphore = asyncio.Semaphore(max_concurrent)

    meta_cfg = agentic_config.get("metadata_agent", {})
    claim_cfg = agentic_config.get("claim_agent", {})

    async with httpx.AsyncClient(follow_redirects=True) as http_client:
        tool_executor = ToolExecutor(http_client, cache)

        # Chunk the current paper for retrieve_current_paper_context tool
        if parsed.body_text:
            from src.verification.comprehension import chunk_text
            current_paper_chunks = chunk_text(parsed.body_text)
            tool_executor.set_current_paper_chunks(current_paper_chunks)

        meta_agent = MetadataAgent(
            openai_client=openai_client,
            tool_executor=tool_executor,
            model=agentic_config.get("model", "gpt-4o-mini"),
            temperature=agentic_config.get("temperature", 0.0),
            max_tool_rounds=meta_cfg.get("max_tool_rounds", 3),
            max_tokens=meta_cfg.get("max_tokens", 1024),
            timeout=agentic_config.get("timeout", 60),
            cost_tracker=cost_tracker,
        )

        claim_agent = ClaimAgent(
            openai_client=openai_client,
            tool_executor=tool_executor,
            model=agentic_config.get("model", "gpt-4o-mini"),
            temperature=agentic_config.get("temperature", 0.0),
            max_tool_rounds=claim_cfg.get("max_tool_rounds", 2),
            max_tokens=claim_cfg.get("max_tokens", 1024),
            timeout=agentic_config.get("timeout", 60),
            cost_tracker=cost_tracker,
        )

        # --- Dispatch metadata agents in parallel ---
        async def _run_metadata(tr):
            async with semaphore:
                return await _dispatch_metadata(tr, meta_agent, parsed)

        meta_tasks = {tr.ref_id: _run_metadata(tr) for tr in needs_metadata}
        if meta_tasks:
            meta_results = await asyncio.gather(
                *meta_tasks.values(), return_exceptions=True,
            )
            for ref_id, result in zip(meta_tasks.keys(), meta_results):
                tr = next(t for t in needs_metadata if t.ref_id == ref_id)
                if isinstance(result, Exception):
                    log.error(f"MetadataAgent failed for {ref_id}: {result}")
                    verdicts[ref_id] = fallback_to_quick(tr)
                else:
                    verdicts[ref_id] = result

        # --- Dispatch claim agents in parallel ---
        async def _run_claim(tr):
            async with semaphore:
                return await _dispatch_claim(tr, claim_agent, parsed)

        claim_tasks = {tr.ref_id: _run_claim(tr) for tr in needs_claim}
        if claim_tasks:
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

        # --- Dispatch NEEDS_BOTH (sequential: metadata → claim) ---
        async def _run_both(tr):
            async with semaphore:
                return await _dispatch_both(tr, meta_agent, claim_agent, parsed)

        both_tasks = {tr.ref_id: _run_both(tr) for tr in needs_both}
        if both_tasks:
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

    await cache.close()

    # --- Return in original reference order (every ref must have a verdict) ---
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
    return ordered, cost_tracker.estimated_cost_usd


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------


async def _dispatch_metadata(tr, meta_agent, parsed) -> CitationVerdict:
    """Run metadata agent for a single triage result."""
    from src.verification.agentic.metadata_agent import build_metadata_user_message
    from src.verification.agentic.verdict_merger import (
        fallback_to_quick,
        merge_metadata_verdict,
    )
    from src.verification.title_comparison import compare_titles
    from src.verification.author_comparison import compare_authors

    ref = next(r for r in parsed.references if r.ref_id == tr.ref_id)
    exist = tr.existence

    # Pre-compute comparison results for the agent
    title_comp = None
    if ref.title and exist and exist.matched_title:
        title_comp = compare_titles(ref.title, exist.matched_title).to_dict()

    author_comp = None
    if ref.authors and exist and exist.matched_authors:
        author_comp = compare_authors(
            ref.authors, exist.matched_authors, ref.citation_format,
        ).to_dict()

    user_msg = build_metadata_user_message(
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

    result = await meta_agent.investigate(user_msg)
    agent_verdict = result.get("verdict", "UNVERIFIABLE")

    # If not VALID and Google Scholar is available, give the agent a second
    # chance with Google Scholar data before finalizing
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


def _build_citing_contexts(tr, ref=None) -> list[dict]:
    """Build citing context dicts from a triage result (shared by claim and both dispatchers)."""
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


def _build_claim_msg(tr, ref, exist) -> str:
    """Build claim agent user message from triage result."""
    from src.verification.agentic.claim_agent import build_claim_user_message

    return build_claim_user_message(
        citing_contexts=_build_citing_contexts(tr),
        paper_title=exist.matched_title or ref.title or "(unknown)",
        paper_abstract=exist.abstract if exist else None,
        full_text_available=bool(tr.full_text_result and tr.full_text_result.full_text),
        full_text_source=tr.full_text_result.source if tr.full_text_result else None,
        paper_doi=exist.matched_doi if exist else ref.doi,
        arxiv_id=exist.matched_arxiv_id if exist else ref.arxiv_id,
    )


async def _dispatch_claim(tr, claim_agent, parsed) -> CitationVerdict:
    """Run claim agent for a single triage result."""
    from src.verification.agentic.verdict_merger import merge_claim_verdicts

    ref = next(r for r in parsed.references if r.ref_id == tr.ref_id)
    user_msg = _build_claim_msg(tr, ref, tr.existence)
    n_contexts = len(tr.substantive_citations)
    claim_verdicts = await claim_agent.verify_claims(user_msg, expected_count=n_contexts)
    return merge_claim_verdicts(triage=tr, claim_verdicts=claim_verdicts)


async def _dispatch_both(tr, meta_agent, claim_agent, parsed) -> CitationVerdict:
    """Run metadata agent first, then claim agent if metadata is OK."""
    from src.verification.agentic.metadata_agent import build_metadata_user_message
    from src.verification.agentic.verdict_merger import merge_both_verdicts
    from src.verification.title_comparison import compare_titles
    from src.verification.author_comparison import compare_authors

    ref = next(r for r in parsed.references if r.ref_id == tr.ref_id)
    exist = tr.existence

    # --- Step 1: Metadata agent ---
    title_comp = None
    if ref.title and exist and exist.matched_title:
        title_comp = compare_titles(ref.title, exist.matched_title).to_dict()

    author_comp = None
    if ref.authors and exist and exist.matched_authors:
        author_comp = compare_authors(
            ref.authors, exist.matched_authors, ref.citation_format,
        ).to_dict()

    meta_msg = build_metadata_user_message(
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

    meta_result = await meta_agent.investigate(meta_msg)
    meta_verdict = meta_result.get("verdict", "UNVERIFIABLE")

    # Google Scholar re-investigation for non-VALID verdicts
    if meta_verdict != "VALID" and ref.title and app_config.serpapi_key():
        gs_result = await _google_scholar_reinvestigate(
            ref, meta_agent, meta_result, tr,
        )
        if gs_result is not None:
            # Google Scholar changed the verdict — check if now VALID
            if gs_result.verdict == "VALID":
                meta_verdict = "VALID"
                # Continue to claim verification below
            else:
                return gs_result

    # --- Step 2: Claim agent (only if metadata OK) ---
    claim_verdicts = None
    if meta_verdict == "VALID" and tr.substantive_citations:
        try:
            user_msg = _build_claim_msg(tr, ref, exist)
            n_contexts = len(tr.substantive_citations)
            claim_verdicts = await claim_agent.verify_claims(user_msg, expected_count=n_contexts)
        except Exception as e:
            log.warning(f"Claim agent failed for NEEDS_BOTH ref {tr.ref_id}: {e}")
            claim_verdicts = None

    return merge_both_verdicts(
        triage=tr,
        metadata_verdict=meta_verdict,
        metadata_explanation=meta_result.get("explanation", ""),
        metadata_flags=meta_result.get("flags", []),
        field_discrepancies=meta_result.get("field_discrepancies", []),
        claim_verdicts=claim_verdicts,
    )


# ---------------------------------------------------------------------------
# Google Scholar safety net
# ---------------------------------------------------------------------------


async def _google_scholar_reinvestigate(
    ref, meta_agent, first_result: dict, tr,
) -> Optional[CitationVerdict]:
    """Give the metadata agent a second call with Google Scholar.

    The agent already investigated and returned non-VALID. Now we let it
    search Google Scholar and reconsider, seeing both the original DB data
    and whatever Google Scholar returns.
    """
    from src.verification.agentic.verdict_merger import merge_metadata_verdict

    first_verdict = first_result.get("verdict", "UNVERIFIABLE")
    first_explanation = first_result.get("explanation", "")

    # Build a follow-up message for the agent
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

    # Only return if verdict changed
    if new_verdict == first_verdict:
        return None

    return merge_metadata_verdict(
        triage=tr,
        agent_verdict=new_verdict,
        agent_explanation=result.get("explanation", ""),
        agent_flags=result.get("flags", []) + ["google_scholar_reinvestigated"],
        field_discrepancies=result.get("field_discrepancies", []),
    )


# ---------------------------------------------------------------------------
# Passage pre-retrieval (deterministic, no LLM)
# ---------------------------------------------------------------------------


async def _pre_retrieve_passages(
    parsed: ParsedPaper,
    exist_map: dict[str, ExistenceResult],
    citation_groups: dict[str, list[Citation]],
    config: dict,
) -> tuple[dict[str, dict[str, list]], dict]:
    """Pre-retrieve passages for all FOUND refs with substantive citations.

    Returns:
        (passages_by_ref, fulltext_by_ref) where:
        - passages_by_ref: {ref_id: {citing_sentence: [ScoredChunk]}}
        - fulltext_by_ref: {ref_id: FullTextResult}
    """
    from src import config as app_cfg
    from src.models.comprehension import FullTextResult
    from src.verification.api_clients.fulltext import get_full_text
    from src.verification.comprehension import (
        build_retrieval_query,
        chunk_text,
        retrieve_passages_bm25,
        retrieve_passages_hybrid,
    )
    from src.verification.filters import is_substantive_citation

    comp_cfg = app_cfg.comprehension()
    top_k = comp_cfg.get("top_k", 3)
    dense_model = comp_cfg.get("dense_model")

    passages_by_ref: dict[str, dict[str, list]] = {}
    fulltext_by_ref: dict[str, FullTextResult] = {}

    # Identify refs that need passage retrieval
    refs_needing_passages = []
    for ref in parsed.references:
        exist = exist_map.get(ref.ref_id)
        if not exist or exist.status != "FOUND":
            continue
        citations = citation_groups.get(ref.ref_id, [])
        substantive = [c for c in citations if is_substantive_citation(c)]
        if not substantive:
            continue
        refs_needing_passages.append((ref, exist, substantive))

    if not refs_needing_passages:
        return passages_by_ref, fulltext_by_ref

    log.info(f"Pre-retrieving passages for {len(refs_needing_passages)} references")

    cache = APICache()
    async with httpx.AsyncClient(follow_redirects=True) as client:
        for ref, exist, substantive in refs_needing_passages:
            try:
                ft = await get_full_text(exist, client, cache)
                fulltext_by_ref[ref.ref_id] = ft

                text = ft.full_text or ft.abstract or ""
                if not text.strip():
                    continue

                sections = ft.sections if ft.sections else None
                chunks = chunk_text(text, sections=sections)
                if not chunks:
                    continue

                ref_passages: dict[str, list] = {}
                for cit in substantive:
                    query = build_retrieval_query(
                        cit.citing_sentence, cit.context_before, cit.context_after,
                        markers=[cit.marker] if cit.marker else None,
                    )
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
                    ref_passages[cit.citing_sentence] = scored

                passages_by_ref[ref.ref_id] = ref_passages

            except Exception as e:
                log.warning(f"Passage pre-retrieval failed for {ref.ref_id}: {e}")
                continue

    await cache.close()
    return passages_by_ref, fulltext_by_ref
