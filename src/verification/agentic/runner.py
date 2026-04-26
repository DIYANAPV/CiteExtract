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
from src.models.comprehension import ClaimVerdict, Chunk, FullTextResult, ScoredChunk
from src.models.parsed_paper import ParsedPaper
from src.models.verdict import ExistenceResult
from src.verification.api_clients.llm_client import CostTracker
from src.utils.timing import stage
from src.verification.cache import APICache
from src.verification.metadata import MetadataResult
from src.verification.agentic.tools import ToolExecutor

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shape aliases for the data that flows through the agentic pipeline
# ---------------------------------------------------------------------------
# Output of fetch_and_chunk_for_refs / input to _pre_retrieve_passages.
# The list of chunks may be empty when the paper has neither full text nor
# abstract — the FullTextResult still carries source/error info.
PrefetchedChunks = dict[str, tuple[FullTextResult, list[Chunk]]]

# Output of _pre_retrieve_passages. Outer key is ref_id; inner key is the
# literal citing sentence (used verbatim as a lookup key by the claim agent).
PassagesByRef = dict[str, dict[str, list[ScoredChunk]]]

# Map from ref_id to the FullTextResult that was fetched for that reference.
FullTextByRef = dict[str, FullTextResult]


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
    ref_pdfs_dir: Optional[str] = None,
    prefetched_chunks: Optional[PrefetchedChunks] = None,
    cache: Optional[APICache] = None,
) -> tuple[list[CitationVerdict], float, PassagesByRef, FullTextByRef]:
    """Run agentic verification: rule-based triage + focused agents where needed.

    Clear-cut cases (exact matches, obvious fabrications) are resolved
    without LLM calls. Ambiguous cases get dispatched to focused
    MetadataAgent and/or ClaimAgent.

    Args:
        parsed: L1 parse output.
        exist_map: Pre-computed L2 existence results (ref_id → ExistenceResult).
        metadata_map: Pre-computed L3 metadata results (ref_id → MetadataResult).
        agentic_config: Config from config.yaml 'agentic' section.
        ref_pdfs_dir: Optional directory of user-uploaded reference PDFs.
        prefetched_chunks: Optional {ref_id: (FullTextResult, list[Chunk])}
            produced by ``fetch_and_chunk_for_refs`` called in parallel
            with L3. Skips the inline fetch when provided.

    Returns:
        (verdicts, cost_usd, passages_by_ref, fulltext_by_ref) — callers that
        also need a ComprehensionReport can feed the last two into
        ``build_comprehension_from_passages`` instead of running the
        comprehension pipeline a second time.
    """
    from src.verification.agentic.claim_agent import ClaimAgent
    from src.verification.agentic.metadata_agent import MetadataAgent
    from src.verification.agentic.verdict_merger import (
        fallback_to_quick,
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

    # Single CostTracker shared by pre-retrieve decompose calls AND by the
    # metadata / claim agents further down. Created here so decompose costs
    # are counted even when triage clears every reference.
    cost_tracker = CostTracker()

    # Cache ownership: caller may supply a shared APICache so fetches done
    # here (pre-retrieve, ToolExecutor) reuse entries across pipeline phases.
    # When not supplied, we own a scoped cache and close it before returning.
    owns_cache = cache is None
    if cache is None:
        cache = APICache()

    # --- Pre-retrieve passages for FOUND refs with substantive citations ---
    with stage("agentic_pre_retrieve", refs=len(parsed.references)):
        passages_by_ref, fulltext_by_ref = await _pre_retrieve_passages(
            parsed, exist_map, citation_groups, agentic_config, cost_tracker,
            ref_pdfs_dir=ref_pdfs_dir,
            prefetched_chunks=prefetched_chunks,
            cache=cache,
        )

    # --- Triage all references ---
    with stage("agentic_triage", refs=len(parsed.references)):
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
        if owns_cache:
            await cache.close()
        ordered = [verdicts[ref.ref_id] for ref in parsed.references if ref.ref_id in verdicts]
        # Return whatever pre-retrieve already spent on decompose calls
        return ordered, cost_tracker.estimated_cost_usd, passages_by_ref, fulltext_by_ref

    # --- Set up shared resources ---
    openai_client = AsyncOpenAI(api_key=api_key)
    max_concurrent = agentic_config.get("max_concurrent_agents", 15)
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
            verdict_classes=int(claim_cfg.get("verdict_classes", 3)),
        )

        # --- Dispatch metadata agents in parallel ---
        async def _run_metadata(tr):
            async with semaphore:
                return await _dispatch_metadata(tr, meta_agent, parsed)

        meta_tasks = {tr.ref_id: _run_metadata(tr) for tr in needs_metadata}
        if meta_tasks:
            with stage("agentic_metadata_dispatch", count=len(meta_tasks),
                       concurrency=max_concurrent):
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

        # --- Dispatch NEEDS_BOTH (sequential: metadata → claim) ---
        async def _run_both(tr):
            async with semaphore:
                return await _dispatch_both(tr, meta_agent, claim_agent, parsed)

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
    return ordered, cost_tracker.estimated_cost_usd, passages_by_ref, fulltext_by_ref


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------


async def _dispatch_metadata(tr, meta_agent, parsed) -> CitationVerdict:
    """Run metadata agent for a single triage result."""
    from src.verification.agentic.verdict_merger import merge_metadata_verdict

    ref = next(r for r in parsed.references if r.ref_id == tr.ref_id)
    user_msg = _build_metadata_msg(tr, ref, tr.existence)

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


def _build_citing_contexts(tr) -> list[dict]:
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


def _align_claim_verdicts(
    tr, claim_verdicts: Optional[list[ClaimVerdict]],
) -> dict[str, ClaimVerdict]:
    """Map each substantive citing sentence to its corresponding ClaimVerdict.

    The claim agent is asked for ``expected_count=len(substantive_citations)``
    verdicts in submission order, so we align by position. If the agent
    returns a different length we still align what we can but log a warning
    so the mismatch is visible in production.
    """
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


def _build_metadata_msg(tr, ref, exist) -> str:
    """Build metadata agent user message from triage result.

    Intentionally re-runs ``compare_titles`` and ``compare_authors`` even
    though L3 already computed similarity scores on the stored
    FieldComparisons. The agent-facing output is richer — word-level title
    diffs and author-truncation analysis — and those richer structures are
    not carried on FieldComparison. Re-running them is cheap (pure Python,
    no I/O) and keeps the agent prompt self-contained.
    """
    from src.verification.agentic.metadata_agent import build_metadata_user_message
    from src.verification.author_comparison import compare_authors
    from src.verification.title_comparison import compare_titles

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
    """Run claim agent for a single triage result."""
    from src.verification.agentic.verdict_merger import merge_claim_verdicts

    ref = next(r for r in parsed.references if r.ref_id == tr.ref_id)
    user_msg = _build_claim_msg(tr, ref, tr.existence)
    n_contexts = len(tr.substantive_citations)
    claim_verdicts = await claim_agent.verify_claims(user_msg, expected_count=n_contexts)
    verdict = merge_claim_verdicts(triage=tr, claim_verdicts=claim_verdicts)
    verdict.per_sentence_claim_verdicts = _align_claim_verdicts(tr, claim_verdicts)
    return verdict


async def _dispatch_both(tr, meta_agent, claim_agent, parsed) -> CitationVerdict:
    """Run metadata and claim agents concurrently, then merge both dimensions.

    No short-circuit: claim verification runs even when metadata comes back
    FABRICATED. The paper was found by L2, so its text is still available
    and the citing sentence can still be checked against it. The two
    verdicts are carried side-by-side on the returned CitationVerdict.

    Exception handling: the two agents run under ``return_exceptions=True``
    so a failure in one does not cancel the other. A meta failure is
    re-raised so the outer ``_run_both`` wrapper falls back; a claim failure
    is captured into ``claim_error`` and surfaced as a ``claim_agent_error``
    flag on the merged verdict (distinguishing it from "claim not
    applicable", where claim_verdicts is legitimately None).
    """
    from src.verification.agentic.verdict_merger import merge_both_verdicts

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

    # Fire both agents concurrently without sibling cancellation on failure.
    meta_raw, claim_raw = await asyncio.gather(
        _run_meta(), _run_claim(), return_exceptions=True,
    )

    # Meta failure: preserve existing behavior — re-raise so the outer
    # _run_both wrapper catches it and calls fallback_to_quick for this ref.
    if isinstance(meta_raw, BaseException):
        raise meta_raw
    meta_result = meta_raw

    # Claim failure: keep the metadata dimension and record the crash so
    # the merger can surface a claim_agent_error flag.
    claim_error: Optional[str] = None
    claim_verdicts: Optional[list] = None
    if isinstance(claim_raw, BaseException):
        log.warning(f"Claim agent failed for NEEDS_BOTH ref {tr.ref_id}: {claim_raw}")
        claim_error = str(claim_raw)[:200]
    else:
        claim_verdicts = claim_raw

    meta_verdict = meta_result.get("verdict", "UNVERIFIABLE")

    # Google Scholar re-investigation for non-VALID metadata verdicts. This
    # only affects the metadata dimension; the already-computed claim
    # verdict is preserved.
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


async def _google_scholar_reinvestigate_meta(
    ref, meta_agent, first_result: dict,
) -> Optional[dict]:
    """Second-chance Google Scholar investigation — dict form for NEEDS_BOTH.

    Returns an updated metadata agent result dict (same schema as
    ``meta_agent.investigate``) with a ``google_scholar_reinvestigated``
    flag appended, or ``None`` if the verdict did not change. Keeps the
    metadata dimension isolated so claim results can be merged separately.
    """
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


# ---------------------------------------------------------------------------
# Passage pre-retrieval (deterministic, no LLM)
# ---------------------------------------------------------------------------


def _identify_refs_needing_passages(
    parsed: ParsedPaper,
    exist_map: dict[str, ExistenceResult],
    citation_groups: dict[str, list[Citation]],
) -> list[tuple]:
    """Return [(ref, existence, substantive_citations)] for FOUND refs w/ claims."""
    from src.verification.filters import is_substantive_citation

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
    """Fetch full text and chunk it for every FOUND ref that has substantive
    citations. Public so the pipeline can launch this in parallel with L3
    metadata validation — the slow I/O is the waterfall fetch (S2 OA,
    Unpaywall, arXiv), not chunking.

    When ``cache`` is supplied, the caller owns its lifecycle. Otherwise a
    scoped cache is created and closed inside this function.

    Returns:
        {ref_id: (FullTextResult, list[Chunk])}
        — chunks may be empty if the paper has neither full text nor abstract.
    """
    from src.verification.api_clients.fulltext import get_full_text
    from src.verification.comprehension import chunk_text

    refs_needing = _identify_refs_needing_passages(parsed, exist_map, citation_groups)
    if not refs_needing:
        return {}

    user_pdf_map: dict[str, str] = {}
    if ref_pdfs_dir:
        try:
            from src.pipeline import _scan_ref_pdfs
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
                text = ft.full_text or ft.abstract or ""
                if not text.strip():
                    return ref.ref_id, ft, []
                sections = ft.sections if ft.sections else None
                chunks = chunk_text(text, sections=sections)
                return ref.ref_id, ft, chunks
            except (httpx.RequestError, httpx.HTTPError, OSError, ValueError) as e:
                # Expected failures: network hiccups, malformed PDFs, GROBID
                # XML parse errors. Anything else is a real bug — let it
                # bubble up to asyncio.gather(return_exceptions=True), which
                # logs it as a task exception instead of silently swallowing.
                log.warning(f"Fetch/chunk failed for {ref.ref_id}: {e}")
                return ref.ref_id, None, []

    # max_workers bounds how many fetch tasks are in flight at once, not the
    # actual throughput: per-API rate limiters (e.g. Semantic Scholar at 1
    # req/s) further serialize calls to the same source, so real wall-clock
    # throughput is min(max_workers, sum of per-source rate-limit budgets).
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
    cost_tracker: Optional[CostTracker] = None,
    ref_pdfs_dir: Optional[str] = None,
    prefetched_chunks: Optional[PrefetchedChunks] = None,
    cache: Optional[APICache] = None,
) -> tuple[PassagesByRef, FullTextByRef]:
    """Pre-retrieve passages for all FOUND refs with substantive citations.

    When ``prefetched_chunks`` is provided (``{ref_id: (FullTextResult,
    list[Chunk])}``), the fetch+chunk stage is skipped — the slow I/O has
    already been done in parallel with L3 by the caller. Otherwise the
    function fetches and chunks inline for backwards compatibility.

    When ``agentic.claim_agent.enable_multiquery`` is true and a dense
    model is configured, each citing sentence is decomposed into
    sub-claims (via one LLM call) and passages are retrieved for the
    full claim AND each sub-claim, then unioned + deduped.

    Returns:
        (passages_by_ref, fulltext_by_ref) where:
        - passages_by_ref: {ref_id: {citing_sentence: [ScoredChunk]}}
        - fulltext_by_ref: {ref_id: FullTextResult}
    """
    import time

    from src import config as app_cfg
    from src.models.comprehension import FullTextResult
    from src.utils.timing import stage
    from src.verification.api_clients.llm_client import create_llm_client
    from src.verification.comprehension import (
        build_retrieval_index,
        build_retrieval_query,
        retrieve_with_index,
    )
    from src.verification.multiquery import decompose, multi_query_retrieve

    comp_cfg = app_cfg.comprehension()
    top_k = comp_cfg.get("top_k", 3)
    dense_model = comp_cfg.get("dense_model")

    claim_cfg = config.get("claim_agent", {}) if config else {}
    multiquery_enabled = bool(claim_cfg.get("enable_multiquery", False)) and bool(dense_model)
    mq_cfg = claim_cfg.get("multiquery", {}) if multiquery_enabled else {}
    mq_n_sub = int(mq_cfg.get("n_sub_claims", 2))
    mq_full_top_k = int(mq_cfg.get("full_top_k", 3))
    mq_sub_top_k = int(mq_cfg.get("sub_top_k", 3))

    mq_llm = None
    if multiquery_enabled:
        try:
            mq_llm = create_llm_client(config)
            if cost_tracker is not None:
                mq_llm.cost_tracker = cost_tracker
        except Exception as e:
            log.warning(f"multi-query decomposition disabled — LLM client init failed: {e}")
            multiquery_enabled = False

    passages_by_ref: PassagesByRef = {}
    fulltext_by_ref: FullTextByRef = {}

    refs_needing = _identify_refs_needing_passages(parsed, exist_map, citation_groups)
    if not refs_needing:
        return passages_by_ref, fulltext_by_ref

    max_concurrent = int(config.get("max_concurrent_agents", 15)) if config else 15

    # Fetch+chunk either inline or take the pre-fetched data handed in by
    # the pipeline (which launched the fetch in parallel with L3). The
    # inline path is wrapped in its own STAGE so we can see how much of
    # the outer agentic_pre_retrieve block was network-bound work vs.
    # in-memory retrieval.
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
        f"(multiquery={multiquery_enabled}, concurrency={max_concurrent})"
    )

    semaphore = asyncio.Semaphore(max_concurrent)

    bm25_candidates = comp_cfg.get("bm25_candidates", 10)
    dense_candidates = comp_cfg.get("dense_candidates", 10)
    rrf_k = comp_cfg.get("rrf_k", 60)

    # Aggregated CPU cost across refs (wall-clock would understate the work
    # because retrieval runs concurrently). These counters give us the real
    # tuning signal: how much time goes into building indexes vs running
    # queries against them.
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
                # Build the index ONCE per ref. Every citing sentence (and
                # every multi-query sub-claim) for this ref reuses it, so
                # we tokenize for BM25 and encode for the dense model only
                # once instead of once per query.
                t_index_start = time.perf_counter()
                index = build_retrieval_index(
                    chunks, model_name=dense_model if dense_model else None,
                )
                ref_index_secs = time.perf_counter() - t_index_start

                t_retrieve_start = time.perf_counter()
                ref_passages: dict[str, list] = {}
                for cit in substantive:
                    query = build_retrieval_query(
                        cit.citing_sentence, cit.context_before, cit.context_after,
                        markers=[cit.marker] if cit.marker else None,
                    )
                    if multiquery_enabled and mq_llm is not None:
                        sub_claims = await decompose(
                            cit.citing_sentence, mq_llm, n=mq_n_sub,
                        )
                        scored = multi_query_retrieve(
                            full_claim=query,
                            sub_claims=sub_claims,
                            index=index,
                            bm25_candidates=bm25_candidates,
                            dense_candidates=dense_candidates,
                            rrf_k=rrf_k,
                            full_top_k=mq_full_top_k,
                            sub_top_k=mq_sub_top_k,
                        )
                    else:
                        scored = retrieve_with_index(
                            query, index,
                            top_k=top_k,
                            bm25_candidates=bm25_candidates,
                            dense_candidates=dense_candidates,
                            rrf_k=rrf_k,
                        )
                    ref_passages[cit.citing_sentence] = scored
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
            except (ValueError, KeyError, RuntimeError) as e:
                log.warning(f"Passage retrieval failed for {ref.ref_id}: {e}")
                return ref.ref_id, ft, None

    with stage("pre_retrieve.retrieve_block", refs=len(refs_needing)):
        tasks = [_retrieve_one(ref, substantive) for ref, _, substantive in refs_needing]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            log.warning(f"Passage retrieval task raised: {result}")
            continue
        ref_id, ft, ref_passages = result
        if ft is not None:
            fulltext_by_ref[ref_id] = ft
        if ref_passages is not None:
            passages_by_ref[ref_id] = ref_passages

    # Emit summed CPU cost across refs through the same logger that
    # stage() uses, so the post-run profile shows in-loop work alongside
    # wall-clock blocks. Wall-clock alone understates the retrieval cost
    # because work runs concurrently across refs.
    timing_log = logging.getLogger("checkcitation.timing")
    timing_log.info(
        "STAGE pre_retrieve.index_build_total seconds=%.3f refs=%d",
        index_build_total, len(refs_needing),
    )
    timing_log.info(
        "STAGE pre_retrieve.retrieve_total seconds=%.3f refs=%d",
        retrieve_total, len(refs_needing),
    )

    return passages_by_ref, fulltext_by_ref
