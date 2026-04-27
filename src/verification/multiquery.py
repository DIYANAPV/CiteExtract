"""Multi-query retrieval with claim decomposition.

Given a citing sentence, decompose it into sub-claims, retrieve passages
for the full claim AND each sub-claim, then union + dedupe the passage
set. The wider passage set is fed to the downstream verifier, which still
judges the *full* original claim (not the sub-claims) — so a hard-to-
retrieve atom cannot sink a record.

Adapted from new_architect/retrieval_v2/{decomposer.py,multi_query.py}.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.models.comprehension import Chunk, ScoredChunk
from src.verification.api_clients.llm_client import LLMClient
from src.verification.comprehension import (
    RetrievalIndex,
    _bm25_top_k,
    _dense_top_k,
    _rerank_pair_batches,
    reciprocal_rank_fusion,
    retrieve_with_index,
)

log = logging.getLogger(__name__)

_DECOMPOSE_PROMPT_PATH = Path(__file__).parent / "prompts" / "decompose.txt"
_DEDUP_PREFIX_CHARS = 100


def _load_decompose_prompt() -> str:
    return _DECOMPOSE_PROMPT_PATH.read_text(encoding="utf-8")


async def decompose(claim: str, llm_client: LLMClient, n: int = 2) -> list[str]:
    """Decompose a claim into `n` sub-claims via one LLM call.

    On failure (malformed JSON, LLM error), returns a singleton list
    containing the original claim — this makes the multi-query path fall
    back cleanly to single-query retrieval.

    On empty or whitespace-only input, returns an empty list (there's
    nothing to retrieve for).
    """
    if not claim or not claim.strip():
        return []
    system = _load_decompose_prompt()
    user = f"Claim: {claim.strip()}"
    try:
        raw = await llm_client.raw_chat(system, user, max_tokens=256)
        data = json.loads(raw)
    except Exception as e:
        log.debug(f"decompose LLM call failed: {e}")
        return [claim.strip()]
    if not isinstance(data, dict):
        return [claim.strip()]
    subs = data.get("sub_claims", [])
    if not isinstance(subs, list):
        return [claim.strip()]
    clean = [str(s).strip() for s in subs if str(s).strip()]
    # Trim to requested count, but always return at least the original on failure.
    return clean[:n] if clean else [claim.strip()]


def _prefix_key(text: str, n: int = _DEDUP_PREFIX_CHARS) -> str:
    return (text or "").strip().lower()[:n]


def _dedup(passages: list[ScoredChunk]) -> list[ScoredChunk]:
    """Dedupe by exact paragraph_index and by first-N-char prefix."""
    seen_idx: set[int] = set()
    seen_prefix: set[str] = set()
    out: list[ScoredChunk] = []
    for sc in passages:
        pidx = sc.chunk.paragraph_index
        if pidx in seen_idx:
            continue
        pref = _prefix_key(sc.chunk.text)
        if pref and pref in seen_prefix:
            continue
        seen_idx.add(pidx)
        if pref:
            seen_prefix.add(pref)
        out.append(sc)
    return out


def _rrf_candidates_for_query(
    query: str,
    index: RetrievalIndex,
    *,
    bm25_candidates: int,
    dense_candidates: int,
    rrf_k: int,
    pool: int,
) -> tuple[list[tuple[int, float]], dict[int, float], dict[int, float]]:
    """Run BM25 + dense + RRF for one query against ``index``, returning
    the top-``pool`` RRF candidates plus per-chunk BM25 / dense lookups
    so the caller can populate ScoredChunk fields after reranking.

    Returns: (rrf_candidates, bm25_scores, dense_scores) where:
      - rrf_candidates: [(chunk_list_index, rrf_score), ...] sorted desc, len <= pool
      - bm25_scores: {chunk_list_index: score}
      - dense_scores: {chunk_list_index: score}
    """
    bm25_results = _bm25_top_k(query, index, top_k=bm25_candidates)
    dense_results = _dense_top_k(query, index, top_k=dense_candidates)

    chunk_idx_map: dict[int, int] = {
        c.paragraph_index: i for i, c in enumerate(index.chunks)
    }
    bm25_ranking = [
        (chunk_idx_map[r.chunk.paragraph_index], r.bm25_score) for r in bm25_results
    ]
    dense_ranking = [
        (chunk_idx_map[r.chunk.paragraph_index], r.dense_score or 0.0)
        for r in dense_results
    ]
    bm25_scores = {idx: score for idx, score in bm25_ranking}
    dense_scores = {idx: score for idx, score in dense_ranking}
    merged = reciprocal_rank_fusion([bm25_ranking, dense_ranking], k=rrf_k)
    return merged[:pool], bm25_scores, dense_scores


def _build_query_list(full_claim: str, sub_claims: list[str]) -> list[str]:
    """Order: full claim first, then sub-claims (excluding empty/duplicate).

    Matches the per-query behavior of ``multi_query_retrieve`` so that
    dedup preserves the same chunk-discovery order.
    """
    queries: list[str] = []
    full = full_claim.strip()
    if full:
        queries.append(full)
    for sub in sub_claims:
        s = sub.strip()
        if not s or s == full:
            continue
        queries.append(s)
    return queries


def multi_query_retrieve(
    full_claim: str,
    sub_claims: list[str],
    index: RetrievalIndex,
    *,
    bm25_candidates: int,
    dense_candidates: int,
    rrf_k: int,
    full_top_k: int = 3,
    sub_top_k: int = 3,
    rerank_pool: int = 10,
) -> list[ScoredChunk]:
    """Retrieve with the full claim AND each sub-claim, union + dedupe.

    All queries hit the same pre-built ``index``, so BM25 + dense are
    computed against pre-tokenized / pre-encoded corpora. With a
    cross-encoder available, all queries' rerank pools are scored in one
    batched forward pass — eliminating the per-query model-call overhead
    that dominated the post-RRF stage.

    When the cross-encoder is unavailable or the index lacks dense
    embeddings, falls back to the per-query path via ``retrieve_with_index``.

    Order: full-claim hits first, then sub-claim hits (dedupe preserves
    the first occurrence of each unique chunk).
    """
    queries = _build_query_list(full_claim, sub_claims)
    if not queries:
        return []

    # BM25-only fallback (no dense embeddings on the index): we have no
    # RRF / rerank stage anyway, so the per-query path produces identical
    # output and there's no batching opportunity to chase.
    if not index.has_dense:
        return _multi_query_per_query(
            full_claim, sub_claims, index,
            bm25_candidates=bm25_candidates,
            dense_candidates=dense_candidates,
            rrf_k=rrf_k,
            full_top_k=full_top_k,
            sub_top_k=sub_top_k,
            rerank_pool=rerank_pool,
        )

    # Compute RRF candidates per query (cheap; same as before but split out).
    rrf_per_query: list[list[tuple[int, float]]] = []
    score_lookups: list[tuple[dict[int, float], dict[int, float]]] = []
    for q in queries:
        cands, bm25_scores, dense_scores = _rrf_candidates_for_query(
            q, index,
            bm25_candidates=bm25_candidates,
            dense_candidates=dense_candidates,
            rrf_k=rrf_k,
            pool=rerank_pool,
        )
        rrf_per_query.append(cands)
        score_lookups.append((bm25_scores, dense_scores))

    # Per-query top_k targets: full claim uses full_top_k, sub-claims sub_top_k.
    per_query_top_k = [full_top_k] + [sub_top_k] * (len(queries) - 1)

    # ONE batched cross-encoder pass across all (query, passage) pairs.
    # Returns per-query reranked top-k (already sorted desc by score).
    reranked_per_query = _rerank_pair_batches(
        queries, rrf_per_query, index.chunks,
        # We pass the max top_k and slice per-query below — the helper
        # only takes one top_k. Use the largest so we don't truncate the
        # full-claim list when sub_top_k differs.
        top_k=max(per_query_top_k) if per_query_top_k else full_top_k,
    )

    # Cross-encoder unavailable → fall back to per-query path which itself
    # falls back further (FlashRank, then RRF-only).
    if reranked_per_query is None:
        return _multi_query_per_query(
            full_claim, sub_claims, index,
            bm25_candidates=bm25_candidates,
            dense_candidates=dense_candidates,
            rrf_k=rrf_k,
            full_top_k=full_top_k,
            sub_top_k=sub_top_k,
            rerank_pool=rerank_pool,
        )

    all_passages: list[ScoredChunk] = []
    for query_idx, reranked in enumerate(reranked_per_query):
        bm25_scores, dense_scores = score_lookups[query_idx]
        k = per_query_top_k[query_idx]
        for chunk_idx, rerank_score in reranked[:k]:
            all_passages.append(ScoredChunk(
                chunk=index.chunks[chunk_idx],
                bm25_score=bm25_scores.get(chunk_idx, 0.0),
                dense_score=dense_scores.get(chunk_idx),
                rrf_score=rerank_score,
            ))

    return _dedup(all_passages)


def _multi_query_per_query(
    full_claim: str,
    sub_claims: list[str],
    index: RetrievalIndex,
    *,
    bm25_candidates: int,
    dense_candidates: int,
    rrf_k: int,
    full_top_k: int,
    sub_top_k: int,
    rerank_pool: int,
) -> list[ScoredChunk]:
    """Per-query fallback path. Used when the batched cross-encoder is
    unavailable or the index has no dense embeddings.

    Behaviorally identical to the pre-batching implementation: each
    query gets its own retrieve_with_index call, results are unioned and
    deduped. Slower but always available.
    """
    all_passages: list[ScoredChunk] = []
    all_passages.extend(retrieve_with_index(
        full_claim.strip(),
        index,
        top_k=full_top_k,
        bm25_candidates=bm25_candidates,
        dense_candidates=dense_candidates,
        rrf_k=rrf_k,
        rerank_pool=rerank_pool,
    ))
    for sub in sub_claims:
        if not sub.strip() or sub.strip() == full_claim.strip():
            continue
        all_passages.extend(retrieve_with_index(
            sub.strip(),
            index,
            top_k=sub_top_k,
            bm25_candidates=bm25_candidates,
            dense_candidates=dense_candidates,
            rrf_k=rrf_k,
            rerank_pool=rerank_pool,
        ))
    return _dedup(all_passages)
