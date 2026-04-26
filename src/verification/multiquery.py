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
from src.verification.comprehension import RetrievalIndex, retrieve_with_index

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
) -> list[ScoredChunk]:
    """Retrieve with the full claim AND each sub-claim, union + dedupe.

    All queries hit the same pre-built ``index``, so we tokenize/encode
    the corpus only once per cited paper regardless of how many
    sub-claims we expand to.

    Order: full-claim hits first, then sub-claim hits (dedupe preserves
    the first occurrence of each unique chunk).
    """
    all_passages: list[ScoredChunk] = []

    all_passages.extend(retrieve_with_index(
        full_claim.strip(),
        index,
        top_k=full_top_k,
        bm25_candidates=bm25_candidates,
        dense_candidates=dense_candidates,
        rrf_k=rrf_k,
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
        ))

    return _dedup(all_passages)
