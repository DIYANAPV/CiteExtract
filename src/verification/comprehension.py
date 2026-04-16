"""Comprehension support — passage retrieval from cited papers.

Given a citing sentence and the text of the cited paper, retrieves the most
relevant passage(s) from the cited paper. This is the core research
contribution for TPDL 2026.

Approach (adapted from Citation Integrity (Sarol et al. 2024) and
SemanticCite (Haan 2025)):
  Phase A: BM25 sparse retrieval
  Phase B: Dense embeddings + reciprocal rank fusion (RRF)

Dense model is lazy-loaded on first call to avoid startup cost when
only BM25 is needed (e.g. --quick mode).
"""

import functools
import logging
import re
from typing import Optional

import bm25s
import numpy as np

from src.models.comprehension import Chunk, ScoredChunk

log = logging.getLogger(__name__)

# Minimum words for a chunk to be useful
_MIN_CHUNK_WORDS = 10

# Target chunk size — 512 chars based on Vectara NAACL 2025 study and
# Feb 2026 benchmarks (recursive 512-char splitting ranked first at 69%
# end-to-end accuracy, outperforming semantic and larger fixed-size chunks).
_MAX_CHUNK_CHARS = 512

# Overlap between consecutive chunks (chars). Ensures sentences near chunk
# boundaries appear fully in at least one chunk. 10-20% of chunk size is
# the recommended range (Vectara NAACL 2025).
_CHUNK_OVERLAP = 50

# Separator hierarchy for recursive splitting — prefer natural boundaries.
# Order: paragraph breaks → sentence endings → clause breaks → word breaks.
_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    sections: Optional[list[dict]] = None,
) -> list[Chunk]:
    """Split paper text into chunks using recursive character splitting.

    Uses a separator hierarchy (paragraph → sentence → clause → word) to
    split text into chunks of ~512 characters with 50-char overlap. Prefers
    natural boundaries so sentences aren't split mid-thought.

    If *sections* is provided (list of {"name": str, "text": str}), chunks
    preserve section names. Otherwise, splits plain text directly.

    Args:
        text: Full text of the paper (or abstract if full text unavailable).
        sections: Optional structured sections from GROBID / parser.

    Returns:
        List of Chunk objects with text, section name, and paragraph index.
    """
    if sections:
        return _chunk_from_sections(sections)
    return _chunk_from_plain_text(text)


def _chunk_from_sections(sections: list[dict]) -> list[Chunk]:
    """Chunk text that has section structure (e.g. from GROBID)."""
    chunks: list[Chunk] = []
    idx = 0
    for section in sections:
        name = section.get("name", "")
        text = section.get("text", "")
        if not text.strip():
            continue
        pieces = _recursive_split(text.strip())
        pieces = _add_overlap(pieces)
        for piece in pieces:
            if len(piece.split()) < _MIN_CHUNK_WORDS:
                continue
            chunks.append(Chunk(text=piece, section_name=name or None, paragraph_index=idx))
            idx += 1
    return chunks


def _chunk_from_plain_text(text: str) -> list[Chunk]:
    """Chunk unstructured text using recursive splitting."""
    chunks: list[Chunk] = []
    idx = 0
    pieces = _recursive_split(text.strip())
    pieces = _add_overlap(pieces)
    for piece in pieces:
        if len(piece.split()) < _MIN_CHUNK_WORDS:
            continue
        chunks.append(Chunk(text=piece, section_name=None, paragraph_index=idx))
        idx += 1
    return chunks


def _recursive_split(
    text: str,
    max_chars: int = _MAX_CHUNK_CHARS,
    separators: list[str] | None = None,
) -> list[str]:
    """Recursively split text using a separator hierarchy.

    Tries each separator in order, preferring paragraph breaks over sentence
    endings over clause breaks over word boundaries. If a piece is still too
    long after splitting on the current separator, recurses with the next
    separator in the hierarchy.

    Args:
        text: Text to split.
        max_chars: Maximum chunk size in characters.
        separators: Remaining separators to try (defaults to _SEPARATORS).

    Returns:
        List of text pieces, each <= max_chars (best effort).
    """
    if not separators:
        separators = list(_SEPARATORS)

    # Base case: text fits in one chunk
    if len(text) <= max_chars:
        return [text] if text.strip() else []

    # Try each separator
    for i, sep in enumerate(separators):
        parts = text.split(sep)
        if len(parts) <= 1:
            continue

        # Merge parts back into chunks that fit under max_chars.
        # Re-attach the separator to the end of each part (except the last)
        # so we don't lose sentence-ending periods, etc.
        result: list[str] = []
        current = ""
        for j, part in enumerate(parts):
            # Re-attach separator (it was consumed by split)
            piece = part + sep if j < len(parts) - 1 else part

            if not current:
                current = piece
            elif len(current) + len(piece) <= max_chars:
                current += piece
            else:
                if current.strip():
                    result.append(current.strip())
                current = piece

        if current.strip():
            result.append(current.strip())

        # If we made progress (more than 1 piece), recurse on oversized pieces
        if len(result) > 1 or (len(result) == 1 and len(result[0]) <= max_chars):
            final: list[str] = []
            remaining_seps = separators[i + 1:]
            for piece in result:
                if len(piece) > max_chars and remaining_seps:
                    final.extend(_recursive_split(piece, max_chars, remaining_seps))
                else:
                    final.append(piece)
            return final

    # No separator worked — return text as-is (best effort)
    return [text] if text.strip() else []


def _add_overlap(pieces: list[str], overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """Add overlap between consecutive chunks.

    The last `overlap` characters of chunk N are prepended to chunk N+1.
    This ensures sentences near chunk boundaries appear fully in at least
    one chunk, making them retrievable.

    Args:
        pieces: List of text chunks from _recursive_split().
        overlap: Number of characters to overlap.

    Returns:
        Chunks with overlap added. First chunk is unchanged.
    """
    if len(pieces) <= 1 or overlap <= 0:
        return pieces

    result = [pieces[0]]
    for i in range(1, len(pieces)):
        prev = pieces[i - 1]
        # Take the last `overlap` chars of previous chunk, break at word boundary
        tail = prev[-overlap:]
        # Don't start mid-word: find first space
        space_idx = tail.find(" ")
        if space_idx >= 0:
            tail = tail[space_idx + 1:]
        result.append(tail + " " + pieces[i] if tail else pieces[i])
    return result


# ---------------------------------------------------------------------------
# BM25 retrieval (Phase A)
# ---------------------------------------------------------------------------


def retrieve_passages_bm25(
    query: str,
    chunks: list[Chunk],
    top_k: int = 3,
) -> list[ScoredChunk]:
    """Retrieve the most relevant chunks for a citing sentence using BM25.

    Uses bm25s (fast sparse retrieval) following the approach from
    Citation Integrity (Sarol et al. 2024) who showed BM25 is a strong
    baseline for citance-to-passage matching.

    Args:
        query: The citing sentence (the claim to match against).
        chunks: List of Chunk objects from chunk_text().
        top_k: Number of top passages to return.

    Returns:
        List of ScoredChunk sorted by descending BM25 score.
    """
    if not chunks:
        return []

    corpus = [c.text for c in chunks]

    # bm25s tokenize + index
    corpus_tokens = bm25s.tokenize(corpus, show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)

    # Query
    query_tokens = bm25s.tokenize([query], show_progress=False)
    k = min(top_k, len(chunks))
    results, scores = retriever.retrieve(query_tokens, k=k)

    scored: list[ScoredChunk] = []
    for i in range(results.shape[1]):
        idx = int(results[0, i])
        score = float(scores[0, i])
        if score <= 0:
            continue
        scored.append(ScoredChunk(chunk=chunks[idx], bm25_score=score))

    return scored


# ---------------------------------------------------------------------------
# Dense retrieval (Phase B)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=2)
def _get_dense_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load a sentence-transformers model. Cached via lru_cache.

    Using all-MiniLM-L6-v2 as default (80MB, CPU-friendly, 384-dim).
    Can be swapped to allenai/aspire-sentence-embedder (science-specific)
    via config.yaml if needed after CL-SciSumm evaluation.
    """
    from sentence_transformers import SentenceTransformer

    log.info(f"Loading dense model: {model_name}")
    return SentenceTransformer(model_name)


def retrieve_passages_dense(
    query: str,
    chunks: list[Chunk],
    top_k: int = 10,
    model_name: str = "all-MiniLM-L6-v2",
) -> list[ScoredChunk]:
    """Retrieve relevant chunks using dense embeddings + cosine similarity.

    Follows SemanticCite's approach of embedding query + chunks with
    sentence-transformers, but without the ChromaDB/LangChain overhead.

    Args:
        query: The citing sentence.
        chunks: Chunks from chunk_text().
        top_k: Number of top passages to return.
        model_name: HuggingFace model name for sentence-transformers.

    Returns:
        List of ScoredChunk sorted by descending cosine similarity.
    """
    if not chunks:
        return []

    model = _get_dense_model(model_name)
    corpus = [c.text for c in chunks]

    # Encode query and corpus
    query_emb = model.encode([query], normalize_embeddings=True)
    corpus_emb = model.encode(corpus, normalize_embeddings=True, show_progress_bar=False)

    # Cosine similarity (embeddings are normalized, so dot product = cosine)
    similarities = (query_emb @ corpus_emb.T)[0]

    # Top-k indices
    k = min(top_k, len(chunks))
    top_indices = np.argsort(similarities)[::-1][:k]

    scored: list[ScoredChunk] = []
    for idx in top_indices:
        sim = float(similarities[idx])
        if sim <= 0:
            continue
        scored.append(ScoredChunk(
            chunk=chunks[idx],
            bm25_score=0.0,
            dense_score=sim,
        ))

    return scored


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    rankings: list[list[tuple[int, float]]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Merge multiple ranked lists via Reciprocal Rank Fusion.

    RRF_score(d) = sum(1 / (k + rank_i(d))) for each ranking system i.

    This is the standard fusion method used by SemanticCite and others.
    Simple, parameter-free (k=60 is standard), and robust.

    Args:
        rankings: List of ranked lists. Each is [(doc_index, score), ...],
                  sorted by descending score.
        k: RRF constant (default 60, from Cormack et al. 2009).

    Returns:
        Merged ranked list [(doc_index, rrf_score), ...], sorted descending.
    """
    rrf_scores: dict[int, float] = {}

    for ranking in rankings:
        for rank, (doc_idx, _score) in enumerate(ranking):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return merged


# ---------------------------------------------------------------------------
# Neural reranking (FlashRank)
# ---------------------------------------------------------------------------


def _rerank_with_flashrank(
    query: str,
    candidates: list[tuple[int, float]],
    chunks: list[Chunk],
    top_k: int = 3,
) -> list[tuple[int, float]] | None:
    """Rerank RRF candidates using FlashRank cross-encoder.

    FlashRank reads each (query, passage) pair jointly, scoring real
    relevance rather than merging ranked lists. This catches passages
    that BM25 or dense retrieval ranked wrong.

    Falls back to None if flashrank is not installed, so the pipeline
    degrades gracefully to RRF-only.

    Args:
        query: The citing sentence (retrieval query).
        candidates: RRF-merged candidates as [(chunk_index, rrf_score), ...].
        chunks: Full chunk list for text lookup.
        top_k: Number of reranked results to return.

    Returns:
        Reranked list [(chunk_index, rerank_score), ...], or None if
        flashrank is unavailable.
    """
    try:
        from flashrank import Ranker, RerankRequest
    except ImportError:
        return None

    if not candidates:
        return None

    # Build passages for FlashRank
    passages = []
    idx_map = {}  # flashrank_position -> chunk_index
    for i, (chunk_idx, _score) in enumerate(candidates):
        passages.append({"id": i, "text": chunks[chunk_idx].text})
        idx_map[i] = chunk_idx

    try:
        ranker = _get_flashrank_ranker()
        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)

        reranked = []
        for r in results[:top_k]:
            orig_idx = idx_map[r["id"]]
            reranked.append((orig_idx, float(r["score"])))
        return reranked
    except Exception as e:
        log.warning(f"FlashRank reranking failed, falling back to RRF: {e}")
        return None


@functools.lru_cache(maxsize=1)
def _get_flashrank_ranker():
    """Lazy-load FlashRank ranker. Cached — loads model once."""
    from flashrank import Ranker
    log.info("Loading FlashRank reranker (ms-marco-MiniLM-L-12-v2)")
    return Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank")


# ---------------------------------------------------------------------------
# Hybrid retrieval (Phase B)
# ---------------------------------------------------------------------------


def retrieve_passages_hybrid(
    query: str,
    chunks: list[Chunk],
    top_k: int = 3,
    model_name: str = "all-MiniLM-L6-v2",
    bm25_candidates: int = 10,
    dense_candidates: int = 10,
    rrf_k: int = 60,
) -> list[ScoredChunk]:
    """Hybrid BM25 + dense retrieval with RRF fusion and neural reranking.

    Pipeline: BM25(10) + Dense(10) → RRF merge → FlashRank rerank → top k.
    Falls back to RRF-only if FlashRank is not installed.

    Args:
        query: The citing sentence.
        chunks: Chunks from chunk_text().
        top_k: Number of final passages to return.
        model_name: Dense embedding model name.
        bm25_candidates: How many BM25 candidates to feed into RRF.
        dense_candidates: How many dense candidates to feed into RRF.
        rrf_k: RRF constant (default 60).

    Returns:
        List of ScoredChunk with bm25_score, dense_score, and rrf_score.
    """
    if not chunks:
        return []

    # Run both retrievers
    bm25_results = retrieve_passages_bm25(query, chunks, top_k=bm25_candidates)
    dense_results = retrieve_passages_dense(query, chunks, top_k=dense_candidates, model_name=model_name)

    # Build index maps: chunk paragraph_index -> position in chunks list
    chunk_idx_map: dict[int, int] = {}
    for i, c in enumerate(chunks):
        chunk_idx_map[c.paragraph_index] = i

    # Convert to (chunk_list_index, score) rankings
    bm25_ranking = [(chunk_idx_map[r.chunk.paragraph_index], r.bm25_score) for r in bm25_results]
    dense_ranking = [(chunk_idx_map[r.chunk.paragraph_index], r.dense_score or 0.0) for r in dense_results]

    # Score lookup maps
    bm25_scores = {idx: score for idx, score in bm25_ranking}
    dense_scores = {idx: score for idx, score in dense_ranking}

    # Fuse
    merged = reciprocal_rank_fusion([bm25_ranking, dense_ranking], k=rrf_k)

    # Neural reranking: take top 15 from RRF, rerank with FlashRank cross-encoder.
    # Falls back to RRF order if FlashRank is not installed.
    reranked = _rerank_with_flashrank(query, merged[:15], chunks, top_k=top_k)
    if reranked is not None:
        final = reranked
    else:
        final = merged[:top_k]

    # Build output
    scored: list[ScoredChunk] = []
    for doc_idx, score in final:
        scored.append(ScoredChunk(
            chunk=chunks[doc_idx],
            bm25_score=bm25_scores.get(doc_idx, 0.0),
            dense_score=dense_scores.get(doc_idx),
            rrf_score=score,
        ))

    return scored


# ---------------------------------------------------------------------------
# Claim verification via retrieved passages (replaces abstract-based L4)
# ---------------------------------------------------------------------------

_CLAIM_SYSTEM_PROMPT = """\
Check whether a citing sentence accurately represents the cited paper, \
based on retrieved passages.

Classify as one of:

- SUPPORTS — the passages confirm what the sentence claims about this \
  paper. For citations that just acknowledge a model, tool, or dataset \
  ("we used X [1]"), it is enough that the paper describes X.
- CONTRADICTS — the passages say something different, or the sentence \
  exaggerates or cherry-picks from this paper.
- NEUTRAL — the passages do not clearly address the specific claim. \
  Normal for background citations.

If a sentence cites multiple papers, focus only on what is attributed \
to this particular reference based on where the citation marker sits.

Ground your judgment in the provided passages only. Quote the relevant \
text."""


def strip_citation_markers(text: str, markers: list[str] | None = None) -> str:
    """Remove known citation markers from text for cleaner retrieval queries.

    Only removes markers that were already detected by the citation detector,
    so no risk of accidentally removing claim content.

    Args:
        text: The text to clean (citing sentence or context).
        markers: Exact marker strings to remove, e.g. ["[23]", "(Smith et al., 2020)"].
                 If None or empty, returns text unchanged.

    Returns:
        Text with markers removed and whitespace normalized.
    """
    if not markers:
        return text
    cleaned = text
    for marker in markers:
        cleaned = cleaned.replace(marker, " ", 1)
    # Collapse multiple spaces into one
    return " ".join(cleaned.split())


def build_retrieval_query(
    citing_sentence: str,
    context_before: str = "",
    context_after: str = "",
    markers: list[str] | None = None,
) -> str:
    """Build a retrieval query from the citing sentence and surrounding context.

    Using the full context (before + citing + after) gives BM25 and dense
    retrieval more keywords and semantic signal to find relevant passages.

    If markers are provided, they are stripped from the query so that
    citation markers (e.g. "[23]", "(Smith et al., 2020)") don't pollute
    BM25 term matching or dense embeddings. The original text is preserved
    for LLM prompts — see _build_claim_prompt().

    Args:
        citing_sentence: The sentence containing the citation.
        context_before: Sentences above the citing sentence.
        context_after: Sentences below the citing sentence.
        markers: Citation marker strings to strip from the query.

    Returns:
        Cleaned query string for retrieval.
    """
    parts = []
    if context_before:
        parts.append(context_before.strip())
    parts.append(citing_sentence.strip())
    if context_after:
        parts.append(context_after.strip())
    query = " ".join(parts)
    return strip_citation_markers(query, markers)


def _build_claim_prompt(
    citing_sentence: str,
    passages: list[ScoredChunk],
    context_before: str = "",
    context_after: str = "",
) -> str:
    """Build the user prompt for claim verification."""
    passage_text = ""
    for i, sc in enumerate(passages, 1):
        section = f" (Section: {sc.chunk.section_name})" if sc.chunk.section_name else ""
        passage_text += f"\n### Passage {i}{section}\n{sc.chunk.text}\n"

    # Show the citing sentence with surrounding context for the LLM
    context_parts = []
    if context_before:
        context_parts.append(context_before.strip())
    context_parts.append(f"**{citing_sentence.strip()}**")
    if context_after:
        context_parts.append(context_after.strip())
    full_context = " ".join(context_parts)

    return f"""## Citing Context
{full_context}

(The bold sentence is the specific claim to verify.)

## Retrieved Passages from Cited Paper
{passage_text}

## Task
Do the passages support what the citing sentence claims about this paper?

Respond in JSON:
{{"verdict": "SUPPORTS or CONTRADICTS or NEUTRAL",
  "explanation": "your reasoning",
  "evidence_quote": "most relevant quote from the passages"}}"""


async def verify_claim_with_passages(
    citing_sentence: str,
    passages: list[ScoredChunk],
    llm_client,
    context_before: str = "",
    context_after: str = "",
) -> dict:
    """LLM compares citing sentence against retrieved passages.

    Args:
        citing_sentence: The claim being checked.
        passages: Top retrieved passages from the cited paper.
        llm_client: An LLMClient instance with raw_chat() method.
        context_before: Sentences before the citing sentence.
        context_after: Sentences after the citing sentence.

    Returns:
        dict with keys: verdict (SUPPORTS/CONTRADICTS/NEUTRAL),
        explanation, evidence_quote.
    """
    import json

    if not passages:
        return {
            "verdict": "NEUTRAL",
            "explanation": "No passages retrieved from the cited paper.",
            "evidence_quote": "",
        }

    prompt = _build_claim_prompt(citing_sentence, passages, context_before, context_after)

    try:
        raw = await llm_client.raw_chat(_CLAIM_SYSTEM_PROMPT, prompt, max_tokens=1024)
        data = json.loads(raw)
        verdict = data.get("verdict", "NEUTRAL").upper().strip()
        if verdict not in ("SUPPORTS", "CONTRADICTS", "NEUTRAL"):
            if "CONTRADICT" in verdict:
                verdict = "CONTRADICTS"
            elif "SUPPORT" in verdict:
                verdict = "SUPPORTS"
            else:
                verdict = "NEUTRAL"
        return {
            "verdict": verdict,
            "explanation": data.get("explanation", ""),
            "evidence_quote": data.get("evidence_quote", ""),
        }
    except Exception as e:
        log.warning(f"Claim verification LLM call failed: {e}")
        return {
            "verdict": "NEUTRAL",
            "explanation": f"LLM claim verification failed: {str(e)[:100]}",
            "evidence_quote": "",
        }
