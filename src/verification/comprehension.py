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

Two retrieval entry points are exposed:

- ``retrieve_passages_bm25 / _dense / _hybrid`` — one-shot calls that build
  the index internally. Convenient for ad-hoc lookups (agent tools, tests,
  the non-agentic comprehension fallback).
- ``build_retrieval_index`` + ``retrieve_with_index`` — split the work so a
  single document's index can be reused across many queries. Used in the
  agentic hot path where one cited paper is queried by every citing
  sentence and every multi-query sub-claim.
"""

import functools
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import bm25s
import numpy as np

from src.models.comprehension import Chunk, ScoredChunk

log = logging.getLogger(__name__)

# Minimum words for a chunk to be useful
_MIN_CHUNK_WORDS = 10

# Minimum chars for a chunk to carry meaningful context. Short tail pieces
# from `_recursive_split` (e.g. a section that ends 90 chars after the last
# sentence boundary) get merged into their neighbour to avoid surfacing
# 1-sentence "passages" that the model and human reader can't reason about.
# Set ~40% of MAX so two short pieces still combine to under 1.4 × MAX.
_MIN_CHUNK_CHARS = 200

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


# Sentence boundary regexes used by `_snap_to_sentences`.
# `_LEADING_SENT_BOUNDARY` finds the first `.|?|!` followed by whitespace and
# a likely sentence-start character (capital letter, opening quote, paren, or
# digit) — used to skip a leading partial sentence.
# `_TRAILING_SENT_END` finds any `.|?|!` followed by whitespace or end-of-text
# — used to trim a trailing partial sentence.
_LEADING_SENT_BOUNDARY = re.compile(r'[.!?]\s+(?=["\'(\[]?[A-Z0-9])')
_TRAILING_SENT_END = re.compile(r'[.!?](?=\s|$)')


def _snap_to_sentences(text: str) -> str:
    """Trim leading/trailing partial sentences from a chunk.

    Recursive splitting + 50-char overlap can leave a chunk starting
    mid-sentence (overlap inherits a fragment from the previous chunk)
    or ending mid-sentence (the splitter fell back to a non-sentence
    separator to fit max_chars). This produces hard-to-read passages
    in the UI.

    Snap forward to the first sentence start when the chunk doesn't
    begin with a capital letter, and trim everything after the last
    sentence-ending punctuation. If snapping would shrink the chunk
    below the min-words threshold, return the original — better an
    imperfect chunk than dropping the content entirely.
    """
    stripped = text.strip()
    if not stripped:
        return stripped

    snapped = stripped

    # Trim leading partial: chunks that genuinely start a sentence begin
    # with a capital letter (or an opening quote/paren wrapping one).
    # Anything else is a fragment — skip to the next sentence boundary.
    if not snapped[:1].isupper() and not (
        snapped[:1] in '"\'([' and snapped[1:2].isupper()
    ):
        m = _LEADING_SENT_BOUNDARY.search(snapped)
        if m:
            snapped = snapped[m.end():].lstrip()

    # Trim trailing partial: cut after the last `.|?|!` followed by
    # whitespace or end-of-text. If the chunk already ends cleanly,
    # the regex matches at the final character and this is a no-op.
    matches = list(_TRAILING_SENT_END.finditer(snapped))
    if matches:
        snapped = snapped[:matches[-1].end()].rstrip()

    if len(snapped.split()) < _MIN_CHUNK_WORDS:
        return stripped
    return snapped


def _merge_short_pieces(
    pieces: list[str], min_chars: int = _MIN_CHUNK_CHARS,
) -> list[str]:
    """Combine pieces shorter than ``min_chars`` with their neighbour.

    ``_recursive_split`` produces a max-512-char piece per slice, but a
    section's tail (or a section that's intrinsically short) can leave
    pieces well under that ceiling — landing in the UI as 1-2 sentence
    "passages" that neither the model nor the human reader can use.

    Strategy: walk forward, and any piece below ``min_chars`` gets merged
    into the previous piece. If the very first piece is short, it is
    merged into the next instead. The result is a list whose pieces are
    each ≥ ``min_chars`` (best effort) — the ceiling can drift above
    ``_MAX_CHUNK_CHARS`` in pathological merges, which is acceptable
    because the alternative is a fragment the system cannot reason about.
    """
    if not pieces:
        return pieces
    out: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        if out and len(piece) < min_chars:
            # Merge into the previous piece. Insert a space if needed so we
            # don't smash sentences together with no separator.
            sep = "" if out[-1].endswith((" ", "\n", "\t")) else " "
            out[-1] = out[-1] + sep + piece
        else:
            out.append(piece)
    # Edge case: the very first piece was short — merged into the second
    # via the walk above only if there was a `out` to merge into. If
    # pieces[0] was the ONLY short one and there are subsequent pieces,
    # the loop already left it as the first entry. Re-check head.
    if len(out) >= 2 and len(out[0]) < min_chars:
        sep = "" if out[0].endswith((" ", "\n", "\t")) else " "
        out[1] = out[0] + sep + out[1]
        out.pop(0)
    return out


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
        pieces = _merge_short_pieces(pieces)
        pieces = _add_overlap(pieces)
        for piece in pieces:
            piece = _snap_to_sentences(piece)
            # Drop pieces too short to give the LLM or the human reader
            # useful context. The merge step above handles multi-piece
            # sections; intrinsically tiny sections (e.g. a 90-char
            # caption-style section) get filtered here.
            if len(piece) < _MIN_CHUNK_CHARS or len(piece.split()) < _MIN_CHUNK_WORDS:
                continue
            chunks.append(Chunk(text=piece, section_name=name or None, paragraph_index=idx))
            idx += 1
    return chunks


def _chunk_from_plain_text(text: str) -> list[Chunk]:
    """Chunk unstructured text using recursive splitting."""
    chunks: list[Chunk] = []
    idx = 0
    pieces = _recursive_split(text.strip())
    pieces = _merge_short_pieces(pieces)
    pieces = _add_overlap(pieces)
    for piece in pieces:
        piece = _snap_to_sentences(piece)
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
# Reusable retrieval index
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalIndex:
    """Pre-built BM25 + (optional) dense state for one document.

    Build once per document, query many times. Reusing the index across
    citing sentences and sub-claims for the same cited paper avoids
    re-tokenizing the BM25 corpus and re-encoding it for the dense model
    on every query.

    Treat the underscore-prefixed fields as private — call
    ``retrieve_with_index`` instead of touching them directly.
    """

    chunks: list[Chunk]
    _bm25: Optional[bm25s.BM25] = None
    _model_name: Optional[str] = None
    _corpus_emb: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def has_dense(self) -> bool:
        return self._corpus_emb is not None


def build_retrieval_index(
    chunks: list[Chunk],
    model_name: Optional[str] = None,
) -> RetrievalIndex:
    """Tokenize + index chunks once for repeated retrieval.

    When ``model_name`` is supplied, the corpus is also pre-encoded so
    dense queries don't pay the encoding cost on every call. Pass
    ``model_name=None`` for BM25-only.
    """
    if not chunks:
        return RetrievalIndex(chunks=[], _model_name=model_name)

    corpus = [c.text for c in chunks]

    corpus_tokens = bm25s.tokenize(corpus, show_progress=False)
    bm25 = bm25s.BM25()
    bm25.index(corpus_tokens, show_progress=False)

    corpus_emb: Optional[np.ndarray] = None
    if model_name:
        model = _get_dense_model(model_name)
        corpus_emb = model.encode(
            corpus, normalize_embeddings=True, show_progress_bar=False,
        )

    return RetrievalIndex(
        chunks=chunks,
        _bm25=bm25,
        _model_name=model_name,
        _corpus_emb=corpus_emb,
    )


def _bm25_top_k(query: str, index: RetrievalIndex, top_k: int) -> list[ScoredChunk]:
    """BM25 retrieval against a pre-built index."""
    if not index.chunks or index._bm25 is None:
        return []
    query_tokens = bm25s.tokenize([query], show_progress=False)
    k = min(top_k, len(index.chunks))
    results, scores = index._bm25.retrieve(query_tokens, k=k)
    scored: list[ScoredChunk] = []
    for i in range(results.shape[1]):
        idx = int(results[0, i])
        score = float(scores[0, i])
        if score <= 0:
            continue
        scored.append(ScoredChunk(chunk=index.chunks[idx], bm25_score=score))
    return scored


def _dense_top_k(query: str, index: RetrievalIndex, top_k: int) -> list[ScoredChunk]:
    """Dense retrieval against a pre-built index."""
    if not index.chunks or index._corpus_emb is None or index._model_name is None:
        return []
    model = _get_dense_model(index._model_name)
    query_emb = model.encode([query], normalize_embeddings=True)
    similarities = (query_emb @ index._corpus_emb.T)[0]
    k = min(top_k, len(index.chunks))
    top_indices = np.argsort(similarities)[::-1][:k]
    scored: list[ScoredChunk] = []
    for idx in top_indices:
        sim = float(similarities[idx])
        if sim <= 0:
            continue
        scored.append(ScoredChunk(
            chunk=index.chunks[idx], bm25_score=0.0, dense_score=sim,
        ))
    return scored


def retrieve_with_index(
    query: str,
    index: RetrievalIndex,
    *,
    top_k: int = 3,
    bm25_candidates: int = 10,
    dense_candidates: int = 10,
    rrf_k: int = 60,
    rerank_pool: int = 10,
) -> list[ScoredChunk]:
    """Hybrid (BM25 + dense + RRF + optional FlashRank) against a pre-built index.

    When the index has no dense embeddings, falls back to BM25-only and
    returns ScoredChunk objects with ``rrf_score=None`` — matching the
    shape returned by ``retrieve_passages_bm25`` so callers can treat
    both paths interchangeably.

    ``rerank_pool`` caps how many top-RRF candidates are sent to the
    cross-encoder. Lower = fewer forward passes per query.
    """
    if not index.chunks:
        return []

    bm25_results = _bm25_top_k(query, index, top_k=bm25_candidates)

    if not index.has_dense:
        # BM25-only: just take the top_k. No RRF, no rerank — matches the
        # shape of the legacy retrieve_passages_bm25 path.
        return bm25_results[:top_k]

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
    reranked = _rerank_with_flashrank(
        query, merged[:rerank_pool], index.chunks, top_k=top_k,
    )
    final = reranked if reranked is not None else merged[:top_k]

    scored: list[ScoredChunk] = []
    for doc_idx, score in final:
        scored.append(ScoredChunk(
            chunk=index.chunks[doc_idx],
            bm25_score=bm25_scores.get(doc_idx, 0.0),
            dense_score=dense_scores.get(doc_idx),
            rrf_score=score,
        ))
    return scored


# ---------------------------------------------------------------------------
# BM25 retrieval (Phase A) — one-shot wrappers
# ---------------------------------------------------------------------------


def retrieve_passages_bm25(
    query: str,
    chunks: list[Chunk],
    top_k: int = 3,
) -> list[ScoredChunk]:
    """One-shot BM25 retrieval (builds an index per call).

    Convenience wrapper around ``build_retrieval_index`` +
    ``_bm25_top_k``. Use ``build_retrieval_index`` directly if you'll
    issue more than one query against the same chunks.

    Uses bm25s (fast sparse retrieval) following the approach from
    Citation Integrity (Sarol et al. 2024) who showed BM25 is a strong
    baseline for citance-to-passage matching.
    """
    if not chunks:
        return []
    index = build_retrieval_index(chunks, model_name=None)
    return _bm25_top_k(query, index, top_k=top_k)


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
    """One-shot dense retrieval (builds an index per call).

    Convenience wrapper around ``build_retrieval_index`` +
    ``_dense_top_k``. Use ``build_retrieval_index`` directly if you'll
    issue more than one query against the same chunks.

    Follows SemanticCite's approach of embedding query + chunks with
    sentence-transformers, but without the ChromaDB/LangChain overhead.
    """
    if not chunks:
        return []
    index = build_retrieval_index(chunks, model_name=model_name)
    return _dense_top_k(query, index, top_k=top_k)


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
# Neural reranking — sentence-transformers CrossEncoder (preferred) with
# FlashRank as fallback
# ---------------------------------------------------------------------------
#
# The cross-encoder reads each (query, passage) pair jointly to score real
# relevance, catching passages that BM25 or dense retrieval ranked wrong.
#
# Two backends are wired up:
#  - sentence-transformers CrossEncoder (PyTorch). Predicts a list of
#    (query, passage) pairs in a single batched forward pass. This lets
#    the multi-query path score all sub-claims with one model invocation
#    instead of one per query — that's where the speedup comes from.
#  - FlashRank (ONNX). Older path. Kept as a fallback when the cross-
#    encoder model isn't available locally so deployments that already
#    rely on FlashRank don't break.
#
# Same model weights underneath (cross-encoder/ms-marco-MiniLM-L-12-v2),
# so scores are within ~1% relative across backends — close enough that
# top-k ranking is preserved on the queries we've seen. The score-parity
# test in tests/ pins this.

_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"


@functools.lru_cache(maxsize=1)
def _get_cross_encoder(model_name: str = _CROSS_ENCODER_MODEL):
    """Lazy-load a sentence-transformers CrossEncoder. Cached — model loads once."""
    from sentence_transformers import CrossEncoder
    log.info(f"Loading CrossEncoder reranker: {model_name}")
    return CrossEncoder(model_name, max_length=512)


def _cross_encoder_score_pairs(
    pairs: list[tuple[str, str]],
) -> Optional[list[float]]:
    """Score (query, passage) pairs in one batched forward pass.

    Returns one float per input pair, in [0, 1], in the same order.
    Returns None if sentence-transformers isn't available or the model
    fails to load — callers should then try ``_rerank_with_flashrank``
    or fall through to RRF-only.

    A sigmoid is applied to the model's raw logits so scores are
    bounded in [0, 1] — this matches FlashRank's prior output range and
    avoids surprising any downstream code (UI display, sort assumptions)
    that expected positive rerank scores. Sigmoid is monotonic, so
    ranking order is preserved.

    Why this exists: the multi-query path expands a citing sentence into
    1 + N sub-claims and reranks each query's candidate list against the
    same cited paper. Scoring all (query_i, passage_j) pairs in one
    invocation lets the cross-encoder batch them into a single forward
    pass instead of paying model-call overhead 1 + N times.
    """
    if not pairs:
        return []
    try:
        ce = _get_cross_encoder()
    except (ImportError, OSError, RuntimeError) as e:
        log.debug(f"CrossEncoder unavailable: {e}")
        return None
    try:
        logits = ce.predict(list(pairs), show_progress_bar=False)
    except (RuntimeError, ValueError) as e:
        log.warning(f"CrossEncoder predict failed: {e}")
        return None
    # Sigmoid to [0,1] — monotonic, preserves ranking, matches prior range.
    arr = np.asarray(logits, dtype=np.float64)
    probs = 1.0 / (1.0 + np.exp(-arr))
    return [float(p) for p in probs]


def _rerank_with_flashrank(
    query: str,
    candidates: list[tuple[int, float]],
    chunks: list[Chunk],
    top_k: int = 3,
) -> list[tuple[int, float]] | None:
    """Rerank RRF candidates with the cross-encoder.

    Prefers ``_cross_encoder_score_pairs`` (sentence-transformers,
    batchable). Falls back to FlashRank's ONNX runtime if the
    cross-encoder isn't available, then to None (caller falls back to
    RRF-only).

    Name kept for back-compat with existing call sites; behavior is now
    cross-encoder-first. Single-query callers don't see a measurable
    speed difference — the win comes from the batched multi-query path
    in ``_rerank_pair_batches``.

    Args:
        query: The citing sentence (retrieval query).
        candidates: RRF-merged candidates as [(chunk_index, rrf_score), ...].
        chunks: Full chunk list for text lookup.
        top_k: Number of reranked results to return.

    Returns:
        Reranked list [(chunk_index, rerank_score), ...], or None if all
        backends are unavailable.
    """
    if not candidates:
        return None

    pairs: list[tuple[str, str]] = [
        (query, chunks[chunk_idx].text) for chunk_idx, _ in candidates
    ]
    scores = _cross_encoder_score_pairs(pairs)
    if scores is not None:
        scored = [(candidates[i][0], scores[i]) for i in range(len(candidates))]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    # Fallback: FlashRank (ONNX). Same model weights, different runtime.
    try:
        from flashrank import RerankRequest
    except ImportError:
        return None

    passages = [
        {"id": i, "text": chunks[chunk_idx].text}
        for i, (chunk_idx, _) in enumerate(candidates)
    ]
    idx_map = {i: candidates[i][0] for i in range(len(candidates))}

    try:
        ranker = _get_flashrank_ranker()
        results = ranker.rerank(RerankRequest(query=query, passages=passages))
    except (RuntimeError, OSError, ValueError) as e:
        log.warning(f"FlashRank reranking failed, falling back to RRF: {e}")
        return None

    return [(idx_map[r["id"]], float(r["score"])) for r in results[:top_k]]


def _rerank_pair_batches(
    queries: list[str],
    candidates_per_query: list[list[tuple[int, float]]],
    chunks: list[Chunk],
    top_k: int = 3,
) -> Optional[list[list[tuple[int, float]]]]:
    """Rerank N queries' candidate lists with ONE cross-encoder forward pass.

    Builds the flat list ``[(q_1, p_1_1), (q_1, p_1_2), ..., (q_N, p_N_M)]``,
    scores it in a single batched call, then splits results back per query.

    Returns one reranked top_k list per input query, or None if the
    cross-encoder is unavailable. When None, callers should fall back to
    per-query reranking via ``_rerank_with_flashrank`` (which itself
    falls back to FlashRank then to RRF-only).
    """
    if not queries or len(queries) != len(candidates_per_query):
        return None

    pairs: list[tuple[str, str]] = []
    offsets: list[tuple[int, int]] = []
    chunk_idx_per_pair: list[int] = []

    for q, cands in zip(queries, candidates_per_query):
        start = len(pairs)
        for chunk_idx, _score in cands:
            pairs.append((q, chunks[chunk_idx].text))
            chunk_idx_per_pair.append(chunk_idx)
        offsets.append((start, len(pairs)))

    if not pairs:
        return [[] for _ in queries]

    scores = _cross_encoder_score_pairs(pairs)
    if scores is None:
        return None

    out: list[list[tuple[int, float]]] = []
    for start, end in offsets:
        per_query = [
            (chunk_idx_per_pair[i], scores[i]) for i in range(start, end)
        ]
        per_query.sort(key=lambda x: -x[1])
        out.append(per_query[:top_k])
    return out


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
    """One-shot hybrid retrieval (builds an index per call).

    Convenience wrapper around ``build_retrieval_index`` +
    ``retrieve_with_index``. Use those directly when issuing more than
    one query against the same chunks.

    Pipeline: BM25(10) + Dense(10) → RRF merge → FlashRank rerank → top k.
    Falls back to RRF-only if FlashRank is not installed.
    """
    if not chunks:
        return []
    index = build_retrieval_index(chunks, model_name=model_name)
    return retrieve_with_index(
        query, index,
        top_k=top_k,
        bm25_candidates=bm25_candidates,
        dense_candidates=dense_candidates,
        rrf_k=rrf_k,
    )


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
