
import functools
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import bm25s
import numpy as np

from citeextract.models.comprehension import Chunk, ScoredChunk

log = logging.getLogger(__name__)

_MIN_CHUNK_WORDS = 10

_MIN_CHUNK_CHARS = 200

_MAX_CHUNK_CHARS = 512

_CHUNK_OVERLAP = 50

_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]


def chunk_text(
    text: str,
    sections: Optional[list[dict]] = None,
) -> list[Chunk]:
    if sections:
        return _chunk_from_sections(sections)
    return _chunk_from_plain_text(text)


_LEADING_SENT_BOUNDARY = re.compile(r'[.!?]\s+(?=["\'(\[]?[A-Z0-9])')
_TRAILING_SENT_END = re.compile(r'[.!?](?=\s|$)')


def _snap_to_sentences(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped

    snapped = stripped

    if not snapped[:1].isupper() and not (
        snapped[:1] in '"\'([' and snapped[1:2].isupper()
    ):
        m = _LEADING_SENT_BOUNDARY.search(snapped)
        if m:
            snapped = snapped[m.end():].lstrip()

    matches = list(_TRAILING_SENT_END.finditer(snapped))
    if matches:
        snapped = snapped[:matches[-1].end()].rstrip()

    if len(snapped.split()) < _MIN_CHUNK_WORDS:
        return stripped
    return snapped


def _merge_short_pieces(
    pieces: list[str], min_chars: int = _MIN_CHUNK_CHARS,
) -> list[str]:
    if not pieces:
        return pieces
    out: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        if out and len(piece) < min_chars:
            sep = "" if out[-1].endswith((" ", "\n", "\t")) else " "
            out[-1] = out[-1] + sep + piece
        else:
            out.append(piece)
    if len(out) >= 2 and len(out[0]) < min_chars:
        sep = "" if out[0].endswith((" ", "\n", "\t")) else " "
        out[1] = out[0] + sep + out[1]
        out.pop(0)
    return out


def _chunk_from_sections(sections: list[dict]) -> list[Chunk]:
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
            if len(piece) < _MIN_CHUNK_CHARS or len(piece.split()) < _MIN_CHUNK_WORDS:
                continue
            chunks.append(Chunk(text=piece, section_name=name or None, paragraph_index=idx))
            idx += 1
    return chunks


def _chunk_from_plain_text(text: str) -> list[Chunk]:
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
    if not separators:
        separators = list(_SEPARATORS)

    if len(text) <= max_chars:
        return [text] if text.strip() else []

    for i, sep in enumerate(separators):
        parts = text.split(sep)
        if len(parts) <= 1:
            continue

        result: list[str] = []
        current = ""
        for j, part in enumerate(parts):
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

        if len(result) > 1 or (len(result) == 1 and len(result[0]) <= max_chars):
            final: list[str] = []
            remaining_seps = separators[i + 1:]
            for piece in result:
                if len(piece) > max_chars and remaining_seps:
                    final.extend(_recursive_split(piece, max_chars, remaining_seps))
                else:
                    final.append(piece)
            return final

    return [text] if text.strip() else []


def _add_overlap(pieces: list[str], overlap: int = _CHUNK_OVERLAP) -> list[str]:
    if len(pieces) <= 1 or overlap <= 0:
        return pieces

    result = [pieces[0]]
    for i in range(1, len(pieces)):
        prev = pieces[i - 1]
        tail = prev[-overlap:]
        space_idx = tail.find(" ")
        if space_idx >= 0:
            tail = tail[space_idx + 1:]
        result.append(tail + " " + pieces[i] if tail else pieces[i])
    return result


@dataclass(frozen=True)
class RetrievalIndex:

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
    if not index.chunks:
        return []

    bm25_results = _bm25_top_k(query, index, top_k=bm25_candidates)

    if not index.has_dense:
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


def retrieve_passages_bm25(
    query: str,
    chunks: list[Chunk],
    top_k: int = 3,
) -> list[ScoredChunk]:
    if not chunks:
        return []
    index = build_retrieval_index(chunks, model_name=None)
    return _bm25_top_k(query, index, top_k=top_k)


@functools.lru_cache(maxsize=2)
def _get_dense_model(model_name: str = "all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer

    log.info(f"Loading dense model: {model_name}")
    return SentenceTransformer(model_name)


def retrieve_passages_dense(
    query: str,
    chunks: list[Chunk],
    top_k: int = 10,
    model_name: str = "all-MiniLM-L6-v2",
) -> list[ScoredChunk]:
    if not chunks:
        return []
    index = build_retrieval_index(chunks, model_name=model_name)
    return _dense_top_k(query, index, top_k=top_k)


def reciprocal_rank_fusion(
    rankings: list[list[tuple[int, float]]],
    k: int = 60,
) -> list[tuple[int, float]]:
    rrf_scores: dict[int, float] = {}

    for ranking in rankings:
        for rank, (doc_idx, _score) in enumerate(ranking):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return merged


_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"


@functools.lru_cache(maxsize=1)
def _get_cross_encoder(model_name: str = _CROSS_ENCODER_MODEL):
    from sentence_transformers import CrossEncoder
    log.info(f"Loading CrossEncoder reranker: {model_name}")
    return CrossEncoder(model_name, max_length=512)


def _cross_encoder_score_pairs(
    pairs: list[tuple[str, str]],
) -> Optional[list[float]]:
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
    arr = np.asarray(logits, dtype=np.float64)
    probs = 1.0 / (1.0 + np.exp(-arr))
    return [float(p) for p in probs]


def _rerank_with_flashrank(
    query: str,
    candidates: list[tuple[int, float]],
    chunks: list[Chunk],
    top_k: int = 3,
) -> list[tuple[int, float]] | None:
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
    from flashrank import Ranker
    log.info("Loading FlashRank reranker (ms-marco-MiniLM-L-12-v2)")
    return Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank")


def retrieve_passages_hybrid(
    query: str,
    chunks: list[Chunk],
    top_k: int = 3,
    model_name: str = "all-MiniLM-L6-v2",
    bm25_candidates: int = 10,
    dense_candidates: int = 10,
    rrf_k: int = 60,
) -> list[ScoredChunk]:
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
    if not markers:
        return text
    cleaned = text
    for marker in markers:
        cleaned = cleaned.replace(marker, " ", 1)
    return " ".join(cleaned.split())


def build_retrieval_query(
    citing_sentence: str,
    context_before: str = "",
    context_after: str = "",
    markers: list[str] | None = None,
) -> str:
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
    passage_text = ""
    for i, sc in enumerate(passages, 1):
        section = f" (Section: {sc.chunk.section_name})" if sc.chunk.section_name else ""
        passage_text += f"\n### Passage {i}{section}\n{sc.chunk.text}\n"

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
