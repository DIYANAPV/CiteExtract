
import pytest

from citeextract.verification.comprehension import (
    RetrievalIndex,
    build_retrieval_index,
    chunk_text,
    reciprocal_rank_fusion,
    retrieve_passages_bm25,
    retrieve_passages_dense,
    retrieve_passages_hybrid,
    retrieve_with_index,
)
from citeextract.models.comprehension import Chunk, ScoredChunk


SAMPLE_ABSTRACT = (
    "We propose a novel attention mechanism that replaces recurrence entirely. "
    "The Transformer architecture relies on self-attention to compute "
    "representations of its input and output. Experiments on machine "
    "translation tasks show that the Transformer is superior in quality "
    "while being more parallelizable."
)

SAMPLE_PAPER_TEXT = """Abstract

We propose a novel attention mechanism that replaces recurrence entirely. The Transformer architecture relies on self-attention to compute representations of its input and output.

Introduction

Recurrent neural networks have long been the dominant approach for sequence modeling tasks. However, they suffer from inherent sequential computation that prevents parallelization within training examples. Attention mechanisms have become an integral part of compelling sequence modeling, allowing modeling of dependencies without regard to their distance in the input or output sequences.

Methods

We propose a new simple network architecture called the Transformer. It is based entirely on attention mechanisms, dispensing with recurrence and convolutions entirely. The encoder maps an input sequence of symbol representations to a continuous representation. Given this representation, the decoder generates an output sequence one element at a time.

The attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

Experiments

We trained our models on the WMT 2014 English-to-German translation task. The training data consisted of about 4.5 million sentence pairs. We also evaluated on the English-to-French translation task. The Transformer achieves 28.4 BLEU on the English-to-German task, improving over the existing best results by more than 2 BLEU.

Results show that the Transformer generalizes well to other tasks. We achieved state-of-the-art results on English constituency parsing with limited training data.

Conclusion

In this work we presented the Transformer, a novel architecture based entirely on attention. The Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. We plan to extend the Transformer to other modalities and to investigate local attention mechanisms."""

SAMPLE_SECTIONS = [
    {"name": "Abstract", "text": "We propose a novel attention mechanism that replaces recurrence entirely. The Transformer architecture relies on self-attention to compute representations of its input and output. Experiments on machine translation tasks demonstrate that the Transformer is superior in quality while being more parallelizable and requiring significantly less time to train."},
    {"name": "Introduction", "text": "Recurrent neural networks have long been the dominant approach for sequence modeling tasks such as language modeling and machine translation. However, they suffer from inherent sequential computation that prevents parallelization within training examples. Attention mechanisms have become an integral part of compelling sequence modeling, allowing modeling of dependencies without regard to their distance in the input or output sequences."},
    {"name": "Methods", "text": "We propose a new simple network architecture called the Transformer. It is based entirely on attention mechanisms, dispensing with recurrence and convolutions entirely. The encoder maps an input sequence of symbol representations to a continuous representation. Given this representation, the decoder generates an output sequence one element at a time."},
    {"name": "Experiments", "text": "We trained our models on the WMT 2014 English-to-German translation task. The training data consisted of about 4.5 million sentence pairs. We also evaluated on the English-to-French translation task. The Transformer achieves 28.4 BLEU on the English-to-German task, improving over the existing best results by more than 2 BLEU."},
    {"name": "Conclusion", "text": "In this work we presented the Transformer, a novel architecture based entirely on attention. The Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. We plan to extend the Transformer to other modalities and to investigate local attention mechanisms."},
]


class TestChunkText:
    def test_plain_text_chunking(self):
        chunks = chunk_text(SAMPLE_PAPER_TEXT)
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        for c in chunks:
            assert len(c.text.split()) >= 10

    def test_section_chunking(self):
        chunks = chunk_text("", sections=SAMPLE_SECTIONS)
        assert len(chunks) > 0
        section_names = {c.section_name for c in chunks}
        assert "Methods" in section_names
        assert "Conclusion" in section_names

    def test_abstract_only(self):
        chunks = chunk_text(SAMPLE_ABSTRACT)
        assert len(chunks) >= 1
        assert "attention mechanism" in chunks[0].text

    def test_empty_text(self):
        assert chunk_text("") == []
        assert chunk_text("  \n\n  ") == []

    def test_paragraph_indices_sequential(self):
        chunks = chunk_text(SAMPLE_PAPER_TEXT)
        indices = [c.paragraph_index for c in chunks]
        assert indices == sorted(indices)
        assert indices == list(range(len(indices)))

    def test_short_fragments_filtered(self):
        text = "Too short.\n\nAlso short.\n\nThis is a paragraph with enough words to pass the minimum word filter threshold easily."
        chunks = chunk_text(text)
        for c in chunks:
            assert len(c.text.split()) >= 10

    def test_chunks_meet_min_chars(self):
        from citeextract.verification.comprehension import _MIN_CHUNK_CHARS
        sections = [
            {"name": "Tiny caption",
             "text": "Just one short sentence that on its own is too brief to be useful."},
            {"name": "Long body",
             "text": ("This is a longer section with enough text. " * 30)},
            {"name": "Short tail",
             "text": "A small section."},
        ]
        chunks = chunk_text("", sections=sections)
        assert chunks, "expected at least the long body to produce chunks"
        for c in chunks:
            assert len(c.text) >= _MIN_CHUNK_CHARS, (
                f"chunk under min_chars surfaced: section={c.section_name!r} "
                f"chars={len(c.text)} text={c.text!r}"
            )

    def test_short_section_with_long_neighbour_via_plain_text(self):
        from citeextract.verification.comprehension import _MIN_CHUNK_CHARS
        text = "This is a longer sentence that should produce a substantial chunk. " * 12
        text += "\n\nTiny tail."
        chunks = chunk_text(text)
        for c in chunks:
            assert len(c.text) >= _MIN_CHUNK_CHARS


class TestRetrieveBm25:
    def test_basic_retrieval(self):
        chunks = chunk_text(SAMPLE_PAPER_TEXT)
        query = "attention mechanism for neural networks"
        results = retrieve_passages_bm25(query, chunks, top_k=3)
        assert len(results) > 0
        assert all(isinstance(r, ScoredChunk) for r in results)
        scores = [r.bm25_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_relevant_passage_ranked_high(self):
        chunks = chunk_text(SAMPLE_PAPER_TEXT)
        query = "BLEU score on English-to-German translation"
        results = retrieve_passages_bm25(query, chunks, top_k=3)
        top_text = results[0].chunk.text
        assert "BLEU" in top_text or "English-to-German" in top_text or "translation" in top_text

    def test_top_k_respected(self):
        chunks = chunk_text(SAMPLE_PAPER_TEXT)
        results = retrieve_passages_bm25("some query", chunks, top_k=2)
        assert len(results) <= 2

    def test_empty_chunks(self):
        results = retrieve_passages_bm25("any query", [], top_k=3)
        assert results == []

    def test_single_chunk(self):
        chunks = [Chunk(text="Attention is all you need in transformer models.", section_name=None, paragraph_index=0)]
        results = retrieve_passages_bm25("attention transformer", chunks, top_k=3)
        assert len(results) <= 1

    def test_abstract_only_retrieval(self):
        chunks = chunk_text(SAMPLE_ABSTRACT)
        query = "self-attention computation in transformers"
        results = retrieve_passages_bm25(query, chunks, top_k=1)
        assert len(results) >= 1
        assert "attention" in results[0].chunk.text.lower()


class TestRRF:
    def test_basic_fusion(self):
        ranking_a = [(0, 0.9), (1, 0.7), (2, 0.5)]
        ranking_b = [(0, 0.8), (2, 0.6), (1, 0.4)]
        merged = reciprocal_rank_fusion([ranking_a, ranking_b], k=60)
        assert merged[0][0] == 0
        assert len(merged) == 3

    def test_disjoint_rankings(self):
        ranking_a = [(0, 0.9), (1, 0.7)]
        ranking_b = [(2, 0.8), (3, 0.6)]
        merged = reciprocal_rank_fusion([ranking_a, ranking_b], k=60)
        assert len(merged) == 4

    def test_single_ranking(self):
        ranking = [(0, 0.9), (1, 0.7)]
        merged = reciprocal_rank_fusion([ranking], k=60)
        assert merged[0][0] == 0
        assert merged[1][0] == 1

    def test_empty(self):
        merged = reciprocal_rank_fusion([], k=60)
        assert merged == []

    def test_scores_are_positive(self):
        ranking = [(0, 0.9), (1, 0.7), (2, 0.5)]
        merged = reciprocal_rank_fusion([ranking], k=60)
        for _, score in merged:
            assert score > 0


class TestRetrieveDense:
    def test_basic_retrieval(self):
        chunks = chunk_text(SAMPLE_PAPER_TEXT)
        query = "attention mechanism for neural networks"
        results = retrieve_passages_dense(query, chunks, top_k=3)
        assert len(results) > 0
        assert all(isinstance(r, ScoredChunk) for r in results)
        for r in results:
            assert r.dense_score is not None
            assert r.dense_score > 0

    def test_relevant_passage_ranked_high(self):
        chunks = chunk_text(SAMPLE_PAPER_TEXT)
        query = "BLEU score on English-to-German machine translation benchmark"
        results = retrieve_passages_dense(query, chunks, top_k=3)
        top_text = results[0].chunk.text.lower()
        assert any(kw in top_text for kw in ["bleu", "translation", "english", "german"])

    def test_empty_chunks(self):
        results = retrieve_passages_dense("any query", [], top_k=3)
        assert results == []

    def test_scores_descending(self):
        chunks = chunk_text(SAMPLE_PAPER_TEXT)
        results = retrieve_passages_dense("attention", chunks, top_k=5)
        scores = [r.dense_score for r in results]
        assert scores == sorted(scores, reverse=True)


class TestRetrieveHybrid:
    def test_basic_retrieval(self):
        chunks = chunk_text(SAMPLE_PAPER_TEXT)
        query = "attention mechanism for neural networks"
        results = retrieve_passages_hybrid(query, chunks, top_k=3)
        assert len(results) > 0
        assert all(isinstance(r, ScoredChunk) for r in results)
        for r in results:
            assert r.rrf_score is not None
            assert r.rrf_score > 0

    def test_has_both_scores(self):
        chunks = chunk_text(SAMPLE_PAPER_TEXT)
        query = "BLEU score on translation"
        results = retrieve_passages_hybrid(query, chunks, top_k=3)
        top = results[0]
        assert top.bm25_score > 0 or (top.dense_score is not None and top.dense_score > 0)

    def test_rrf_scores_descending(self):
        chunks = chunk_text(SAMPLE_PAPER_TEXT)
        results = retrieve_passages_hybrid(query="transformer model", chunks=chunks, top_k=5)
        rrf_scores = [r.rrf_score for r in results]
        assert rrf_scores == sorted(rrf_scores, reverse=True)

    def test_empty_chunks(self):
        results = retrieve_passages_hybrid("any query", [], top_k=3)
        assert results == []

    def test_top_k_respected(self):
        chunks = chunk_text(SAMPLE_PAPER_TEXT)
        results = retrieve_passages_hybrid("attention", chunks, top_k=2)
        assert len(results) <= 2


PARITY_QUERIES = [
    "attention mechanism for neural networks",
    "BLEU score on English-to-German machine translation benchmark",
    "transformer model architecture",
    "recurrent networks sequential computation",
    "weighted sum of values with compatibility function",
]


def _passage_keys(results):
    return [
        (
            r.chunk.paragraph_index,
            round(r.bm25_score, 6),
            round(r.dense_score, 6) if r.dense_score is not None else None,
            round(r.rrf_score, 6) if r.rrf_score is not None else None,
        )
        for r in results
    ]


class TestRetrievalIndexParity:
    def test_bm25_only_index_matches_oneshot(self):
        chunks = chunk_text(SAMPLE_PAPER_TEXT)
        index = build_retrieval_index(chunks, model_name=None)
        for query in PARITY_QUERIES:
            from_index = retrieve_with_index(query, index, top_k=3)
            from_oneshot = retrieve_passages_bm25(query, chunks, top_k=3)
            assert _passage_keys(from_index) == _passage_keys(from_oneshot), (
                f"BM25 parity failed for query: {query!r}"
            )

    def test_hybrid_index_matches_oneshot(self):
        chunks = chunk_text(SAMPLE_PAPER_TEXT)
        index = build_retrieval_index(chunks, model_name="all-MiniLM-L6-v2")
        for query in PARITY_QUERIES:
            from_index = retrieve_with_index(query, index, top_k=3)
            from_oneshot = retrieve_passages_hybrid(
                query, chunks, top_k=3, model_name="all-MiniLM-L6-v2",
            )
            assert _passage_keys(from_index) == _passage_keys(from_oneshot), (
                f"Hybrid parity failed for query: {query!r}"
            )

    def test_index_reuse_is_idempotent(self):
        chunks = chunk_text(SAMPLE_PAPER_TEXT)
        index = build_retrieval_index(chunks, model_name="all-MiniLM-L6-v2")
        first = retrieve_with_index("attention", index, top_k=3)
        second = retrieve_with_index("attention", index, top_k=3)
        assert _passage_keys(first) == _passage_keys(second)

    def test_empty_chunks(self):
        index = build_retrieval_index([], model_name="all-MiniLM-L6-v2")
        assert index.chunks == []
        assert index.has_dense is False
        assert retrieve_with_index("anything", index, top_k=3) == []

    def test_has_dense_flag(self):
        chunks = chunk_text(SAMPLE_PAPER_TEXT)
        bm25_only = build_retrieval_index(chunks, model_name=None)
        hybrid = build_retrieval_index(chunks, model_name="all-MiniLM-L6-v2")
        assert bm25_only.has_dense is False
        assert hybrid.has_dense is True

    def test_bm25_only_index_returns_no_rrf_score(self):
        chunks = chunk_text(SAMPLE_PAPER_TEXT)
        index = build_retrieval_index(chunks, model_name=None)
        results = retrieve_with_index("attention", index, top_k=3)
        assert len(results) > 0
        assert all(r.rrf_score is None for r in results)
        assert all(r.dense_score is None for r in results)

    def test_index_is_frozen(self):
        chunks = chunk_text(SAMPLE_PAPER_TEXT)
        index = build_retrieval_index(chunks, model_name=None)
        with pytest.raises(Exception):
            index.chunks = []  # type: ignore[misc]


class TestRerankPool:
    def test_rerank_pool_passed_to_flashrank(self, monkeypatch):
        from citeextract.verification import comprehension as comp_module

        chunks = chunk_text(SAMPLE_PAPER_TEXT)
        index = build_retrieval_index(chunks, model_name="all-MiniLM-L6-v2")

        seen_pool_sizes: list[int] = []

        def _spy_rerank(query, candidates, chunks_arg, top_k=3):
            seen_pool_sizes.append(len(candidates))
            return [(c[0], 1.0 - i * 0.01) for i, c in enumerate(candidates[:top_k])]

        monkeypatch.setattr(comp_module, "_rerank_with_flashrank", _spy_rerank)

        retrieve_with_index("attention", index, top_k=3, rerank_pool=5)
        retrieve_with_index("attention", index, top_k=3, rerank_pool=15)

        assert seen_pool_sizes[0] <= 5
        assert seen_pool_sizes[1] <= 15
        assert seen_pool_sizes[1] >= seen_pool_sizes[0]

    def test_rerank_pool_default_is_ten(self):
        import inspect
        from citeextract.verification.comprehension import retrieve_with_index
        sig = inspect.signature(retrieve_with_index)
        assert sig.parameters["rerank_pool"].default == 10
