"""Tests for comprehension support — chunking, BM25, dense, and hybrid retrieval."""

import pytest

from src.verification.comprehension import (
    chunk_text,
    retrieve_passages_bm25,
    retrieve_passages_dense,
    retrieve_passages_hybrid,
    reciprocal_rank_fusion,
    _split_paragraphs,
    _split_if_long,
)
from src.models.comprehension import Chunk, ScoredChunk


# ---- Sample data ----

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
    {"name": "Abstract", "text": "We propose a novel attention mechanism that replaces recurrence entirely. The Transformer architecture relies on self-attention."},
    {"name": "Introduction", "text": "Recurrent neural networks have long been the dominant approach for sequence modeling tasks. However, they suffer from inherent sequential computation."},
    {"name": "Methods", "text": "We propose a new simple network architecture called the Transformer. It is based entirely on attention mechanisms, dispensing with recurrence and convolutions entirely."},
    {"name": "Experiments", "text": "We trained our models on the WMT 2014 English-to-German translation task. The Transformer achieves 28.4 BLEU on the English-to-German task."},
    {"name": "Conclusion", "text": "In this work we presented the Transformer, a novel architecture based entirely on attention. We plan to extend the Transformer to other modalities."},
]


# ---- Paragraph splitting ----

class TestSplitParagraphs:
    def test_double_newline(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        parts = _split_paragraphs(text)
        assert len(parts) == 3

    def test_extra_whitespace(self):
        text = "First.\n  \n  Second.\n\n\n\nThird."
        parts = _split_paragraphs(text)
        assert len(parts) == 3

    def test_single_newline_not_split(self):
        text = "Line one.\nLine two still same paragraph."
        parts = _split_paragraphs(text)
        assert len(parts) == 1

    def test_empty(self):
        assert _split_paragraphs("") == []
        assert _split_paragraphs("   ") == []


class TestSplitIfLong:
    def test_short_text_unchanged(self):
        text = "This is short."
        assert _split_if_long(text) == [text]

    def test_long_text_splits(self):
        # Build a 1200+ char text
        sentences = [f"Sentence number {i} with some padding text to make it longer." for i in range(30)]
        text = " ".join(sentences)
        assert len(text) > 800
        parts = _split_if_long(text)
        assert len(parts) > 1
        # All parts should be non-empty
        assert all(p.strip() for p in parts)


# ---- Chunking ----

class TestChunkText:
    def test_plain_text_chunking(self):
        chunks = chunk_text(SAMPLE_PAPER_TEXT)
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        # Each chunk should have reasonable content
        for c in chunks:
            assert len(c.text.split()) >= 10

    def test_section_chunking(self):
        chunks = chunk_text("", sections=SAMPLE_SECTIONS)
        assert len(chunks) > 0
        # Section names should be preserved
        section_names = {c.section_name for c in chunks}
        assert "Methods" in section_names
        assert "Conclusion" in section_names

    def test_abstract_only(self):
        chunks = chunk_text(SAMPLE_ABSTRACT)
        assert len(chunks) >= 1
        # Short abstract should be one chunk
        assert "attention mechanism" in chunks[0].text

    def test_empty_text(self):
        assert chunk_text("") == []
        assert chunk_text("  \n\n  ") == []

    def test_paragraph_indices_sequential(self):
        chunks = chunk_text(SAMPLE_PAPER_TEXT)
        indices = [c.paragraph_index for c in chunks]
        assert indices == sorted(indices)
        # No gaps — sequential from 0
        assert indices == list(range(len(indices)))

    def test_short_fragments_filtered(self):
        text = "Too short.\n\nAlso short.\n\nThis is a paragraph with enough words to pass the minimum word filter threshold easily."
        chunks = chunk_text(text)
        # Short fragments should be filtered out
        for c in chunks:
            assert len(c.text.split()) >= 10


# ---- BM25 retrieval ----

class TestRetrieveBm25:
    def test_basic_retrieval(self):
        chunks = chunk_text(SAMPLE_PAPER_TEXT)
        query = "attention mechanism for neural networks"
        results = retrieve_passages_bm25(query, chunks, top_k=3)
        assert len(results) > 0
        assert all(isinstance(r, ScoredChunk) for r in results)
        # Scores should be descending
        scores = [r.bm25_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_relevant_passage_ranked_high(self):
        chunks = chunk_text(SAMPLE_PAPER_TEXT)
        query = "BLEU score on English-to-German translation"
        results = retrieve_passages_bm25(query, chunks, top_k=3)
        # The experiments section should rank high
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
        """When only abstract is available, should still work."""
        chunks = chunk_text(SAMPLE_ABSTRACT)
        query = "self-attention computation in transformers"
        results = retrieve_passages_bm25(query, chunks, top_k=1)
        assert len(results) >= 1
        assert "attention" in results[0].chunk.text.lower()


# ---- Reciprocal Rank Fusion ----

class TestRRF:
    def test_basic_fusion(self):
        # Two rankings that agree on top item
        ranking_a = [(0, 0.9), (1, 0.7), (2, 0.5)]
        ranking_b = [(0, 0.8), (2, 0.6), (1, 0.4)]
        merged = reciprocal_rank_fusion([ranking_a, ranking_b], k=60)
        # Doc 0 should be top (rank 1 in both)
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


# ---- Dense retrieval ----

class TestRetrieveDense:
    def test_basic_retrieval(self):
        chunks = chunk_text(SAMPLE_PAPER_TEXT)
        query = "attention mechanism for neural networks"
        results = retrieve_passages_dense(query, chunks, top_k=3)
        assert len(results) > 0
        assert all(isinstance(r, ScoredChunk) for r in results)
        # Dense scores should be set
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


# ---- Hybrid retrieval ----

class TestRetrieveHybrid:
    def test_basic_retrieval(self):
        chunks = chunk_text(SAMPLE_PAPER_TEXT)
        query = "attention mechanism for neural networks"
        results = retrieve_passages_hybrid(query, chunks, top_k=3)
        assert len(results) > 0
        assert all(isinstance(r, ScoredChunk) for r in results)
        # Should have RRF scores
        for r in results:
            assert r.rrf_score is not None
            assert r.rrf_score > 0

    def test_has_both_scores(self):
        chunks = chunk_text(SAMPLE_PAPER_TEXT)
        query = "BLEU score on translation"
        results = retrieve_passages_hybrid(query, chunks, top_k=3)
        # Top result should have at least one non-zero BM25 or dense score
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
