"""Tests for multi-query retrieval and the batched cross-encoder rerank path."""

import pytest

from src.models.comprehension import Chunk
from src.verification.comprehension import (
    _cross_encoder_score_pairs,
    _rerank_pair_batches,
    build_retrieval_index,
    chunk_text,
)
from src.verification.multiquery import (
    _build_query_list,
    multi_query_retrieve,
)


SAMPLE_TEXT = """We propose the Transformer, a novel attention-based architecture for sequence modeling.

Attention mechanisms have become an integral part of compelling sequence modeling, allowing modeling of dependencies without regard to their distance in the input or output sequences.

Recurrent neural networks have long been the dominant approach for sequence modeling tasks. They suffer from inherent sequential computation that prevents parallelization within training examples.

The Transformer is based entirely on attention mechanisms, dispensing with recurrence and convolutions entirely. Its encoder maps input tokens to continuous representations, decoder generates one element at a time.

Experiments on machine translation: WMT 2014 English-to-German. Training data was about 4.5 million sentence pairs. The Transformer achieves 28.4 BLEU on the English-to-German task.

Results show that the Transformer generalizes well to other tasks, achieving state-of-the-art results on English constituency parsing with limited training data."""


# ---- Build query list ----

class TestBuildQueryList:
    def test_orders_full_first(self):
        q = _build_query_list("full claim", ["sub one", "sub two"])
        assert q == ["full claim", "sub one", "sub two"]

    def test_strips_whitespace(self):
        q = _build_query_list("  full  ", ["  sub  "])
        assert q == ["full", "sub"]

    def test_dedupes_sub_matching_full(self):
        q = _build_query_list("identical", ["identical", "different"])
        assert q == ["identical", "different"]

    def test_skips_empty_subs(self):
        q = _build_query_list("full", ["", "  ", "real"])
        assert q == ["full", "real"]

    def test_empty_full_skipped(self):
        q = _build_query_list("", ["sub one"])
        assert q == ["sub one"]


# ---- Cross-encoder pair scoring ----

class TestCrossEncoderPairs:
    def test_returns_one_score_per_pair(self):
        pairs = [
            ("transformer attention", "We propose the Transformer architecture."),
            ("translation BLEU", "BLEU on English-to-German is 28.4."),
            ("recurrent networks", "Recurrent networks have sequential computation."),
        ]
        scores = _cross_encoder_score_pairs(pairs)
        assert scores is not None
        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)

    def test_relevance_ordering_holds(self):
        # The cross-encoder should score on-topic pairs higher than
        # mismatched ones — this is the core premise of using it.
        pairs = [
            ("transformer attention", "We propose the Transformer architecture."),
            ("transformer attention", "Cooking recipes for vegetable soup."),
        ]
        scores = _cross_encoder_score_pairs(pairs)
        assert scores is not None
        assert scores[0] > scores[1]

    def test_empty_pairs(self):
        assert _cross_encoder_score_pairs([]) == []


# ---- Batched rerank parity ----

class TestRerankPairBatchesParity:
    """Pin the core safety property: scoring the SAME (query, passage)
    pair in a batched call returns the SAME score as scoring it alone.
    Without this, the batched multi-query path could silently shift
    rankings.
    """

    def test_batched_scores_match_per_pair_scores(self):
        chunks = chunk_text(SAMPLE_TEXT)
        index = build_retrieval_index(chunks, model_name="all-MiniLM-L6-v2")

        queries = [
            "transformer attention architecture",
            "BLEU translation English German",
        ]
        # Reuse the same RRF candidates for both queries to make scores
        # directly comparable per-query-vs-batched.
        candidates = [(i, 1.0 / (i + 1)) for i in range(min(5, len(chunks)))]

        # Per-query reranks (one cross-encoder call each)
        per_query_results = []
        for q in queries:
            r = _rerank_pair_batches([q], [candidates], index.chunks, top_k=5)
            assert r is not None
            per_query_results.append(r[0])

        # Single batched rerank covering all queries
        batched = _rerank_pair_batches(
            queries, [candidates, candidates], index.chunks, top_k=5,
        )
        assert batched is not None
        assert len(batched) == 2

        # Same chunk_idx → same score, modulo fp32 noise.
        for q_i in range(len(queries)):
            per_query_map = dict(per_query_results[q_i])
            batched_map = dict(batched[q_i])
            assert set(per_query_map) == set(batched_map)
            for chunk_idx in per_query_map:
                assert abs(per_query_map[chunk_idx] - batched_map[chunk_idx]) < 1e-3, (
                    f"score drift for query {q_i} chunk {chunk_idx}: "
                    f"per-query={per_query_map[chunk_idx]} batched={batched_map[chunk_idx]}"
                )

    def test_returns_none_when_cross_encoder_unavailable(self, monkeypatch):
        from src.verification import comprehension as comp_module
        monkeypatch.setattr(
            comp_module, "_cross_encoder_score_pairs", lambda _pairs: None,
        )
        chunks = chunk_text(SAMPLE_TEXT)
        candidates = [(i, 1.0) for i in range(min(3, len(chunks)))]
        out = _rerank_pair_batches(
            ["q"], [candidates], chunks, top_k=3,
        )
        assert out is None


# ---- multi_query_retrieve end-to-end ----

class TestMultiQueryRetrieve:
    def test_returns_deduped_passages(self):
        chunks = chunk_text(SAMPLE_TEXT)
        index = build_retrieval_index(chunks, model_name="all-MiniLM-L6-v2")
        out = multi_query_retrieve(
            full_claim="transformer attention model",
            sub_claims=["BLEU translation", "recurrent networks"],
            index=index,
            bm25_candidates=10, dense_candidates=10, rrf_k=60,
            full_top_k=2, sub_top_k=2, rerank_pool=5,
        )
        # Each query takes top_k=2 → up to 6 passages, but dedupe across
        # queries usually shrinks this. Just guard against empty output
        # and against duplicate paragraph indices in the result.
        assert len(out) > 0
        para_indices = [sc.chunk.paragraph_index for sc in out]
        assert len(para_indices) == len(set(para_indices)), "duplicate passages"

    def test_falls_back_to_per_query_when_cross_encoder_unavailable(self, monkeypatch):
        # When _rerank_pair_batches returns None, the function must take
        # the per-query path so the pipeline still works on hosts where
        # the cross-encoder failed to load.
        from src.verification import multiquery as mq_module

        chunks = chunk_text(SAMPLE_TEXT)
        index = build_retrieval_index(chunks, model_name="all-MiniLM-L6-v2")

        monkeypatch.setattr(
            mq_module, "_rerank_pair_batches",
            lambda *_args, **_kwargs: None,
        )

        out = multi_query_retrieve(
            full_claim="transformer attention",
            sub_claims=["translation BLEU"],
            index=index,
            bm25_candidates=10, dense_candidates=10, rrf_k=60,
            full_top_k=2, sub_top_k=2, rerank_pool=5,
        )
        assert len(out) > 0

    def test_bm25_only_index_falls_back(self):
        # No dense embeddings → no RRF/rerank stage to batch. Falls
        # straight through to per-query path which itself just returns
        # BM25 top_k. Should still produce passages.
        chunks = chunk_text(SAMPLE_TEXT)
        index = build_retrieval_index(chunks, model_name=None)
        out = multi_query_retrieve(
            full_claim="transformer attention",
            sub_claims=["BLEU"],
            index=index,
            bm25_candidates=10, dense_candidates=10, rrf_k=60,
            full_top_k=2, sub_top_k=2, rerank_pool=5,
        )
        assert len(out) > 0
        for sc in out:
            assert sc.dense_score is None
            assert sc.rrf_score is None  # BM25-only path

    def test_empty_index(self):
        index = build_retrieval_index([], model_name="all-MiniLM-L6-v2")
        out = multi_query_retrieve(
            full_claim="anything",
            sub_claims=["sub"],
            index=index,
            bm25_candidates=10, dense_candidates=10, rrf_k=60,
            full_top_k=2, sub_top_k=2, rerank_pool=5,
        )
        assert out == []
