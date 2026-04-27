"""Tests for per-citation isolation in agentic pre-retrieve.

The previous implementation wrapped the entire citation loop in a single
``try/except``. A transient OpenAI failure on multi-query decomposition for
ONE citing sentence threw away passages for *every* citation on that
reference. The fix isolates each citation: a failure records ``[]`` for
that key only, and the rest still get their top-k passages.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from src.models.citation import Citation
from src.models.comprehension import Chunk, FullTextResult, ScoredChunk
from src.models.parsed_paper import ParsedPaper
from src.models.reference import Reference
from src.models.verdict import ExistenceResult


def _run(coro):
    return asyncio.run(coro)


def _ref(rid: str = "1", title: str = "Test paper") -> Reference:
    return Reference(ref_id=rid, title=title, authors=[], source_format="text")


def _cit(rid: str, sentence: str, marker: str = "[1]") -> Citation:
    return Citation(
        ref_id=rid,
        marker=marker,
        citing_sentence=sentence,
        context_before="",
        context_after="",
    )


def _chunk(text: str, idx: int = 0) -> Chunk:
    return Chunk(text=text, paragraph_index=idx, section_name=None)


def _scored(text: str) -> ScoredChunk:
    return ScoredChunk(chunk=_chunk(text), bm25_score=1.0, dense_score=None, rrf_score=None)


def _exist(rid: str = "1") -> ExistenceResult:
    return ExistenceResult(
        ref_id=rid, status="FOUND", source="arxiv",
        matched_title="Test paper", matched_authors=[],
        databases_checked=["arxiv"],
    )


class TestPerCitationIsolation:
    """Goal: a failure on one citation must not nuke passages for sibling
    citations on the same reference."""

    def _setup_pre_retrieve(self, decompose_side_effects, *, multiquery_enabled=True):
        """Build the args + mocks for a 3-citation pre-retrieve call.

        Returns (parsed, exist_map, citation_groups, prefetched_chunks, config).
        """
        ref = _ref(rid="r1")
        cits = [
            _cit("r1", "Sentence A about topic one with enough words to qualify."),
            _cit("r1", "Sentence B about topic two with enough words to qualify."),
            _cit("r1", "Sentence C about topic three with enough words to qualify."),
        ]
        parsed = ParsedPaper(
            input_format="text",
            references=[ref],
            citations=cits,
            body_text="",
            has_body_text=False,
            warnings=[],
        )
        exist_map = {"r1": _exist("r1")}
        citation_groups = {"r1": cits}
        prefetched_chunks = {
            "r1": (
                FullTextResult(source="arxiv", full_text="some body text"),
                [_chunk("Body chunk 1", 0), _chunk("Body chunk 2", 1)],
            ),
        }
        # Minimal agentic config — multiquery enabled
        config = {
            "max_concurrent_agents": 5,
            "claim_agent": {
                "enable_multiquery": multiquery_enabled,
                "multiquery": {"n_sub_claims": 2, "full_top_k": 2, "sub_top_k": 2},
            },
        }
        return parsed, exist_map, citation_groups, prefetched_chunks, config

    def test_decompose_failure_on_one_citation_doesnt_kill_the_ref(self):
        """When ``decompose`` raises for the second citation only, the
        first and third still get their passages."""
        from src.verification.agentic.runner import _pre_retrieve_passages

        parsed, exist_map, citation_groups, prefetched, cfg = self._setup_pre_retrieve(
            decompose_side_effects=None,
        )

        # Mocks: index always builds; retrieve always returns one passage;
        # decompose succeeds for cit A and C, raises httpx.RequestError on B.
        decompose_mock = AsyncMock(side_effect=[
            ["sub1", "sub2"],                                 # cit A
            httpx.RequestError("connection refused"),         # cit B — fails
            ["sub3", "sub4"],                                 # cit C
        ])

        index_mock = MagicMock()
        # multi_query_retrieve and retrieve_with_index both return one ScoredChunk
        mq_retrieve_mock = MagicMock(return_value=[_scored("passage")])
        single_retrieve_mock = MagicMock(return_value=[_scored("passage")])

        # An LLM client is required when multiquery is enabled. Stub it.
        llm_client_mock = MagicMock()

        with patch(
            "src.verification.comprehension.build_retrieval_index",
            return_value=index_mock,
        ), patch(
            "src.verification.comprehension.build_retrieval_query",
            side_effect=lambda s, *a, **kw: s,
        ), patch(
            "src.verification.multiquery.decompose",
            new=decompose_mock,
        ), patch(
            "src.verification.multiquery.multi_query_retrieve",
            new=mq_retrieve_mock,
        ), patch(
            "src.verification.comprehension.retrieve_with_index",
            new=single_retrieve_mock,
        ), patch(
            "src.verification.api_clients.llm_client.create_llm_client",
            return_value=llm_client_mock,
        ):
            passages_by_ref, _ft = _run(_pre_retrieve_passages(
                parsed, exist_map, citation_groups,
                config=cfg, cost_tracker=None,
                prefetched_chunks=prefetched, cache=None,
            ))

        # The ref must be present in the result — not dropped due to the
        # one citation that failed.
        assert "r1" in passages_by_ref, (
            "regression: one citation's decompose failure dropped the entire ref"
        )

        ref_passages = passages_by_ref["r1"]
        # All three citations must have an entry.
        sentences = [c.citing_sentence for c in citation_groups["r1"]]
        for s in sentences:
            assert s in ref_passages, f"missing passage entry for: {s!r}"

        # The first and third citations should have passages (multi_query
        # path succeeded). The middle one fell back to single-query
        # retrieval after decompose raised — single-query retrieve was
        # patched to return [_scored("passage")] too, so it also has 1.
        assert len(ref_passages[sentences[0]]) == 1
        assert len(ref_passages[sentences[1]]) == 1, (
            "decompose failure on cit B should fall back to single-query, "
            "still returning passages"
        )
        assert len(ref_passages[sentences[2]]) == 1

    def test_index_build_failure_aborts_ref_cleanly(self):
        """An ``OSError`` from index build (model load failure) must abort
        the ref but not raise — the ref's entry simply isn't added."""
        from src.verification.agentic.runner import _pre_retrieve_passages

        parsed, exist_map, citation_groups, prefetched, cfg = self._setup_pre_retrieve(
            decompose_side_effects=None, multiquery_enabled=False,
        )

        with patch(
            "src.verification.comprehension.build_retrieval_index",
            side_effect=OSError("model file not found"),
        ), patch(
            "src.verification.comprehension.build_retrieval_query",
            side_effect=lambda s, *a, **kw: s,
        ):
            passages_by_ref, _ft = _run(_pre_retrieve_passages(
                parsed, exist_map, citation_groups,
                config=cfg, cost_tracker=None,
                prefetched_chunks=prefetched, cache=None,
            ))

        # Index build failed → ref was NOT added to passages_by_ref. Caller
        # gets an empty mapping, downstream code handles the absent entry
        # by yielding ``[]`` per citation in _build_comprehension_from_passages.
        assert "r1" not in passages_by_ref

    def test_per_citation_runtime_error_doesnt_kill_others(self):
        """A non-multiquery path: one citation's retrieve_with_index throws
        a RuntimeError; the others must still return passages."""
        from src.verification.agentic.runner import _pre_retrieve_passages

        parsed, exist_map, citation_groups, prefetched, cfg = self._setup_pre_retrieve(
            decompose_side_effects=None, multiquery_enabled=False,
        )

        retrieve_mock = MagicMock(side_effect=[
            [_scored("passage A")],
            RuntimeError("flashrank model crashed"),
            [_scored("passage C")],
        ])
        index_mock = MagicMock()

        with patch(
            "src.verification.comprehension.build_retrieval_index",
            return_value=index_mock,
        ), patch(
            "src.verification.comprehension.build_retrieval_query",
            side_effect=lambda s, *a, **kw: s,
        ), patch(
            "src.verification.comprehension.retrieve_with_index",
            new=retrieve_mock,
        ):
            passages_by_ref, _ft = _run(_pre_retrieve_passages(
                parsed, exist_map, citation_groups,
                config=cfg, cost_tracker=None,
                prefetched_chunks=prefetched, cache=None,
            ))

        ref_passages = passages_by_ref["r1"]
        sentences = [c.citing_sentence for c in citation_groups["r1"]]
        # Cits A and C succeed, B records empty list.
        assert len(ref_passages[sentences[0]]) == 1
        assert ref_passages[sentences[1]] == [], (
            "RuntimeError on cit B should record [] for that citation only"
        )
        assert len(ref_passages[sentences[2]]) == 1
