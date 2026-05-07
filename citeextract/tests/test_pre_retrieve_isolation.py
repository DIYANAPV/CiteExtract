
import asyncio
from unittest.mock import MagicMock, patch

from citeextract.models.citation import Citation
from citeextract.models.comprehension import Chunk, FullTextResult, ScoredChunk
from citeextract.models.parsed_paper import ParsedPaper
from citeextract.models.reference import Reference
from citeextract.models.verdict import ExistenceResult


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

    def _setup_pre_retrieve(self):
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
        config = {"max_concurrent_agents": 5, "claim_agent": {}}
        return parsed, exist_map, citation_groups, prefetched_chunks, config

    def test_index_build_failure_aborts_ref_cleanly(self):
        from citeextract.verification.agentic.runner import _pre_retrieve_passages

        parsed, exist_map, citation_groups, prefetched, cfg = self._setup_pre_retrieve()

        with patch(
            "citeextract.verification.comprehension.build_retrieval_index",
            side_effect=OSError("model file not found"),
        ), patch(
            "citeextract.verification.comprehension.build_retrieval_query",
            side_effect=lambda s, *a, **kw: s,
        ):
            passages_by_ref, _ft = _run(_pre_retrieve_passages(
                parsed, exist_map, citation_groups,
                config=cfg,
                prefetched_chunks=prefetched, cache=None,
            ))

        assert "r1" not in passages_by_ref

    def test_per_citation_runtime_error_doesnt_kill_others(self):
        from citeextract.verification.agentic.runner import _pre_retrieve_passages

        parsed, exist_map, citation_groups, prefetched, cfg = self._setup_pre_retrieve()

        retrieve_mock = MagicMock(side_effect=[
            [_scored("passage A")],
            RuntimeError("flashrank model crashed"),
            [_scored("passage C")],
        ])
        index_mock = MagicMock()

        with patch(
            "citeextract.verification.comprehension.build_retrieval_index",
            return_value=index_mock,
        ), patch(
            "citeextract.verification.comprehension.build_retrieval_query",
            side_effect=lambda s, *a, **kw: s,
        ), patch(
            "citeextract.verification.comprehension.retrieve_with_index",
            new=retrieve_mock,
        ):
            passages_by_ref, _ft = _run(_pre_retrieve_passages(
                parsed, exist_map, citation_groups,
                config=cfg,
                prefetched_chunks=prefetched, cache=None,
            ))

        ref_passages = passages_by_ref["r1"]
        sentences = [c.citing_sentence for c in citation_groups["r1"]]
        assert len(ref_passages[sentences[0]]) == 1
        assert ref_passages[sentences[1]] == [], (
            "RuntimeError on cit B should record [] for that citation only"
        )
        assert len(ref_passages[sentences[2]]) == 1
