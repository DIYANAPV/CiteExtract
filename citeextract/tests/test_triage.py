
import pytest

from citeextract.models.citation import Citation
from citeextract.models.comprehension import Chunk, FullTextResult, ScoredChunk
from citeextract.models.verdict import ExistenceResult
from citeextract.verification.metadata import FieldComparison, MetadataResult
from citeextract.verification.triage import (
    TriageRoute,
    TriageResult,
    _assess_context_quality,
    triage_reference,
    triage_all,
)


def _exist(
    ref_id: str = "1",
    status: str = "FOUND",
    source: str = "semantic_scholar",
    title_sim: float = 1.0,
    dbs: list[str] | None = None,
    flags: list[str] | None = None,
) -> ExistenceResult:
    return ExistenceResult(
        ref_id=ref_id,
        status=status,
        source=source,
        title_similarity=title_sim,
        databases_checked=dbs or ["semantic_scholar", "openalex"],
        flags=flags or [],
    )


def _meta(
    ref_id: str = "1",
    title_status: str = "MATCH",
    title_sim: float = 1.0,
    author_status: str = "MATCH",
    author_sim: float = 1.0,
    has_mismatch: bool = False,
    is_retracted: bool = False,
    flags: list[str] | None = None,
) -> MetadataResult:
    return MetadataResult(
        ref_id=ref_id,
        comparisons=[
            FieldComparison(field="title", status=title_status, similarity=title_sim),
            FieldComparison(field="authors", status=author_status, similarity=author_sim),
            FieldComparison(field="year", status="MATCH", similarity=1.0),
            FieldComparison(field="venue", status="MATCH", similarity=1.0),
        ],
        metadata_score=0.95,
        flags=flags or [],
        is_retracted=is_retracted,
        has_metadata_mismatch=has_mismatch,
    )


def _citation(
    ref_id: str = "1",
    sentence: str = "Smith et al. showed that transformers outperform RNNs on translation tasks.",
    before: str = "Many approaches have been proposed for neural machine translation.",
    after: str = "This finding was later confirmed by several independent studies.",
) -> Citation:
    return Citation(
        ref_id=ref_id,
        citing_sentence=sentence,
        context_before=before,
        context_after=after,
        marker="[1]",
        position=100,
    )


def _trivial_citation(ref_id: str = "1") -> Citation:
    return Citation(
        ref_id=ref_id,
        citing_sentence="see [1]",
        context_before="",
        context_after="",
        marker="[1]",
        position=50,
    )


class TestClearValid:

    def test_found_match_no_claims(self):
        result = triage_reference(
            ref_id="1",
            existence=_exist(),
            metadata=_meta(),
            citations=[_trivial_citation()],
        )
        assert result.route == TriageRoute.CLEAR_VALID
        assert "no substantive claims" in result.triage_reason.lower()

    def test_found_match_no_citations_at_all(self):
        result = triage_reference(
            ref_id="1",
            existence=_exist(),
            metadata=_meta(),
            citations=[],
        )
        assert result.route == TriageRoute.CLEAR_VALID

    def test_web_source_no_claims(self):
        result = triage_reference(
            ref_id="1",
            existence=_exist(source="web"),
            metadata=None,
            citations=[_trivial_citation()],
        )
        assert result.route == TriageRoute.CLEAR_VALID
        assert "web source" in result.triage_reason.lower()


class TestClearFabricated:

    def test_not_found_multiple_dbs(self):
        result = triage_reference(
            ref_id="1",
            existence=_exist(
                status="NOT_FOUND",
                source=None,
                title_sim=None,
                dbs=["semantic_scholar", "openalex", "crossref"],
            ),
            metadata=None,
            citations=[],
        )
        assert result.route == TriageRoute.CLEAR_FABRICATED

    def test_retracted(self):
        result = triage_reference(
            ref_id="1",
            existence=_exist(),
            metadata=_meta(is_retracted=True),
            citations=[],
        )
        assert result.route == TriageRoute.CLEAR_FABRICATED
        assert "retracted" in result.triage_reason.lower()

    def test_retracted_with_substantive_citations_routes_to_claim(self):
        result = triage_reference(
            ref_id="1",
            existence=_exist(),
            metadata=_meta(is_retracted=True),
            citations=[_citation()],
        )
        assert result.route == TriageRoute.NEEDS_CLAIM
        assert "retracted" in result.triage_reason.lower()


class TestUnverifiable:

    def test_not_found_single_db(self):
        result = triage_reference(
            ref_id="1",
            existence=_exist(
                status="NOT_FOUND",
                source=None,
                title_sim=None,
                dbs=["semantic_scholar"],
            ),
            metadata=None,
            citations=[],
        )
        assert result.route == TriageRoute.UNVERIFIABLE
        assert "insufficient" in result.triage_reason.lower()

    def test_no_existence_result(self):
        result = triage_reference(
            ref_id="1",
            existence=None,
            metadata=None,
            citations=[],
        )
        assert result.route == TriageRoute.UNVERIFIABLE


class TestNeedsMetadata:

    def test_metadata_mismatch_no_claims(self):
        result = triage_reference(
            ref_id="1",
            existence=_exist(),
            metadata=_meta(
                has_mismatch=True,
                author_status="MISMATCH",
                author_sim=0.3,
            ),
            citations=[_trivial_citation()],
        )
        assert result.route == TriageRoute.NEEDS_METADATA

    def test_title_mismatch_no_claims(self):
        result = triage_reference(
            ref_id="1",
            existence=_exist(title_sim=0.70),
            metadata=_meta(title_status="MISMATCH", title_sim=0.70),
            citations=[],
        )
        assert result.route == TriageRoute.NEEDS_METADATA
        assert "title mismatch" in result.triage_reason.lower()

    def test_doi_title_mismatch_no_claims(self):
        result = triage_reference(
            ref_id="1",
            existence=_exist(flags=["doi_title_mismatch"]),
            metadata=_meta(),
            citations=[],
        )
        assert result.route == TriageRoute.NEEDS_METADATA
        assert "doi" in result.triage_reason.lower()

    def test_unmatched_authors_close_match(self):
        result = triage_reference(
            ref_id="1",
            existence=_exist(),
            metadata=_meta(
                author_status="CLOSE_MATCH",
                author_sim=0.7,
                flags=["2 unmatched authors: Smith J, Doe A"],
            ),
            citations=[],
        )
        assert result.route == TriageRoute.NEEDS_METADATA


class TestNeedsClaim:

    def test_clean_metadata_with_claims(self):
        cit = _citation()
        passages = {cit.citing_sentence: [
            ScoredChunk(chunk=Chunk(text="passage1", paragraph_index=0), bm25_score=0.8),
            ScoredChunk(chunk=Chunk(text="passage2", paragraph_index=1), bm25_score=0.5),
        ]}
        result = triage_reference(
            ref_id="1",
            existence=_exist(),
            metadata=_meta(),
            citations=[cit],
            pre_retrieved_passages=passages,
        )
        assert result.route == TriageRoute.NEEDS_CLAIM
        assert len(result.substantive_citations) == 1

    def test_clean_metadata_claims_no_passages(self):
        result = triage_reference(
            ref_id="1",
            existence=_exist(),
            metadata=_meta(),
            citations=[_citation()],
            pre_retrieved_passages={},
        )
        assert result.route == TriageRoute.NEEDS_CLAIM

    def test_web_source_with_claims(self):
        result = triage_reference(
            ref_id="1",
            existence=_exist(source="web"),
            metadata=None,
            citations=[_citation()],
        )
        assert result.route == TriageRoute.NEEDS_CLAIM


class TestNeedsBoth:

    def test_metadata_mismatch_with_claims(self):
        result = triage_reference(
            ref_id="1",
            existence=_exist(),
            metadata=_meta(has_mismatch=True),
            citations=[_citation()],
        )
        assert result.route == TriageRoute.NEEDS_BOTH

    def test_title_mismatch_with_claims(self):
        result = triage_reference(
            ref_id="1",
            existence=_exist(title_sim=0.70),
            metadata=_meta(title_status="MISMATCH", title_sim=0.70),
            citations=[_citation()],
        )
        assert result.route == TriageRoute.NEEDS_BOTH

    def test_near_match_title_with_claims_routes_to_claim(self):
        cit = _citation()
        result = triage_reference(
            ref_id="1",
            existence=_exist(title_sim=0.90),
            metadata=_meta(title_sim=0.90),
            citations=[cit],
        )
        assert result.route == TriageRoute.NEEDS_CLAIM

    def test_doi_mismatch_with_claims(self):
        result = triage_reference(
            ref_id="1",
            existence=_exist(flags=["doi_title_mismatch"]),
            metadata=_meta(),
            citations=[_citation()],
        )
        assert result.route == TriageRoute.NEEDS_BOTH


class TestContextQuality:

    def test_full_context(self):
        cit = _citation(before="Sentence one. Sentence two.", after="Sentence three.")
        quality = _assess_context_quality(cit)
        assert quality.has_before is True
        assert quality.has_after is True
        assert quality.total_sentences >= 3

    def test_no_context(self):
        cit = _citation(before="", after="")
        quality = _assess_context_quality(cit)
        assert quality.has_before is False
        assert quality.has_after is False
        assert quality.total_sentences == 1

    def test_before_only(self):
        cit = _citation(before="Previous sentence.", after="")
        quality = _assess_context_quality(cit)
        assert quality.has_before is True
        assert quality.has_after is False
        assert quality.total_sentences >= 2


class TestTriageAll:

    def test_mixed_batch(self):
        exist_map = {
            "1": _exist(ref_id="1"),
            "2": _exist(ref_id="2", status="NOT_FOUND", source=None, title_sim=None,
                        dbs=["semantic_scholar", "openalex"]),
            "3": _exist(ref_id="3", title_sim=0.88),
        }
        metadata_map = {
            "1": _meta(ref_id="1"),
            "3": _meta(ref_id="3", title_sim=0.88),
        }
        citations_by_ref = {
            "1": [_trivial_citation(ref_id="1")],
            "2": [],
            "3": [_citation(ref_id="3")],
        }

        results = triage_all(exist_map, metadata_map, citations_by_ref)

        by_id = {r.ref_id: r for r in results}
        assert by_id["1"].route == TriageRoute.CLEAR_VALID
        assert by_id["2"].route == TriageRoute.CLEAR_FABRICATED
        assert by_id["3"].route == TriageRoute.NEEDS_CLAIM
