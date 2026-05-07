
import pytest

from citeextract.models.citation import Citation
from citeextract.models.comprehension import ClaimVerdict
from citeextract.models.verdict import ExistenceResult
from citeextract.verification.metadata import FieldComparison, MetadataResult
from citeextract.verification.triage import TriageResult, TriageRoute, ContextQuality
from citeextract.verification.agentic.verdict_merger import (
    resolve_clear_route,
    merge_metadata_verdict,
    merge_claim_verdicts,
    merge_both_verdicts,
    fallback_to_quick,
)


def _exist(ref_id="1", status="FOUND", dbs=None, flags=None):
    return ExistenceResult(
        ref_id=ref_id, status=status,
        source="semantic_scholar" if status == "FOUND" else None,
        databases_checked=dbs or ["semantic_scholar", "openalex"],
        flags=flags or [],
    )


def _meta(ref_id="1", has_mismatch=False, is_retracted=False, flags=None):
    return MetadataResult(
        ref_id=ref_id,
        comparisons=[
            FieldComparison(field="title", status="MATCH", similarity=1.0),
            FieldComparison(field="authors", status="MATCH", similarity=1.0),
        ],
        metadata_score=0.95,
        flags=flags or [],
        is_retracted=is_retracted,
        has_metadata_mismatch=has_mismatch,
    )


def _triage(route, ref_id="1", existence=None, metadata=None, citations=None):
    return TriageResult(
        ref_id=ref_id,
        route=route,
        triage_reason="test reason",
        existence=existence or _exist(ref_id),
        metadata=metadata,
        substantive_citations=citations or [],
    )


class TestResolveClearRoute:

    def test_clear_valid(self):
        v = resolve_clear_route(_triage(TriageRoute.CLEAR_VALID))
        assert v.verdict == "VALID"
        assert v.mode == "agentic"
        assert v.action == "no_action"

    def test_clear_fabricated(self):
        v = resolve_clear_route(_triage(TriageRoute.CLEAR_FABRICATED))
        assert v.verdict == "FABRICATED"
        assert v.action == "remove_citation"

    def test_unverifiable(self):
        v = resolve_clear_route(_triage(TriageRoute.UNVERIFIABLE))
        assert v.verdict == "UNVERIFIABLE"
        assert "insufficient_data" in v.flags

    def test_needs_metadata_returns_none(self):
        assert resolve_clear_route(_triage(TriageRoute.NEEDS_METADATA)) is None

    def test_needs_claim_returns_none(self):
        assert resolve_clear_route(_triage(TriageRoute.NEEDS_CLAIM)) is None

    def test_needs_both_returns_none(self):
        assert resolve_clear_route(_triage(TriageRoute.NEEDS_BOTH)) is None


class TestMergeMetadata:

    def test_valid_metadata(self):
        t = _triage(TriageRoute.NEEDS_METADATA)
        v = merge_metadata_verdict(t, "VALID", "Title difference is cosmetic", [])
        assert v.verdict == "VALID"
        assert v.mode == "agentic"
        assert "cosmetic" in v.explanation

    def test_fabricated_metadata(self):
        t = _triage(TriageRoute.NEEDS_METADATA)
        v = merge_metadata_verdict(t, "FABRICATED", "Authors are from different paper", ["author_mismatch"])
        assert v.verdict == "FABRICATED"
        assert v.action == "remove_citation"
        assert "author_mismatch" in v.flags
        assert "agentic_agent_used" in v.flags

    def test_with_discrepancies(self):
        t = _triage(TriageRoute.NEEDS_METADATA)
        discs = [{"field": "authors", "status": "mismatch", "investigation_notes": "wrong people"}]
        v = merge_metadata_verdict(t, "FABRICATED", "Metadata wrong", [], discs)
        assert "Discrepancies" in v.explanation
        assert "wrong people" in v.explanation

    def test_preserves_existence_from_l2(self):
        exist = _exist(ref_id="5")
        t = _triage(TriageRoute.NEEDS_METADATA, ref_id="5", existence=exist)
        v = merge_metadata_verdict(t, "VALID", "OK", [])
        assert v.existence is exist
        assert v.existence.source == "semantic_scholar"


class TestMergeClaims:

    def test_all_supports(self):
        t = _triage(TriageRoute.NEEDS_CLAIM)
        claims = [
            ClaimVerdict(verdict="SUPPORTS", explanation="Matches", evidence_quote="quote"),
        ]
        v = merge_claim_verdicts(t, claims)
        assert v.verdict == "VALID"
        assert v.claim_verdict == "SUPPORTED"
        assert "1 supported" in v.explanation

    def test_all_neutral(self):
        t = _triage(TriageRoute.NEEDS_CLAIM)
        claims = [
            ClaimVerdict(verdict="NEUTRAL", explanation="Tangential"),
        ]
        v = merge_claim_verdicts(t, claims)
        assert v.claim_verdict == "NEUTRAL"
        assert "claim_neutral" in v.flags
        assert "neutral" in v.explanation

    def test_one_contradicts(self):
        t = _triage(TriageRoute.NEEDS_CLAIM)
        claims = [
            ClaimVerdict(verdict="SUPPORTS", explanation="OK"),
            ClaimVerdict(verdict="CONTRADICTS", explanation="Paper says opposite", evidence_quote="..."),
        ]
        v = merge_claim_verdicts(t, claims)
        assert v.claim_verdict == "CONTRADICTS"
        assert v.action == "verify_claim"
        assert "claim_contradicts" in v.flags

    def test_multiple_contradicts(self):
        t = _triage(TriageRoute.NEEDS_CLAIM)
        claims = [
            ClaimVerdict(verdict="CONTRADICTS", explanation="Wrong claim 1"),
            ClaimVerdict(verdict="CONTRADICTS", explanation="Wrong claim 2"),
        ]
        v = merge_claim_verdicts(t, claims)
        assert v.claim_verdict == "CONTRADICTS"
        assert "2 citing sentence(s)" in v.explanation

    def test_empty_claim_list(self):
        t = _triage(TriageRoute.NEEDS_CLAIM)
        v = merge_claim_verdicts(t, [])
        assert v.claim_verdict == "UNVERIFIABLE"
        assert "claim_verification_empty" in v.flags


    def test_binary_all_supported(self):
        t = _triage(TriageRoute.NEEDS_CLAIM)
        claims = [
            ClaimVerdict(verdict="SUPPORTED", explanation="Matches"),
            ClaimVerdict(verdict="SUPPORTED", explanation="Matches 2"),
        ]
        v = merge_claim_verdicts(t, claims)
        assert v.claim_verdict == "SUPPORTED"
        assert "binary scheme" in v.explanation

    def test_binary_one_not_supported(self):
        t = _triage(TriageRoute.NEEDS_CLAIM)
        claims = [
            ClaimVerdict(verdict="SUPPORTED", explanation="OK"),
            ClaimVerdict(verdict="NOT_SUPPORTED", explanation="Paper doesn't say this"),
        ]
        v = merge_claim_verdicts(t, claims)
        assert v.claim_verdict == "CONTRADICTS"
        assert v.action == "verify_claim"
        assert "claim_not_supported" in v.flags

    def test_binary_all_not_supported(self):
        t = _triage(TriageRoute.NEEDS_CLAIM)
        claims = [
            ClaimVerdict(verdict="NOT_SUPPORTED", explanation="not addressed"),
            ClaimVerdict(verdict="NOT_SUPPORTED", explanation="contradicted"),
        ]
        v = merge_claim_verdicts(t, claims)
        assert v.claim_verdict == "CONTRADICTS"
        assert "2 citing sentence(s) not supported" in v.claim_explanation

    def test_retracted_paper_with_supported_claim(self):
        t = _triage(
            TriageRoute.NEEDS_CLAIM,
            metadata=_meta(is_retracted=True),
        )
        claims = [ClaimVerdict(verdict="SUPPORTS", explanation="Matches")]
        v = merge_claim_verdicts(t, claims)
        assert v.metadata_verdict == "FABRICATED"
        assert v.claim_verdict == "SUPPORTED"
        assert v.verdict == "FABRICATED"
        assert v.action == "remove_citation"
        assert "retracted" in v.metadata_explanation.lower()

    def test_retracted_paper_with_contradicted_claim(self):
        t = _triage(
            TriageRoute.NEEDS_CLAIM,
            metadata=_meta(is_retracted=True),
        )
        claims = [ClaimVerdict(verdict="CONTRADICTS", explanation="Paper says opposite")]
        v = merge_claim_verdicts(t, claims)
        assert v.metadata_verdict == "FABRICATED"
        assert v.claim_verdict == "CONTRADICTS"
        assert v.verdict == "FABRICATED"


class TestClaimAgentConfig:

    def test_schema_3class(self):
        from citeextract.verification.agentic.claim_agent import build_verdict_schema
        schema = build_verdict_schema(3)
        enum = schema["json_schema"]["schema"]["properties"]["verdicts"]["items"]["properties"]["verdict"]["enum"]
        assert enum == ["SUPPORTS", "CONTRADICTS", "NEUTRAL"]

    def test_schema_2class(self):
        from citeextract.verification.agentic.claim_agent import build_verdict_schema
        schema = build_verdict_schema(2)
        enum = schema["json_schema"]["schema"]["properties"]["verdicts"]["items"]["properties"]["verdict"]["enum"]
        assert enum == ["SUPPORTED", "NOT_SUPPORTED"]

    def test_schema_invalid_classes_raises(self):
        from citeextract.verification.agentic.claim_agent import build_verdict_schema
        with pytest.raises(ValueError):
            build_verdict_schema(4)

    def test_prompt_3class_mentions_contradicts(self):
        from citeextract.verification.agentic.claim_agent import load_claim_prompt
        assert "CONTRADICTS" in load_claim_prompt(3)

    def test_prompt_2class_mentions_supported(self):
        from citeextract.verification.agentic.claim_agent import load_claim_prompt
        p = load_claim_prompt(2)
        assert "SUPPORTED" in p and "NOT_SUPPORTED" in p
        assert "CONTRADICTS" not in p


class TestMergeBoth:

    def test_metadata_fabricated_claim_still_runs(self):
        t = _triage(TriageRoute.NEEDS_BOTH)
        v = merge_both_verdicts(
            t,
            metadata_verdict="FABRICATED",
            metadata_explanation="Wrong paper",
            metadata_flags=["author_mismatch"],
            claim_verdicts=[ClaimVerdict(verdict="SUPPORTS", explanation="...")],
        )
        assert v.verdict == "FABRICATED"
        assert v.metadata_verdict == "FABRICATED"
        assert v.claim_verdict == "SUPPORTED"

    def test_metadata_valid_claims_support(self):
        t = _triage(TriageRoute.NEEDS_BOTH)
        v = merge_both_verdicts(
            t,
            metadata_verdict="VALID",
            metadata_explanation="Cosmetic difference",
            metadata_flags=[],
            claim_verdicts=[ClaimVerdict(verdict="SUPPORTS", explanation="Matches")],
        )
        assert v.verdict == "VALID"
        assert v.metadata_verdict == "VALID"
        assert v.claim_verdict == "SUPPORTED"

    def test_metadata_valid_claims_contradict(self):
        t = _triage(TriageRoute.NEEDS_BOTH)
        v = merge_both_verdicts(
            t,
            metadata_verdict="VALID",
            metadata_explanation="OK",
            metadata_flags=[],
            claim_verdicts=[ClaimVerdict(verdict="CONTRADICTS", explanation="Misrepresented")],
        )
        assert v.verdict == "VALID"
        assert v.metadata_verdict == "VALID"
        assert v.claim_verdict == "CONTRADICTS"

    def test_metadata_unverifiable_claims_support(self):
        t = _triage(TriageRoute.NEEDS_BOTH)
        v = merge_both_verdicts(
            t,
            metadata_verdict="UNVERIFIABLE",
            metadata_explanation="Can't determine",
            metadata_flags=[],
            claim_verdicts=[ClaimVerdict(verdict="SUPPORTS", explanation="Matches")],
        )
        assert v.verdict == "UNVERIFIABLE"
        assert v.metadata_verdict == "UNVERIFIABLE"
        assert v.claim_verdict == "SUPPORTED"

    def test_metadata_valid_no_claims(self):
        t = _triage(TriageRoute.NEEDS_BOTH)
        v = merge_both_verdicts(
            t,
            metadata_verdict="VALID",
            metadata_explanation="OK",
            metadata_flags=[],
            claim_verdicts=None,
        )
        assert v.verdict == "VALID"
        assert v.metadata_verdict == "VALID"
        assert v.claim_verdict is None

    def test_claim_agent_error_surfaced_as_flag(self):
        t = _triage(TriageRoute.NEEDS_BOTH)
        v = merge_both_verdicts(
            t,
            metadata_verdict="VALID",
            metadata_explanation="OK",
            metadata_flags=[],
            claim_verdicts=None,
            claim_error="timeout after 60s",
        )
        assert v.metadata_verdict == "VALID"
        assert v.claim_verdict == "UNVERIFIABLE"
        assert "claim_agent_error" in v.claim_flags
        assert "timeout" in v.claim_explanation.lower()
        assert v.verdict == "VALID"

    def test_claim_agent_error_distinguishable_from_not_applicable(self):
        t = _triage(TriageRoute.NEEDS_BOTH)
        not_applicable = merge_both_verdicts(
            t, metadata_verdict="VALID", metadata_explanation="OK",
            metadata_flags=[], claim_verdicts=None,
        )
        errored = merge_both_verdicts(
            t, metadata_verdict="VALID", metadata_explanation="OK",
            metadata_flags=[], claim_verdicts=None, claim_error="boom",
        )
        assert not_applicable.claim_verdict is None
        assert "claim_agent_error" not in not_applicable.claim_flags
        assert errored.claim_verdict == "UNVERIFIABLE"
        assert "claim_agent_error" in errored.claim_flags


class TestFallback:

    def test_fallback_found_is_unverifiable(self):
        t = _triage(TriageRoute.NEEDS_METADATA, metadata=_meta())
        v = fallback_to_quick(t)
        assert v.verdict == "UNVERIFIABLE"
        assert v.mode == "agentic"
        assert "agent_fallback_to_quick" in v.flags
        assert "Manual review" in v.explanation

    def test_fallback_found_with_mismatch_is_still_unverifiable(self):
        t = _triage(TriageRoute.NEEDS_METADATA, metadata=_meta(has_mismatch=True))
        v = fallback_to_quick(t)
        assert v.verdict == "UNVERIFIABLE"
        assert "agent_fallback_to_quick" in v.flags

    def test_fallback_not_found(self):
        exist = _exist(status="NOT_FOUND", dbs=["s2", "openalex", "crossref"])
        t = _triage(TriageRoute.NEEDS_METADATA, existence=exist)
        v = fallback_to_quick(t)
        assert v.verdict == "FABRICATED"
        assert "agent_fallback_to_quick" in v.flags

    def test_fallback_no_existence(self):
        t = TriageResult(
            ref_id="1", route=TriageRoute.NEEDS_METADATA,
            triage_reason="test", existence=None,
        )
        v = fallback_to_quick(t)
        assert v.verdict == "UNVERIFIABLE"
