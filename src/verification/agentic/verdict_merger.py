"""Verdict merger — combines triage routes + agent outputs into CitationVerdict.

Each CitationVerdict carries two independent verdict dimensions — metadata
(does the paper exist and match the reference?) and claim (does the paper
support the citing sentence?). The top-level ``verdict`` is a roll-up so
existing reports still work, but flags and explanations are kept separate
per dimension.

For clear-cut routes (CLEAR_VALID, CLEAR_FABRICATED, UNVERIFIABLE), the
verdict is produced directly from L2/L3 results without any LLM call.

For agent-routed cases, the merger integrates the agent's output with the
existing evidence trail (ExistenceResult, MetadataResult).
"""

import logging
from typing import Optional

from src.classification.classifier import (
    CitationVerdict,
    classify_quick,
    rollup_verdict,
)
from src.models.comprehension import ClaimVerdict
from src.verification.triage import TriageResult, TriageRoute

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Flag helpers
# ---------------------------------------------------------------------------

def _dedup(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _existence_metadata_flags(triage: TriageResult) -> list[str]:
    """Flags from L2 (existence) + L3 (metadata) — belong to the metadata dimension."""
    flags: list[str] = []
    if triage.existence:
        flags.extend(triage.existence.flags)
    if triage.metadata:
        flags.extend(triage.metadata.flags)
    return _dedup(flags)


def _combined_explanation(metadata_expl: Optional[str], claim_expl: Optional[str]) -> str:
    parts = []
    if metadata_expl:
        parts.append(f"[metadata] {metadata_expl}")
    if claim_expl:
        parts.append(f"[claim] {claim_expl}")
    return " | ".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Action chooser
# ---------------------------------------------------------------------------

def _choose_action(metadata_verdict: Optional[str], claim_verdict: Optional[str]) -> str:
    """Action depends on both dimensions independently.

    FABRICATED metadata trumps everything (the cited paper isn't real,
    so the citation must go regardless of claim status). Otherwise a
    CONTRADICTS claim asks the user to revisit their wording.
    """
    if metadata_verdict == "FABRICATED":
        return "remove_citation"
    if claim_verdict == "CONTRADICTS":
        return "verify_claim"
    return "no_action"


# ---------------------------------------------------------------------------
# Clear-cut verdict builders (no LLM)
# ---------------------------------------------------------------------------

def _verdict_clear_valid(triage: TriageResult) -> CitationVerdict:
    meta_flags = _existence_metadata_flags(triage)
    meta_expl = f"Reference exists and metadata matches. ({triage.triage_reason})"
    return CitationVerdict(
        ref_id=triage.ref_id,
        verdict="VALID",
        mode="agentic",
        action="no_action",
        explanation=meta_expl,
        flags=meta_flags,
        metadata_verdict="VALID",
        metadata_flags=meta_flags,
        metadata_explanation=meta_expl,
        claim_verdict=None,
        claim_flags=[],
        claim_explanation=None,
        existence=triage.existence,
        metadata=triage.metadata,
    )


def _verdict_clear_fabricated(triage: TriageResult) -> CitationVerdict:
    meta_flags = _existence_metadata_flags(triage)
    return CitationVerdict(
        ref_id=triage.ref_id,
        verdict="FABRICATED",
        mode="agentic",
        action="remove_citation",
        explanation=triage.triage_reason,
        flags=meta_flags,
        metadata_verdict="FABRICATED",
        metadata_flags=meta_flags,
        metadata_explanation=triage.triage_reason,
        claim_verdict=None,
        claim_flags=[],
        claim_explanation=None,
        existence=triage.existence,
        metadata=triage.metadata,
    )


def _verdict_unverifiable(triage: TriageResult) -> CitationVerdict:
    meta_flags = _existence_metadata_flags(triage) + ["insufficient_data"]
    meta_flags = _dedup(meta_flags)
    return CitationVerdict(
        ref_id=triage.ref_id,
        verdict="UNVERIFIABLE",
        mode="agentic",
        action="no_action",
        explanation=triage.triage_reason,
        flags=meta_flags,
        metadata_verdict="UNVERIFIABLE",
        metadata_flags=meta_flags,
        metadata_explanation=triage.triage_reason,
        claim_verdict=None,
        claim_flags=[],
        claim_explanation=None,
        existence=triage.existence,
        metadata=triage.metadata,
    )


# ---------------------------------------------------------------------------
# Resolve clear-cut routes (no agent needed)
# ---------------------------------------------------------------------------

def resolve_clear_route(triage: TriageResult) -> Optional[CitationVerdict]:
    """Return a verdict if the route is clear-cut, None if an agent is needed.

    This is the fast path — no LLM calls.
    """
    if triage.route == TriageRoute.CLEAR_VALID:
        return _verdict_clear_valid(triage)
    if triage.route == TriageRoute.CLEAR_FABRICATED:
        return _verdict_clear_fabricated(triage)
    if triage.route == TriageRoute.UNVERIFIABLE:
        return _verdict_unverifiable(triage)
    return None  # needs agent


# ---------------------------------------------------------------------------
# Metadata dimension builder
# ---------------------------------------------------------------------------

def _build_metadata_dimension(
    triage: TriageResult,
    agent_verdict: str,
    agent_explanation: str,
    agent_flags: list[str],
    field_discrepancies: list[dict] | None = None,
) -> tuple[str, str, list[str]]:
    """Return (metadata_verdict, metadata_explanation, metadata_flags)."""
    explanation = agent_explanation or ""
    if field_discrepancies:
        disc_notes = []
        for d in field_discrepancies:
            note = f"{d.get('field', '?')}: {d.get('status', '?')}"
            if d.get("investigation_notes"):
                note += f" — {d['investigation_notes']}"
            disc_notes.append(note)
        if disc_notes:
            explanation += f" | Discrepancies: {'; '.join(disc_notes)}"

    flags = _existence_metadata_flags(triage) + list(agent_flags) + ["agentic_agent_used"]
    return agent_verdict, explanation, _dedup(flags)


# ---------------------------------------------------------------------------
# Claim dimension builder
# ---------------------------------------------------------------------------

def _build_claim_dimension(
    claim_verdicts: Optional[list[ClaimVerdict]],
    claim_error: Optional[str] = None,
) -> tuple[Optional[str], Optional[str], list[str]]:
    """Return (claim_verdict, claim_explanation, claim_flags).

    Three distinct states, in priority order:
    - ``claim_error`` set → the claim agent crashed. UNVERIFIABLE with an
      explicit ``claim_agent_error`` flag so the report surfaces the
      attempt-and-failure instead of silently reporting "not applicable".
    - ``claim_verdicts`` is None → claim verification was not applicable
      (e.g. no substantive citations). Dimension is absent on the verdict.
    - ``claim_verdicts`` is [] → agent ran but returned nothing.
      UNVERIFIABLE with ``claim_verification_empty`` flag.

    When a non-empty list is supplied, auto-detects 2-class
    (SUPPORTED/NOT_SUPPORTED) vs 3-class scheme.
    """
    if claim_error is not None:
        return (
            "UNVERIFIABLE",
            f"Claim agent failed: {claim_error}",
            ["claim_agent_error", "agentic_agent_used"],
        )

    if claim_verdicts is None:
        return None, None, []

    if not claim_verdicts:
        return (
            "UNVERIFIABLE",
            "Claim verification returned no results.",
            ["claim_verification_empty", "agentic_agent_used"],
        )

    # Per-reference rollup uses a severity ladder normalized across schemes:
    #   any CONTRADICTS / NOT_SUPPORTED  → CONTRADICTS
    #   else any NEUTRAL                 → NEUTRAL
    #   else                             → SUPPORTED
    is_binary = all(cv.verdict in ("SUPPORTED", "NOT_SUPPORTED") for cv in claim_verdicts)
    if is_binary:
        not_supported = [cv for cv in claim_verdicts if cv.verdict == "NOT_SUPPORTED"]
        if not_supported:
            explanations = [cv.explanation for cv in not_supported if cv.explanation]
            expl = (
                f"{len(not_supported)} citing sentence(s) not supported by the cited paper. "
            )
            if explanations:
                expl += explanations[0]
            return (
                "CONTRADICTS",
                expl,
                ["claim_not_supported", "agentic_agent_used"],
            )
        support_count = len(claim_verdicts)
        return (
            "SUPPORTED",
            f"Claim verification: {support_count} supported (binary scheme).",
            ["agentic_agent_used"],
        )

    has_contradiction = any(cv.verdict == "CONTRADICTS" for cv in claim_verdicts)
    if has_contradiction:
        contradicted = [cv for cv in claim_verdicts if cv.verdict == "CONTRADICTS"]
        explanations = [cv.explanation for cv in contradicted if cv.explanation]
        expl = (
            f"{len(contradicted)} citing sentence(s) contradict the cited paper. "
        )
        if explanations:
            expl += explanations[0]
        return (
            "CONTRADICTS",
            expl,
            ["claim_contradicts", "agentic_agent_used"],
        )

    support_count = sum(1 for cv in claim_verdicts if cv.verdict == "SUPPORTS")
    neutral_count = sum(1 for cv in claim_verdicts if cv.verdict == "NEUTRAL")
    if neutral_count > 0:
        return (
            "NEUTRAL",
            (
                f"Claim verification: {support_count} supported, {neutral_count} neutral "
                f"(cited paper does not directly address the citing sentence)."
            ),
            ["claim_neutral", "agentic_agent_used"],
        )

    return (
        "SUPPORTED",
        f"Claim verification: {support_count} supported.",
        ["agentic_agent_used"],
    )


# ---------------------------------------------------------------------------
# Assemble final verdict from dimensions
# ---------------------------------------------------------------------------

def _assemble(
    triage: TriageResult,
    metadata_verdict: Optional[str],
    metadata_explanation: Optional[str],
    metadata_flags: list[str],
    claim_verdict: Optional[str],
    claim_explanation: Optional[str],
    claim_flags: list[str],
) -> CitationVerdict:
    rolled = rollup_verdict(metadata_verdict, claim_verdict)
    union_flags = _dedup(list(metadata_flags) + list(claim_flags))
    return CitationVerdict(
        ref_id=triage.ref_id,
        verdict=rolled,
        mode="agentic",
        action=_choose_action(metadata_verdict, claim_verdict),
        explanation=_combined_explanation(metadata_explanation, claim_explanation),
        flags=union_flags,
        metadata_verdict=metadata_verdict,
        metadata_flags=list(metadata_flags),
        metadata_explanation=metadata_explanation,
        claim_verdict=claim_verdict,
        claim_flags=list(claim_flags),
        claim_explanation=claim_explanation,
        existence=triage.existence,
        metadata=triage.metadata,
    )


# ---------------------------------------------------------------------------
# Public mergers used by the dispatch helpers
# ---------------------------------------------------------------------------

def merge_metadata_verdict(
    triage: TriageResult,
    agent_verdict: str,
    agent_explanation: str,
    agent_flags: list[str],
    field_discrepancies: list[dict] | None = None,
) -> CitationVerdict:
    """NEEDS_METADATA path: metadata agent only, claim not applicable."""
    mv, me, mf = _build_metadata_dimension(
        triage, agent_verdict, agent_explanation, agent_flags, field_discrepancies,
    )
    return _assemble(
        triage,
        metadata_verdict=mv,
        metadata_explanation=me,
        metadata_flags=mf,
        claim_verdict=None,
        claim_explanation=None,
        claim_flags=[],
    )


def merge_claim_verdicts(
    triage: TriageResult,
    claim_verdicts: list[ClaimVerdict],
) -> CitationVerdict:
    """NEEDS_CLAIM path: metadata dimension is deterministic, claim from agent.

    Metadata is VALID on the normal path. When the paper was retracted,
    triage still routes here (so the claim agent can check whether the
    citing text is supported), and the metadata dimension is forced to
    FABRICATED — retraction is a factual DB signal, not an LLM judgment.
    The top-level rollup then yields FABRICATED regardless of claim verdict.
    """
    cv, ce, cf = _build_claim_dimension(claim_verdicts if claim_verdicts is not None else [])

    meta_flags = _existence_metadata_flags(triage)
    if triage.metadata and triage.metadata.is_retracted:
        metadata_verdict = "FABRICATED"
        metadata_explanation = (
            f"Paper has been retracted. ({triage.triage_reason})"
        )
    else:
        metadata_verdict = "VALID"
        metadata_explanation = (
            f"Reference exists, metadata matches. ({triage.triage_reason})"
        )
    return _assemble(
        triage,
        metadata_verdict=metadata_verdict,
        metadata_explanation=metadata_explanation,
        metadata_flags=meta_flags,
        claim_verdict=cv,
        claim_explanation=ce,
        claim_flags=cf,
    )


def merge_both_verdicts(
    triage: TriageResult,
    metadata_verdict: str,
    metadata_explanation: str,
    metadata_flags: list[str],
    field_discrepancies: list[dict] | None = None,
    claim_verdicts: list[ClaimVerdict] | None = None,
    claim_error: str | None = None,
) -> CitationVerdict:
    """NEEDS_BOTH path: always run both dimensions independently.

    Claim verification runs even when metadata is FABRICATED — the paper
    was found, so its text can still be compared against the citing
    sentence. The two verdicts live side by side; the top-level ``verdict``
    is metadata-only (FABRICATED / VALID / UNVERIFIABLE), with the claim
    dimension reported separately as SUPPORTED / CONTRADICTS / NEUTRAL /
    UNVERIFIABLE.

    ``claim_error`` is set by the dispatcher when the claim agent raised an
    exception; the merger marks the claim dimension UNVERIFIABLE with a
    ``claim_agent_error`` flag so the failure is visible in the report
    rather than silently collapsing into "not applicable".
    """
    mv, me, mf = _build_metadata_dimension(
        triage, metadata_verdict, metadata_explanation, metadata_flags, field_discrepancies,
    )
    cv, ce, cf = _build_claim_dimension(claim_verdicts, claim_error=claim_error)
    return _assemble(
        triage,
        metadata_verdict=mv,
        metadata_explanation=me,
        metadata_flags=mf,
        claim_verdict=cv,
        claim_explanation=ce,
        claim_flags=cf,
    )


# ---------------------------------------------------------------------------
# Fallback on agent failure
# ---------------------------------------------------------------------------

def fallback_to_quick(triage: TriageResult) -> CitationVerdict:
    """Fall back when an agent fails.

    Conservative: if the reference was routed to an agent, it means the
    rule-based pipeline found it ambiguous. If the agent then fails, the
    honest answer is UNVERIFIABLE — not FABRICATED from a threshold that
    already flagged the case as uncertain.
    """
    if triage.existence is None:
        return _verdict_unverifiable(triage)

    if triage.existence.status == "FOUND":
        meta_flags = _existence_metadata_flags(triage) + ["agent_fallback_to_quick"]
        meta_flags = _dedup(meta_flags)
        meta_expl = (
            f"Paper found via {triage.existence.source} but the "
            f"verification agent failed to complete its analysis. "
            f"Manual review recommended."
        )
        return _assemble(
            triage,
            metadata_verdict="UNVERIFIABLE",
            metadata_explanation=meta_expl,
            metadata_flags=meta_flags,
            claim_verdict=None,
            claim_explanation=None,
            claim_flags=[],
        )

    # Paper not found — classify_quick handles NOT_FOUND reliably.
    verdict = classify_quick(triage.existence, triage.metadata)
    verdict.mode = "agentic"
    verdict.flags = _dedup(verdict.flags + ["agent_fallback_to_quick"])
    # Populate two-dimension fields so callers can rely on them everywhere.
    verdict.metadata_verdict = verdict.verdict
    verdict.metadata_flags = list(verdict.flags)
    verdict.metadata_explanation = verdict.explanation
    verdict.claim_verdict = None
    verdict.claim_flags = []
    verdict.claim_explanation = None
    return verdict
