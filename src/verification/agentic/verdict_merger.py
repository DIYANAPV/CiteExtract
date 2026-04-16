"""Verdict merger — combines triage routes + agent outputs into CitationVerdict.

For clear-cut routes (CLEAR_VALID, CLEAR_FABRICATED, UNVERIFIABLE), the
verdict is produced directly from L2/L3 results without any LLM call.

For agent-routed cases, the merger integrates the agent's output with the
existing evidence trail (ExistenceResult, MetadataResult) into a single
CitationVerdict compatible with the existing report pipeline.
"""

import logging
from typing import Optional

from src.classification.classifier import CitationVerdict, classify_quick
from src.models.comprehension import ClaimVerdict
from src.verification.metadata import MetadataResult
from src.verification.triage import TriageResult, TriageRoute

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Clear-cut verdict builders (no LLM)
# ---------------------------------------------------------------------------

def _verdict_clear_valid(triage: TriageResult) -> CitationVerdict:
    return CitationVerdict(
        ref_id=triage.ref_id,
        verdict="VALID",
        mode="agentic",
        action="no_action",
        explanation=f"Reference exists and metadata matches. ({triage.triage_reason})",
        flags=_collect_flags(triage),
        existence=triage.existence,
        metadata=triage.metadata,
    )


def _verdict_clear_fabricated(triage: TriageResult) -> CitationVerdict:
    return CitationVerdict(
        ref_id=triage.ref_id,
        verdict="FABRICATED",
        mode="agentic",
        action="remove_citation",
        explanation=triage.triage_reason,
        flags=_collect_flags(triage),
        existence=triage.existence,
        metadata=triage.metadata,
    )


def _verdict_unverifiable(triage: TriageResult) -> CitationVerdict:
    return CitationVerdict(
        ref_id=triage.ref_id,
        verdict="UNVERIFIABLE",
        mode="agentic",
        action="no_action",
        explanation=triage.triage_reason,
        flags=_collect_flags(triage) + ["insufficient_data"],
        existence=triage.existence,
        metadata=triage.metadata,
    )


def _collect_flags(triage: TriageResult) -> list[str]:
    """Collect all flags from existence + metadata results."""
    flags: list[str] = []
    if triage.existence:
        flags.extend(triage.existence.flags)
    if triage.metadata:
        flags.extend(triage.metadata.flags)
    return list(dict.fromkeys(flags))  # deduplicate preserving order


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
# Merge metadata agent output
# ---------------------------------------------------------------------------

def merge_metadata_verdict(
    triage: TriageResult,
    agent_verdict: str,
    agent_explanation: str,
    agent_flags: list[str],
    field_discrepancies: list[dict] | None = None,
) -> CitationVerdict:
    """Merge metadata agent output into a CitationVerdict.

    The ExistenceResult comes from L2 (real database source), not the agent.
    The agent's judgment determines the final verdict.
    """
    action = "remove_citation" if agent_verdict == "FABRICATED" else "no_action"

    # Build explanation combining triage reason + agent reasoning
    explanation = f"{agent_explanation}"
    if field_discrepancies:
        disc_notes = []
        for d in field_discrepancies:
            note = f"{d.get('field', '?')}: {d.get('status', '?')}"
            if d.get("investigation_notes"):
                note += f" — {d['investigation_notes']}"
            disc_notes.append(note)
        if disc_notes:
            explanation += f" | Discrepancies: {'; '.join(disc_notes)}"

    flags = _collect_flags(triage) + agent_flags + ["agentic_agent_used"]
    flags = list(dict.fromkeys(flags))

    return CitationVerdict(
        ref_id=triage.ref_id,
        verdict=agent_verdict,
        mode="agentic",
        action=action,
        explanation=explanation,
        flags=flags,
        existence=triage.existence,
        metadata=triage.metadata,
    )


# ---------------------------------------------------------------------------
# Merge claim agent output
# ---------------------------------------------------------------------------

def merge_claim_verdicts(
    triage: TriageResult,
    claim_verdicts: list[ClaimVerdict],
) -> CitationVerdict:
    """Merge claim agent output into a CitationVerdict.

    If any claim is CONTRADICTS → overall MISREPRESENTED.
    If all SUPPORTS or NEUTRAL → VALID.
    If no verdicts returned → VALID with note (claims couldn't be evaluated).
    """
    if not claim_verdicts:
        log.warning(f"No claim verdicts returned for ref {triage.ref_id}")
        return CitationVerdict(
            ref_id=triage.ref_id,
            verdict="VALID",
            mode="agentic",
            action="no_action",
            explanation="Reference exists, metadata matches. Claim verification returned no results.",
            flags=_collect_flags(triage) + ["claim_verification_empty"],
            existence=triage.existence,
            metadata=triage.metadata,
        )

    has_contradiction = any(cv.verdict == "CONTRADICTS" for cv in claim_verdicts)

    if has_contradiction:
        contradicted = [cv for cv in claim_verdicts if cv.verdict == "CONTRADICTS"]
        explanations = [cv.explanation for cv in contradicted if cv.explanation]
        evidence = [cv.evidence_quote for cv in contradicted if cv.evidence_quote]

        explanation = (
            f"Paper exists and metadata matches, but "
            f"{len(contradicted)} citing sentence(s) misrepresent the cited paper. "
        )
        if explanations:
            explanation += explanations[0]

        flags = _collect_flags(triage) + ["claim_contradicts", "agentic_agent_used"]

        return CitationVerdict(
            ref_id=triage.ref_id,
            verdict="MISREPRESENTED",
            mode="agentic",
            action="verify_claim",
            explanation=explanation,
            flags=list(dict.fromkeys(flags)),
            existence=triage.existence,
            metadata=triage.metadata,
        )

    # All SUPPORTS or NEUTRAL → VALID
    support_count = sum(1 for cv in claim_verdicts if cv.verdict == "SUPPORTS")
    neutral_count = sum(1 for cv in claim_verdicts if cv.verdict == "NEUTRAL")

    explanation = (
        f"Reference exists, metadata matches. "
        f"Claim verification: {support_count} supported, {neutral_count} neutral."
    )

    return CitationVerdict(
        ref_id=triage.ref_id,
        verdict="VALID",
        mode="agentic",
        action="no_action",
        explanation=explanation,
        flags=_collect_flags(triage) + ["agentic_agent_used"],
        existence=triage.existence,
        metadata=triage.metadata,
    )


# ---------------------------------------------------------------------------
# Merge NEEDS_BOTH (metadata + claim, sequential)
# ---------------------------------------------------------------------------

def merge_both_verdicts(
    triage: TriageResult,
    metadata_verdict: str,
    metadata_explanation: str,
    metadata_flags: list[str],
    field_discrepancies: list[dict] | None = None,
    claim_verdicts: list[ClaimVerdict] | None = None,
) -> CitationVerdict:
    """Merge NEEDS_BOTH results: metadata first, then claims if metadata is OK.

    If metadata agent says FABRICATED → return FABRICATED (skip claims).
    If metadata agent says VALID → check claims.
    If metadata agent says UNVERIFIABLE → return with metadata info only.
    """
    if metadata_verdict == "FABRICATED":
        return merge_metadata_verdict(
            triage, metadata_verdict, metadata_explanation,
            metadata_flags, field_discrepancies,
        )

    if metadata_verdict == "UNVERIFIABLE":
        return merge_metadata_verdict(
            triage, metadata_verdict, metadata_explanation,
            metadata_flags + ["metadata_unverifiable_claims_skipped"],
            field_discrepancies,
        )

    # Metadata is VALID — check claims
    if claim_verdicts:
        result = merge_claim_verdicts(triage, claim_verdicts)
        # Enrich explanation with metadata context
        if field_discrepancies:
            disc_summary = "; ".join(
                f"{d.get('field')}: {d.get('status')}"
                for d in field_discrepancies
                if d.get("status") not in ("match", "MATCH")
            )
            if disc_summary:
                result.explanation += f" (Metadata notes: {disc_summary})"
        return result

    # No claims to check (shouldn't happen for NEEDS_BOTH, but be safe)
    return merge_metadata_verdict(
        triage, "VALID", metadata_explanation, metadata_flags, field_discrepancies,
    )


# ---------------------------------------------------------------------------
# Fallback on agent failure
# ---------------------------------------------------------------------------

def fallback_to_quick(triage: TriageResult) -> CitationVerdict:
    """Fall back when an agent fails.

    Conservative: if the reference was routed to an agent, it means the
    rule-based pipeline found it ambiguous.  If the agent then fails, the
    honest answer is UNVERIFIABLE — not FABRICATED from a threshold that
    already flagged the case as uncertain.

    Only produce VALID (paper clearly exists) or UNVERIFIABLE (can't tell).
    Never produce FABRICATED from a fallback — that's the agent's job.
    """
    if triage.existence is None:
        return _verdict_unverifiable(triage)

    # Paper was found in a database — we know it exists, agent just
    # couldn't finish judging metadata.  Return UNVERIFIABLE with context.
    if triage.existence.status == "FOUND":
        flags = _collect_flags(triage) + ["agent_fallback_to_quick"]
        return CitationVerdict(
            ref_id=triage.ref_id,
            verdict="UNVERIFIABLE",
            mode="agentic",
            action="no_action",
            explanation=(
                f"Paper found via {triage.existence.source} but the "
                f"verification agent failed to complete its analysis. "
                f"Manual review recommended."
            ),
            flags=flags,
            existence=triage.existence,
            metadata=triage.metadata,
        )

    # Paper not found — let classify_quick decide (NOT_FOUND logic is reliable)
    verdict = classify_quick(triage.existence, triage.metadata)
    verdict.mode = "agentic"
    verdict.flags = list(dict.fromkeys(verdict.flags + ["agent_fallback_to_quick"]))
    return verdict
