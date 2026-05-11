from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from citeextract.models.reference import Reference
from citeextract.verification.cache import APICache
from citeextract.verification.existence import check_existence
from citeextract.verification.metadata import validate_metadata
from citeextract.verification.triage import TriageRoute, triage_reference

log = logging.getLogger(__name__)


@dataclass
class ProductionResult:
    predicted_verdict: str
    raw_verdict: str
    triage_route: str
    metadata_agent_called: bool
    db_source_matched: Optional[str]
    explanation: str
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    error: Optional[str]


def _record_to_reference(record: dict) -> Reference:
    rec_id = f"bench-{record['instance_id']:04d}"
    src = record.get("source", "")
    source_format = "bibtex" if src == "s2_real" else "text"
    return Reference(
        ref_id=rec_id,
        title=(record.get("title") or "") or None,
        authors=list(record.get("authors") or []),
        year=record.get("year"),
        venue=record.get("venue"),
        doi=record.get("doi"),
        arxiv_id=record.get("arxiv_id"),
        raw_text=record.get("raw_reference_string") or "",
        source_format=source_format,
    )


async def run_one_production(
    record: dict,
    *,
    http_client: httpx.AsyncClient,
    cache: APICache,
    metadata_agent_factory,
) -> ProductionResult:
    t0 = time.perf_counter()
    ref = _record_to_reference(record)

    try:
        existence = await check_existence(ref, http_client, cache)
    except Exception as e:
        return ProductionResult(
            predicted_verdict="fabricated", raw_verdict="ERROR",
            triage_route="ERROR", metadata_agent_called=False,
            db_source_matched=None,
            explanation=f"existence cascade failed: {e}",
            cost_usd=0.0, prompt_tokens=0, completion_tokens=0,
            latency_seconds=time.perf_counter() - t0,
            error=f"existence_error:{type(e).__name__}",
        )

    metadata = validate_metadata(ref, existence)
    triage = triage_reference(
        ref_id=ref.ref_id,
        existence=existence,
        metadata=metadata,
        citations=[],
        pre_retrieved_passages={},
        full_text_result=None,
    )

    if triage.route == TriageRoute.CLEAR_VALID:
        return ProductionResult(
            predicted_verdict="valid", raw_verdict="CLEAR_VALID",
            triage_route=triage.route.name, metadata_agent_called=False,
            db_source_matched=existence.source,
            explanation=triage.triage_reason,
            cost_usd=0.0, prompt_tokens=0, completion_tokens=0,
            latency_seconds=time.perf_counter() - t0, error=None,
        )
    if triage.route == TriageRoute.CLEAR_FABRICATED:
        return ProductionResult(
            predicted_verdict="fabricated", raw_verdict="CLEAR_FABRICATED",
            triage_route=triage.route.name, metadata_agent_called=False,
            db_source_matched=existence.source if existence.status == "FOUND" else None,
            explanation=triage.triage_reason,
            cost_usd=0.0, prompt_tokens=0, completion_tokens=0,
            latency_seconds=time.perf_counter() - t0, error=None,
        )
    if triage.route == TriageRoute.UNVERIFIABLE:
        return ProductionResult(
            predicted_verdict="fabricated", raw_verdict="UNVERIFIABLE",
            triage_route=triage.route.name, metadata_agent_called=False,
            db_source_matched=existence.source if existence.status == "FOUND" else None,
            explanation=triage.triage_reason,
            cost_usd=0.0, prompt_tokens=0, completion_tokens=0,
            latency_seconds=time.perf_counter() - t0, error=None,
        )

    if triage.route in (TriageRoute.NEEDS_METADATA, TriageRoute.NEEDS_BOTH):
        agent = metadata_agent_factory()
        from citeextract.verification.agentic.metadata_agent import build_metadata_user_message

        title_comp = next(
            (c for c in (metadata.comparisons or []) if c.field == "title"), None,
        )
        author_comp = next(
            (c for c in (metadata.comparisons or []) if c.field == "authors"), None,
        )
        msg = build_metadata_user_message(
            ref_title=ref.title,
            ref_authors=ref.authors or [],
            ref_year=ref.year,
            ref_venue=ref.venue,
            ref_doi=ref.doi,
            ref_raw_text=ref.raw_text or "",
            citation_format=existence.citation_format,
            db_title=existence.matched_title,
            db_authors=existence.matched_authors or [],
            db_year=existence.matched_year,
            db_venue=existence.matched_venue,
            db_doi=existence.matched_doi,
            db_source=existence.source,
            title_comparison=_dict_or_none(title_comp),
            author_comparison=_dict_or_none(author_comp),
            triage_reason=triage.triage_reason,
        )

        try:
            verdict_payload = await agent.investigate(msg)
            agent_verdict = str(verdict_payload.get("verdict", "UNVERIFIABLE")).upper()
            explanation = str(verdict_payload.get("explanation", "") or "")[:600]
            err = None
        except Exception as e:
            agent_verdict = "UNVERIFIABLE"
            explanation = f"metadata agent crashed: {e}"
            err = f"agent_error:{type(e).__name__}"

        mapped = "valid" if agent_verdict == "VALID" else "fabricated"
        ct = agent.cost_tracker
        return ProductionResult(
            predicted_verdict=mapped, raw_verdict=agent_verdict,
            triage_route=triage.route.name, metadata_agent_called=True,
            db_source_matched=existence.source if existence.status == "FOUND" else None,
            explanation=explanation,
            cost_usd=float(ct.estimated_cost_usd),
            prompt_tokens=int(ct.total_input_tokens),
            completion_tokens=int(ct.total_output_tokens),
            latency_seconds=time.perf_counter() - t0,
            error=err,
        )

    return ProductionResult(
        predicted_verdict="fabricated", raw_verdict=triage.route.name,
        triage_route=triage.route.name, metadata_agent_called=False,
        db_source_matched=existence.source if existence.status == "FOUND" else None,
        explanation=f"unexpected route: {triage.route.name}",
        cost_usd=0.0, prompt_tokens=0, completion_tokens=0,
        latency_seconds=time.perf_counter() - t0,
        error=f"unexpected_route:{triage.route.name}",
    )


def _dict_or_none(comp) -> Optional[dict]:
    if comp is None:
        return None
    return {
        "exact_match": getattr(comp, "exact_match", None),
        "similarity": getattr(comp, "similarity", None),
        "differences": list(getattr(comp, "differences", []) or []),
    }
