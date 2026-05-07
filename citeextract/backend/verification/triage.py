
import enum
import logging
from typing import Optional

from pydantic import BaseModel, Field

from citeextract.models.citation import Citation
from citeextract.models.comprehension import FullTextResult, ScoredChunk
from citeextract.models.verdict import ExistenceResult
from citeextract.verification.filters import is_substantive_citation
from citeextract.verification.metadata import FieldComparison, MetadataResult

from citeextract.verification.matching import WEB_SOURCES

log = logging.getLogger(__name__)

_DEFAULT_MIN_DBS_FOR_FABRICATED = 2


class TriageRoute(str, enum.Enum):

    CLEAR_VALID = "CLEAR_VALID"
    CLEAR_FABRICATED = "CLEAR_FABRICATED"
    UNVERIFIABLE = "UNVERIFIABLE"
    NEEDS_METADATA = "NEEDS_METADATA"
    NEEDS_CLAIM = "NEEDS_CLAIM"
    NEEDS_BOTH = "NEEDS_BOTH"


class ContextQuality(BaseModel):

    total_sentences: int = Field(
        description="Total sentence count: citing + before + after",
    )
    has_before: bool = Field(description="Whether context_before is non-empty")
    has_after: bool = Field(description="Whether context_after is non-empty")


def _assess_context_quality(citation: Citation) -> ContextQuality:
    from citeextract.citation.context_extractor import split_sentences

    count = 1
    has_before = bool(citation.context_before and citation.context_before.strip())
    has_after = bool(citation.context_after and citation.context_after.strip())
    if has_before:
        count += len(split_sentences(citation.context_before.strip()))
    if has_after:
        count += len(split_sentences(citation.context_after.strip()))
    return ContextQuality(total_sentences=count, has_before=has_before, has_after=has_after)


class TriageResult(BaseModel):

    ref_id: str
    route: TriageRoute
    triage_reason: str = Field(description="Human-readable reason for the route")
    existence: Optional[ExistenceResult] = None
    metadata: Optional[MetadataResult] = None
    substantive_citations: list[Citation] = Field(default_factory=list)
    pre_retrieved_passages: dict[str, list[ScoredChunk]] = Field(
        default_factory=dict,
        description="Citing sentence → list[ScoredChunk] (pre-retrieved, keyed by sentence text)",
    )
    full_text_result: Optional[FullTextResult] = None
    context_quality: dict[str, ContextQuality] = Field(
        default_factory=dict,
        description="Citing sentence → ContextQuality",
    )

    model_config = {"arbitrary_types_allowed": True}


def triage_reference(
    ref_id: str,
    existence: Optional[ExistenceResult],
    metadata: Optional[MetadataResult],
    citations: list[Citation],
    pre_retrieved_passages: Optional[dict[str, list]] = None,
    full_text_result: Optional[FullTextResult] = None,
    min_dbs_for_fabricated: int = _DEFAULT_MIN_DBS_FOR_FABRICATED,
) -> TriageResult:
    if pre_retrieved_passages is None:
        pre_retrieved_passages = {}

    substantive = [c for c in citations if is_substantive_citation(c)]
    has_claims = len(substantive) > 0

    ctx_quality = {c.citing_sentence: _assess_context_quality(c) for c in substantive}

    def _result(route: TriageRoute, reason: str) -> TriageResult:
        return TriageResult(
            ref_id=ref_id,
            route=route,
            triage_reason=reason,
            existence=existence,
            metadata=metadata,
            substantive_citations=substantive,
            pre_retrieved_passages=pre_retrieved_passages,
            full_text_result=full_text_result,
            context_quality=ctx_quality,
        )

    if existence is None:
        return _result(TriageRoute.UNVERIFIABLE, "No existence result available")

    if existence.status == "NOT_FOUND":
        checked = len(existence.databases_checked)
        if checked >= min_dbs_for_fabricated:
            return _result(
                TriageRoute.CLEAR_FABRICATED,
                f"Not found in {checked} databases: {', '.join(existence.databases_checked)}",
            )
        return _result(
            TriageRoute.UNVERIFIABLE,
            f"Not found, but only {checked} database(s) checked — insufficient coverage",
        )

    if metadata and metadata.is_retracted:
        if substantive:
            return _result(
                TriageRoute.NEEDS_CLAIM,
                "Paper retracted — still verify whether citing text is supported",
            )
        return _result(TriageRoute.CLEAR_FABRICATED, "Paper has been retracted")

    if "doi_title_mismatch" in existence.flags:
        route = TriageRoute.NEEDS_BOTH if has_claims else TriageRoute.NEEDS_METADATA
        return _result(route, "DOI resolves but title does not match — needs investigation")

    if metadata and metadata.has_metadata_mismatch:
        route = TriageRoute.NEEDS_BOTH if has_claims else TriageRoute.NEEDS_METADATA
        mismatched = [c.field for c in metadata.comparisons if c.status == "MISMATCH" and c.field != "title"]
        return _result(route, f"Metadata mismatch in: {', '.join(mismatched)}")

    title_comp = _get_title_comparison(metadata)
    if title_comp and title_comp.status == "MISMATCH":
        route = TriageRoute.NEEDS_BOTH if has_claims else TriageRoute.NEEDS_METADATA
        sim_str = f"{title_comp.similarity:.2f}" if title_comp.similarity is not None else "n/a"
        return _result(route, f"Title mismatch (similarity={sim_str})")

    author_comp = _get_author_comparison(metadata)
    if author_comp and author_comp.status == "CLOSE_MATCH":
        unmatched = _count_unmatched_authors(metadata)
        if unmatched > 0:
            route = TriageRoute.NEEDS_BOTH if has_claims else TriageRoute.NEEDS_METADATA
            return _result(route, f"{unmatched} unmatched author(s) in CLOSE_MATCH — needs investigation")

    if existence.source in WEB_SOURCES:
        if has_claims:
            return _result(TriageRoute.NEEDS_CLAIM, "Web source verified, but citing claims need checking")
        return _result(TriageRoute.CLEAR_VALID, "Web source verified")

    if has_claims and pre_retrieved_passages:
        return _result(TriageRoute.NEEDS_CLAIM, "Metadata matches, substantive citing claims to verify")

    if has_claims and not pre_retrieved_passages:
        return _result(TriageRoute.NEEDS_CLAIM, "Metadata matches, claims present but no passages pre-retrieved")

    return _result(TriageRoute.CLEAR_VALID, "Reference exists, metadata matches, no substantive claims")


def _get_title_comparison(metadata: Optional[MetadataResult]) -> Optional[FieldComparison]:
    if not metadata:
        return None
    for comp in metadata.comparisons:
        if comp.field == "title":
            return comp
    return None


def _get_author_comparison(metadata: Optional[MetadataResult]) -> Optional[FieldComparison]:
    if not metadata:
        return None
    for comp in metadata.comparisons:
        if comp.field == "authors":
            return comp
    return None


def _count_unmatched_authors(metadata: Optional[MetadataResult]) -> int:
    if not metadata:
        return 0

    import re
    for flag in metadata.flags:
        if "unmatched" in flag.lower() and "author" in flag.lower():
            match = re.search(r"(\d+)\s+unmatched\s+author", flag, re.IGNORECASE)
            if match:
                return int(match.group(1))
            return 1

    author_comp = _get_author_comparison(metadata)
    if author_comp and author_comp.status == "CLOSE_MATCH" and author_comp.similarity is not None:
        if author_comp.similarity < 0.9:
            return 1
    return 0


def triage_all(
    exist_map: dict[str, ExistenceResult],
    metadata_map: dict[str, MetadataResult],
    citations_by_ref: dict[str, list[Citation]],
    passages_by_ref: Optional[dict[str, dict[str, list]]] = None,
    fulltext_by_ref: Optional[dict[str, FullTextResult]] = None,
    triage_config: Optional[dict] = None,
) -> list[TriageResult]:
    if passages_by_ref is None:
        passages_by_ref = {}
    if fulltext_by_ref is None:
        fulltext_by_ref = {}
    if triage_config is None:
        triage_config = {}

    min_dbs = triage_config.get("min_databases_for_fabricated", _DEFAULT_MIN_DBS_FOR_FABRICATED)

    results = []
    all_ref_ids = set(exist_map.keys()) | set(citations_by_ref.keys())

    for ref_id in sorted(all_ref_ids):
        existence = exist_map.get(ref_id)
        metadata = metadata_map.get(ref_id)
        citations = citations_by_ref.get(ref_id, [])
        passages = passages_by_ref.get(ref_id, {})
        ft = fulltext_by_ref.get(ref_id)

        result = triage_reference(
            ref_id=ref_id,
            existence=existence,
            metadata=metadata,
            citations=citations,
            pre_retrieved_passages=passages,
            full_text_result=ft,
            min_dbs_for_fabricated=min_dbs,
        )
        results.append(result)

    route_counts = {}
    for r in results:
        route_counts[r.route.value] = route_counts.get(r.route.value, 0) + 1
    log.info(f"Triage complete: {route_counts}")

    return results
