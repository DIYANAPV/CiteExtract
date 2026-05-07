
from typing import Literal, Optional

from pydantic import BaseModel, Field


class FetchAttempt(BaseModel):

    source: str = Field(
        description=(
            "Which fetch path: oa_url | unpaywall | arxiv | s2_fallback | "
            "arxiv_fallback | html_meta_pdf | abstract_only | user_pdf"
        )
    )
    status: str = Field(
        description=(
            "Outcome bucket: ok | http_error | timeout | not_pdf | "
            "grobid_failed | unsafe_url | skipped | empty_text"
        )
    )
    ms: int = Field(default=0, description="Wall-clock milliseconds the step took")
    detail: Optional[str] = Field(
        default=None,
        description=(
            "Free-form context — HTTP status code, redirect target, "
            "exception type, etc. Truncated to 200 chars."
        ),
    )


class Chunk(BaseModel):

    text: str = Field(description="The chunk text content")
    section_name: Optional[str] = Field(
        default=None, description="Section heading this chunk belongs to"
    )
    paragraph_index: int = Field(description="Index of this chunk within the paper")


class ScoredChunk(BaseModel):

    chunk: Chunk
    bm25_score: float = Field(default=0.0, description="BM25 sparse retrieval score")
    dense_score: Optional[float] = Field(
        default=None, description="Dense embedding similarity (Phase B)"
    )
    rrf_score: Optional[float] = Field(
        default=None, description="Reciprocal rank fusion score (Phase B)"
    )


class FullTextResult(BaseModel):

    source: Literal[
        "user_pdf", "s2_api", "unpaywall", "core", "arxiv",
        "oa_url",
        "html_meta_pdf",
        "s2_fallback", "arxiv_fallback",
        "abstract_only", "not_found",
    ] = Field(description="Where the text came from")
    full_text: Optional[str] = Field(default=None, description="Raw full text")
    sections: list[dict] = Field(
        default_factory=list,
        description="Structured sections: [{name: str, text: str}]",
    )
    abstract: Optional[str] = Field(default=None)
    truncated: bool = Field(
        default=False,
        description=(
            "True when ``full_text`` was truncated before caching (very long "
            "papers). The in-flight result returned by ``get_full_text`` may "
            "carry the untruncated body; cached entries always carry the "
            "truncated version."
        ),
    )
    attempts: list[FetchAttempt] = Field(
        default_factory=list,
        description=(
            "Per-step trace of the retrieval waterfall. Empty when served "
            "from a cache entry written before telemetry shipped."
        ),
    )


class ClaimVerdict(BaseModel):

    verdict: Literal[
        "SUPPORTS", "CONTRADICTS", "NEUTRAL",
        "SUPPORTED", "NOT_SUPPORTED",
    ] = Field(description="Verdict in either 3-class or 2-class scheme")
    explanation: str = Field(default="", description="LLM reasoning")
    evidence_quote: str = Field(default="", description="Most relevant quote from passages")


class ComprehensionResult(BaseModel):

    ref_id: str = Field(description="Links to Reference.ref_id")
    citing_sentence: str = Field(description="The claim being checked")
    context_before: str = Field(default="", description="Sentences before the citing sentence")
    context_after: str = Field(default="", description="Sentences after the citing sentence")
    paper_found: bool = Field(description="Whether the cited paper was found in L2")
    full_text_available: bool = Field(
        description="Whether full text (not just abstract) was available"
    )
    full_text_source: Optional[str] = Field(
        default=None, description="Source of text: s2_api, unpaywall, user_pdf, abstract_only, etc."
    )
    top_passages: list[ScoredChunk] = Field(
        default_factory=list, description="Top-k retrieved passages ranked by relevance"
    )
    claim_verdict: Optional[ClaimVerdict] = Field(
        default=None, description="LLM analysis of citing sentence vs passages"
    )
    paper_metadata: dict = Field(
        default_factory=dict, description="Title, authors, year, doi of cited paper"
    )
    fetch_attempts: list[FetchAttempt] = Field(
        default_factory=list,
        description=(
            "Per-reference trace of the full-text retrieval waterfall. "
            "Mirrors ``FullTextResult.attempts`` so a downstream consumer "
            "can render the trace without re-plumbing through the agentic "
            "data-flow path."
        ),
    )


class ComprehensionReport(BaseModel):

    input_file: str
    timestamp: str
    total_citations: int = Field(description="Total citing sentence / reference pairs processed")
    results: list[ComprehensionResult] = Field(default_factory=list)
    coverage: dict = Field(
        default_factory=dict,
        description="Stats: {full_text: N, abstract_only: N, not_found: N}",
    )
    warnings: list[str] = Field(default_factory=list)
