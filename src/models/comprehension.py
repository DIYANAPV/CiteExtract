"""Comprehension and claim verification models — passage retrieval + LLM analysis."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A text chunk from a cited paper, typically one paragraph."""

    text: str = Field(description="The chunk text content")
    section_name: Optional[str] = Field(
        default=None, description="Section heading this chunk belongs to"
    )
    paragraph_index: int = Field(description="Index of this chunk within the paper")


class ScoredChunk(BaseModel):
    """A chunk with retrieval scores."""

    chunk: Chunk
    bm25_score: float = Field(default=0.0, description="BM25 sparse retrieval score")
    dense_score: Optional[float] = Field(
        default=None, description="Dense embedding similarity (Phase B)"
    )
    rrf_score: Optional[float] = Field(
        default=None, description="Reciprocal rank fusion score (Phase B)"
    )


class FullTextResult(BaseModel):
    """Result of attempting to retrieve the full text of a cited paper."""

    source: Literal[
        "user_pdf", "s2_api", "unpaywall", "core", "arxiv", "abstract_only", "not_found"
    ] = Field(description="Where the text came from")
    full_text: Optional[str] = Field(default=None, description="Raw full text")
    sections: list[dict] = Field(
        default_factory=list,
        description="Structured sections: [{name: str, text: str}]",
    )
    abstract: Optional[str] = Field(default=None)


class ClaimVerdict(BaseModel):
    """LLM verdict for a citing sentence checked against retrieved passages."""

    verdict: Literal["SUPPORTS", "CONTRADICTS", "NEUTRAL"] = Field(
        description="Whether the passages support, contradict, or don't address the claim"
    )
    explanation: str = Field(default="", description="LLM reasoning")
    evidence_quote: str = Field(default="", description="Most relevant quote from passages")


class ComprehensionResult(BaseModel):
    """Result of comprehension check for one (citing_sentence, reference) pair."""

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


class ComprehensionReport(BaseModel):
    """Full comprehension report for all citations in a paper."""

    input_file: str
    timestamp: str
    total_citations: int = Field(description="Total citing sentence / reference pairs processed")
    results: list[ComprehensionResult] = Field(default_factory=list)
    coverage: dict = Field(
        default_factory=dict,
        description="Stats: {full_text: N, abstract_only: N, not_found: N}",
    )
    warnings: list[str] = Field(default_factory=list)
