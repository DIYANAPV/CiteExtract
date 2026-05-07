
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ExistenceResult(BaseModel):

    ref_id: str = Field(description="Links to Reference.ref_id")
    status: Literal["FOUND", "NOT_FOUND"] = Field(description="Whether the paper was found")
    source: Optional[str] = Field(
        default=None,
        description="Database that matched: crossref, semantic_scholar, openalex, pubmed",
    )

    matched_title: Optional[str] = Field(default=None)
    matched_authors: list[str] = Field(default_factory=list)
    matched_year: Optional[int] = Field(default=None)
    matched_venue: Optional[str] = Field(default=None)
    venue_aliases: list[str] = Field(
        default_factory=list,
        description="Known alternate names for the venue (from S2/OpenAlex)",
    )
    matched_doi: Optional[str] = Field(default=None)
    matched_arxiv_id: Optional[str] = Field(
        default=None, description="arXiv ID from DB (used to bridge arXiv vs publisher DOIs)"
    )
    anthology_id: Optional[str] = Field(
        default=None,
        description=(
            "ACL Anthology ID from DB. Used to bridge cross-system "
            "aliases — e.g. a citation that includes the anthology URL "
            "vs a DB record that holds the publisher DOI separately."
        ),
    )
    abstract: Optional[str] = Field(default=None, description="Cached abstract from DB")
    oa_url: Optional[str] = Field(
        default=None, description="Open-access URL from Semantic Scholar"
    )
    retraction_status: Optional[bool] = Field(
        default=None, description="True if retracted, False if not, None if unknown"
    )

    title_similarity: Optional[float] = Field(default=None, description="0.0-1.0")
    citation_format: Optional[str] = Field(
        default=None,
        description="Detected bibliography format: apa, vancouver, ieee, chicago, harvard, mla, bibtex",
    )

    databases_checked: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
