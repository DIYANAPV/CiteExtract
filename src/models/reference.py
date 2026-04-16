"""Reference data model — represents a single bibliographic reference."""

from typing import Optional
from pydantic import BaseModel, Field


class Reference(BaseModel):
    """A bibliographic reference extracted from a paper.

    All parsers normalize their output to this structure.
    """

    ref_id: str = Field(description="Unique ID within the paper, e.g. '1', 'smith2020'")
    title: Optional[str] = Field(default=None, description="Paper title")
    authors: list[str] = Field(default_factory=list, description="Author names (normalized)")
    year: Optional[int] = Field(default=None, description="Publication year")
    venue: Optional[str] = Field(default=None, description="Journal or conference name")
    doi: Optional[str] = Field(default=None, description="DOI if available")
    arxiv_id: Optional[str] = Field(default=None, description="arXiv ID if available")
    url: Optional[str] = Field(default=None, description="URL if available (blog posts, web sources)")
    raw_text: str = Field(default="", description="Original reference string from the paper")
    source_format: str = Field(description="Parser that produced this: grobid, bibtex, latex, text")
    citation_format: Optional[str] = Field(
        default=None,
        description="Detected bibliography format: apa, vancouver, ieee, chicago, harvard, mla, bibtex",
    )
