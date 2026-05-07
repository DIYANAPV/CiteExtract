
from pydantic import BaseModel, Field

from citeextract.models.reference import Reference
from citeextract.models.citation import Citation


class ParsedPaper(BaseModel):

    references: list[Reference] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    has_body_text: bool = Field(description="False for BibTeX-only input")
    body_text: str = Field(default="", description="Full body text of the paper (for current-paper context retrieval)")
    input_format: str = Field(description="'pdf', 'latex', 'bibtex', 'text'")
    metadata: dict = Field(
        default_factory=dict,
        description="Paper-level metadata: title, authors if extractable",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Parsing issues to surface in the report",
    )
