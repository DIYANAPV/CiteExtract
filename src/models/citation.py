"""Citation context model — represents an in-text citation with surrounding context."""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """An in-text citation occurrence with its surrounding context.

    Links back to a Reference via ref_id.
    A single Reference may have multiple Citation objects if cited in multiple places.
    Context window is paragraph-bounded (default: 2 sentences before, 1 after).
    """

    ref_id: str = Field(description="Links to Reference.ref_id")
    citing_sentence: str = Field(description="The sentence containing the citation marker")
    context_before: str = Field(default="", description="Sentences above the citing sentence")
    context_after: str = Field(default="", description="Sentences below the citing sentence")
    section: Optional[str] = Field(default=None, description="Section heading if detectable")
    marker: str = Field(description="Citation marker as it appears, e.g. '[1]', '(Smith, 2020)'")
    position: int = Field(default=0, description="Character offset in full text")
    link_confidence: Literal["high", "medium", "low"] = Field(
        default="high",
        description="Confidence that marker↔reference linking is correct. Internal signal; "
        "only 'low' is surfaced in UI and only when verdict is 'supported'.",
    )
    grobid_ref_id: Optional[str] = Field(
        default=None, description="GROBID <ref target> vote (debug only)"
    )
    rule_ref_id: Optional[str] = Field(
        default=None, description="Deterministic rule-check vote (debug only)"
    )
    llm_ref_id: Optional[str] = Field(
        default=None, description="LLM matched_markers vote (debug only)"
    )
