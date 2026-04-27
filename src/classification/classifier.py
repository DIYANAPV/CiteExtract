"""Classification (L5) — decision tree combining L2 + L3 results.

The top-level verdict is metadata-driven only: FABRICATED, VALID, or
UNVERIFIABLE. The claim dimension (does the cited paper actually support
the citing sentence?) is reported separately as ``claim_verdict`` —
SUPPORTED / CONTRADICTS / NEUTRAL / UNVERIFIABLE — so the user reads it
as a second, orthogonal signal on the verdict card rather than seeing it
collapsed into the top-level rollup.
"""

from typing import Optional

from pydantic import BaseModel, Field

from src.models.comprehension import ClaimVerdict
from src.models.verdict import ExistenceResult
from src.verification.metadata import MetadataResult


class CitationVerdict(BaseModel):
    """Final verdict for a single reference.

    Carries two independent verdict dimensions — metadata (does the cited
    paper exist and match the reference metadata?) and claim (does the
    cited paper actually support the citing sentence?). The top-level
    ``verdict`` is a roll-up of the two for back-compat with reports.
    """

    ref_id: str
    verdict: str = Field(description="FABRICATED, VALID, UNVERIFIABLE (metadata-driven top-level)")
    mode: str = Field(description="'quick' (rule-based) or 'agentic'")
    action: str = Field(description="no_action, verify_claim, remove_citation")
    explanation: str = Field(description="Human-readable summary of why this verdict was given")
    flags: list[str] = Field(default_factory=list)

    # Two-dimension verdicts (agentic mode populates both independently)
    metadata_verdict: Optional[str] = Field(
        default=None,
        description="VALID, FABRICATED, UNVERIFIABLE (metadata dimension)",
    )
    metadata_flags: list[str] = Field(default_factory=list)
    metadata_explanation: Optional[str] = None
    claim_verdict: Optional[str] = Field(
        default=None,
        description="SUPPORTED, CONTRADICTS, NEUTRAL, UNVERIFIABLE, or None if not applicable",
    )
    claim_flags: list[str] = Field(default_factory=list)
    claim_explanation: Optional[str] = None

    # Per-citing-sentence claim agent output, keyed by literal citing sentence.
    # Empty in quick mode and in agentic when the route did not run the claim
    # agent. Surfaces individual SUPPORTS/CONTRADICTS/NEUTRAL judgments and
    # evidence quotes that the claim_verdict roll-up necessarily flattens.
    per_sentence_claim_verdicts: dict[str, ClaimVerdict] = Field(default_factory=dict)

    # Evidence trail
    existence: Optional[ExistenceResult] = None
    metadata: Optional[MetadataResult] = None

    # Corrected citation from database metadata (when match found)
    corrected_apa: Optional[str] = Field(
        default=None,
        description="APA 7th edition citation generated from database metadata",
    )
    corrected_bibtex: Optional[str] = Field(
        default=None,
        description="BibTeX entry generated from database metadata",
    )


def rollup_verdict(
    metadata_verdict: Optional[str],
    claim_verdict: Optional[str] = None,  # kept for back-compat call sites
) -> str:
    """Top-level verdict is metadata-driven only.

    The claim dimension is shown separately on the verdict card so users
    can read it as an independent signal — collapsing CONTRADICTS into
    the top-level was conflating a "wrong paper cited" finding with a
    "right paper, mismatched claim" finding, two materially different
    actions for the user.

    Returns the metadata verdict, or UNVERIFIABLE if metadata never ran.
    """
    return metadata_verdict or "UNVERIFIABLE"


def classify_quick(
    existence: ExistenceResult,
    metadata: Optional[MetadataResult],
) -> CitationVerdict:
    """Classification based on existence + metadata (L2 + L3).

    Priority order:
    1. FABRICATED (not found, retracted, or metadata from wrong paper)
    2. VALID (all clear)
    3. UNVERIFIABLE (insufficient database coverage)
    """
    ref_id = existence.ref_id
    all_flags = list(existence.flags)
    if metadata:
        all_flags.extend(metadata.flags)
    all_flags = list(dict.fromkeys(all_flags))

    def _build(
        verdict: str,
        action: str,
        explanation: str,
        flags: list[str],
        **extra,
    ) -> CitationVerdict:
        # Quick mode only evaluates metadata; mirror the fields so both
        # quick and agentic verdicts expose the same two-dimension shape.
        return CitationVerdict(
            ref_id=ref_id,
            verdict=verdict,
            mode="quick",
            action=action,
            explanation=explanation,
            flags=flags,
            metadata_verdict=verdict,
            metadata_flags=flags,
            metadata_explanation=explanation,
            claim_verdict=None,
            claim_flags=[],
            claim_explanation=None,
            existence=existence,
            metadata=metadata,
            **extra,
        )

    # 1. NOT_FOUND → FABRICATED (if enough databases checked) or UNVERIFIABLE
    if existence.status == "NOT_FOUND":
        checked = existence.databases_checked
        if len(checked) >= 2:
            return _build(
                verdict="FABRICATED",
                action="remove_citation",
                explanation=(
                    f"Reference not found in any database. "
                    f"Checked: {', '.join(checked)}."
                ),
                flags=all_flags,
            )
        else:
            return _build(
                verdict="UNVERIFIABLE",
                action="no_action",
                explanation=(
                    f"Could not verify this reference — only "
                    f"{len(checked)} database(s) responded "
                    f"({', '.join(checked) if checked else 'none'}). "
                    f"This may be a real paper that was missed due to API "
                    f"errors or rate limits."
                ),
                flags=all_flags + ["insufficient_database_coverage"],
            )

    # 2. Retracted → FABRICATED
    if metadata and metadata.is_retracted:
        return _build(
            verdict="FABRICATED",
            action="remove_citation",
            explanation="This paper has been retracted. Check the retraction notice before citing.",
            flags=all_flags + ["retracted"],
        )

    # 3. Metadata mismatch (blended or misattributed) → FABRICATED
    if metadata and metadata.has_metadata_mismatch:
        mismatched = [c for c in metadata.comparisons if c.status == "MISMATCH" and c.field != "title"]
        fields = ", ".join(c.field for c in mismatched)
        return _build(
            verdict="FABRICATED",
            action="remove_citation",
            explanation=(
                f"Title matches a real paper but {fields} "
                f"{'does' if len(mismatched) == 1 else 'do'} not match "
                f"the database record. The reference metadata appears "
                f"incorrect or fabricated."
            ),
            flags=all_flags,
        )

    # 4. All clear — generate corrected citation from DB metadata
    apa, bibtex = _format_corrected_citation(existence)
    return _build(
        verdict="VALID",
        action="no_action",
        explanation="Reference exists and metadata matches.",
        flags=all_flags,
        corrected_apa=apa,
        corrected_bibtex=bibtex,
    )


def _format_corrected_citation(
    existence: ExistenceResult,
) -> tuple[Optional[str], Optional[str]]:
    """Generate corrected APA and BibTeX from database metadata.

    Uses the authoritative DB record, not the user's reference. This means
    the output has correct spelling, complete author lists, and canonical
    venue names.

    Returns:
        (apa_string, bibtex_string) — either may be None if insufficient data.
    """
    title = existence.matched_title
    authors = existence.matched_authors
    year = existence.matched_year
    venue = existence.matched_venue
    doi = existence.matched_doi

    if not title or not authors:
        return None, None

    # --- APA 7th edition ---
    # Format: Author, A. A., & Author, B. B. (Year). Title. Venue. DOI
    apa_authors = _format_apa_authors(authors)
    year_str = f"({year})" if year else "(n.d.)"
    apa_parts = [f"{apa_authors} {year_str}. {title}."]
    if venue:
        apa_parts.append(f" *{venue}*.")
    if doi:
        apa_parts.append(f" https://doi.org/{doi}")
    apa = "".join(apa_parts)

    # --- BibTeX ---
    # Generate a cite key from first author surname + year
    first_surname = authors[0].split(",")[0].split()[-1] if authors else "unknown"
    key = f"{first_surname.lower()}{year or 'nd'}"
    bib_authors = " and ".join(authors)
    bibtex_lines = [
        f"@article{{{key},",
        f"  title = {{{title}}},",
        f"  author = {{{bib_authors}}},",
    ]
    if year:
        bibtex_lines.append(f"  year = {{{year}}},")
    if venue:
        bibtex_lines.append(f"  journal = {{{venue}}},")
    if doi:
        bibtex_lines.append(f"  doi = {{{doi}}},")
    bibtex_lines.append("}")
    bibtex = "\n".join(bibtex_lines)

    return apa, bibtex


def _format_apa_authors(authors: list[str]) -> str:
    """Format author list in APA 7th edition style.

    Rules:
    - 1-2 authors: list all, joined by "&"
    - 3-20 authors: list all, last preceded by "&"
    - 21+ authors: first 19, "...", last author
    """
    if not authors:
        return ""

    formatted = []
    for name in authors:
        # Try to convert "First Last" → "Last, F."
        if "," in name:
            # Already "Last, First" format
            parts = name.split(",", 1)
            surname = parts[0].strip()
            given = parts[1].strip()
            initials = ". ".join(w[0].upper() for w in given.split() if w) + "."
            formatted.append(f"{surname}, {initials}")
        else:
            parts = name.split()
            if len(parts) >= 2:
                surname = parts[-1]
                initials = ". ".join(p[0].upper() for p in parts[:-1]) + "."
                formatted.append(f"{surname}, {initials}")
            else:
                formatted.append(name)

    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]}, & {formatted[1]}"
    if len(formatted) <= 20:
        return ", ".join(formatted[:-1]) + f", & {formatted[-1]}"
    # 21+ authors
    return ", ".join(formatted[:19]) + f", ... {formatted[-1]}"
