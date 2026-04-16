"""Classification (L5) — decision tree combining L2 + L3 results.

Maps existence + metadata results into final verdicts:
FABRICATED, VALID, UNVERIFIABLE.

Claim verification (MISREPRESENTED) is handled separately via
passage-based analysis in the comprehension pipeline.
"""

from typing import Optional

from pydantic import BaseModel, Field

from src.models.verdict import ExistenceResult
from src.verification.metadata import MetadataResult


class CitationVerdict(BaseModel):
    """Final verdict for a single reference."""

    ref_id: str
    verdict: str = Field(description="FABRICATED, MISREPRESENTED, VALID, UNVERIFIABLE")
    mode: str = Field(description="'quick', 'standard', or 'agentic'")
    action: str = Field(description="no_action, verify_claim, remove_citation")
    explanation: str = Field(description="Human-readable summary of why this verdict was given")
    flags: list[str] = Field(default_factory=list)

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

    # 1. NOT_FOUND → FABRICATED (if enough databases checked) or UNVERIFIABLE
    if existence.status == "NOT_FOUND":
        checked = existence.databases_checked
        if len(checked) >= 2:
            return CitationVerdict(
                ref_id=ref_id,
                verdict="FABRICATED",
                mode="quick",
                action="remove_citation",
                explanation=(
                    f"Reference not found in any database. "
                    f"Checked: {', '.join(checked)}."
                ),
                flags=all_flags,
                existence=existence,
                metadata=metadata,
            )
        else:
            return CitationVerdict(
                ref_id=ref_id,
                verdict="UNVERIFIABLE",
                mode="quick",
                action="no_action",
                explanation=(
                    f"Could not verify this reference — only "
                    f"{len(checked)} database(s) responded "
                    f"({', '.join(checked) if checked else 'none'}). "
                    f"This may be a real paper that was missed due to API "
                    f"errors or rate limits."
                ),
                flags=all_flags + ["insufficient_database_coverage"],
                existence=existence,
                metadata=metadata,
            )

    # 2. Retracted → FABRICATED
    if metadata and metadata.is_retracted:
        return CitationVerdict(
            ref_id=ref_id,
            verdict="FABRICATED",
            mode="quick",
            action="remove_citation",
            explanation="This paper has been retracted. Check the retraction notice before citing.",
            flags=all_flags + ["retracted"],
            existence=existence,
            metadata=metadata,
        )

    # 3. Metadata mismatch (blended or misattributed) → FABRICATED
    if metadata and metadata.has_metadata_mismatch:
        mismatched = [c for c in metadata.comparisons if c.status == "MISMATCH" and c.field != "title"]
        fields = ", ".join(c.field for c in mismatched)
        return CitationVerdict(
            ref_id=ref_id,
            verdict="FABRICATED",
            mode="quick",
            action="remove_citation",
            explanation=(
                f"Title matches a real paper but {fields} "
                f"{'does' if len(mismatched) == 1 else 'do'} not match "
                f"the database record. The reference metadata appears "
                f"incorrect or fabricated."
            ),
            flags=all_flags,
            existence=existence,
            metadata=metadata,
        )

    # 4. All clear — generate corrected citation from DB metadata
    apa, bibtex = _format_corrected_citation(existence)
    return CitationVerdict(
        ref_id=ref_id,
        verdict="VALID",
        mode="quick",
        action="no_action",
        explanation="Reference exists and metadata matches.",
        flags=all_flags,
        existence=existence,
        metadata=metadata,
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
