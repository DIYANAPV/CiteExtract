
from typing import Optional

from pydantic import BaseModel, Field

from citeextract.models.comprehension import ClaimVerdict
from citeextract.models.verdict import ExistenceResult
from citeextract.verification.metadata import MetadataResult


class CitationVerdict(BaseModel):

    ref_id: str
    verdict: str = Field(description="FABRICATED, VALID, UNVERIFIABLE (metadata-driven top-level)")
    mode: str = Field(description="'quick' (rule-based) or 'agentic'")
    action: str = Field(description="no_action, verify_claim, remove_citation")
    explanation: str = Field(description="Human-readable summary of why this verdict was given")
    flags: list[str] = Field(default_factory=list)

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

    per_sentence_claim_verdicts: dict[str, ClaimVerdict] = Field(default_factory=dict)

    existence: Optional[ExistenceResult] = None
    metadata: Optional[MetadataResult] = None

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
    claim_verdict: Optional[str] = None,
) -> str:
    return metadata_verdict or "UNVERIFIABLE"


def classify_quick(
    existence: ExistenceResult,
    metadata: Optional[MetadataResult],
) -> CitationVerdict:
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

    if existence.status == "NOT_FOUND":
        errored = list(existence.errored_databases)
        errored_sources = {e.split(":")[0] for e in errored}
        responded = [db for db in existence.databases_checked if db not in errored_sources]
        if len(responded) >= 2:
            tail = (
                f" ({len(errored)} other source(s) errored: {'; '.join(errored)})"
                if errored else ""
            )
            return _build(
                verdict="FABRICATED",
                action="remove_citation",
                explanation=(
                    f"Reference not found in any database. "
                    f"Responded: {', '.join(responded)}.{tail}"
                ),
                flags=all_flags,
            )
        else:
            extra_flag = "transient_api_failures" if errored else "insufficient_database_coverage"
            errored_note = (
                f" {len(errored)} source(s) errored ({'; '.join(errored)})."
                if errored else ""
            )
            return _build(
                verdict="UNVERIFIABLE",
                action="no_action",
                explanation=(
                    f"Could not verify this reference — only "
                    f"{len(responded)} database(s) responded "
                    f"({', '.join(responded) if responded else 'none'}).{errored_note} "
                    f"This may be a real paper missed due to API errors or rate limits."
                ),
                flags=all_flags + [extra_flag],
            )

    if metadata and metadata.is_retracted:
        return _build(
            verdict="FABRICATED",
            action="remove_citation",
            explanation="This paper has been retracted. Check the retraction notice before citing.",
            flags=all_flags + ["retracted"],
        )

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
    title = existence.matched_title
    authors = existence.matched_authors
    year = existence.matched_year
    venue = existence.matched_venue
    doi = existence.matched_doi

    if not title or not authors:
        return None, None

    apa_authors = _format_apa_authors(authors)
    year_str = f"({year})" if year else "(n.d.)"
    apa_parts = [f"{apa_authors} {year_str}. {title}."]
    if venue:
        apa_parts.append(f" *{venue}*.")
    if doi:
        apa_parts.append(f" https://doi.org/{doi}")
    apa = "".join(apa_parts)

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
    if not authors:
        return ""

    formatted = []
    for name in authors:
        if "," in name:
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
    return ", ".join(formatted[:19]) + f", ... {formatted[-1]}"
