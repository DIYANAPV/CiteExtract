
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field
from unidecode import unidecode

from citeextract.models.reference import Reference
from citeextract.models.verdict import ExistenceResult
from citeextract import config
from citeextract.verification.matching import (
    PREPRINT_RE,
    author_similarity,
    compare_year,
    is_author_truncation,
    title_similarity,
)


class FieldComparison(BaseModel):

    field: str
    ref_value: Optional[str] = None
    db_value: Optional[str] = None
    status: Literal["MATCH", "CLOSE_MATCH", "MISMATCH", "MISSING", "MISSING_REF", "MISSING_BOTH"]
    similarity: Optional[float] = None
    flag: Optional[str] = None


class MetadataResult(BaseModel):

    ref_id: str
    comparisons: list[FieldComparison] = Field(default_factory=list)
    metadata_score: float = 0.0
    flags: list[str] = Field(default_factory=list)
    is_retracted: bool = False
    has_metadata_mismatch: bool = False


def validate_metadata(reference: Reference, existence: ExistenceResult) -> MetadataResult:
    comparisons: list[FieldComparison] = []
    flags: list[str] = []

    thresholds_cfg = config.thresholds()

    title_sim = title_similarity(reference.title or "", existence.matched_title or "")
    if not reference.title or not existence.matched_title:
        comparisons.append(FieldComparison(
            field="title", ref_value=reference.title,
            db_value=existence.matched_title, status="MISSING",
        ))
    elif title_sim >= thresholds_cfg["title_match"]:
        flag = None
        if title_sim < 1.0:
            flag = (
                f"Title not exact match (similarity={title_sim:.2f}). "
                f"Found: '{existence.matched_title[:80]}'"
            )
            flags.append(flag)
        comparisons.append(FieldComparison(
            field="title", ref_value=reference.title,
            db_value=existence.matched_title, status="MATCH",
            similarity=title_sim, flag=flag,
        ))
    else:
        comparisons.append(FieldComparison(
            field="title", ref_value=reference.title,
            db_value=existence.matched_title, status="MISMATCH",
            similarity=title_sim,
        ))

    author_sim = author_similarity(reference.authors, existence.matched_authors)
    if not reference.authors or not existence.matched_authors:
        comparisons.append(FieldComparison(
            field="authors",
            ref_value=", ".join(reference.authors) if reference.authors else None,
            db_value=", ".join(existence.matched_authors) if existence.matched_authors else None,
            status="MISSING",
        ))
    elif author_sim >= thresholds_cfg["author_match"]:
        flag = None
        if author_sim < 1.0:
            flag = (
                f"Authors partial match (similarity={author_sim:.2f}). "
                f"Ref: {', '.join(reference.authors[:3])}; "
                f"DB: {', '.join(existence.matched_authors[:3])}"
            )
            flags.append(flag)
        comparisons.append(FieldComparison(
            field="authors",
            ref_value=", ".join(reference.authors),
            db_value=", ".join(existence.matched_authors),
            status="MATCH", similarity=author_sim, flag=flag,
        ))
    elif is_author_truncation(reference.authors, existence.matched_authors):
        flag = (
            f"Author list truncated: ref lists {len(reference.authors)} of "
            f"{len(existence.matched_authors)} authors (Jaccard={author_sim:.2f}, "
            f"but listed authors are a subset of the real author list)"
        )
        flags.append(flag)
        comparisons.append(FieldComparison(
            field="authors",
            ref_value=", ".join(reference.authors),
            db_value=", ".join(existence.matched_authors),
            status="CLOSE_MATCH", similarity=author_sim, flag=flag,
        ))
    else:
        flag = (
            f"Authors mismatch (similarity={author_sim:.2f}). "
            f"Ref: {', '.join(reference.authors[:3])}; "
            f"DB: {', '.join(existence.matched_authors[:3])}"
        )
        flags.append(flag)
        comparisons.append(FieldComparison(
            field="authors",
            ref_value=", ".join(reference.authors),
            db_value=", ".join(existence.matched_authors),
            status="MISMATCH", similarity=author_sim, flag=flag,
        ))

    strong_year_match = (
        title_sim >= thresholds_cfg["title_match"]
        and reference.authors and existence.matched_authors
        and author_sim >= thresholds_cfg["author_match"]
    )
    year_cmp = compare_year(
        reference.year, existence.matched_year,
        strong_match=strong_year_match,
    )
    if reference.year is None or existence.matched_year is None:
        comparisons.append(FieldComparison(
            field="year",
            ref_value=str(reference.year) if reference.year else None,
            db_value=str(existence.matched_year) if existence.matched_year else None,
            status="MISSING",
        ))
    elif year_cmp["match"] and not year_cmp["close_match"]:
        comparisons.append(FieldComparison(
            field="year",
            ref_value=str(reference.year),
            db_value=str(existence.matched_year),
            status="MATCH",
        ))
    elif year_cmp["close_match"]:
        flags.append(year_cmp["flag"])
        comparisons.append(FieldComparison(
            field="year",
            ref_value=str(reference.year),
            db_value=str(existence.matched_year),
            status="CLOSE_MATCH", flag=year_cmp["flag"],
        ))
    else:
        flags.append(year_cmp["flag"])
        comparisons.append(FieldComparison(
            field="year",
            ref_value=str(reference.year),
            db_value=str(existence.matched_year),
            status="MISMATCH", flag=year_cmp["flag"],
        ))

    ref_venue = (reference.venue or "").strip()
    db_venue = (existence.matched_venue or "").strip()
    if not ref_venue or not db_venue:
        comparisons.append(FieldComparison(
            field="venue", ref_value=ref_venue or None,
            db_value=db_venue or None, status="MISSING",
        ))
    else:
        alias_matched = _check_venue_aliases(ref_venue, existence.venue_aliases)
        if alias_matched:
            comparisons.append(FieldComparison(
                field="venue", ref_value=ref_venue, db_value=db_venue,
                status="MATCH", similarity=1.0,
            ))
        else:
            venue_sim = title_similarity(ref_venue, db_venue)
            if venue_sim >= thresholds_cfg["venue_match"]:
                comparisons.append(FieldComparison(
                    field="venue", ref_value=ref_venue, db_value=db_venue,
                    status="MATCH", similarity=venue_sim,
                ))
            elif _normalized_venue_match(ref_venue, db_venue):
                comparisons.append(FieldComparison(
                    field="venue", ref_value=ref_venue, db_value=db_venue,
                    status="MATCH", similarity=venue_sim,
                    flag="venue_matched_via_normalization",
                ))
            elif _is_preprint_vs_publication(ref_venue, db_venue):
                flag = (
                    f"venue_preprint_vs_publication: Ref: '{ref_venue}'; "
                    f"DB: '{db_venue}' (likely same paper at different stages)"
                )
                flags.append(flag)
                comparisons.append(FieldComparison(
                    field="venue", ref_value=ref_venue, db_value=db_venue,
                    status="CLOSE_MATCH", similarity=venue_sim, flag=flag,
                ))
            else:
                flag = f"Venue mismatch. Ref: '{ref_venue}'; DB: '{db_venue}'"
                flags.append(flag)
                comparisons.append(FieldComparison(
                    field="venue", ref_value=ref_venue, db_value=db_venue,
                    status="MISMATCH", similarity=venue_sim, flag=flag,
                ))

    ref_doi = _normalize_doi((reference.doi or "").strip().lower())
    db_doi = _normalize_doi((existence.matched_doi or "").strip().lower())
    if not ref_doi and not db_doi:
        comparisons.append(FieldComparison(
            field="doi", ref_value=None,
            db_value=None, status="MISSING_BOTH",
        ))
    elif not ref_doi and db_doi:
        comparisons.append(FieldComparison(
            field="doi", ref_value=None,
            db_value=db_doi, status="MISSING_REF",
        ))
    elif ref_doi and not db_doi:
        comparisons.append(FieldComparison(
            field="doi", ref_value=ref_doi,
            db_value=None, status="MISSING",
        ))
    elif ref_doi == db_doi or _doi_prefix_match(ref_doi, db_doi):
        comparisons.append(FieldComparison(
            field="doi", ref_value=ref_doi, db_value=db_doi, status="MATCH",
        ))
    elif _arxiv_doi_bridge(ref_doi, db_doi, reference, existence):
        comparisons.append(FieldComparison(
            field="doi", ref_value=ref_doi, db_value=db_doi, status="MATCH",
            flag="doi_matched_via_arxiv_id: different DOI registrations for same paper",
        ))
    elif _canonical_id_match(ref_doi, db_doi, reference, existence):
        comparisons.append(FieldComparison(
            field="doi", ref_value=ref_doi, db_value=db_doi, status="MATCH",
            flag="doi_matched_via_canonical_id: same paper, different identifier system",
        ))
    else:
        flag = f"DOI mismatch. Ref: {ref_doi}; DB: {db_doi}"
        flags.append(flag)
        comparisons.append(FieldComparison(
            field="doi", ref_value=ref_doi, db_value=db_doi,
            status="MISMATCH", flag=flag,
        ))

    is_retracted = existence.retraction_status is True

    score = _compute_score(comparisons)

    has_mismatch = _has_metadata_mismatch(comparisons)

    all_flags = list(existence.flags) + flags

    return MetadataResult(
        ref_id=reference.ref_id,
        comparisons=comparisons,
        metadata_score=score,
        flags=all_flags,
        is_retracted=is_retracted,
        has_metadata_mismatch=has_mismatch,
    )


def _compute_score(comparisons: list[FieldComparison]) -> float:
    weights = {"title": 0.5, "authors": 0.3, "venue": 0.2}
    score = 0.0
    for comp in comparisons:
        w = weights.get(comp.field, 0)
        if w > 0 and comp.similarity is not None:
            score += w * comp.similarity
        elif w > 0 and comp.status == "MATCH":
            score += w
    return round(score, 3)


def _has_metadata_mismatch(comparisons: list[FieldComparison]) -> bool:
    by_field = {c.field: c for c in comparisons}
    title = by_field.get("title")

    if not title or title.status != "MATCH":
        return False

    mismatched = [
        c for c in comparisons
        if c.status == "MISMATCH" and c.field != "title"
    ]

    return len(mismatched) > 0


def _normalize_doi(doi: str) -> str:
    if not doi:
        return ""
    doi = re.sub(r'^https?://(dx\.)?doi\.org/', '', doi)
    doi = re.sub(r'\.full$', '', doi)
    doi = doi.replace('_', '')
    doi = doi.strip('. ')
    return doi


def _extract_arxiv_id(doi: str) -> Optional[str]:
    if not doi:
        return None
    match = re.match(r'10\.48550/arxiv\.(\d{4}\.\d{4,5})', doi, re.IGNORECASE)
    return match.group(1) if match else None


def _arxiv_doi_bridge(
    ref_doi: str, db_doi: str,
    reference: "Reference", existence: "ExistenceResult",
) -> bool:
    ref_arxiv_from_doi = _extract_arxiv_id(ref_doi)
    db_arxiv_from_doi = _extract_arxiv_id(db_doi)

    ref_arxiv = ref_arxiv_from_doi or (reference.arxiv_id or "").strip()
    db_arxiv = db_arxiv_from_doi or (getattr(existence, 'matched_arxiv_id', None) or "").strip()

    if not (ref_arxiv_from_doi or db_arxiv_from_doi):
        return False
    if not ref_arxiv or not db_arxiv:
        return False

    return ref_arxiv.lower().strip() == db_arxiv.lower().strip()


def _doi_prefix_match(doi_a: str, doi_b: str) -> bool:
    if not doi_a or not doi_b:
        return False
    shorter = min(doi_a, doi_b, key=len)
    longer = max(doi_a, doi_b, key=len)
    if len(shorter) < 15:
        return False
    return longer.startswith(shorter)


def _canonical_id_match(
    ref_doi: str, db_doi: str,
    reference: "Reference", existence: "ExistenceResult",
) -> bool:
    from citeextract.verification.matching import canonical_id

    def _all_canonicals(*raws):
        seen: set = set()
        for r in raws:
            if not r:
                continue
            cid = canonical_id(r)
            if cid is not None:
                seen.add(cid)
        return seen

    ref_ids = _all_canonicals(ref_doi, getattr(reference, "arxiv_id", None))
    db_ids = _all_canonicals(
        db_doi,
        getattr(existence, "matched_arxiv_id", None),
        getattr(existence, "anthology_id", None),
    )
    return bool(ref_ids & db_ids)


_VENUE_ABBREVIATIONS = {
    "int": "international",
    "intl": "international",
    "natl": "national",
    "j": "journal",
    "trans": "transactions",
    "proc": "proceedings",
    "conf": "conference",
    "symp": "symposium",
    "rev": "review",
    "bull": "bulletin",
    "mag": "magazine",
    "ann": "annual",
    "annu": "annual",
    "econ": "economics",
    "comp": "computing",
    "comput": "computing",
    "sci": "science",
    "tech": "technology",
    "eng": "engineering",
    "engr": "engineering",
    "math": "mathematics",
    "stat": "statistics",
    "phys": "physics",
    "chem": "chemistry",
    "biol": "biology",
    "med": "medicine",
    "soc": "society",
    "acad": "academy",
    "assoc": "association",
    "appl": "applied",
    "anal": "analysis",
    "vis": "vision",
    "intell": "intelligence",
    "mach": "machine",
    "lang": "language",
    "ling": "linguistics",
    "lett": "letters",
    "res": "research",
    "syst": "systems",
    "softw": "software",
    "inf": "information",
    "commun": "communications",
    "netw": "networks",
}

_VENUE_ABBREV_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _VENUE_ABBREVIATIONS) + r")\.?\b",
    re.IGNORECASE,
)


def _expand_venue_abbreviations(venue: str) -> str:
    return _VENUE_ABBREV_RE.sub(
        lambda m: _VENUE_ABBREVIATIONS[m.group(1).lower()], venue,
    )


def _normalize_venue(venue: str) -> str:
    v = venue.lower().strip()
    v = unidecode(v)
    v = _expand_venue_abbreviations(v)
    v = re.sub(r"^proceedings of (the )?([\d]+(st|nd|rd|th) )?", "", v)
    v = re.sub(r"^proc\.?\s+", "", v)
    v = re.sub(r"\([^)]*\)", "", v)
    v = re.sub(r"\s*\(?\d{4}\)?$", "", v)
    v = re.sub(r"\b\d+(st|nd|rd|th)\b", "", v)
    v = re.sub(
        r"\b(conference|symposium|workshop|proceedings|annual|international"
        r"|meeting|congress|convention|transactions on|journal of)\b",
        "", v,
    )
    v = re.sub(r"\be[\s-]?prints\b", "", v)
    v = re.sub(r"[^\w\s]", " ", v)
    v = re.sub(r"\s+", " ", v).strip()
    return v


def _normalized_venue_match(ref_venue: str, db_venue: str) -> bool:
    norm_ref = _normalize_venue(ref_venue)
    norm_db = _normalize_venue(db_venue)
    if not norm_ref or not norm_db:
        return False
    shorter = min(norm_ref, norm_db, key=len)
    if len(shorter) < 3:
        return False
    if norm_ref == norm_db:
        return True
    if norm_ref in norm_db or norm_db in norm_ref:
        return True
    return title_similarity(norm_ref, norm_db) >= 0.85


def _is_preprint_vs_publication(ref_venue: str, db_venue: str) -> bool:
    ref_is_preprint = bool(PREPRINT_RE.search(ref_venue))
    db_is_preprint = bool(PREPRINT_RE.search(db_venue))
    return ref_is_preprint != db_is_preprint


def _check_venue_aliases(ref_venue: str, aliases: list[str]) -> bool:
    if not aliases:
        return False
    norm_ref = _normalize_venue(ref_venue)
    if not norm_ref:
        return False
    for alias in aliases:
        norm_alias = _normalize_venue(alias)
        if not norm_alias:
            continue
        if norm_ref == norm_alias:
            return True
        if len(norm_ref) >= 3 and len(norm_alias) >= 3:
            if norm_ref in norm_alias or norm_alias in norm_ref:
                return True
    return False
