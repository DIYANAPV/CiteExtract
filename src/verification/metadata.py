"""Metadata validation (L3) — field-by-field comparison + mismatch detection.

For each FOUND reference, compares extracted metadata against the DB record.
Detects metadata mismatches where the title matches a real paper but other
fields (authors, year, venue) are incorrect.
"""

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field
from unidecode import unidecode

from src.models.reference import Reference
from src.models.verdict import ExistenceResult
from src import config
from src.verification.matching import (
    PREPRINT_RE,
    author_similarity,
    compare_year,
    is_author_truncation,
    title_similarity,
)


class FieldComparison(BaseModel):
    """Result of comparing a single metadata field between ref and DB."""

    field: str
    ref_value: Optional[str] = None
    db_value: Optional[str] = None
    status: Literal["MATCH", "CLOSE_MATCH", "MISMATCH", "MISSING", "MISSING_REF", "MISSING_BOTH"]
    similarity: Optional[float] = None
    flag: Optional[str] = None


class MetadataResult(BaseModel):
    """Full metadata validation result for one reference."""

    ref_id: str
    comparisons: list[FieldComparison] = Field(default_factory=list)
    metadata_score: float = 0.0
    flags: list[str] = Field(default_factory=list)
    is_retracted: bool = False
    has_metadata_mismatch: bool = False


def validate_metadata(reference: Reference, existence: ExistenceResult) -> MetadataResult:
    """Compare all metadata fields between a reference and its matched DB record.

    Returns MetadataResult with field-level comparisons, blended detection,
    and aggregated flags.
    """
    comparisons: list[FieldComparison] = []
    flags: list[str] = []

    thresholds_cfg = config.thresholds()

    # --- Title ---
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

    # --- Authors ---
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
        # Low Jaccard but all listed authors are real — truncated "et al." list
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

    # --- Year (context-aware: preprint vs publication dates) ---
    # When title and authors both agree strongly with the DB record we
    # treat a 2-year gap as a close match (versioned arXiv repost or slow
    # journal pipeline). Without that signal, diff==2 would flag the
    # citation as broken even though the title+author evidence already
    # made it clear the paper is the same.
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

    # --- Venue ---
    ref_venue = (reference.venue or "").strip()
    db_venue = (existence.matched_venue or "").strip()
    if not ref_venue or not db_venue:
        comparisons.append(FieldComparison(
            field="venue", ref_value=ref_venue or None,
            db_value=db_venue or None, status="MISSING",
        ))
    else:
        # First, check if the reference venue matches any known alias
        # from the database (S2 publicationVenue.alternate_names,
        # OpenAlex source.alternate_titles). This handles cases like
        # "NeurIPS" vs "Advances in Neural Information Processing Systems".
        alias_matched = _check_venue_aliases(ref_venue, existence.venue_aliases)
        if alias_matched:
            comparisons.append(FieldComparison(
                field="venue", ref_value=ref_venue, db_value=db_venue,
                status="MATCH", similarity=1.0,
            ))
        else:
            # Fall back to fuzzy string matching
            venue_sim = title_similarity(ref_venue, db_venue)
            if venue_sim >= thresholds_cfg["venue_match"]:
                comparisons.append(FieldComparison(
                    field="venue", ref_value=ref_venue, db_value=db_venue,
                    status="MATCH", similarity=venue_sim,
                ))
            elif _normalized_venue_match(ref_venue, db_venue):
                # Normalized forms match (e.g. both reduce to "arxiv",
                # or one is a substring like "usenix" in "usenix security")
                comparisons.append(FieldComparison(
                    field="venue", ref_value=ref_venue, db_value=db_venue,
                    status="MATCH", similarity=venue_sim,
                    flag="venue_matched_via_normalization",
                ))
            elif _is_preprint_vs_publication(ref_venue, db_venue):
                # One side is a preprint server, the other a conference/journal.
                # This is a normal publishing pattern (arXiv → ICML, or DB
                # only has preprint version). NOT a mismatch.
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

    # --- DOI (normalized comparison) ---
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
        # DOIs differ but arXiv ID confirms it's the same paper
        # (e.g. arXiv DOI 10.48550/arxiv.XXXX vs publisher DOI 10.1109/...)
        comparisons.append(FieldComparison(
            field="doi", ref_value=ref_doi, db_value=db_doi, status="MATCH",
            flag="doi_matched_via_arxiv_id: different DOI registrations for same paper",
        ))
    elif _canonical_id_match(ref_doi, db_doi, reference, existence):
        # Canonical-ID bridge catches cross-system aliasing the simple
        # string comparison misses: ACL anthology URL ↔ ACL DOI ↔ ACL
        # ID, arXiv URL ↔ arXiv ID, and the case where the DB has a
        # secondary identifier (anthology_id) that matches the ref's
        # primary ID (DOI URL).
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

    # --- Retraction ---
    is_retracted = existence.retraction_status is True

    # --- Metadata score ---
    score = _compute_score(comparisons)

    # --- Metadata mismatch detection ---
    has_mismatch = _has_metadata_mismatch(comparisons)

    # Aggregate existence flags too
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
    """Weighted metadata score: title(0.5) + authors(0.3) + venue(0.2).

    Year excluded from score — flagged separately.
    """
    weights = {"title": 0.5, "authors": 0.3, "venue": 0.2}
    score = 0.0
    for comp in comparisons:
        w = weights.get(comp.field, 0)
        if w > 0 and comp.similarity is not None:
            score += w * comp.similarity
        elif w > 0 and comp.status == "MATCH":
            score += w  # binary match with no similarity = full weight
    return round(score, 3)


def _has_metadata_mismatch(comparisons: list[FieldComparison]) -> bool:
    """Detect whether reference metadata has field mismatches.

    Returns True if the title matches a real paper but one or more other
    fields (authors, year, venue) are genuinely wrong — indicating the
    reference may be fabricated or have incorrect metadata.

    CLOSE_MATCH (e.g. year off by 1 due to preprint-vs-publication) is NOT
    treated as a mismatch — it's a known, systematic pattern in academic
    publishing, not evidence of fabrication.
    """
    by_field = {c.field: c for c in comparisons}
    title = by_field.get("title")

    # Only trigger if title matched
    if not title or title.status != "MATCH":
        return False

    # Only hard MISMATCH counts — CLOSE_MATCH is excluded
    mismatched = [
        c for c in comparisons
        if c.status == "MISMATCH" and c.field != "title"
    ]

    return len(mismatched) > 0


def _normalize_doi(doi: str) -> str:
    """Normalize a DOI for comparison.

    Strips URL prefixes, trailing artifacts from GROBID (.full, extra path
    segments), and normalizes underscores. Underscores in DOIs are formatting
    separators (e.g. chapter numbers) — GROBID often strips them, producing
    '10.1007/3-540-36415-313' instead of '10.1007/3-540-36415-3_13'.
    Stripping all underscores from both sides makes them comparable.
    """
    if not doi:
        return ""
    # Strip URL prefix
    doi = re.sub(r'^https?://(dx\.)?doi\.org/', '', doi)
    # Strip trailing .full (SPIE/journal artifact)
    doi = re.sub(r'\.full$', '', doi)
    # Strip underscores (GROBID drops them from chapter DOIs)
    doi = doi.replace('_', '')
    # Strip trailing whitespace/periods
    doi = doi.strip('. ')
    return doi


def _extract_arxiv_id(doi: str) -> Optional[str]:
    """Extract arXiv ID from an arXiv DOI.

    arXiv DOIs follow the pattern: 10.48550/arxiv.XXXX.XXXXX
    Returns the arXiv ID (e.g. '2210.07321') or None.
    """
    if not doi:
        return None
    match = re.match(r'10\.48550/arxiv\.(\d{4}\.\d{4,5})', doi, re.IGNORECASE)
    return match.group(1) if match else None


def _arxiv_doi_bridge(
    ref_doi: str, db_doi: str,
    reference: "Reference", existence: "ExistenceResult",
) -> bool:
    """Check if mismatched DOIs refer to the same paper via arXiv ID.

    When a reference has an arXiv DOI (10.48550/arxiv.XXXX) and the DB
    returns a different publisher DOI, the papers may still be identical.
    We verify by checking if the arXiv ID extracted from the arXiv DOI
    matches the arXiv ID from the other source.

    This is a CORRECT verification (not ignoring the mismatch) — we're
    confirming identity through a different identifier.
    """
    # Extract arXiv IDs from DOIs
    ref_arxiv_from_doi = _extract_arxiv_id(ref_doi)
    db_arxiv_from_doi = _extract_arxiv_id(db_doi)

    # Collect all available arXiv IDs from both sides
    ref_arxiv = ref_arxiv_from_doi or (reference.arxiv_id or "").strip()
    db_arxiv = db_arxiv_from_doi or (getattr(existence, 'matched_arxiv_id', None) or "").strip()

    # Need at least one arXiv DOI involved AND both sides to have an arXiv ID
    if not (ref_arxiv_from_doi or db_arxiv_from_doi):
        return False  # Neither DOI is arXiv — can't bridge
    if not ref_arxiv or not db_arxiv:
        return False  # Missing arXiv ID on one side — can't verify

    # Normalize and compare
    return ref_arxiv.lower().strip() == db_arxiv.lower().strip()


def _doi_prefix_match(doi_a: str, doi_b: str) -> bool:
    """Check if one DOI is a prefix of the other.

    Handles GROBID truncation where e.g. '10.1016/j.csl.2008.04' is
    a truncated version of '10.1016/j.csl.2008.04.001'. The shorter
    must be at least 15 chars to avoid false matches on short prefixes.
    """
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
    """Detect cross-system identifier equivalence the string compare misses.

    Three patterns this catches that ``ref_doi == db_doi`` misses:

    1. **URL ↔ bare DOI** —
       ``https://doi.org/10.1234/foo`` vs ``10.1234/foo``.
    2. **arXiv DOI ↔ arXiv ID** —
       ``10.48550/arxiv.2407.21783`` vs the arXiv ID ``2407.21783``
       stored on the DB record.
    3. **ACL anthology URL ↔ ACL DOI** —
       ``aclanthology.org/Q16-1026`` (ref) vs
       ``10.18653/v1/Q16-1026`` (DB), or vs an anthology ID stored as
       a secondary identifier on the DB record.

    Each "side" contributes every identifier we have for the paper —
    DOI, arXiv ID, and any other identifier the DB exposed. If ANY
    canonical form on the ref side equals ANY canonical form on the DB
    side, we accept the match. Comparing single fields would miss the
    cross-system case where two valid identifier systems describe the
    same paper.
    """
    from src.verification.matching import canonical_id

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
        # Anthology and other secondary identifiers will surface here
        # once the DB clients are updated to populate these fields on
        # ``ExistenceResult``. Until then ``getattr`` quietly falls back
        # to ``None`` and the canonical set just won't contain them.
        getattr(existence, "anthology_id", None),
    )
    return bool(ref_ids & db_ids)


# Common academic-venue abbreviations expanded to their full form.
#
# Citations frequently abbreviate venue names to save space — "Int. J.
# Comput. Vis." stands for "International Journal of Computer Vision".
# The DB record may hold the full form, the abbreviation, or a mix, so
# the metadata layer normalizes both sides through this expansion before
# any string-based comparison. Without it, "Int. Econ." vs
# "International Economics" looks like a venue mismatch and the verdict
# falsely flags a correct citation.
#
# Keys are matched as whole words (with an optional trailing period)
# and replaced with the expanded form. Order doesn't matter — the
# regex is built once with alternation so each token is replaced in
# a single pass.
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

# Built once so per-call normalization is regex-substitution-fast.
_VENUE_ABBREV_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _VENUE_ABBREVIATIONS) + r")\.?\b",
    re.IGNORECASE,
)


def _expand_venue_abbreviations(venue: str) -> str:
    """Replace academic abbreviations with their full form.

    Operates token-by-token with whole-word boundaries so unrelated
    substrings (e.g. "international" already containing "int") aren't
    double-expanded — ``re.sub`` only fires on standalone matches.
    A trailing period is allowed and consumed by the pattern.
    """
    return _VENUE_ABBREV_RE.sub(
        lambda m: _VENUE_ABBREVIATIONS[m.group(1).lower()], venue,
    )


def _normalize_venue(venue: str) -> str:
    """Normalize a venue name for alias comparison.

    Strips common prefixes, suffixes, ordinals, publisher info,
    and generic terms to expose the core venue identity. Expands
    common abbreviations (Int. → International, J. → Journal,
    Trans. → Transactions, …) up front so abbreviated and full
    forms collapse to the same normalized string.
    """
    v = venue.lower().strip()
    v = unidecode(v)
    v = _expand_venue_abbreviations(v)
    # Strip common prefixes
    v = re.sub(r"^proceedings of (the )?([\d]+(st|nd|rd|th) )?", "", v)
    v = re.sub(r"^proc\.?\s+", "", v)
    # Strip parenthetical publisher/location info: "(Cornell University)", "(Springer)"
    v = re.sub(r"\([^)]*\)", "", v)
    # Strip trailing year like "2020" or "(2020)"
    v = re.sub(r"\s*\(?\d{4}\)?$", "", v)
    # Strip ordinal conference numbers: "27th", "58th", "2021"
    v = re.sub(r"\b\d+(st|nd|rd|th)\b", "", v)
    # Strip generic venue type words that add noise to matching
    v = re.sub(
        r"\b(conference|symposium|workshop|proceedings|annual|international"
        r"|meeting|congress|convention|transactions on|journal of)\b",
        "", v,
    )
    # Strip "e-prints" (arXiv artifact)
    v = re.sub(r"\be[\s-]?prints\b", "", v)
    # Remove punctuation
    v = re.sub(r"[^\w\s]", " ", v)
    v = re.sub(r"\s+", " ", v).strip()
    return v


def _normalized_venue_match(ref_venue: str, db_venue: str) -> bool:
    """Check if two venue names match after aggressive normalization.

    Strips generic terms (conference, symposium, etc.) and checks if the
    normalized core names are equal or one is a substring of the other.
    Requires the shorter normalized form to be at least 3 chars to avoid
    false matches on very short strings.
    """
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
    # Token-level near-match for normalized forms — catches cases like
    # "national academy science" vs "national academy of sciences"
    # where neither is a substring of the other but token overlap is
    # near-complete. Threshold deliberately strict (0.85) so we don't
    # paper over genuine venue mismatches.
    return title_similarity(norm_ref, norm_db) >= 0.85




def _is_preprint_vs_publication(ref_venue: str, db_venue: str) -> bool:
    """Detect when one venue is a preprint server and the other is not.

    This is a normal publishing pattern: authors post on arXiv first, then
    the paper appears at a conference/journal. The reference may cite either
    version while the DB has the other. This should be CLOSE_MATCH, not
    MISMATCH.

    Returns True when exactly one of the two venues is a preprint server.
    Returns False if both are preprints or neither is (genuine mismatch).
    """
    ref_is_preprint = bool(PREPRINT_RE.search(ref_venue))
    db_is_preprint = bool(PREPRINT_RE.search(db_venue))
    # Exactly one side is a preprint → preprint-vs-publication pattern
    return ref_is_preprint != db_is_preprint


def _check_venue_aliases(ref_venue: str, aliases: list[str]) -> bool:
    """Check if the reference venue matches any known alias.

    Uses normalized comparison so 'NeurIPS' matches
    'Advances in Neural Information Processing Systems'.
    """
    if not aliases:
        return False
    norm_ref = _normalize_venue(ref_venue)
    if not norm_ref:
        return False
    for alias in aliases:
        norm_alias = _normalize_venue(alias)
        if not norm_alias:
            continue
        # Exact match after normalization
        if norm_ref == norm_alias:
            return True
        # Check if one is a substring of the other (handles abbreviations
        # that are contained in the full name, like "AAAI" in
        # "AAAI Conference on Artificial Intelligence")
        if len(norm_ref) >= 3 and len(norm_alias) >= 3:
            if norm_ref in norm_alias or norm_alias in norm_ref:
                return True
    return False
