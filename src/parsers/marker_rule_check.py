"""Deterministic marker↔reference rule check.

Used as the second of three votes for citation-marker linking, alongside
GROBID's ``<ref target>`` and the LLM's ``matched_markers``. Pure Python,
no external calls.

A marker like ``(Smith et al., 2020)`` is its own partial ground truth:
the surname + year can be verified against ``Reference.authors`` / ``year``.
Numeric markers like ``[5]`` can only be checked against ``ref_id`` position.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from src.citation.detector import normalize_author_name
from src.models.citation import Citation
from src.models.reference import Reference

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarkerSurface:
    """Parsed surface form of a citation marker."""

    surname: Optional[str] = None
    year: Optional[int] = None
    num: Optional[int] = None


_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2}|21\d{2})[a-z]?\b")
_NUM_RE = re.compile(r"\d+")
_SURNAME_RE = re.compile(
    r"(?:[Dd]e|[Vv]an|[Vv]on|[Dd]el|[Dd]er|[Dd]en|[Dd]i|[Ll]a|[Ll]e|[Aa]l|[Ee]l|[Bb]in|[Ii]bn)\s+"
    r"[A-ZÀ-Ý][\wÀ-ÿ'\-]+|[A-ZÀ-Ý][\wÀ-ÿ'\-]+",
    re.UNICODE,
)


def parse_marker_surface(marker_text: str) -> MarkerSurface:
    """Extract surname / year / numeric index from a raw marker string.

    Returns a ``MarkerSurface`` with whichever fields could be parsed; missing
    fields are ``None``. Unparseable markers return an all-``None`` surface.
    """
    if not marker_text:
        return MarkerSurface()

    text = marker_text.strip()

    # Numeric-only: [5], [1, 2, 3], [3-5], 5
    stripped = text.strip("[](){}")
    if stripped and all(ch.isdigit() or ch in ", -;" for ch in stripped):
        m = _NUM_RE.search(stripped)
        if m:
            return MarkerSurface(num=int(m.group()))
        return MarkerSurface()

    # Author-year: find first year, first surname-like token
    year: Optional[int] = None
    y = _YEAR_RE.search(text)
    if y:
        year = int(y.group(1))

    surname: Optional[str] = None
    # Strip common non-name tokens that would otherwise match the capitalized pattern.
    scan = re.sub(r"\bet\s+al\.?", " ", text)
    scan = re.sub(r"\band\b", " ", scan, flags=re.IGNORECASE)
    for cand in _SURNAME_RE.finditer(scan):
        token = cand.group()
        if token.lower() in {"and", "the", "of"}:
            continue
        surname = normalize_author_name(token.split()[-1])
        break

    num: Optional[int] = None
    n = _NUM_RE.search(stripped) if stripped else None
    if n and year is None and surname is None:
        num = int(n.group())

    return MarkerSurface(surname=surname, year=year, num=num)


def _ref_surnames(ref: Reference) -> set[str]:
    out: set[str] = set()
    for author in ref.authors:
        parts = author.strip().split()
        if not parts:
            continue
        out.add(normalize_author_name(parts[-1]))
    return out


def rule_vote(marker_text: str, ref: Optional[Reference]) -> Optional[bool]:
    """Does the marker match this reference?

    Returns:
        True  — marker's author+year both match the reference
        False — at least one signal conflicts
        None  — not enough info on marker or reference to decide
    """
    if ref is None:
        return None

    surface = parse_marker_surface(marker_text)

    if surface.num is not None and surface.surname is None and surface.year is None:
        try:
            return ref.ref_id == str(surface.num)
        except Exception:
            return None

    if surface.surname is None and surface.year is None:
        return None

    surnames = _ref_surnames(ref)
    year_ok: Optional[bool] = None
    surname_ok: Optional[bool] = None

    if surface.year is not None:
        if ref.year is None:
            year_ok = None
        else:
            year_ok = ref.year == surface.year

    if surface.surname is not None:
        if not surnames:
            surname_ok = None
        else:
            surname_ok = surface.surname in surnames

    checks = [c for c in (year_ok, surname_ok) if c is not None]
    if not checks:
        return None
    return all(checks)


def unique_rule_match(
    marker_text: str, references: dict[str, Reference]
) -> Optional[str]:
    """Scan all references for one that passes rule_vote; return its ref_id if unique."""
    hits: list[str] = []
    for ref_id, ref in references.items():
        if rule_vote(marker_text, ref) is True:
            hits.append(ref_id)
    return hits[0] if len(hits) == 1 else None


def numeric_fallback(
    marker_text: str, refs_list: list[Reference]
) -> Optional[str]:
    """For numeric markers [N], return refs_list[N-1].ref_id if in range."""
    surface = parse_marker_surface(marker_text)
    if surface.num is None:
        return None
    idx = surface.num - 1
    if 0 <= idx < len(refs_list):
        return refs_list[idx].ref_id
    return None


def compute_rule_ref_id(
    marker_text: str,
    references_by_id: dict[str, Reference],
    refs_list: Optional[list[Reference]] = None,
) -> Optional[str]:
    """Best-effort deterministic ref_id for a marker.

    - Author-year markers: unique_rule_match.
    - Numeric markers: numeric_fallback (positional).
    - Returns None when nothing is confidently resolvable.
    """
    surface = parse_marker_surface(marker_text)
    if surface.surname is not None or surface.year is not None:
        return unique_rule_match(marker_text, references_by_id)
    if surface.num is not None and refs_list is not None:
        return numeric_fallback(marker_text, refs_list)
    return None


def normalize_marker_key(marker: str) -> str:
    """Normalize a marker string for cross-source lookup.

    Different sources (body text, LLM output) drift on whitespace, unicode
    quotes, and dashes. Collapsing those so keys hash-match.
    """
    if not marker:
        return ""
    s = marker.strip()
    s = s.replace("‘", "'").replace("’", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def annotate_citation_confidence(
    citations: list[Citation],
    references: dict[str, Reference],
    marker_to_ref_norm: Optional[dict[str, str]] = None,
) -> list[Citation]:
    """Populate link_confidence + vote fields on each citation.

    Three votes per citation (all free):
      1. GROBID — ``citation.ref_id`` as already set by whoever built it.
      2. Rule check — deterministic author+year / numeric match.
      3. LLM — ``matched_markers`` from the LLM ref-parsing call (may be absent).

    Then three fixes on disagreements (all deterministic):
      - Fix 1 (tiebreaker): if GROBID ≠ LLM, pick whichever passes the rule.
      - Fix 2 (rule search): if neither passes, scan all refs for a unique match.
      - Fix 3 (numeric fallback): for ``[N]`` markers, fall back to refs_list[N-1].

    Mutates each citation's ``ref_id`` to the chosen vote and stamps confidence.
    """
    if not citations:
        return citations

    marker_to_ref_norm = marker_to_ref_norm or {}
    refs_list = list(references.values())

    n = len(citations)
    low = med = dis = 0

    for c in citations:
        grobid = c.ref_id
        llm = marker_to_ref_norm.get(normalize_marker_key(c.marker))
        rule = compute_rule_ref_id(c.marker, references, refs_list)

        c.grobid_ref_id = grobid
        c.rule_ref_id = rule
        c.llm_ref_id = llm

        votes = [v for v in (grobid, llm, rule) if v is not None]
        distinct = set(votes)

        final: str = grobid
        conf: str = "high"

        if len(distinct) == 1 and len(votes) >= 2:
            # Two or more independent sources agree.
            final, conf = grobid, "high"

        elif len(distinct) > 1:
            dis += 1
            # Fix 1: rule-check tiebreaker.
            grobid_ref = references.get(grobid) if grobid else None
            llm_ref = references.get(llm) if llm else None

            if rule_vote(c.marker, grobid_ref) is True:
                final, conf = grobid, "medium"
            elif llm is not None and rule_vote(c.marker, llm_ref) is True:
                final, conf = llm, "medium"
            else:
                # Fix 2: rule search across all refs.
                uniq = unique_rule_match(c.marker, references)
                if uniq is not None:
                    final, conf = uniq, "medium"
                else:
                    # Fix 3: numeric positional fallback.
                    positional = numeric_fallback(c.marker, refs_list)
                    if positional is not None and positional in references:
                        final, conf = positional, "low"
                    else:
                        final, conf = grobid, "low"

        else:
            # Single vote (typically GROBID only). Use rule to confirm/downgrade.
            grobid_ref = references.get(grobid) if grobid else None
            check = rule_vote(c.marker, grobid_ref)
            if check is True:
                final, conf = grobid, "high"
            elif check is False:
                uniq = unique_rule_match(c.marker, references)
                if uniq is not None and uniq != grobid:
                    final, conf = uniq, "medium"
                else:
                    final, conf = grobid, "low"
            else:
                final, conf = grobid, "high"

        c.ref_id = final
        c.link_confidence = conf  # type: ignore[assignment]

        if conf == "medium":
            med += 1
        elif conf == "low":
            low += 1

        log.info(
            "link_vote marker=%r grobid=%s rule=%s llm=%s final=%s conf=%s",
            c.marker[:60], grobid, rule, llm, final, conf,
        )

    log.info(
        "link_stats n=%d medium=%d low=%d disagreements=%d", n, med, low, dis
    )
    return citations
