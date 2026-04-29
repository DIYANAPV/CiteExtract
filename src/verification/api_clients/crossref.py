"""CrossRef API client — DOI lookup + title search.

Used as the first step when a reference has a DOI extracted by L1, and as
a late-cascade fallback for refs that S2/OpenAlex/PubMed/arXiv all missed.
Also provides retraction status for free.
"""

import logging
import re
from typing import Optional
from urllib.parse import quote

import httpx

from src import config
from src.verification.api_clients.rate_limiter import fetch_with_retry

log = logging.getLogger(__name__)

BASE_URL = "https://api.crossref.org/works"

# CrossRef ``type`` values that almost never represent the actual paper a
# reference is pointing at. ``reference-entry`` and ``component`` are common
# false-positives for short queries (e.g. "Perceptron" alone matches the
# encyclopedia entry "Perceptron Algorithm, 1959; Rosenblatt"). We still
# accept them when nothing else clears the threshold — the caller decides.
_LOW_QUALITY_CROSSREF_TYPES = frozenset({
    "reference-entry",
    "component",
    "dataset",
})


async def lookup_doi(
    doi: str, client: httpx.AsyncClient
) -> Optional[dict]:
    """Look up a DOI in CrossRef.

    Returns dict with title, authors, year, venue, retraction_status, or None.
    """
    cfg = config.api("crossref")
    mailto = config.crossref_mailto()
    timeout = cfg.get("timeout", 15)

    url = f"{BASE_URL}/{quote(doi, safe='')}"
    try:
        resp = await fetch_with_retry(
            client, "crossref", "GET", url,
            params={"mailto": mailto},
            headers={"User-Agent": f"CheckCitation/1.0 (mailto:{mailto})"},
            timeout=timeout,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
    except (httpx.HTTPStatusError, httpx.RequestError):
        return None

    try:
        data = resp.json().get("message", {})
    except Exception:
        return None
    if not data:
        return None

    # Extract fields
    title_list = data.get("title", [])
    title = title_list[0] if title_list else ""

    authors = []
    for a in data.get("author", []):
        given = a.get("given", "")
        family = a.get("family", "")
        name = f"{given} {family}".strip()
        if name:
            authors.append(name)

    year: Optional[int] = None
    date_parts = data.get("published-print", data.get("published-online", data.get("issued", {})))
    if date_parts and "date-parts" in date_parts:
        parts = date_parts["date-parts"]
        if parts and parts[0] and parts[0][0]:
            year = int(parts[0][0])

    venue_list = data.get("container-title", [])
    venue = venue_list[0] if venue_list else ""

    # Retraction detection
    retracted = _check_retraction(data)

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "venue": venue,
        "doi": doi,
        "retraction_status": retracted,
    }


async def search_by_title(
    title: str, client: httpx.AsyncClient,
    *,
    ref_authors: Optional[list[str]] = None,
    ref_year: Optional[int] = None,
) -> Optional[dict]:
    """Search CrossRef by title and pick the best candidate.

    Two-axis selection:

    1. **Type-quality bucketing.** Encyclopedia / dataset / reference-entry
       types are demoted to a fallback bucket so they don't beat real
       articles for short titles like "Perceptron".
    2. **Composite scoring within each bucket** (see
       :func:`matching.pick_best_candidate`). High-quality bucket runs
       first; only when it has no match do we try the low-quality
       bucket. Inside each bucket the picker prefers candidates whose
       authors and year align with the reference, falling back to
       title-only when nothing strong matches (with
       ``match_strategy="title_only"`` so verdicts can flag low
       author confidence).
    """
    if not title or len(title.strip()) < 5:
        return None

    from src.verification.matching import title_similarity, pick_best_candidate

    cfg = config.api("crossref")
    mailto = config.crossref_mailto()
    timeout = cfg.get("timeout", 15)

    # ``query.bibliographic`` is CrossRef's recommended scoring field for
    # general bibliographic queries — better than ``query.title`` for
    # tolerating subtitles, capitalisation, and punctuation drift.
    try:
        resp = await fetch_with_retry(
            client, "crossref", "GET", BASE_URL,
            params={
                "query.bibliographic": title,
                "rows": 5,
                "mailto": mailto,
            },
            headers={"User-Agent": f"CheckCitation/1.0 (mailto:{mailto})"},
            timeout=timeout,
        )
        if resp.status_code != 200:
            return None
    except (httpx.HTTPStatusError, httpx.RequestError):
        return None

    try:
        items = resp.json().get("message", {}).get("items", []) or []
    except Exception:
        return None
    if not items:
        return None

    # Split into quality buckets, parsing each item to the structured
    # record shape the picker expects.
    high_quality: list[dict] = []
    low_quality: list[dict] = []
    for item in items:
        item_title = (item.get("title") or [""])[0]
        if not item_title:
            continue
        parsed = _parse_works_item(item, similarity=0.0)
        if item.get("type") in _LOW_QUALITY_CROSSREF_TYPES:
            low_quality.append(parsed)
        else:
            high_quality.append(parsed)

    # Try high-quality first; fall back to low-quality only when it's empty.
    for bucket in (high_quality, low_quality):
        if not bucket:
            continue
        chosen, strategy, score = pick_best_candidate(
            title, ref_authors or [], ref_year, bucket,
        )
        if chosen is not None:
            chosen["title_similarity"] = title_similarity(
                title, chosen.get("title", ""),
            )
            chosen["match_strategy"] = strategy
            chosen["match_score"] = score
            return chosen

    return None


def _parse_works_item(data: dict, similarity: float) -> dict:
    """Parse one CrossRef ``works`` item into our standard record dict.

    Mirrors :func:`lookup_doi` so cascade callers can treat both shapes
    interchangeably.
    """
    title_list = data.get("title", [])
    title = title_list[0] if title_list else ""

    authors = []
    for a in data.get("author", []):
        given = a.get("given", "")
        family = a.get("family", "")
        name = f"{given} {family}".strip()
        if name:
            authors.append(name)

    year: Optional[int] = None
    date_parts = data.get("published-print") or data.get("published-online") or data.get("issued") or {}
    if date_parts and "date-parts" in date_parts:
        parts = date_parts["date-parts"]
        if parts and parts[0] and parts[0][0]:
            year = int(parts[0][0])

    venue_list = data.get("container-title", [])
    venue = venue_list[0] if venue_list else ""

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "venue": venue,
        "doi": data.get("DOI"),
        "retraction_status": _check_retraction(data),
        "title_similarity": similarity,
        "type": data.get("type"),
    }


def _check_retraction(data: dict) -> bool:
    """Check if CrossRef record indicates retraction."""
    # Method 1: update-to field
    for update in data.get("update-to", []):
        if update.get("type") == "retraction":
            return True
    # Method 2: type field
    if data.get("type") == "retraction":
        return True
    # Method 3: check relation
    if data.get("relation", {}).get("is-retracted-by"):
        return True
    return False
