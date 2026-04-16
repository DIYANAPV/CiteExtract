"""CrossRef API client — DOI lookup only (no title search).

Used as the first step when a reference has a DOI extracted by L1.
Also provides retraction status for free.
"""

import re
from typing import Optional
from urllib.parse import quote

import httpx

from src import config
from src.verification.api_clients.rate_limiter import fetch_with_retry

BASE_URL = "https://api.crossref.org/works"


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
