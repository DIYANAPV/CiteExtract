
import logging
from typing import Optional
from urllib.parse import quote

import httpx

from citeextract import config
from citeextract.verification.api_clients.rate_limiter import fetch_with_retry

log = logging.getLogger(__name__)

BASE_URL = "https://api.crossref.org/works"

_LOW_QUALITY_CROSSREF_TYPES = frozenset({
    "reference-entry",
    "component",
    "dataset",
})


async def lookup_doi(
    doi: str, client: httpx.AsyncClient,
    *,
    errors: Optional[list[str]] = None,
) -> Optional[dict]:
    cfg = config.api("crossref")
    mailto = config.crossref_mailto()
    timeout = cfg.get("timeout", 15)

    url = f"{BASE_URL}/{quote(doi, safe='')}"
    try:
        resp = await fetch_with_retry(
            client, "crossref", "GET", url,
            params={"mailto": mailto},
            headers={"User-Agent": f"CiteExtract/1.0 (mailto:{mailto})"},
            timeout=timeout,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        if errors is not None:
            errors.append(f"crossref: {type(exc).__name__}")
        return None

    try:
        data = resp.json().get("message", {})
    except Exception as exc:
        if errors is not None:
            errors.append(f"crossref: {type(exc).__name__}")
        return None
    if not data:
        return None

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
    errors: Optional[list[str]] = None,
) -> Optional[dict]:
    if not title or len(title.strip()) < 5:
        return None

    from citeextract.verification.matching import title_similarity, pick_best_candidate

    cfg = config.api("crossref")
    mailto = config.crossref_mailto()
    timeout = cfg.get("timeout", 15)

    try:
        resp = await fetch_with_retry(
            client, "crossref", "GET", BASE_URL,
            params={
                "query.bibliographic": title,
                "rows": 5,
                "mailto": mailto,
            },
            headers={"User-Agent": f"CiteExtract/1.0 (mailto:{mailto})"},
            timeout=timeout,
        )
        if resp.status_code != 200:
            if errors is not None and resp.status_code >= 500:
                errors.append(f"crossref: HTTP {resp.status_code}")
            return None
    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        if errors is not None:
            errors.append(f"crossref: {type(exc).__name__}")
        return None

    try:
        items = resp.json().get("message", {}).get("items", []) or []
    except Exception as exc:
        if errors is not None:
            errors.append(f"crossref: {type(exc).__name__}")
        return None
    if not items:
        return None

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
    for update in data.get("update-to", []):
        if update.get("type") == "retraction":
            return True
    if data.get("type") == "retraction":
        return True
    if data.get("relation", {}).get("is-retracted-by"):
        return True
    return False
