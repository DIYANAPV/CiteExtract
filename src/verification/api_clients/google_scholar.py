"""Google Scholar search via SerpAPI — safety net for false FABRICATED verdicts.

Only called when the metadata agent returns FABRICATED, to double-check
against what people actually see on Google Scholar.  Uses SerpAPI's free
tier (100 searches/month).  If no API key is configured, silently skips.
"""

import logging
import re
from typing import Optional

import httpx

from src import config

log = logging.getLogger(__name__)

_SERPAPI_ENDPOINT = "https://serpapi.com/search"


def _get_api_key() -> Optional[str]:
    return config.serpapi_key()


async def search_google_scholar(
    title: str,
    client: httpx.AsyncClient,
    api_key: Optional[str] = None,
) -> Optional[dict]:
    """Search Google Scholar for a paper by title via SerpAPI.

    Args:
        title: Paper title to search for.
        client: Shared httpx async client.
        api_key: SerpAPI key. Falls back to env if not provided.

    Returns:
        Dict with title, authors, year, venue, link — or None if not found
        or no API key.
    """
    key = api_key or _get_api_key()
    if not key:
        return None

    if not title or len(title.strip()) < 5:
        return None

    # Clean query — remove citation markers, brackets
    query = re.sub(r"[\[\]{}()]", "", title)
    query = re.sub(r"\s+", " ", query).strip()

    try:
        resp = await client.get(
            _SERPAPI_ENDPOINT,
            params={
                "engine": "google_scholar",
                "q": query,
                "api_key": key,
                "num": 3,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

    except Exception as e:
        log.warning(f"Google Scholar search failed: {e}")
        return None

    # Parse organic results
    results = data.get("organic_results", [])
    if not results:
        return None

    # Return the top result in our standard format
    top = results[0]
    return _parse_result(top)


def _parse_result(result: dict) -> dict:
    """Parse a SerpAPI Google Scholar result into our standard format."""
    # Title
    title = result.get("title", "")

    # Authors + year + venue from the "publication_info" snippet
    pub_info = result.get("publication_info", {})
    summary = pub_info.get("summary", "")

    authors = _extract_authors(pub_info)
    year = _extract_year(summary)
    venue = _extract_venue(summary)

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "venue": venue,
        "link": result.get("link"),
        "snippet": result.get("snippet", ""),
        "cited_by_count": result.get("inline_links", {}).get("cited_by", {}).get("total"),
    }


def _extract_authors(pub_info: dict) -> list[str]:
    """Extract author names from publication_info."""
    authors_list = pub_info.get("authors", [])
    if authors_list:
        return [a.get("name", "") for a in authors_list if a.get("name")]

    # Fallback: parse from summary string ("D Guo, D Yang, ... - arXiv, 2025")
    summary = pub_info.get("summary", "")
    if " - " in summary:
        author_part = summary.split(" - ")[0]
        return [a.strip() for a in author_part.split(",") if a.strip() and not a.strip().isdigit()]
    return []


def _extract_year(summary: str) -> Optional[int]:
    """Extract year from summary string."""
    # Look for 4-digit year (typically at the end: "..., 2025")
    match = re.search(r"\b(19|20)\d{2}\b", summary)
    if match:
        return int(match.group(0))
    return None


def _extract_venue(summary: str) -> Optional[str]:
    """Extract venue from summary string.

    Summary format is typically: "Authors - Venue, Year" or "Authors - Year"
    """
    if " - " not in summary:
        return None

    after_dash = summary.split(" - ", 1)[1].strip()

    # Remove year from the end
    venue = re.sub(r",?\s*(19|20)\d{2}\s*$", "", after_dash).strip()

    if not venue or venue == "…":
        return None
    return venue
