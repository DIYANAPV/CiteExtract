
import logging
import re
from typing import Optional

import httpx

from citeextract import config

log = logging.getLogger(__name__)

_SERPAPI_ENDPOINT = "https://serpapi.com/search"


def _get_api_key() -> Optional[str]:
    return config.serpapi_key()


async def search_google_scholar(
    title: str,
    client: httpx.AsyncClient,
    api_key: Optional[str] = None,
) -> Optional[dict]:
    key = api_key or _get_api_key()
    if not key:
        return None

    if not title or len(title.strip()) < 5:
        return None

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

    results = data.get("organic_results", [])
    if not results:
        return None

    top = results[0]
    return _parse_result(top)


def _parse_result(result: dict) -> dict:
    title = result.get("title", "")

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
    authors_list = pub_info.get("authors", [])
    if authors_list:
        return [a.get("name", "") for a in authors_list if a.get("name")]

    summary = pub_info.get("summary", "")
    if " - " in summary:
        author_part = summary.split(" - ")[0]
        return [a.strip() for a in author_part.split(",") if a.strip() and not a.strip().isdigit()]
    return []


def _extract_year(summary: str) -> Optional[int]:
    match = re.search(r"\b(19|20)\d{2}\b", summary)
    if match:
        return int(match.group(0))
    return None


def _extract_venue(summary: str) -> Optional[str]:
    if " - " not in summary:
        return None

    after_dash = summary.split(" - ", 1)[1].strip()

    venue = re.sub(r",?\s*(19|20)\d{2}\s*$", "", after_dash).strip()

    if not venue or venue == "…":
        return None
    return venue
