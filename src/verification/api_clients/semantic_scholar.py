"""Semantic Scholar API client — primary title search.

Returns abstracts (critical for L4 semantic verification reuse).
Supports both title search and direct ID lookups (DOI, arXiv, S2 ID).
"""

import re
from typing import Optional

import httpx

from src import config
from src.verification.api_clients.rate_limiter import fetch_with_retry
from src.verification.matching import title_similarity

BASE_URL = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,authors,year,venue,abstract,externalIds,openAccessPdf,publicationVenue"


def _get_api_key() -> Optional[str]:
    return config.s2_api_key()


def _headers() -> dict:
    headers = {"User-Agent": "CheckCitation/1.0"}
    key = _get_api_key()
    if key:
        headers["x-api-key"] = key
    return headers


async def search_by_title(
    title: str, client: httpx.AsyncClient
) -> Optional[dict]:
    """Search Semantic Scholar by title. Returns best matching paper or None.

    The returned dict includes 'abstract' which is cached for L4 reuse.
    """
    if not title or len(title.strip()) < 5:
        return None

    # Clean query
    query = re.sub(r"[^\w\s]", " ", title)
    query = re.sub(r"\s+", " ", query).strip()

    cfg = config.api("semantic_scholar")
    timeout = cfg.get("timeout", 20)
    results_per_page = cfg.get("results_per_page", 5)

    try:
        resp = await fetch_with_retry(
            client, "semantic_scholar", "GET",
            f"{BASE_URL}/paper/search",
            params={"query": query, "fields": FIELDS, "limit": results_per_page},
            headers=_headers(),
            timeout=timeout,
        )
        if resp.status_code == 429:
            return None  # rate limited after retries, let cascade continue
        resp.raise_for_status()
    except (httpx.HTTPStatusError, httpx.RequestError):
        return None

    data = resp.json().get("data", [])
    if not data:
        return None

    # Find best title match
    best_match: Optional[dict] = None
    best_sim = 0.0
    for paper in data:
        paper_title = paper.get("title", "")
        sim = title_similarity(title, paper_title)
        if sim > best_sim:
            best_sim = sim
            best_match = paper

    if best_match is None or best_sim < config.thresholds()["title_match"]:
        return None

    return _parse_paper(best_match, best_sim)


async def lookup_by_id(
    paper_id: str, client: httpx.AsyncClient
) -> Optional[dict]:
    """Look up a single paper by S2 ID, DOI, or arXiv ID.

    paper_id formats: "DOI:10.xxx", "ARXIV:2301.xxx", or raw S2 paper ID.
    """
    try:
        cfg_s2 = config.api("semantic_scholar")
        resp = await fetch_with_retry(
            client, "semantic_scholar", "GET",
            f"{BASE_URL}/paper/{paper_id}",
            params={"fields": FIELDS},
            headers=_headers(),
            timeout=cfg_s2.get("timeout", 20),
        )
        if resp.status_code in (404, 429):
            return None
        resp.raise_for_status()
    except (httpx.HTTPStatusError, httpx.RequestError):
        return None

    paper = resp.json()
    if not paper or not paper.get("title"):
        return None

    return _parse_paper(paper, similarity=1.0)



def _parse_paper(paper: dict, similarity: float) -> dict:
    """Parse S2 paper response into our standard format."""
    authors = [
        a.get("name", "") for a in paper.get("authors", []) if a.get("name")
    ]

    ext_ids = paper.get("externalIds", {}) or {}
    doi = ext_ids.get("DOI")
    arxiv_id = ext_ids.get("ArXiv")

    # Extract venue aliases from publicationVenue (structured, disambiguated)
    venue_aliases: list[str] = []
    pub_venue = paper.get("publicationVenue") or {}
    if pub_venue:
        if pub_venue.get("name"):
            venue_aliases.append(pub_venue["name"])
        venue_aliases.extend(pub_venue.get("alternate_names", []))

    return {
        "title": paper.get("title", ""),
        "authors": authors,
        "year": paper.get("year"),
        "venue": paper.get("venue", ""),
        "abstract": paper.get("abstract"),
        "doi": doi,
        "arxiv_id": arxiv_id,
        "s2_id": paper.get("paperId"),
        "openAccessPdf": paper.get("openAccessPdf"),
        "venue_aliases": venue_aliases,
        "title_similarity": similarity,
    }
