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
    title: str, client: httpx.AsyncClient,
    *,
    ref_authors: Optional[list[str]] = None,
    ref_year: Optional[int] = None,
) -> Optional[dict]:
    """Search Semantic Scholar by title and pick the best candidate.

    Uses :func:`matching.pick_best_candidate` for two-pass scoring —
    composite (title + authors + year) first, title-only fallback when
    nothing strong matches. The returned dict carries a
    ``match_strategy`` key so downstream knows whether the match was
    composite-strong or title-only (i.e. authors didn't agree, possibly
    because they're hallucinated in the citing paper).

    Author and year are optional: passing them raises match precision
    (rejects vocabulary-overlap false-positives), omitting them falls
    back gracefully to title-only behavior.
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

    # Pre-parse every candidate so the picker sees structured fields
    # (title, authors, year) instead of raw S2 dicts. The
    # ``title_similarity`` is rewritten after the picker chooses.
    parsed = [_parse_paper(p, similarity=0.0) for p in data]

    from src.verification.matching import pick_best_candidate

    chosen, strategy, score = pick_best_candidate(
        title, ref_authors or [], ref_year, parsed,
    )
    if chosen is None:
        return None

    # Stamp the chosen candidate with both ``title_similarity`` (for
    # back-compat with legacy display code) and the new ``match_strategy``
    # / ``match_score`` fields so downstream verdicts can flag a
    # title-only fallback as low-author-confidence.
    chosen["title_similarity"] = (
        title_similarity(title, chosen.get("title", ""))
    )
    chosen["match_strategy"] = strategy
    chosen["match_score"] = score
    return chosen


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
    # ACL Anthology ID — let the metadata layer's canonical-id matcher
    # bridge ``aclanthology.org/Q16-1026`` (cited) with the publisher
    # DOI ``10.1162/tacla00104`` (DB), since the ACL ID is what S2 holds
    # in common.
    anthology_id = ext_ids.get("ACL")

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
        "anthology_id": anthology_id,
        "s2_id": paper.get("paperId"),
        "openAccessPdf": paper.get("openAccessPdf"),
        "venue_aliases": venue_aliases,
        "title_similarity": similarity,
    }
