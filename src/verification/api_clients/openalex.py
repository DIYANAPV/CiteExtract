"""OpenAlex API client — broadest coverage (250M+ works).

Fallback when Semantic Scholar misses. Reconstructs abstracts from
OpenAlex's inverted-index format for downstream semantic verification.
"""

import re
from typing import Optional

import httpx

from src import config
from src.verification.api_clients.rate_limiter import fetch_with_retry
from src.verification.matching import title_similarity

BASE_URL = "https://api.openalex.org/works"


async def search_by_title(
    title: str, client: httpx.AsyncClient,
    *,
    ref_authors: Optional[list[str]] = None,
    ref_year: Optional[int] = None,
) -> Optional[dict]:
    """Search OpenAlex by title and pick the best candidate.

    Uses :func:`matching.pick_best_candidate` for two-pass scoring —
    composite (title + authors + year) first, title-only fallback when
    nothing strong matches. The returned dict carries
    ``match_strategy`` + ``match_score`` so downstream verdicts can
    flag title-only matches as low-author-confidence.
    """
    if not title or len(title.strip()) < 5:
        return None

    cfg = config.api("openalex")
    mailto = config.openalex_mailto()
    timeout = cfg.get("timeout", 20)
    per_page = cfg.get("results_per_page", 5)

    # Clean query for filter param
    query = re.sub(r"[^\w\s]", " ", title)
    query = re.sub(r"\s+", " ", query).strip()

    try:
        resp = await fetch_with_retry(
            client, "openalex", "GET", BASE_URL,
            params={
                "filter": f"title.search:{query}",
                "per_page": per_page,
                "mailto": mailto,
            },
            headers={"User-Agent": "CheckCitation/1.0"},
            timeout=timeout,
        )
        resp.raise_for_status()
    except (httpx.HTTPStatusError, httpx.RequestError):
        return None

    results = resp.json().get("results", [])
    if not results:
        return None

    # Pre-parse so the picker scores against structured fields.
    parsed = [_parse_work(w, similarity=0.0) for w in results]

    from src.verification.matching import pick_best_candidate

    chosen, strategy, score = pick_best_candidate(
        title, ref_authors or [], ref_year, parsed,
    )
    if chosen is None:
        return None

    chosen["title_similarity"] = title_similarity(
        title, chosen.get("title", ""),
    )
    chosen["match_strategy"] = strategy
    chosen["match_score"] = score
    return chosen


def _parse_work(work: dict, similarity: float) -> dict:
    """Parse OpenAlex work response into our standard format."""
    # Authors
    authors = []
    for authorship in work.get("authorships", []):
        author = authorship.get("author", {})
        name = author.get("display_name", "")
        if name:
            authors.append(name)

    # Abstract reconstruction from inverted index
    abstract = None
    abstract_inv = work.get("abstract_inverted_index")
    if abstract_inv:
        abstract = _reconstruct_abstract(abstract_inv)

    # DOI
    doi = None
    raw_doi = work.get("doi", "")
    if raw_doi:
        # OpenAlex returns full URL: "https://doi.org/10.xxx"
        doi = raw_doi.replace("https://doi.org/", "")

    # Extract open-access URL if available
    oa_url = None
    primary_loc = work.get("primary_location", {}) or {}
    if primary_loc.get("is_oa"):
        oa_url = primary_loc.get("landing_page_url")
    if not oa_url:
        best_oa = work.get("best_oa_location", {}) or {}
        oa_url = best_oa.get("landing_page_url")

    # Extract venue aliases from source metadata
    venue_aliases = _extract_venue_aliases(work)

    return {
        "title": work.get("title", ""),
        "authors": authors,
        "year": work.get("publication_year"),
        "venue": _extract_venue(work),
        "abstract": abstract,
        "doi": doi,
        "openalex_id": work.get("id"),
        "oa_url": oa_url,
        "venue_aliases": venue_aliases,
        "title_similarity": similarity,
    }


def _extract_venue(work: dict) -> str:
    """Extract venue name from OpenAlex work."""
    primary_loc = work.get("primary_location", {}) or {}
    source = primary_loc.get("source", {}) or {}
    return source.get("display_name", "")


def _extract_venue_aliases(work: dict) -> list[str]:
    """Extract all known venue aliases from OpenAlex source metadata."""
    primary_loc = work.get("primary_location", {}) or {}
    source = primary_loc.get("source", {}) or {}
    aliases: list[str] = []
    display_name = source.get("display_name", "")
    if display_name:
        aliases.append(display_name)
    aliases.extend(source.get("alternate_titles", []))
    return aliases


def _reconstruct_abstract(inverted_index: dict) -> str:
    """Reconstruct abstract from OpenAlex inverted index format.

    OpenAlex stores abstracts as: {"word": [position0, position1, ...]}
    We rebuild by mapping positions to words and joining.
    """
    if not inverted_index:
        return ""
    position_map: dict[int, str] = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            position_map[pos] = word
    sorted_positions = sorted(position_map.keys())
    return " ".join(position_map[pos] for pos in sorted_positions)
