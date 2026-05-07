
import re
from typing import Optional

import httpx

from citeextract import config
from citeextract.verification.api_clients.rate_limiter import fetch_with_retry
from citeextract.verification.matching import title_similarity

BASE_URL = "https://api.openalex.org/works"


async def search_by_title(
    title: str, client: httpx.AsyncClient,
    *,
    ref_authors: Optional[list[str]] = None,
    ref_year: Optional[int] = None,
) -> Optional[dict]:
    if not title or len(title.strip()) < 5:
        return None

    cfg = config.api("openalex")
    mailto = config.openalex_mailto()
    timeout = cfg.get("timeout", 20)
    per_page = cfg.get("results_per_page", 5)

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
            headers={"User-Agent": "CiteExtract/1.0"},
            timeout=timeout,
        )
        resp.raise_for_status()
    except (httpx.HTTPStatusError, httpx.RequestError):
        return None

    results = resp.json().get("results", [])
    if not results:
        return None

    parsed = [_parse_work(w, similarity=0.0) for w in results]

    from citeextract.verification.matching import pick_best_candidate

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
    authors = []
    for authorship in work.get("authorships", []):
        author = authorship.get("author", {})
        name = author.get("display_name", "")
        if name:
            authors.append(name)

    abstract = None
    abstract_inv = work.get("abstract_inverted_index")
    if abstract_inv:
        abstract = _reconstruct_abstract(abstract_inv)

    doi = None
    raw_doi = work.get("doi", "")
    if raw_doi:
        doi = raw_doi.replace("https://doi.org/", "")

    oa_url = None
    primary_loc = work.get("primary_location", {}) or {}
    if primary_loc.get("is_oa"):
        oa_url = primary_loc.get("landing_page_url")
    if not oa_url:
        best_oa = work.get("best_oa_location", {}) or {}
        oa_url = best_oa.get("landing_page_url")

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
    primary_loc = work.get("primary_location", {}) or {}
    source = primary_loc.get("source", {}) or {}
    return source.get("display_name", "")


def _extract_venue_aliases(work: dict) -> list[str]:
    primary_loc = work.get("primary_location", {}) or {}
    source = primary_loc.get("source", {}) or {}
    aliases: list[str] = []
    display_name = source.get("display_name", "")
    if display_name:
        aliases.append(display_name)
    aliases.extend(source.get("alternate_titles", []))
    return aliases


def _reconstruct_abstract(inverted_index: dict) -> str:
    if not inverted_index:
        return ""
    position_map: dict[int, str] = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            position_map[pos] = word
    sorted_positions = sorted(position_map.keys())
    return " ".join(position_map[pos] for pos in sorted_positions)
