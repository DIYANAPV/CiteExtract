
import os
import re
from typing import Optional
from xml.etree import ElementTree

import httpx
from defusedxml.ElementTree import fromstring as _safe_fromstring

from citeextract.verification.api_clients.rate_limiter import fetch_with_retry
from citeextract.verification.matching import title_similarity

BASE_URL = "https://export.arxiv.org/api/query"

_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _arxiv_disabled() -> bool:
    """True when the caller has explicitly disabled arXiv via env var.

    Used to skip arXiv when the daily quota is exhausted. Same pattern as
    the SERPAPI_KEY and CITEEXTRACT_SKIP_OPENALEX gates.
    """
    return os.environ.get("CITEEXTRACT_SKIP_ARXIV") == "1"


async def search_by_title(
    title: str, client: httpx.AsyncClient,
    *,
    errors: Optional[list[str]] = None,
) -> Optional[dict]:
    if _arxiv_disabled():
        return None
    if not title or len(title.strip()) < 5:
        return None

    clean = re.sub(r"[^\w\s]", " ", title)
    clean = re.sub(r"\s+", " ", clean).strip()
    query = f'ti:"{clean}"'

    try:
        resp = await fetch_with_retry(
            client, "arxiv", "GET",
            BASE_URL,
            params={
                "search_query": query,
                "start": 0,
                "max_results": 5,
            },
            timeout=15,
        )
        resp.raise_for_status()
    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        if errors is not None:
            errors.append(f"arxiv: {type(exc).__name__}")
        return None

    return _parse_response(resp.text, title)


async def search_by_authors_year(
    authors: list[str],
    year: Optional[int],
    title_hint: str,
    client: httpx.AsyncClient,
    *,
    errors: Optional[list[str]] = None,
) -> Optional[dict]:
    if _arxiv_disabled():
        return None
    from citeextract.verification.matching import _author_tokens, author_containment

    surnames = sorted(_author_tokens(authors))
    if len(surnames) < 2:
        return None

    au_query = f"au:{surnames[0]} AND au:{surnames[1]}"

    try:
        resp = await fetch_with_retry(
            client, "arxiv", "GET",
            BASE_URL,
            params={
                "search_query": au_query,
                "start": 0,
                "max_results": 10,
            },
            timeout=15,
        )
        resp.raise_for_status()
    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        if errors is not None:
            errors.append(f"arxiv: {type(exc).__name__}")
        return None

    try:
        root = _safe_fromstring(resp.text)
    except ElementTree.ParseError:
        return None

    entries = root.findall("atom:entry", _NS)
    if not entries:
        return None

    best: Optional[tuple[float, dict]] = None
    for entry in entries:
        paper = _parse_entry(entry)
        if not paper:
            continue
        if year is not None and paper.get("year") is not None:
            if abs(year - paper["year"]) > 1:
                continue
        overlap = author_containment(authors, paper.get("authors", []))
        if overlap < 0.6:
            continue
        sim = title_similarity(title_hint, paper["title"])
        if sim < 0.30:
            continue
        score = overlap + sim * 0.5
        if best is None or score > best[0]:
            paper["title_similarity"] = sim
            paper["title_evolved"] = True
            best = (score, paper)

    return best[1] if best else None


def _parse_response(xml_text: str, query_title: str) -> Optional[dict]:
    try:
        root = _safe_fromstring(xml_text)
    except ElementTree.ParseError:
        return None

    entries = root.findall("atom:entry", _NS)
    if not entries:
        return None

    from citeextract import config
    threshold = config.thresholds()["title_match"]

    best_match: Optional[dict] = None
    best_sim = 0.0

    for entry in entries:
        paper = _parse_entry(entry)
        if not paper:
            continue
        sim = title_similarity(query_title, paper["title"])
        if sim > best_sim:
            best_sim = sim
            best_match = paper

    if best_match is None or best_sim < threshold:
        return None

    best_match["title_similarity"] = best_sim
    return best_match


def _parse_entry(entry: ElementTree.Element) -> Optional[dict]:
    title_el = entry.find("atom:title", _NS)
    if title_el is None or not title_el.text:
        return None

    title = re.sub(r"\s+", " ", title_el.text).strip()

    authors = []
    for author_el in entry.findall("atom:author", _NS):
        name_el = author_el.find("atom:name", _NS)
        if name_el is not None and name_el.text:
            authors.append(name_el.text.strip())

    published = entry.find("atom:published", _NS)
    year = None
    if published is not None and published.text:
        year_match = re.match(r"(\d{4})", published.text)
        if year_match:
            year = int(year_match.group(1))

    arxiv_id = None
    id_el = entry.find("atom:id", _NS)
    if id_el is not None and id_el.text:
        id_match = re.search(r"(\d{4}\.\d{4,5})", id_el.text)
        if id_match:
            arxiv_id = id_match.group(1)

    doi = None
    for link in entry.findall("atom:link", _NS):
        if link.get("title") == "doi":
            href = link.get("href", "")
            doi_match = re.search(r"doi\.org/(.+)", href)
            if doi_match:
                doi = doi_match.group(1)

    summary = entry.find("atom:summary", _NS)
    abstract = None
    if summary is not None and summary.text:
        abstract = re.sub(r"\s+", " ", summary.text).strip()

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "venue": "arXiv",
        "abstract": abstract,
        "doi": f"10.48550/arXiv.{arxiv_id}" if arxiv_id else doi,
        "arxiv_id": arxiv_id,
    }
