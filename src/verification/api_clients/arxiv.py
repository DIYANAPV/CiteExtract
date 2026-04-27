"""arXiv API client — last-resort title search for preprints.

arXiv's API is free, requires no key, and covers papers that may not yet
be indexed by Semantic Scholar or OpenAlex. Added as the final scholarly
source before the web fallback.

API docs: https://info.arxiv.org/help/api/basics.html
"""

import re
from typing import Optional
from xml.etree import ElementTree

import httpx
from defusedxml.ElementTree import fromstring as _safe_fromstring

from src.verification.api_clients.rate_limiter import fetch_with_retry
from src.verification.matching import title_similarity

BASE_URL = "https://export.arxiv.org/api/query"

# Atom namespace used in arXiv API responses
_NS = {"atom": "http://www.w3.org/2005/Atom"}


async def search_by_title(
    title: str, client: httpx.AsyncClient
) -> Optional[dict]:
    """Search arXiv by title. Returns best matching paper or None.

    Uses the arXiv search API which returns Atom XML.
    """
    if not title or len(title.strip()) < 5:
        return None

    # Clean query: arXiv search works best with quoted title
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
    except (httpx.HTTPStatusError, httpx.RequestError):
        return None

    return _parse_response(resp.text, title)


async def search_by_authors_year(
    authors: list[str],
    year: Optional[int],
    title_hint: str,
    client: httpx.AsyncClient,
) -> Optional[dict]:
    """Last-resort arXiv search by author surnames + year, with title verification.

    Closes a real-world gap: papers whose arXiv title was renamed across
    versions (e.g. ``2505.14376`` was uploaded as "AutoRev: Automatic Peer
    Review System…" in v1, renamed to "Graph-Guided Passage Retrieval…" in
    v3). The arXiv search API only returns the latest title, so a strict
    title-similarity check fails — but the authors and year are stable.

    Verification is layered to keep precision:

    - Need at least **two** distinct reference author surnames before we
      try this — single-author matches on a common surname are too risky.
    - Year must be within **one** of the reference year (or unknown
      either side) to handle preprint→publication shift.
    - Author **containment** (fraction of reference authors found on the
      candidate) must be ≥ 0.6 — i.e. most of the user's listed authors
      genuinely appear on the paper.
    - A relaxed ``title_hint`` similarity floor of 0.30 catches the case
      where the title was renamed entirely (e.g. AutoRev → Graph-Guided…)
      but still rejects unrelated papers that happen to share authors.

    Returns the matched record (with ``title_evolved=True`` flag) or
    ``None`` when nothing meets the bar.
    """
    from src.verification.matching import _author_tokens, author_containment

    surnames = sorted(_author_tokens(authors))
    if len(surnames) < 2:
        # Single-author searches on common surnames are noisy and
        # frequently false-positive on arXiv. Skip rather than risk
        # returning the wrong paper.
        return None

    # Search for the two most distinctive (alphabetically first, which
    # roughly correlates with token rarity for non-Western names) surnames
    # — pairing two surnames almost always uniquely identifies the paper.
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
    except (httpx.HTTPStatusError, httpx.RequestError):
        return None

    # Parse all entries, then post-filter by year + author overlap +
    # title-hint similarity. ``_parse_response`` enforces a strict title
    # threshold — we need the broader entry list, so iterate manually.
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
        # Year filter: tolerate ±1 year for preprint vs publication date
        if year is not None and paper.get("year") is not None:
            if abs(year - paper["year"]) > 1:
                continue
        # Author containment: most of user's listed authors must appear
        overlap = author_containment(authors, paper.get("authors", []))
        if overlap < 0.6:
            continue
        # Title hint: relaxed floor — a renamed paper still shouldn't
        # share zero token-level overlap with the user's title.
        sim = title_similarity(title_hint, paper["title"])
        if sim < 0.30:
            continue
        # Score by author overlap + title sim — pick the strongest
        # candidate among entries that pass all filters.
        score = overlap + sim * 0.5
        if best is None or score > best[0]:
            paper["title_similarity"] = sim
            paper["title_evolved"] = True
            best = (score, paper)

    return best[1] if best else None


def _parse_response(xml_text: str, query_title: str) -> Optional[dict]:
    """Parse arXiv Atom XML response and find best title match."""
    try:
        root = _safe_fromstring(xml_text)
    except ElementTree.ParseError:
        return None

    entries = root.findall("atom:entry", _NS)
    if not entries:
        return None

    from src import config
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
    """Parse a single arXiv Atom entry into our standard format."""
    title_el = entry.find("atom:title", _NS)
    if title_el is None or not title_el.text:
        return None

    # Title comes with newlines and extra whitespace from XML
    title = re.sub(r"\s+", " ", title_el.text).strip()

    # Authors
    authors = []
    for author_el in entry.findall("atom:author", _NS):
        name_el = author_el.find("atom:name", _NS)
        if name_el is not None and name_el.text:
            authors.append(name_el.text.strip())

    # Published date → year
    published = entry.find("atom:published", _NS)
    year = None
    if published is not None and published.text:
        year_match = re.match(r"(\d{4})", published.text)
        if year_match:
            year = int(year_match.group(1))

    # arXiv ID from the entry ID URL
    # e.g. "http://arxiv.org/abs/2504.01928v1" → "2504.01928"
    arxiv_id = None
    id_el = entry.find("atom:id", _NS)
    if id_el is not None and id_el.text:
        id_match = re.search(r"(\d{4}\.\d{4,5})", id_el.text)
        if id_match:
            arxiv_id = id_match.group(1)

    # DOI link (if present)
    doi = None
    for link in entry.findall("atom:link", _NS):
        if link.get("title") == "doi":
            href = link.get("href", "")
            doi_match = re.search(r"doi\.org/(.+)", href)
            if doi_match:
                doi = doi_match.group(1)

    # Abstract
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
