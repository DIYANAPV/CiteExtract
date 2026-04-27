"""OpenReview API client — final-resort title search for tech reports.

Closes the L2 blind spot where a paper exists nowhere in CrossRef / S2 /
OpenAlex / PubMed / arXiv because it was published as an OpenReview tech
report rather than via a journal/conference proceedings record. Concrete
example: LeCun's "A Path Towards Autonomous Machine Intelligence" sits at
``https://openreview.net/forum?id=BZ5a1r-kVsf`` with no DOI anywhere — only
this client can find it.

We use the v1 endpoint (``api.openreview.net``) rather than v2 because v1
returns ``content.title`` as a plain string. v2 ships titles wrapped in
``{"value": "..."}`` envelopes which were inconsistent across record
types in our probe runs.

Free, no API key required. Rate limited to 3 req/s in our config to be a
polite citizen.
"""

import logging
from typing import Optional

import httpx

from src import config
from src.verification.api_clients.rate_limiter import fetch_with_retry
from src.verification.matching import title_similarity

log = logging.getLogger(__name__)

BASE_URL = "https://api.openreview.net"


async def search_by_title(
    title: str, client: httpx.AsyncClient
) -> Optional[dict]:
    """Search OpenReview by title. Returns best matching paper or None.

    Filters candidates by the global ``thresholds.title_match`` (0.80) so
    weak matches don't surface as false positives — OpenReview indexes
    every comment and review alongside the actual papers, and the search
    endpoint mixes them all together.
    """
    if not title or len(title.strip()) < 5:
        return None

    cfg = config.api("openreview") or {}
    timeout = cfg.get("timeout", 15)
    threshold = config.thresholds()["title_match"]

    try:
        resp = await fetch_with_retry(
            client, "openreview", "GET",
            f"{BASE_URL}/notes/search",
            params={"term": title, "limit": 10},
            timeout=timeout,
        )
        if resp.status_code != 200:
            return None
    except (httpx.HTTPStatusError, httpx.RequestError):
        return None

    try:
        notes = resp.json().get("notes", []) or []
    except Exception:
        return None
    if not notes:
        return None

    # Walk the notes and keep the highest-similarity hit that clears the
    # threshold. Many OpenReview notes are reviews / comments without a
    # ``title`` field — skip them silently.
    best: Optional[tuple[float, dict]] = None
    for note in notes:
        content = note.get("content", {}) or {}
        note_title = content.get("title")
        # v2 sometimes returns ``{"value": "..."}``; tolerate both forms.
        if isinstance(note_title, dict):
            note_title = note_title.get("value")
        if not note_title:
            continue
        sim = title_similarity(title, note_title)
        if sim < threshold:
            continue
        if best is None or sim > best[0]:
            best = (sim, note)

    if best is None:
        return None

    sim, note = best
    return _parse_note(note, sim)


def _parse_note(note: dict, similarity: float) -> dict:
    """Parse one OpenReview ``note`` into our standard record shape."""
    content = note.get("content", {}) or {}

    def _val(field: str) -> Optional[object]:
        v = content.get(field)
        if isinstance(v, dict):
            return v.get("value")
        return v

    title = _val("title") or ""
    authors_field = _val("authors") or []
    if isinstance(authors_field, str):
        authors = [a.strip() for a in authors_field.split(",") if a.strip()]
    elif isinstance(authors_field, list):
        authors = [a for a in authors_field if isinstance(a, str)]
    else:
        authors = []

    abstract = _val("abstract")

    # OpenReview notes carry a ``cdate`` timestamp (millis since epoch).
    year: Optional[int] = None
    cdate = note.get("cdate")
    if isinstance(cdate, (int, float)) and cdate > 0:
        try:
            from datetime import datetime, timezone
            year = datetime.fromtimestamp(cdate / 1000, tz=timezone.utc).year
        except Exception:
            year = None

    venue = _val("venue") or "OpenReview"
    forum_id = note.get("forum") or note.get("id")
    landing_url = (
        f"https://openreview.net/forum?id={forum_id}" if forum_id else None
    )
    pdf_field = _val("pdf")
    pdf_url: Optional[str] = None
    if isinstance(pdf_field, str) and pdf_field:
        # OpenReview stores PDFs at ``/pdf?id=<forum>`` for v1 entries; the
        # ``pdf`` content field is the relative path or a full URL depending
        # on age. Normalize.
        if pdf_field.startswith("http"):
            pdf_url = pdf_field
        else:
            pdf_url = f"https://openreview.net{pdf_field}" if pdf_field.startswith("/") else f"https://openreview.net/{pdf_field}"
    elif forum_id:
        # Most OpenReview papers are openly accessible at this URL pattern.
        pdf_url = f"https://openreview.net/pdf?id={forum_id}"

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "venue": str(venue) if venue else "OpenReview",
        "abstract": abstract,
        "doi": None,
        "arxiv_id": None,
        "oa_url": pdf_url or landing_url,
        "title_similarity": similarity,
    }
