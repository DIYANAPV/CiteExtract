
import logging
import os
from typing import Optional

import httpx

from citeextract import config
from citeextract.verification.api_clients.rate_limiter import fetch_with_retry
from citeextract.verification.matching import title_similarity

log = logging.getLogger(__name__)

BASE_URL = "https://api.openreview.net"


async def search_by_title(
    title: str, client: httpx.AsyncClient,
    *,
    errors: Optional[list[str]] = None,
) -> Optional[dict]:
    if os.environ.get("CITEEXTRACT_SKIP_OPENREVIEW") == "1":
        return None
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
            if errors is not None and resp.status_code >= 500:
                errors.append(f"openreview: HTTP {resp.status_code}")
            return None
    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        if errors is not None:
            errors.append(f"openreview: {type(exc).__name__}")
        return None

    try:
        notes = resp.json().get("notes", []) or []
    except Exception as exc:
        if errors is not None:
            errors.append(f"openreview: {type(exc).__name__}")
        return None
    if not notes:
        return None

    best: Optional[tuple[float, dict]] = None
    for note in notes:
        content = note.get("content", {}) or {}
        note_title = content.get("title")
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
        if pdf_field.startswith("http"):
            pdf_url = pdf_field
        else:
            pdf_url = f"https://openreview.net{pdf_field}" if pdf_field.startswith("/") else f"https://openreview.net/{pdf_field}"
    elif forum_id:
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
