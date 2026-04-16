"""Full-text retrieval for cited papers.

Waterfall strategy to get the body text of a cited paper:
  1. User-uploaded PDF (best quality, no API call)
  2. Semantic Scholar openAccessPdf (already have URL from L2)
  3. Unpaywall (by DOI)
  4. arXiv e-print / PDF (if arxiv_id present)
  5. Abstract-only fallback (from L2 ExistenceResult)

Text is extracted from PDFs via GROBID (structured TEI XML with sections).

Approach informed by:
  - Citation Integrity (Sarol et al. 2024): pre-parsed sentences
  - SemanticCite (Haan 2025): live PDF extraction
"""

import asyncio
import logging
import re
import tempfile
from pathlib import Path
from typing import Optional

import httpx
import requests

from src import config
from src.models.comprehension import FullTextResult
from src.models.verdict import ExistenceResult
from src.verification.api_clients.rate_limiter import fetch_with_retry
from src.verification.cache import APICache

log = logging.getLogger(__name__)

# Cache TTL for full text (7 days — PDFs don't change often)
TTL_FULLTEXT = 7 * 86400

# User-Agent for polite access
_UA = "CheckCitation/1.0 (academic citation verification; mailto:{email})"

# PDF download retry settings
_PDF_MAX_RETRIES = 2
_PDF_BASE_DELAY = 2.0
_PDF_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# GROBID availability flag (checked once per import / pipeline run)
_grobid_available: Optional[bool] = None


def _user_agent() -> str:
    email = config.openalex_mailto()
    return _UA.format(email=email)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def get_full_text(
    existence_result: ExistenceResult,
    client: httpx.AsyncClient,
    cache: APICache,
    user_pdf_path: Optional[str] = None,
) -> FullTextResult:
    """Retrieve full text of a cited paper via waterfall strategy.

    Uses information from L2 ExistenceResult (oa_url, doi, arxiv_id,
    abstract) to avoid redundant API lookups.

    Args:
        existence_result: L2 result for this reference (contains oa_url, doi, etc.)
        client: Shared async HTTP client.
        cache: API cache (reuses existing SQLite cache).
        user_pdf_path: Optional path to a user-uploaded PDF for this reference.

    Returns:
        FullTextResult with source, text, sections, and abstract.
    """
    ref_id = existence_result.ref_id

    # Check cache first
    cached = await cache.get(f"fulltext:{ref_id}")
    if cached:
        return FullTextResult(**cached)

    # 1. User-uploaded PDF (highest quality, no API call)
    if user_pdf_path and Path(user_pdf_path).exists():
        result = _extract_from_pdf(user_pdf_path, source="user_pdf")
        if result.full_text:
            await _cache_fulltext(cache, ref_id, result)
            return result

    # Paper not found in L2 — can only return abstract or not_found
    if existence_result.status != "FOUND":
        return FullTextResult(source="not_found", abstract=existence_result.abstract)

    # 2. Semantic Scholar openAccessPdf (URL already available from L2)
    if existence_result.oa_url:
        result = await _download_and_extract(
            existence_result.oa_url, client, source="s2_api"
        )
        if result and result.full_text:
            result.abstract = result.abstract or existence_result.abstract
            await _cache_fulltext(cache, ref_id, result)
            return result

    # 3. Unpaywall (by DOI)
    if existence_result.matched_doi:
        pdf_url = await _unpaywall_pdf_url(existence_result.matched_doi, client)
        if pdf_url:
            result = await _download_and_extract(pdf_url, client, source="unpaywall")
            if result and result.full_text:
                result.abstract = result.abstract or existence_result.abstract
                await _cache_fulltext(cache, ref_id, result)
                return result

    # 4. arXiv (if arxiv_id available on the original reference)
    arxiv_id = _get_arxiv_id(existence_result)
    if arxiv_id:
        result = await _download_and_extract_arxiv(arxiv_id, client)
        if result and result.full_text:
            result.abstract = result.abstract or existence_result.abstract
            await _cache_fulltext(cache, ref_id, result)
            return result

    # 5. Abstract-only fallback
    result = FullTextResult(
        source="abstract_only",
        abstract=existence_result.abstract,
    )
    # Only cache the fallback if the paper had no OA sources at all.
    # If it had sources but downloads failed transiently, leave uncached
    # so the next run retries fresh.
    had_oa_sources = bool(
        existence_result.oa_url
        or existence_result.matched_doi
        or arxiv_id
    )
    if not had_oa_sources:
        await _cache_fulltext(cache, ref_id, result)
    return result


# ---------------------------------------------------------------------------
# Unpaywall API
# ---------------------------------------------------------------------------


async def _unpaywall_pdf_url(doi: str, client: httpx.AsyncClient) -> Optional[str]:
    """Query Unpaywall for a PDF URL. Returns URL or None.

    Endpoint: GET https://api.unpaywall.org/v2/{doi}?email=...
    No API key needed, just a real email address.
    Rate limit: 100K/day (generous).
    """
    email = config.openalex_mailto()
    url = f"https://api.unpaywall.org/v2/{doi}"

    try:
        resp = await fetch_with_retry(
            client, "unpaywall", "GET", url,
            params={"email": email},
            headers={"User-Agent": _user_agent()},
            timeout=15,
        )
        if resp.status_code != 200:
            return None

        data = resp.json()

        # Try best_oa_location first
        best = data.get("best_oa_location") or {}
        pdf_url = best.get("url_for_pdf")
        if pdf_url:
            return pdf_url

        # Fall back to scanning all locations for direct PDF links
        for loc in data.get("oa_locations", []):
            pdf_url = loc.get("url_for_pdf")
            if pdf_url:
                return pdf_url

        # Last resort: try landing page URLs — many redirect to the actual PDF
        # and httpx follow_redirects=True in _download_and_extract handles it.
        best_url = best.get("url")
        if best_url:
            return best_url
        for loc in data.get("oa_locations", []):
            page_url = loc.get("url")
            if page_url:
                return page_url

        return None

    except (httpx.RequestError, httpx.HTTPStatusError, Exception) as e:
        log.debug(f"Unpaywall lookup failed for DOI {doi}: {e}")
        return None


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------


def _get_arxiv_id(existence_result: ExistenceResult) -> Optional[str]:
    """Extract arXiv ID from the existence result or DOI."""
    # Check matched_arxiv_id from L2 (Semantic Scholar externalIds)
    if existence_result.matched_arxiv_id:
        return existence_result.matched_arxiv_id

    # Check if the DOI is an arXiv DOI (case-insensitive)
    doi = existence_result.matched_doi or ""
    doi_lower = doi.lower()
    if doi_lower.startswith("10.48550/arxiv."):
        return doi[len("10.48550/arxiv."):]

    # Check oa_url for arXiv pattern
    oa_url = existence_result.oa_url or ""
    match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)", oa_url)
    if match:
        return match.group(1)

    return None


async def _download_and_extract_arxiv(
    arxiv_id: str, client: httpx.AsyncClient
) -> Optional[FullTextResult]:
    """Download arXiv PDF and extract text.

    Uses https://arxiv.org/pdf/{arxiv_id} directly.
    Rate limit: ~1 request per 3 seconds (be polite).
    """
    pdf_url = f"https://export.arxiv.org/pdf/{arxiv_id}"
    return await _download_and_extract(pdf_url, client, source="arxiv")


# ---------------------------------------------------------------------------
# PDF download + text extraction
# ---------------------------------------------------------------------------


async def _download_and_extract(
    pdf_url: str,
    client: httpx.AsyncClient,
    source: str,
) -> Optional[FullTextResult]:
    """Download a PDF from a URL and extract text via GROBID.

    Retries on connection errors and 5xx/429 with exponential backoff.
    """
    tmp_path: Optional[str] = None
    last_error: Optional[Exception] = None

    for attempt in range(_PDF_MAX_RETRIES + 1):
        try:
            resp = await client.get(
                pdf_url,
                headers={"User-Agent": _user_agent()},
                timeout=30,
                follow_redirects=True,
            )

            if resp.status_code in _PDF_RETRYABLE_STATUS and attempt < _PDF_MAX_RETRIES:
                delay = _PDF_BASE_DELAY * (2 ** attempt)
                log.debug(
                    f"PDF download HTTP {resp.status_code} for {pdf_url} "
                    f"(attempt {attempt + 1}/{_PDF_MAX_RETRIES + 1}), retrying in {delay:.0f}s"
                )
                await asyncio.sleep(delay)
                continue

            if resp.status_code != 200:
                log.debug(f"PDF download failed: {pdf_url} -> HTTP {resp.status_code}")
                return None

            content_type = resp.headers.get("content-type", "")
            if "pdf" not in content_type and "octet-stream" not in content_type:
                log.debug(f"Not a PDF: {pdf_url} -> {content_type}")
                return None

            # Write to temp file for parsing
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(resp.content)
                tmp_path = f.name

            return _extract_from_pdf(tmp_path, source=source)

        except httpx.RequestError as e:
            last_error = e
            if attempt < _PDF_MAX_RETRIES:
                delay = _PDF_BASE_DELAY * (2 ** attempt)
                log.debug(
                    f"PDF download error for {pdf_url}: {e} "
                    f"(attempt {attempt + 1}/{_PDF_MAX_RETRIES + 1}), retrying in {delay:.0f}s"
                )
                await asyncio.sleep(delay)
                continue
            log.debug(f"PDF download/extract failed for {pdf_url}: {e}")
            return None
        except Exception as e:
            log.debug(f"PDF download/extract failed for {pdf_url}: {e}")
            return None
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)
                tmp_path = None

    log.debug(f"PDF download exhausted retries for {pdf_url}: {last_error}")
    return None


def _check_grobid_once() -> bool:
    """Check GROBID availability once and cache the result for this process."""
    global _grobid_available
    if _grobid_available is not None:
        return _grobid_available

    cfg = config.grobid()
    try:
        health = requests.get(
            f"{cfg['service_url']}/api/isalive",
            timeout=cfg["health_check_timeout"],
        )
        _grobid_available = health.status_code == 200
    except Exception:
        _grobid_available = False

    if not _grobid_available:
        log.warning(
            "GROBID is not available — cannot extract text from PDFs. "
            "Start GROBID with: docker run -d --name grobid -p 8070:8070 grobid/grobid:0.8.2-crf"
        )
    return _grobid_available


def _extract_from_pdf(pdf_path: str, source: str) -> FullTextResult:
    """Extract text from a local PDF file using GROBID.

    GROBID must be running as a Docker container. If it's not available,
    returns not_found with a warning — the user should start GROBID.
    """
    if not _check_grobid_once():
        return FullTextResult(source="not_found")

    text, sections = _extract_via_grobid(pdf_path)
    if text:
        return FullTextResult(
            source=source,
            full_text=text,
            sections=sections,
        )

    log.debug("GROBID returned no text for PDF — possibly corrupt or empty.")
    return FullTextResult(source="not_found")


def _extract_via_grobid(pdf_path: str) -> tuple[Optional[str], list[dict]]:
    """Extract structured text via GROBID. Returns (full_text, sections).

    Reuses the GROBID service already running for L1 parsing.
    Caller must check _check_grobid_once() before calling this.
    """
    from defusedxml.ElementTree import fromstring as safe_fromstring

    cfg = config.grobid()
    service_url = cfg["service_url"]
    timeout = cfg["timeout"]

    try:
        with open(pdf_path, "rb") as f:
            resp = requests.post(
                f"{service_url}/api/processFulltextDocument",
                files={"input": f},
                timeout=timeout,
            )
        if resp.status_code != 200:
            return None, []

        # Parse TEI XML to extract sections
        tei_ns = {"tei": "http://www.tei-c.org/ns/1.0"}
        root = safe_fromstring(resp.text)
        body = root.find(".//tei:body", tei_ns)
        if body is None:
            return None, []

        sections: list[dict] = []
        current_heading = ""
        current_text_parts: list[str] = []

        for div in body.findall(".//tei:div", tei_ns):
            # Get section heading
            head = div.find("tei:head", tei_ns)
            heading = ""
            if head is not None:
                heading = "".join(head.itertext()).strip()

            # Get paragraph text
            paragraphs = []
            for p in div.findall("tei:p", tei_ns):
                p_text = "".join(p.itertext()).strip()
                if p_text:
                    paragraphs.append(p_text)

            if paragraphs:
                section_text = "\n\n".join(paragraphs)
                if heading:
                    # New section — flush previous if any
                    if current_text_parts and current_heading:
                        sections.append({
                            "name": current_heading,
                            "text": "\n\n".join(current_text_parts),
                        })
                        current_text_parts = []
                    current_heading = heading
                    current_text_parts.append(section_text)
                else:
                    current_text_parts.append(section_text)

        # Flush last section
        if current_text_parts:
            sections.append({
                "name": current_heading or "Body",
                "text": "\n\n".join(current_text_parts),
            })

        if not sections:
            return None, []

        full_text = "\n\n".join(
            f"{s['name']}\n\n{s['text']}" if s["name"] else s["text"]
            for s in sections
        )
        return full_text, sections

    except Exception as e:
        log.debug(f"GROBID extraction failed for {pdf_path}: {e}")
        return None, []


# ---------------------------------------------------------------------------
# Cache helper
# ---------------------------------------------------------------------------


async def _cache_fulltext(cache: APICache, ref_id: str, result: FullTextResult) -> None:
    """Cache a FullTextResult. Truncates full_text to avoid bloating the cache."""
    data = result.model_dump()
    # Cap cached text at 200K chars (~50 pages) to keep SQLite healthy
    if data.get("full_text") and len(data["full_text"]) > 200_000:
        data["full_text"] = data["full_text"][:200_000]
    await cache.set(f"fulltext:{ref_id}", data, TTL_FULLTEXT)
