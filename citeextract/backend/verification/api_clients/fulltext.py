
import asyncio
import logging
import re
import tempfile
import time
from pathlib import Path
from typing import Optional

import httpx
import requests

from citeextract import config
from citeextract.models.comprehension import FetchAttempt, FullTextResult
from citeextract.models.verdict import ExistenceResult
from citeextract.verification.api_clients.rate_limiter import fetch_with_retry
from citeextract.verification.url_safety import is_safe_external_url
from citeextract.verification.cache import APICache, cited_paper_cache_key

log = logging.getLogger(__name__)

TTL_FULLTEXT = 7 * 86400

_FULLTEXT_CACHE_MAX_CHARS = 800_000

_UA = "CiteExtract/1.0 (academic citation verification; mailto:{email})"

_PDF_MAX_RETRIES = 2
_PDF_BASE_DELAY = 2.0
_PDF_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

_HTML_PDF_REDIRECT_MAX_DEPTH = 1

_CITATION_PDF_URL_RE = re.compile(
    r'<meta\s+name=["\']citation_pdf_url["\']\s+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_CITATION_PDF_URL_RE_REVERSED = re.compile(
    r'<meta\s+content=["\']([^"\']+)["\']\s+name=["\']citation_pdf_url["\']',
    re.IGNORECASE,
)
_HTML_PARSE_MAX_BYTES = 256 * 1024

_PDF_HOST_CONCURRENCY: dict[str, int] = {
    "arxiv.org": 4,
    "export.arxiv.org": 4,
    "www.arxiv.org": 4,
}
_PDF_DEFAULT_CONCURRENCY = 8

_pdf_host_semaphores: dict[str, asyncio.Semaphore] = {}

_grobid_available: Optional[bool] = None

_grobid_extract_semaphore: Optional[asyncio.Semaphore] = None
_grobid_check_lock: Optional[asyncio.Lock] = None


def _user_agent() -> str:
    email = config.openalex_mailto()
    return _UA.format(email=email)


def _get_extract_semaphore() -> asyncio.Semaphore:
    global _grobid_extract_semaphore
    if _grobid_extract_semaphore is None:
        cfg = config.grobid()
        n = int(cfg.get("extract_concurrency", cfg.get("concurrency", 4)))
        _grobid_extract_semaphore = asyncio.Semaphore(n)
    return _grobid_extract_semaphore


def _get_check_lock() -> asyncio.Lock:
    global _grobid_check_lock
    if _grobid_check_lock is None:
        _grobid_check_lock = asyncio.Lock()
    return _grobid_check_lock


def _reset_grobid_state_for_tests() -> None:
    global _grobid_available, _grobid_extract_semaphore, _grobid_check_lock
    _grobid_available = None
    _grobid_extract_semaphore = None
    _grobid_check_lock = None


def _reset_pdf_host_semaphores_for_tests() -> None:
    _pdf_host_semaphores.clear()


def _get_pdf_host_semaphore(pdf_url: str) -> asyncio.Semaphore:
    from urllib.parse import urlparse

    host = (urlparse(pdf_url).hostname or "").lower()
    sem = _pdf_host_semaphores.get(host)
    if sem is None:
        cap = _PDF_HOST_CONCURRENCY.get(host, _PDF_DEFAULT_CONCURRENCY)
        sem = asyncio.Semaphore(cap)
        _pdf_host_semaphores[host] = sem
    return sem


async def get_full_text(
    existence_result: ExistenceResult,
    client: httpx.AsyncClient,
    cache: APICache,
    user_pdf_path: Optional[str] = None,
) -> FullTextResult:
    timeout_s = float(
        config.comprehension().get("per_ref_fetch_timeout_s", 10)
    )
    try:
        return await asyncio.wait_for(
            _run_waterfall(existence_result, client, cache, user_pdf_path),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        log.warning(
            "get_full_text: ref_id=%s exceeded %.0fs fetch budget; "
            "returning abstract_only.",
            existence_result.ref_id, timeout_s,
        )
        return FullTextResult(
            source="abstract_only",
            abstract=existence_result.abstract,
            attempts=[FetchAttempt(
                source="fetch_budget",
                status="timeout",
                detail=f"per_ref_fetch_timeout_s={timeout_s:g}",
            )],
        )


async def _run_waterfall(
    existence_result: ExistenceResult,
    client: httpx.AsyncClient,
    cache: APICache,
    user_pdf_path: Optional[str] = None,
) -> FullTextResult:
    ref_id = existence_result.ref_id

    cache_key = cited_paper_cache_key(
        "fulltext_v2",
        doi=existence_result.matched_doi,
        arxiv_id=existence_result.matched_arxiv_id,
        title=existence_result.matched_title,
    )

    if cache_key:
        cached = await cache.get(cache_key)
        if cached:
            return FullTextResult(**cached)

    attempts: list[FetchAttempt] = []

    def _finalize(result: FullTextResult) -> FullTextResult:
        result.attempts = list(attempts)
        return result

    if user_pdf_path and Path(user_pdf_path).exists():
        started = time.perf_counter()
        result = await _extract_from_pdf(user_pdf_path, source="user_pdf")
        if result.full_text:
            attempts.append(_make_attempt("user_pdf", "ok", started))
            result = _finalize(result)
            await _cache_fulltext(cache, cache_key, ref_id, result)
            return result
        attempts.append(_make_attempt(
            "user_pdf",
            "grobid_failed" if result.source == "not_found" else "empty_text",
            started,
            user_pdf_path,
        ))

    if existence_result.status != "FOUND":
        return _finalize(FullTextResult(
            source="not_found", abstract=existence_result.abstract,
        ))

    if existence_result.oa_url:
        result, atts = await _download_and_extract(
            existence_result.oa_url, client, source="oa_url",
        )
        attempts.extend(atts)
        if result and result.full_text:
            result.abstract = result.abstract or existence_result.abstract
            result = _finalize(result)
            await _cache_fulltext(cache, cache_key, ref_id, result)
            return result

    if existence_result.matched_doi:
        pdf_url = await _unpaywall_pdf_url(existence_result.matched_doi, client)
        if pdf_url:
            result, atts = await _download_and_extract(
                pdf_url, client, source="unpaywall",
            )
            attempts.extend(atts)
            if result and result.full_text:
                result.abstract = result.abstract or existence_result.abstract
                result = _finalize(result)
                await _cache_fulltext(cache, cache_key, ref_id, result)
                return result
        else:
            attempts.append(FetchAttempt(
                source="unpaywall", status="skipped",
                detail="no_oa_pdf_url",
            ))

    arxiv_id = _get_arxiv_id(existence_result)
    if arxiv_id:
        result, atts = await _download_and_extract_arxiv(arxiv_id, client)
        attempts.extend(atts)
        if result and result.full_text:
            result.abstract = result.abstract or existence_result.abstract
            result = _finalize(result)
            await _cache_fulltext(cache, cache_key, ref_id, result)
            return result

    result = await _try_s2_fallback_pdf(existence_result, client, attempts)
    if result and result.full_text:
        result.abstract = result.abstract or existence_result.abstract
        result = _finalize(result)
        await _cache_fulltext(cache, cache_key, ref_id, result)
        return result

    result = await _try_arxiv_fallback_pdf(
        existence_result, arxiv_id, client, attempts,
    )
    if result and result.full_text:
        result.abstract = result.abstract or existence_result.abstract
        result = _finalize(result)
        await _cache_fulltext(cache, cache_key, ref_id, result)
        return result

    attempts.append(FetchAttempt(
        source="abstract_only",
        status="ok" if existence_result.abstract else "empty_text",
    ))
    result = _finalize(FullTextResult(
        source="abstract_only",
        abstract=existence_result.abstract,
    ))
    had_oa_sources = bool(
        existence_result.oa_url
        or existence_result.matched_doi
        or arxiv_id
        or existence_result.matched_title
    )
    if not had_oa_sources:
        await _cache_fulltext(cache, cache_key, ref_id, result)
    return result


async def _unpaywall_pdf_url(doi: str, client: httpx.AsyncClient) -> Optional[str]:
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

        best = data.get("best_oa_location") or {}
        pdf_url = best.get("url_for_pdf")
        if pdf_url:
            return pdf_url

        for loc in data.get("oa_locations", []):
            pdf_url = loc.get("url_for_pdf")
            if pdf_url:
                return pdf_url

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


def _get_arxiv_id(existence_result: ExistenceResult) -> Optional[str]:
    if existence_result.matched_arxiv_id:
        return existence_result.matched_arxiv_id

    doi = existence_result.matched_doi or ""
    doi_lower = doi.lower()
    if doi_lower.startswith("10.48550/arxiv."):
        return doi[len("10.48550/arxiv."):]

    oa_url = existence_result.oa_url or ""
    match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)", oa_url)
    if match:
        return match.group(1)

    return None


async def _download_and_extract_arxiv(
    arxiv_id: str, client: httpx.AsyncClient
) -> tuple[Optional[FullTextResult], list[FetchAttempt]]:
    pdf_url = f"https://export.arxiv.org/pdf/{arxiv_id}"
    return await _download_and_extract(pdf_url, client, source="arxiv")


def _truncate_detail(detail: Optional[str]) -> Optional[str]:
    if detail is None:
        return None
    s = str(detail)
    return s if len(s) <= 200 else s[:197] + "..."


def _make_attempt(
    source: str,
    status: str,
    started_at: float,
    detail: Optional[str] = None,
) -> FetchAttempt:
    return FetchAttempt(
        source=source,
        status=status,
        ms=int((time.perf_counter() - started_at) * 1000),
        detail=_truncate_detail(detail),
    )


def _extract_citation_pdf_url(
    body: bytes, base_url: str
) -> Optional[str]:
    if not body:
        return None
    snippet = body[:_HTML_PARSE_MAX_BYTES]
    try:
        html = snippet.decode("utf-8", errors="replace")
    except Exception:
        return None
    m = _CITATION_PDF_URL_RE.search(html) or _CITATION_PDF_URL_RE_REVERSED.search(html)
    if not m:
        return None
    pdf_url = m.group(1).strip()
    if not pdf_url:
        return None
    if pdf_url.startswith("//"):
        pdf_url = "https:" + pdf_url
    elif pdf_url.startswith("/"):
        from urllib.parse import urlparse

        parsed = urlparse(base_url)
        if parsed.scheme and parsed.netloc:
            pdf_url = f"{parsed.scheme}://{parsed.netloc}{pdf_url}"
    return pdf_url


async def _download_and_extract(
    pdf_url: str,
    client: httpx.AsyncClient,
    source: str,
    _html_depth: int = 0,
) -> tuple[Optional[FullTextResult], list[FetchAttempt]]:
    started = time.perf_counter()

    if not await asyncio.to_thread(is_safe_external_url, pdf_url):
        log.warning(f"PDF URL rejected as unsafe (private/non-http): {pdf_url}")
        return None, [_make_attempt(source, "unsafe_url", started, pdf_url)]

    tmp_path: Optional[str] = None
    last_error: Optional[Exception] = None
    last_status: Optional[int] = None
    host_sem = _get_pdf_host_semaphore(pdf_url)

    for attempt in range(_PDF_MAX_RETRIES + 1):
        try:
            async with host_sem:
                resp = await client.get(
                    pdf_url,
                    headers={"User-Agent": _user_agent()},
                    timeout=30,
                    follow_redirects=True,
                )

            if not is_safe_external_url(str(resp.url)):
                log.warning(
                    f"PDF download redirected to unsafe URL "
                    f"(initial={pdf_url}, final={resp.url})"
                )
                return None, [_make_attempt(
                    source, "unsafe_url", started,
                    f"redirect_to:{resp.url}",
                )]

            last_status = resp.status_code

            if resp.status_code in _PDF_RETRYABLE_STATUS and attempt < _PDF_MAX_RETRIES:
                delay = _PDF_BASE_DELAY * (2 ** attempt)
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass
                log.debug(
                    f"PDF download HTTP {resp.status_code} for {pdf_url} "
                    f"(attempt {attempt + 1}/{_PDF_MAX_RETRIES + 1}), retrying in {delay:.0f}s"
                )
                await asyncio.sleep(delay)
                continue

            if resp.status_code != 200:
                log.debug(f"PDF download failed: {pdf_url} -> HTTP {resp.status_code}")
                return None, [_make_attempt(
                    source, "http_error", started,
                    f"http_{resp.status_code}",
                )]

            content_type = resp.headers.get("content-type", "")
            if "pdf" not in content_type and "octet-stream" not in content_type:
                landing_attempt = _make_attempt(
                    source, "not_pdf", started,
                    f"content-type:{content_type}|url:{resp.url}",
                )
                if (
                    _html_depth < _HTML_PDF_REDIRECT_MAX_DEPTH
                    and ("html" in content_type or "text" in content_type)
                ):
                    embedded = _extract_citation_pdf_url(resp.content, str(resp.url))
                    if embedded and embedded != pdf_url:
                        log.debug(
                            f"HTML landing page at {pdf_url} advertised "
                            f"citation_pdf_url={embedded} — recursing."
                        )
                        result, child_attempts = await _download_and_extract(
                            embedded,
                            client,
                            source="html_meta_pdf",
                            _html_depth=_html_depth + 1,
                        )
                        return result, [landing_attempt, *child_attempts]
                log.debug(f"Not a PDF: {pdf_url} -> {content_type}")
                return None, [landing_attempt]

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(resp.content)
                tmp_path = f.name

            extracted = await _extract_from_pdf(tmp_path, source=source)
            if extracted.full_text:
                return extracted, [_make_attempt(source, "ok", started)]
            return extracted, [_make_attempt(
                source,
                "grobid_failed" if extracted.source == "not_found" else "empty_text",
                started,
            )]

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
            status = "timeout" if isinstance(e, httpx.TimeoutException) else "http_error"
            return None, [_make_attempt(source, status, started, f"{type(e).__name__}: {e}")]
        except (httpx.HTTPError, OSError, ValueError) as e:
            log.warning(
                f"PDF download/extract failed for {pdf_url}: "
                f"{type(e).__name__}: {e}"
            )
            return None, [_make_attempt(source, "http_error", started, f"{type(e).__name__}: {e}")]
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)
                tmp_path = None

    log.debug(f"PDF download exhausted retries for {pdf_url}: {last_error}")
    detail = f"retries_exhausted:last_status={last_status}"
    if last_error is not None:
        detail += f":{type(last_error).__name__}"
    return None, [_make_attempt(source, "http_error", started, detail)]


def grobid_was_probed_unavailable() -> bool:
    return _grobid_available is False


async def _ensure_grobid_available() -> bool:
    global _grobid_available
    if _grobid_available is not None:
        return _grobid_available

    async with _get_check_lock():
        if _grobid_available is not None:
            return _grobid_available

        cfg = config.grobid()
        try:
            health = await asyncio.to_thread(
                requests.get,
                f"{cfg['service_url']}/api/isalive",
                timeout=cfg["health_check_timeout"],
            )
            _grobid_available = health.status_code == 200
        except (requests.RequestException, OSError):
            _grobid_available = False

        if not _grobid_available:
            log.error(
                "GROBID is NOT running — PDF text extraction will fail. "
                "Start GROBID with: docker run -d --name grobid -p 8070:8070 grobid/grobid:0.8.2-crf"
            )
        return _grobid_available


async def _extract_from_pdf(pdf_path: str, source: str) -> FullTextResult:
    if not await _ensure_grobid_available():
        return FullTextResult(source="not_found")

    async with _get_extract_semaphore():
        text, sections = await asyncio.to_thread(_extract_via_grobid, pdf_path)

    if text:
        return FullTextResult(
            source=source,
            full_text=text,
            sections=sections,
        )

    log.debug("GROBID returned no text for PDF — possibly corrupt or empty.")
    return FullTextResult(source="not_found")


def _extract_via_grobid(pdf_path: str) -> tuple[Optional[str], list[dict]]:
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

        tei_ns = {"tei": "http://www.tei-c.org/ns/1.0"}
        root = safe_fromstring(resp.text)
        body = root.find(".//tei:body", tei_ns)
        if body is None:
            return None, []

        sections: list[dict] = []
        current_heading = ""
        current_text_parts: list[str] = []

        for div in body.findall(".//tei:div", tei_ns):
            head = div.find("tei:head", tei_ns)
            heading = ""
            if head is not None:
                heading = "".join(head.itertext()).strip()

            paragraphs = []
            for p in div.findall("tei:p", tei_ns):
                p_text = "".join(p.itertext()).strip()
                if p_text:
                    paragraphs.append(p_text)

            if paragraphs:
                section_text = "\n\n".join(paragraphs)
                if heading:
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


async def _try_s2_fallback_pdf(
    existence_result: ExistenceResult,
    client: httpx.AsyncClient,
    attempts: list[FetchAttempt],
) -> Optional[FullTextResult]:
    if existence_result.source in {"semantic_scholar", "web"}:
        attempts.append(FetchAttempt(
            source="s2_fallback", status="skipped",
            detail=f"primary_source={existence_result.source}",
        ))
        return None
    if not existence_result.matched_doi and not existence_result.matched_title:
        attempts.append(FetchAttempt(
            source="s2_fallback", status="skipped", detail="no_doi_or_title",
        ))
        return None

    from citeextract.verification.api_clients import semantic_scholar

    started = time.perf_counter()
    paper: Optional[dict] = None
    if existence_result.matched_doi:
        paper = await semantic_scholar.lookup_by_id(
            f"DOI:{existence_result.matched_doi}", client,
        )
    if paper is None and existence_result.matched_title:
        paper = await semantic_scholar.search_by_title(
            existence_result.matched_title, client,
        )
    if not paper:
        attempts.append(_make_attempt("s2_fallback", "http_error", started, "s2_lookup_miss"))
        return None

    oa = paper.get("openAccessPdf")
    pdf_url = oa.get("url") if isinstance(oa, dict) else None
    if not pdf_url:
        attempts.append(_make_attempt("s2_fallback", "skipped", started, "no_openAccessPdf"))
        return None

    result, atts = await _download_and_extract(pdf_url, client, source="s2_fallback")
    attempts.extend(atts)
    if result and result.full_text:
        return result
    return None


async def _try_arxiv_fallback_pdf(
    existence_result: ExistenceResult,
    arxiv_id_already_tried: Optional[str],
    client: httpx.AsyncClient,
    attempts: list[FetchAttempt],
) -> Optional[FullTextResult]:
    if arxiv_id_already_tried:
        attempts.append(FetchAttempt(
            source="arxiv_fallback", status="skipped",
            detail="arxiv_id_already_tried",
        ))
        return None
    if existence_result.source in {"arxiv", "web"}:
        attempts.append(FetchAttempt(
            source="arxiv_fallback", status="skipped",
            detail=f"primary_source={existence_result.source}",
        ))
        return None
    if not existence_result.matched_title:
        attempts.append(FetchAttempt(
            source="arxiv_fallback", status="skipped",
            detail="no_matched_title",
        ))
        return None

    from citeextract.verification.api_clients import arxiv as arxiv_client

    started = time.perf_counter()
    paper = await arxiv_client.search_by_title(
        existence_result.matched_title, client,
    )
    if not paper or not paper.get("arxiv_id"):
        attempts.append(_make_attempt(
            "arxiv_fallback", "skipped", started, "title_search_miss",
        ))
        return None

    result, atts = await _download_and_extract_arxiv(paper["arxiv_id"], client)
    attempts.extend(atts)
    if result and result.full_text:
        result.source = "arxiv_fallback"
        return result
    return None


async def _cache_fulltext(
    cache: APICache,
    cache_key: Optional[str],
    ref_id: str,
    result: FullTextResult,
) -> None:
    if cache_key is None:
        return
    data = result.model_dump()
    full_text = data.get("full_text") or ""
    if full_text and len(full_text) > _FULLTEXT_CACHE_MAX_CHARS:
        original_len = len(full_text)
        data["full_text"] = full_text[:_FULLTEXT_CACHE_MAX_CHARS]
        data["truncated"] = True
        log.info(
            f"Full text for {ref_id} truncated from {original_len} to "
            f"{_FULLTEXT_CACHE_MAX_CHARS} chars before caching."
        )
        kept_sections: list[dict] = []
        running_len = 0
        for sec in data.get("sections", []):
            sec_text = sec.get("text", "") or ""
            if running_len + len(sec_text) > _FULLTEXT_CACHE_MAX_CHARS:
                break
            kept_sections.append(sec)
            running_len += len(sec_text)
        data["sections"] = kept_sections
    await cache.set(cache_key, data, TTL_FULLTEXT)
