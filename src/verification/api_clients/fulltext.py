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
import time
from pathlib import Path
from typing import Optional

import httpx
import requests

from src import config
from src.models.comprehension import FetchAttempt, FullTextResult
from src.models.verdict import ExistenceResult
from src.verification.api_clients.rate_limiter import fetch_with_retry
from src.verification.url_safety import is_safe_external_url
from src.verification.cache import APICache, cited_paper_cache_key

log = logging.getLogger(__name__)

# Cache TTL for full text (7 days — PDFs don't change often)
TTL_FULLTEXT = 7 * 86400

# Soft truncation point for cached full text.
# 800K chars ≈ 200 pages of body text — covers every modern frontier-model
# paper (PaLM, Llama 3, OLMo, GPT-4 tech report) without bloating the
# SQLite cache. The previous 200K hard cap was poisoning the cache for
# every paper longer than ~50 pages.
_FULLTEXT_CACHE_MAX_CHARS = 800_000

# User-Agent for polite access
_UA = "CheckCitation/1.0 (academic citation verification; mailto:{email})"

# PDF download retry settings
_PDF_MAX_RETRIES = 2
_PDF_BASE_DELAY = 2.0
_PDF_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Maximum depth for HTML→PDF citation_pdf_url chasing. Set to 1 because
# institutional landing pages (HAL, ResearchGate, university repos) typically
# point straight at the PDF; nested chains are usually loops.
_HTML_PDF_REDIRECT_MAX_DEPTH = 1

# Google Scholar's published convention for "this page is about a paper, here
# is the PDF": ``<meta name="citation_pdf_url" content="...">``. Both attribute
# orders are seen in the wild — match permissively.
# See: https://scholar.google.com/intl/en/scholar/inclusion.html#indexing
_CITATION_PDF_URL_RE = re.compile(
    r'<meta\s+name=["\']citation_pdf_url["\']\s+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_CITATION_PDF_URL_RE_REVERSED = re.compile(
    r'<meta\s+content=["\']([^"\']+)["\']\s+name=["\']citation_pdf_url["\']',
    re.IGNORECASE,
)
# HTML body we'll scan for the meta tag — past 256K it's almost certainly
# either not a paper landing page or has the meta tag in the head already.
_HTML_PARSE_MAX_BYTES = 256 * 1024

# Per-host concurrency caps for PDF downloads. Keys are URL hosts
# (lowercased; both ``arxiv.org`` and ``export.arxiv.org`` route to the
# same arXiv backend so they share a budget). Without a cap, fanning out
# 13+ concurrent arXiv downloads triggers 429s that silently fall through
# to abstract_only — which is the leak this dict closes.
#
# Numbers picked empirically: arXiv tolerates ~4–8 sustained concurrent
# downloads before throttling kicks in. ``_PDF_DEFAULT_CONCURRENCY`` for
# unknown hosts is intentionally generous since publishers are usually fine.
_PDF_HOST_CONCURRENCY: dict[str, int] = {
    "arxiv.org": 4,
    "export.arxiv.org": 4,
    "www.arxiv.org": 4,
}
_PDF_DEFAULT_CONCURRENCY = 8

# Cache of host-keyed semaphores. Lazy-init per loop, like _grobid_extract_semaphore.
_pdf_host_semaphores: dict[str, asyncio.Semaphore] = {}

# GROBID availability flag (checked once per process — guarded by _grobid_check_lock).
_grobid_available: Optional[bool] = None

# Lazily-created on first use because asyncio.Semaphore / asyncio.Lock
# can only be constructed inside a running event loop.
_grobid_extract_semaphore: Optional[asyncio.Semaphore] = None
_grobid_check_lock: Optional[asyncio.Lock] = None


def _user_agent() -> str:
    email = config.openalex_mailto()
    return _UA.format(email=email)


def _get_extract_semaphore() -> asyncio.Semaphore:
    """Return the bounded semaphore capping concurrent GROBID extractions.

    The GROBID image ships with an internal worker pool (~10 threads).
    Capping our in-flight extractions at ``grobid.extract_concurrency``
    matches the container's capacity and prevents queue buildup that
    would otherwise surface as request timeouts.
    """
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
    """Clear cached GROBID state so tests can re-probe with a clean slate."""
    global _grobid_available, _grobid_extract_semaphore, _grobid_check_lock
    _grobid_available = None
    _grobid_extract_semaphore = None
    _grobid_check_lock = None


def _reset_pdf_host_semaphores_for_tests() -> None:
    """Clear cached per-host PDF semaphores between tests."""
    _pdf_host_semaphores.clear()


def _get_pdf_host_semaphore(pdf_url: str) -> asyncio.Semaphore:
    """Return the bounded semaphore that caps concurrent downloads to a host.

    Per-host caps keep us under arXiv's 429 ceiling without serializing —
    other hosts get a more generous default. Created lazily on first use
    because ``asyncio.Semaphore`` requires a running event loop.
    """
    from urllib.parse import urlparse

    host = (urlparse(pdf_url).hostname or "").lower()
    sem = _pdf_host_semaphores.get(host)
    if sem is None:
        cap = _PDF_HOST_CONCURRENCY.get(host, _PDF_DEFAULT_CONCURRENCY)
        sem = asyncio.Semaphore(cap)
        _pdf_host_semaphores[host] = sem
    return sem


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def get_full_text(
    existence_result: ExistenceResult,
    client: httpx.AsyncClient,
    cache: APICache,
    user_pdf_path: Optional[str] = None,
) -> FullTextResult:
    """Retrieve full text with a per-ref wall-time budget.

    Wraps the underlying waterfall in :func:`asyncio.wait_for` so a
    single ref can't burn more than ``comprehension.per_ref_fetch_timeout_s``
    seconds. On timeout we abandon any in-flight downloads and return an
    abstract-only result — the user-visible signal that the OA path took
    too long, plus a ``fetch_budget`` attempt entry for the audit trail.
    A timed-out result is NOT cached; the next run gets a fresh chance
    in case the slowness was transient (network, slow OA mirror).
    """
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
    """Underlying waterfall: user_pdf → oa_url → unpaywall → arxiv →
    s2_fallback → arxiv_fallback → abstract_only.

    Uses information from L2 ExistenceResult (oa_url, doi, arxiv_id,
    abstract) to avoid redundant API lookups. Wrapped by
    :func:`get_full_text` for the per-ref budget; tests can call this
    directly to bypass the timeout layer.

    Args:
        existence_result: L2 result for this reference (contains oa_url, doi, etc.)
        client: Shared async HTTP client.
        cache: API cache (reuses existing SQLite cache).
        user_pdf_path: Optional path to a user-uploaded PDF for this reference.

    Returns:
        FullTextResult with source, text, sections, and abstract.
    """
    ref_id = existence_result.ref_id

    # Cache key is the cited paper's content identity (DOI / arXiv / title
    # hash) — not the per-input ``ref_id``, which collides across runs and
    # would serve a previous paper's full text. ``cache_key`` is None when
    # we have no identifier at all, in which case we skip caching rather
    # than fall back to a colliding key.
    #
    # ``fulltext_v2`` bumps past the legacy ``fulltext:`` keys that were
    # written with the old 200K-char cap. Long papers (PaLM, Llama 3, OLMo)
    # got cached as ``not_found`` under the v1 prefix; the v2 keyspace skips
    # those poisoned entries entirely. ``scripts/purge_legacy_fulltext_cache.py``
    # can drop the v1 rows for an immediate disk cleanup.
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

    # Per-call trace of every waterfall step. Attached to whatever
    # FullTextResult we end up returning (and persisted in the cache).
    attempts: list[FetchAttempt] = []

    def _finalize(result: FullTextResult) -> FullTextResult:
        result.attempts = list(attempts)
        return result

    # 1. User-uploaded PDF (highest quality, no API call)
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

    # Paper not found in L2 — can only return abstract or not_found
    if existence_result.status != "FOUND":
        return _finalize(FullTextResult(
            source="not_found", abstract=existence_result.abstract,
        ))

    # 2. Semantic Scholar openAccessPdf (URL already available from L2)
    if existence_result.oa_url:
        result, atts = await _download_and_extract(
            existence_result.oa_url, client, source="oa_url",
        )
        attempts.extend(atts)
        if result and result.full_text:
            result.abstract = result.abstract or existence_result.abstract
            # NB: ``result.source`` here is whatever path actually succeeded
            # — usually ``oa_url``, but ``html_meta_pdf`` when the URL was
            # an HTML landing page that we rescued via the meta-tag scrape.
            # We deliberately preserve the rescue-path label so the public
            # source field credits the recovery rather than masquerading
            # the HAL/ResearchGate hit as a direct OA download.
            result = _finalize(result)
            await _cache_fulltext(cache, cache_key, ref_id, result)
            return result

    # 3. Unpaywall (by DOI)
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

    # 4. arXiv (if arxiv_id available on the original reference)
    arxiv_id = _get_arxiv_id(existence_result)
    if arxiv_id:
        result, atts = await _download_and_extract_arxiv(arxiv_id, client)
        attempts.extend(atts)
        if result and result.full_text:
            result.abstract = result.abstract or existence_result.abstract
            result = _finalize(result)
            await _cache_fulltext(cache, cache_key, ref_id, result)
            return result

    # 5. Semantic Scholar fallback re-query.
    #    L2 stops at the first DB that returns a strong match. When that's
    #    crossref/openalex/pubmed, S2's openAccessPdf field is never
    #    consulted — even though S2 frequently has an OA copy of the same
    #    paper. Retrying S2 here covers the dead-DOI long tail and any paper
    #    L2 happened to find via a non-S2 source.
    result = await _try_s2_fallback_pdf(existence_result, client, attempts)
    if result and result.full_text:
        result.abstract = result.abstract or existence_result.abstract
        result = _finalize(result)
        await _cache_fulltext(cache, cache_key, ref_id, result)
        return result

    # 6. arXiv-by-title fallback.
    #    No arxiv_id on the L2 record AND the DOI didn't map to one. Many
    #    papers are on arXiv but never get linked from publisher metadata
    #    (e.g. ACM DOIs without arxiv preprint cross-refs). Search arXiv
    #    by the matched title with the existing similarity threshold.
    result = await _try_arxiv_fallback_pdf(
        existence_result, arxiv_id, client, attempts,
    )
    if result and result.full_text:
        result.abstract = result.abstract or existence_result.abstract
        result = _finalize(result)
        await _cache_fulltext(cache, cache_key, ref_id, result)
        return result

    # 7. Abstract-only fallback
    attempts.append(FetchAttempt(
        source="abstract_only",
        status="ok" if existence_result.abstract else "empty_text",
    ))
    result = _finalize(FullTextResult(
        source="abstract_only",
        abstract=existence_result.abstract,
    ))
    # Only cache the fallback if the paper had no OA sources at all.
    # If it had sources but downloads failed transiently, leave uncached
    # so the next run retries fresh.
    had_oa_sources = bool(
        existence_result.oa_url
        or existence_result.matched_doi
        or arxiv_id
        or existence_result.matched_title  # S2/arxiv title fallbacks were also tried
    )
    if not had_oa_sources:
        await _cache_fulltext(cache, cache_key, ref_id, result)
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
) -> tuple[Optional[FullTextResult], list[FetchAttempt]]:
    """Download arXiv PDF and extract text.

    Uses https://export.arxiv.org/pdf/{arxiv_id} directly.
    Rate limit: ~1 request per 3 seconds (be polite).
    """
    pdf_url = f"https://export.arxiv.org/pdf/{arxiv_id}"
    return await _download_and_extract(pdf_url, client, source="arxiv")


# ---------------------------------------------------------------------------
# PDF download + text extraction
# ---------------------------------------------------------------------------


def _truncate_detail(detail: Optional[str]) -> Optional[str]:
    """Cap detail strings at 200 chars so attempts payloads stay small."""
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
    """Build a FetchAttempt, computing wall-clock from a perf_counter mark."""
    return FetchAttempt(
        source=source,
        status=status,
        ms=int((time.perf_counter() - started_at) * 1000),
        detail=_truncate_detail(detail),
    )


def _extract_citation_pdf_url(
    body: bytes, base_url: str
) -> Optional[str]:
    """Find a ``citation_pdf_url`` meta tag on an HTML landing page.

    Returns an absolute URL or ``None``. Resolves protocol-relative
    (``//host/path``) and host-relative (``/path``) URLs against the
    page that served the HTML so callers don't have to.
    """
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
    # Resolve relative forms.
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
    """Download a PDF from a URL and extract text via GROBID.

    Retries on connection errors and 5xx/429 with exponential backoff.
    Returns ``(result, attempts)`` so callers can record what happened —
    ``result`` is ``None`` on failure (network, non-PDF, GROBID). The
    ``attempts`` list normally contains one entry but is multi-element
    when the HTML-landing-page rescue chains a second download (the
    first ``not_pdf`` attempt is preserved alongside the recursive
    ``html_meta_pdf`` attempt, so the trace doesn't lose the initial
    landing-page hop).

    SSRF guard: ``pdf_url`` may originate from third-party metadata
    sources (Unpaywall, Semantic Scholar) which we don't fully trust to
    return only public URLs. Reject anything pointing at private /
    loopback / link-local / cloud-metadata addresses up front, and
    re-check the resolved URL after redirect-following.

    HTML-landing-page rescue: when the response is HTML (not a PDF), we
    look for the Google-Scholar-standard ``citation_pdf_url`` meta tag
    and recursively try the URL it points at. ``_html_depth`` caps the
    recursion at one redirect (HAL, ResearchGate, university repos
    redirect once; deeper chains are usually loops).
    """
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
            # Cap concurrent in-flight downloads per host so arXiv (and
            # similar) don't 429 us into the abstract_only fallthrough. The
            # semaphore is held only for the network GET; PDF parsing
            # afterwards has its own (GROBID) semaphore.
            async with host_sem:
                resp = await client.get(
                    pdf_url,
                    headers={"User-Agent": _user_agent()},
                    timeout=30,
                    follow_redirects=True,
                )

            # Redirects could land on a private IP — reject the response.
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
                # Honor server's Retry-After header if it asks for longer.
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
                # HTML landing pages (HAL, ResearchGate, university repos)
                # often advertise the real PDF via a ``citation_pdf_url``
                # meta tag. Try that one level deep before giving up. We
                # preserve the original ``not_pdf`` attempt alongside the
                # recursive call's attempts so the trace shows the full
                # landing-page → PDF hop, not just the rescue half.
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

            # Write to temp file for parsing
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(resp.content)
                tmp_path = f.name

            extracted = await _extract_from_pdf(tmp_path, source=source)
            if extracted.full_text:
                return extracted, [_make_attempt(source, "ok", started)]
            # GROBID returned no text — _extract_from_pdf already logged.
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


async def _ensure_grobid_available() -> bool:
    """Probe GROBID once per process; cache the result.

    Concurrent first-callers serialize through ``_grobid_check_lock`` so
    the health probe runs exactly once even when many extraction tasks
    fire off in parallel. Subsequent calls return the cached flag without
    acquiring the lock.
    """
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
    """Extract text from a local PDF file using GROBID.

    GROBID's HTTP call + XML parse is a sync block (``requests`` doesn't
    yield), so we offload it to a worker thread via ``asyncio.to_thread``
    and gate concurrency with ``_get_extract_semaphore`` to match
    GROBID's internal pool. Without that, every async caller would line
    up single-file at the blocking ``requests.post`` regardless of how
    many tasks ``asyncio.gather`` fanned out.

    Returns ``not_found`` if GROBID isn't reachable or returns no text;
    callers treat that as "no full text available" without crashing.
    """
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
    """Extract structured text via GROBID. Returns (full_text, sections).

    Synchronous on purpose — invoked from ``_extract_from_pdf`` through
    ``asyncio.to_thread`` so the blocking ``requests`` call doesn't stall
    the event loop. Reuses the GROBID service already running for L1
    parsing. Caller must check ``_ensure_grobid_available()`` first.
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
# Fallback fetch paths
# ---------------------------------------------------------------------------
#
# These run AFTER the primary cascade (S2 oa_url → Unpaywall → arXiv-by-id)
# has failed. They exist because L2's existence cascade short-circuits on
# the first matching database and never asks the others for an OA URL —
# so a paper that is fully open via S2 may be marked abstract_only just
# because L2 happened to match it via crossref first.
#
# Both fallbacks tag their successful FullTextResult with a distinct
# ``source`` label (``s2_fallback`` / ``arxiv_fallback``) so we can
# measure how often each rescues a paper that would otherwise have been
# abstract_only.


async def _try_s2_fallback_pdf(
    existence_result: ExistenceResult,
    client: httpx.AsyncClient,
    attempts: list[FetchAttempt],
) -> Optional[FullTextResult]:
    """Re-query Semantic Scholar for openAccessPdf when L2 didn't go through it.

    Skipped when:
      - L2 already used S2 (we'd be re-asking the same DB for the same answer)
      - L2 source is "web" (the cited thing isn't a scholarly paper — a blog
        post or social-media URL — so an S2 lookup is pointless)
      - We have no DOI and no title to search by

    Returns a FullTextResult with full_text on success, or None on miss /
    download failure (caller proceeds to the next fallback step).
    Appends a FetchAttempt to ``attempts`` describing the outcome.
    """
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

    from src.verification.api_clients import semantic_scholar

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
    """Search arXiv by title for papers with no arxiv_id on the L2 record.

    Skipped when:
      - We already tried an arxiv_id in the primary cascade
      - L2 source is already "arxiv" (no point re-asking) or "web"
      - There is no matched_title to search by

    The underlying ``arxiv.search_by_title`` already enforces the
    configured title-similarity threshold, so a weak match silently
    returns None instead of fetching an unrelated paper.
    """
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

    from src.verification.api_clients import arxiv as arxiv_client

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
        # Distinct source label so logs / cache audits can attribute the
        # rescue to the title-search fallback rather than a primary hit.
        result.source = "arxiv_fallback"
        return result
    return None


# ---------------------------------------------------------------------------
# Cache helper
# ---------------------------------------------------------------------------


async def _cache_fulltext(
    cache: APICache,
    cache_key: Optional[str],
    ref_id: str,
    result: FullTextResult,
) -> None:
    """Persist a FullTextResult under its content-identity cache key.

    ``cache_key`` may be None when the cited paper has no DOI, arXiv ID,
    or title to identify it (only happens for very malformed references).
    In that case we silently skip caching — better to refetch on the next
    run than risk serving the wrong paper's content under a colliding
    per-input ``ref_id`` key.

    ``ref_id`` is kept only for the size-limit log message so it remains
    traceable to the source reference.
    """
    if cache_key is None:
        return
    data = result.model_dump()
    # Soft truncation: legitimate frontier-model papers (PaLM, Llama 3,
    # OLMo) routinely exceed 200K chars and the previous hard cap was
    # writing them back as ``source: not_found``, which silently broke
    # passage retrieval on every subsequent run. We now keep the first
    # ``_FULLTEXT_CACHE_MAX_CHARS`` and tag ``truncated=True`` so the
    # downstream chunker still has plenty of body to work with and
    # consumers can tell that they're looking at a partial copy.
    full_text = data.get("full_text") or ""
    if full_text and len(full_text) > _FULLTEXT_CACHE_MAX_CHARS:
        original_len = len(full_text)
        data["full_text"] = full_text[:_FULLTEXT_CACHE_MAX_CHARS]
        data["truncated"] = True
        log.info(
            f"Full text for {ref_id} truncated from {original_len} to "
            f"{_FULLTEXT_CACHE_MAX_CHARS} chars before caching."
        )
        # Sections may also be huge — drop any whose ``text`` runs past
        # the truncation point so the cached entry doesn't carry duplicate
        # body text in two places. The chunker reads ``full_text`` first.
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
