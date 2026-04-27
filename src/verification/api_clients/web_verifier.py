"""Web verification for non-scholarly sources.

Verifies that blog posts, tech reports, and other web sources actually exist
by checking if their URL resolves and/or if the Wayback Machine has archived them.

No API keys required. Uses existing httpx async client.
"""

import asyncio
import logging
import re
from typing import Optional

import httpx

from src.verification.url_safety import is_safe_external_url

log = logging.getLogger(__name__)

# Known non-scholarly web platforms
_WEB_PLATFORMS = re.compile(
    r'(medium\.com|github\.com|github\.io|substack\.com|wordpress\.com|'
    r'blog\.|blogspot\.com|twitter\.com|x\.com|openai\.com/blog|'
    r'deepmind\.com/blog|ai\.googleblog\.com|huggingface\.co|'
    r'arxiv\.org/abs)',
    re.IGNORECASE,
)

# Patterns in raw_text suggesting a web source
_WEB_TEXT_PATTERNS = re.compile(
    r'\b(blog|website|webpage|accessed|retrieved from|available at|online)\b',
    re.IGNORECASE,
)

# A URL is "live" if the server responds — even with 403 (access denied).
# 403 means the server EXISTS and recognized the URL, just blocked our bot.
# This is common for OpenAI, Twitter, etc. Only 404/410 means "page gone."
# Connection errors mean the server doesn't exist at all.
_LIVE_CODES = frozenset(range(200, 400)) | {403}  # 2xx, 3xx, and 403


def classify_source_type(
    title: Optional[str],
    doi: Optional[str],
    arxiv_id: Optional[str],
    url: Optional[str],
    raw_text: str,
) -> str:
    """Classify a reference as scholarly, web, or unknown.

    Returns:
        "scholarly" — has DOI or arXiv ID (use scholarly DB cascade)
        "web"       — has URL or web indicators (use web verification)
        "unknown"   — try scholarly first, web fallback if NOT_FOUND
    """
    if doi or arxiv_id:
        return "scholarly"

    if url:
        if _WEB_PLATFORMS.search(url):
            return "web"
        # URL present but could be a publisher site — still "unknown"
        # so scholarly DBs get tried first

    if _WEB_TEXT_PATTERNS.search(raw_text):
        return "web"

    # Check if raw_text contains a URL even if not extracted to url field
    if re.search(r'https?://\S+', raw_text):
        return "web"

    return "unknown"


async def verify_url(
    url: str,
    client: httpx.AsyncClient,
    timeout: float = 10.0,
) -> dict | None:
    """Check if a URL resolves to a live page.

    Returns dict with page info if reachable, None otherwise. URLs that
    point at private, loopback, link-local, or otherwise non-public
    addresses are rejected up front to prevent SSRF: a reference of the
    form `http://169.254.169.254/...` must not be probed.
    """
    if not await asyncio.to_thread(is_safe_external_url, url):
        log.debug(f"URL rejected as unsafe (private/non-http): {url}")
        return None

    try:
        resp = await client.head(url, timeout=timeout, follow_redirects=True)
        if resp.status_code in _LIVE_CODES and is_safe_external_url(str(resp.url)):
            return {
                "url": str(resp.url),
                "status_code": resp.status_code,
                "content_type": resp.headers.get("content-type", ""),
            }
    except (httpx.HTTPError, httpx.InvalidURL) as e:
        log.debug(f"URL check failed for {url}: {e}")

    # Try GET as fallback (some servers don't support HEAD)
    try:
        resp = await client.get(
            url, timeout=timeout, follow_redirects=True,
            headers={"Range": "bytes=0-1024"},  # limit download
        )
        if resp.status_code in _LIVE_CODES and is_safe_external_url(str(resp.url)):
            return {
                "url": str(resp.url),
                "status_code": resp.status_code,
                "content_type": resp.headers.get("content-type", ""),
            }
    except (httpx.HTTPError, httpx.InvalidURL) as e:
        log.debug(f"URL GET fallback failed for {url}: {e}")

    return None


async def check_wayback_machine(
    url: str,
    client: httpx.AsyncClient,
    timeout: float = 10.0,
) -> dict | None:
    """Check if a URL is archived in the Wayback Machine.

    Free API, no key required.
    Returns dict with archive info if found, None otherwise.
    """
    api_url = f"https://archive.org/wayback/available?url={url}"
    try:
        resp = await client.get(api_url, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            snapshots = data.get("archived_snapshots", {})
            closest = snapshots.get("closest")
            if closest and closest.get("available"):
                return {
                    "archive_url": closest["url"],
                    "timestamp": closest.get("timestamp", ""),
                    "status": closest.get("status", ""),
                }
    except httpx.HTTPError as e:
        log.debug(f"Wayback Machine check failed for {url}: {e}")

    return None


async def verify_web_source(
    url: Optional[str],
    raw_text: str,
    client: httpx.AsyncClient,
) -> dict | None:
    """Verify a web source exists via URL resolution or Wayback Machine.

    Tries:
    1. Direct URL check (if URL provided)
    2. Wayback Machine (if URL provided but dead, or extracted from raw_text)

    Returns dict with verification info if found, None if not verifiable.
    """
    # Extract URL from raw_text if not provided
    if not url:
        url_match = re.search(r'https?://\S+', raw_text)
        if url_match:
            url = url_match.group(0).rstrip('.,;)')

    if not url:
        return None

    # Try direct URL
    result = await verify_url(url, client)
    if result:
        return {
            "source": "web",
            "verified_url": result["url"],
            "method": "url_resolution",
        }

    # Try Wayback Machine
    wayback = await check_wayback_machine(url, client)
    if wayback:
        return {
            "source": "web_archive",
            "verified_url": wayback["archive_url"],
            "method": "wayback_machine",
        }

    return None
