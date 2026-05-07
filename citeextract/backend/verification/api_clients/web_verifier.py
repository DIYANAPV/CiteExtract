
import asyncio
import logging
import re
from typing import Optional

import httpx

from citeextract.verification.url_safety import is_safe_external_url

log = logging.getLogger(__name__)

_WEB_PLATFORMS = re.compile(
    r'(medium\.com|github\.com|github\.io|substack\.com|wordpress\.com|'
    r'blog\.|blogspot\.com|twitter\.com|x\.com|openai\.com/blog|'
    r'deepmind\.com/blog|ai\.googleblog\.com|huggingface\.co|'
    r'arxiv\.org/abs)',
    re.IGNORECASE,
)

_WEB_TEXT_PATTERNS = re.compile(
    r'\b(blog|website|webpage|accessed|retrieved from|available at|online)\b',
    re.IGNORECASE,
)

_LIVE_CODES = frozenset(range(200, 400)) | {403}


def classify_source_type(
    title: Optional[str],
    doi: Optional[str],
    arxiv_id: Optional[str],
    url: Optional[str],
    raw_text: str,
) -> str:
    if doi or arxiv_id:
        return "scholarly"

    if url:
        if _WEB_PLATFORMS.search(url):
            return "web"

    if _WEB_TEXT_PATTERNS.search(raw_text):
        return "web"

    if re.search(r'https?://\S+', raw_text):
        return "web"

    return "unknown"


async def verify_url(
    url: str,
    client: httpx.AsyncClient,
    timeout: float = 10.0,
) -> dict | None:
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

    try:
        resp = await client.get(
            url, timeout=timeout, follow_redirects=True,
            headers={"Range": "bytes=0-1024"},
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
    if not url:
        url_match = re.search(r'https?://\S+', raw_text)
        if url_match:
            url = url_match.group(0).rstrip('.,;)')

    if not url:
        return None

    result = await verify_url(url, client)
    if result:
        return {
            "source": "web",
            "verified_url": result["url"],
            "method": "url_resolution",
        }

    wayback = await check_wayback_machine(url, client)
    if wayback:
        return {
            "source": "web_archive",
            "verified_url": wayback["archive_url"],
            "method": "wayback_machine",
        }

    return None
