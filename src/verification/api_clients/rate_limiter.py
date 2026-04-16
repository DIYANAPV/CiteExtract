"""Global async rate limiters and retry logic shared across all API clients.

Each API has its own limiter. Limiters are created per event loop
to avoid cross-loop reuse issues.
"""

import asyncio
import logging
from typing import Callable, Optional, TypeVar

import httpx
from aiolimiter import AsyncLimiter

from src import config

log = logging.getLogger(__name__)

T = TypeVar("T")

# Rate limit configs loaded lazily on first access (not at import time)
_CONFIGS: Optional[dict[str, tuple[int, int]]] = None


def _get_configs() -> dict[str, tuple[int, int]]:
    global _CONFIGS
    if _CONFIGS is None:
        _CONFIGS = config.rate_limits()
    return _CONFIGS

# Cache limiters per event loop
_limiters: dict[int, dict[str, AsyncLimiter]] = {}


def _get_limiter(api_name: str) -> Optional[AsyncLimiter]:
    """Get or create a rate limiter for the current event loop."""
    config = _get_configs().get(api_name)
    if not config:
        return None

    try:
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
    except RuntimeError:
        return None

    # Clean up limiters from previous (now-dead) event loops.
    # Keep only the current loop's limiters to prevent memory leaks.
    stale = [lid for lid in _limiters if lid != loop_id]
    for lid in stale:
        del _limiters[lid]

    if loop_id not in _limiters:
        _limiters[loop_id] = {}

    if api_name not in _limiters[loop_id]:
        _limiters[loop_id][api_name] = AsyncLimiter(*config)

    return _limiters[loop_id][api_name]


async def acquire(api_name: str) -> None:
    """Acquire a rate limit slot for the given API. Blocks if at limit."""
    limiter = _get_limiter(api_name)
    if limiter:
        await limiter.acquire()


# Retryable HTTP status codes
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


async def fetch_with_retry(
    client: httpx.AsyncClient,
    api_name: str,
    method: str,
    url: str,
    max_retries: int = 2,
    base_delay: float = 1.0,
    **kwargs,
) -> httpx.Response:
    """HTTP request with rate limiting, exponential backoff, and retry.

    Retries on 429 (rate limit), 5xx (server errors), and connection errors.
    Raises the last exception if all retries fail.

    Args:
        client: httpx.AsyncClient to use
        api_name: API name for rate limiting (e.g., "crossref")
        method: HTTP method ("GET" or "POST")
        url: Request URL
        max_retries: Maximum number of retry attempts (default 2, so 3 total)
        base_delay: Base delay in seconds, doubled on each retry
        **kwargs: Passed to client.request()

    Returns:
        httpx.Response on success

    Raises:
        httpx.HTTPStatusError: Non-retryable HTTP error
        httpx.RequestError: Connection error after all retries
    """
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        await acquire(api_name)

        try:
            resp = await client.request(method, url, **kwargs)

            if resp.status_code not in _RETRYABLE_STATUS:
                return resp

            # Retryable HTTP status
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                # Respect Retry-After header if present
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass
                log.warning(
                    f"{api_name}: HTTP {resp.status_code} (attempt {attempt + 1}/{max_retries + 1}), "
                    f"retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
                continue

            # Last attempt, return whatever we got
            return resp

        except httpx.RequestError as e:
            last_error = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                log.warning(
                    f"{api_name}: {type(e).__name__} (attempt {attempt + 1}/{max_retries + 1}), "
                    f"retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
                continue
            raise

    raise last_error or RuntimeError(f"{api_name}: all retries exhausted")
