
import asyncio
import logging
from typing import Optional, TypeVar

import httpx
from aiolimiter import AsyncLimiter

from citeextract import config

log = logging.getLogger(__name__)

T = TypeVar("T")

_CONFIGS: Optional[dict[str, tuple[int, int]]] = None


def _get_configs() -> dict[str, tuple[int, int]]:
    global _CONFIGS
    if _CONFIGS is None:
        _CONFIGS = config.rate_limits()
    return _CONFIGS

_limiters: dict[int, dict[str, AsyncLimiter]] = {}


def _get_limiter(api_name: str) -> Optional[AsyncLimiter]:
    config = _get_configs().get(api_name)
    if not config:
        return None

    try:
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
    except RuntimeError:
        return None

    stale = [lid for lid in _limiters if lid != loop_id]
    for lid in stale:
        del _limiters[lid]

    if loop_id not in _limiters:
        _limiters[loop_id] = {}

    if api_name not in _limiters[loop_id]:
        _limiters[loop_id][api_name] = AsyncLimiter(*config)

    return _limiters[loop_id][api_name]


async def acquire(api_name: str) -> None:
    limiter = _get_limiter(api_name)
    if limiter:
        await limiter.acquire()


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
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        await acquire(api_name)

        try:
            resp = await client.request(method, url, **kwargs)

            if resp.status_code not in _RETRYABLE_STATUS:
                return resp

            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass
                delay = min(delay, 120.0)
                log.warning(
                    f"{api_name}: HTTP {resp.status_code} (attempt {attempt + 1}/{max_retries + 1}), "
                    f"retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
                continue

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
