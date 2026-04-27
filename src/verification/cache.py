"""SQLite-based async cache for API responses.

Caches metadata, abstracts, and existence results with configurable TTL.
Abstracts cached here are reused in L4 semantic verification at zero extra cost.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import aiosqlite

from src import config

# TTL constants in seconds — loaded lazily on first access.
# These are stable integers once config.yaml is read, so caching is safe.
_ttl_cache: Optional[dict] = None


def _ttls() -> dict:
    global _ttl_cache
    if _ttl_cache is None:
        _ttl_cache = config.cache()
    return _ttl_cache


# Public TTL names kept for backward compatibility with external imports
TTL_METADATA: int = 30 * 86400    # overridden at first use
TTL_ABSTRACT: int = 30 * 86400
TTL_RETRACTION: int = 7 * 86400
TTL_NOT_FOUND: int = 7 * 86400


def _init_ttls() -> None:
    """Initialise TTL module constants from config. Called on first APICache use."""
    global TTL_METADATA, TTL_ABSTRACT, TTL_RETRACTION, TTL_NOT_FOUND
    cfg = _ttls()
    TTL_METADATA = cfg["ttl_metadata"]
    TTL_ABSTRACT = cfg["ttl_abstract"]
    TTL_RETRACTION = cfg["ttl_retraction"]
    TTL_NOT_FOUND = cfg["ttl_not_found"]


# ---------------------------------------------------------------------------
# Cited-paper cache keys
# ---------------------------------------------------------------------------
#
# Resources scoped to a *cited paper* (full text, abstract, etc.) must be
# keyed by what the paper IS, not by where it happens to appear in some
# input paper. Per-input ``ref_id`` values like "13" collide across runs
# of different input papers and serve cached content from the wrong paper
# back on the next run — corrupting passage retrieval downstream.
#
# Identity priority: DOI > arXiv ID > normalised-title hash. Returns None
# when no identifier is available so the caller can decide whether to skip
# caching entirely (recommended) or fall back to a per-input scope (loses
# cross-paper sharing and risks collisions — only acceptable if the data
# can never poison a future run).


def cited_paper_identity(
    *,
    doi: Optional[str] = None,
    arxiv_id: Optional[str] = None,
    title: Optional[str] = None,
) -> Optional[str]:
    """Build a stable identity string for a cited paper, or None.

    DOIs and arXiv IDs are normalised to lowercase since both are
    case-insensitive in practice (DOI spec; arXiv numeric IDs are digits
    + lowercase ``v``). Titles are run through ``normalize_title`` so
    capitalisation, punctuation, and minor whitespace differences hash
    to the same identity.
    """
    if doi:
        return f"doi:{doi.strip().lower()}"
    if arxiv_id:
        return f"arxiv:{arxiv_id.strip().lower()}"
    if title:
        from src.verification.matching import normalize_title
        norm = normalize_title(title)
        if norm:
            digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]
            return f"title:{digest}"
    return None


def cited_paper_cache_key(
    prefix: str,
    *,
    doi: Optional[str] = None,
    arxiv_id: Optional[str] = None,
    title: Optional[str] = None,
) -> Optional[str]:
    """Cache key for any per-cited-paper resource. Returns None when no
    identity is available — caller should skip caching in that case.

    Two different input papers citing the same work will produce the
    same key, so the second run reuses the first's fetch. Two input
    papers whose ``ref_id`` 13 happens to point at different works
    will produce different keys, eliminating the cross-paper collision
    that the legacy ``f"{prefix}:{ref_id}"`` scheme suffered from.
    """
    identity = cited_paper_identity(doi=doi, arxiv_id=arxiv_id, title=title)
    if identity is None:
        return None
    return f"{prefix}:{identity}"


class APICache:
    """Async SQLite cache with TTL expiration."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = _ttls()["db_path"]
        _init_ttls()
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def _ensure_db(self) -> aiosqlite.Connection:
        if self._db is None:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._db = await aiosqlite.connect(self.db_path, timeout=10)
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires_at REAL NOT NULL
                )
            """)
            await self._db.commit()
        return self._db

    async def get(self, key: str) -> Optional[dict]:
        """Lookup by key. Returns None if missing or expired."""
        db = await self._ensure_db()
        cursor = await db.execute(
            "SELECT value, expires_at FROM cache WHERE key = ?", (key,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        value_str, expires_at = row
        if time.time() > expires_at:
            await db.execute("DELETE FROM cache WHERE key = ?", (key,))
            await db.commit()
            return None
        return json.loads(value_str)

    async def set(self, key: str, value: dict, ttl_seconds: int = 0) -> None:
        """Store with TTL. Overwrites existing. Uses TTL_METADATA if ttl_seconds is 0."""
        if ttl_seconds == 0:
            ttl_seconds = TTL_METADATA
        db = await self._ensure_db()
        expires_at = time.time() + ttl_seconds
        await db.execute(
            "INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)",
            (key, json.dumps(value), expires_at),
        )
        await db.commit()

    async def set_abstract(self, ref_id: str, abstract: str) -> None:
        """Store an abstract for L4 reuse."""
        await self.set(f"abstract:{ref_id}", {"abstract": abstract}, TTL_ABSTRACT)

    async def set_title_index(self, normalized_title: str, ref_id: str) -> None:
        """Map a normalized title to its ref_id for fuzzy cache lookups."""
        await self.set(f"title_idx:{normalized_title}", {"ref_id": ref_id}, TTL_METADATA)

    async def get_title_ref_id(self, normalized_title: str) -> Optional[str]:
        """Look up a single normalized title in the index. O(1) SQLite lookup."""
        result = await self.get(f"title_idx:{normalized_title}")
        if result:
            return result.get("ref_id")
        return None

    async def get_all_title_keys(self) -> list[tuple[str, str]]:
        """Return all (normalized_title, ref_id) pairs from the title index.

        Only returns non-expired entries. Used for fuzzy matching against
        incoming titles to avoid redundant API calls.
        """
        db = await self._ensure_db()
        cursor = await db.execute(
            "SELECT key, value FROM cache WHERE key LIKE 'title_idx:%' AND expires_at > ?",
            (time.time(),),
        )
        rows = await cursor.fetchall()
        results = []
        for key, value_str in rows:
            norm_title = key[len("title_idx:"):]
            data = json.loads(value_str)
            results.append((norm_title, data["ref_id"]))
        return results

    async def clear_not_found(self) -> int:
        """Delete all cached NOT_FOUND existence results.

        Returns the number of entries cleared.
        """
        db = await self._ensure_db()
        cursor = await db.execute(
            "DELETE FROM cache WHERE key LIKE 'existence:%' AND value LIKE '%NOT_FOUND%'"
        )
        await db.commit()
        return cursor.rowcount

    async def purge_legacy_fulltext_keys(self) -> int:
        """Delete legacy ``fulltext:{ref_id}`` entries left over from the
        per-input ref_id scheme. Those entries served content from a
        different cited paper when the same ref_id appeared in a later
        run on a different input paper. Identity-keyed entries
        (``fulltext:doi:…``, ``fulltext:arxiv:…``, ``fulltext:title:…``)
        are kept. Safe to run repeatedly. Returns rows deleted.
        """
        db = await self._ensure_db()
        cursor = await db.execute(
            "DELETE FROM cache "
            "WHERE key LIKE 'fulltext:%' "
            "  AND key NOT LIKE 'fulltext:doi:%' "
            "  AND key NOT LIKE 'fulltext:arxiv:%' "
            "  AND key NOT LIKE 'fulltext:title:%'"
        )
        await db.commit()
        return cursor.rowcount

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None
