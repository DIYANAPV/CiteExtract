
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

import aiosqlite

from citeextract import config

log = logging.getLogger(__name__)

_ttl_cache: Optional[dict] = None


def _ttls() -> dict:
    global _ttl_cache
    if _ttl_cache is None:
        _ttl_cache = config.cache()
    return _ttl_cache


TTL_METADATA: int = 30 * 86400
TTL_ABSTRACT: int = 30 * 86400
TTL_RETRACTION: int = 7 * 86400
TTL_NOT_FOUND: int = 7 * 86400


def _init_ttls() -> None:
    global TTL_METADATA, TTL_ABSTRACT, TTL_RETRACTION, TTL_NOT_FOUND
    cfg = _ttls()
    TTL_METADATA = cfg["ttl_metadata"]
    TTL_ABSTRACT = cfg["ttl_abstract"]
    TTL_RETRACTION = cfg["ttl_retraction"]
    TTL_NOT_FOUND = cfg["ttl_not_found"]


def cited_paper_identity(
    *,
    doi: Optional[str] = None,
    arxiv_id: Optional[str] = None,
    title: Optional[str] = None,
) -> Optional[str]:
    if doi:
        return f"doi:{doi.strip().lower()}"
    if arxiv_id:
        return f"arxiv:{arxiv_id.strip().lower()}"
    if title:
        from citeextract.verification.matching import normalize_title
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
    identity = cited_paper_identity(doi=doi, arxiv_id=arxiv_id, title=title)
    if identity is None:
        return None
    return f"{prefix}:{identity}"


class APICache:

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
            await self._apply_pragmas(self._db)
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires_at REAL NOT NULL
                )
            """)
            await self._db.commit()
        return self._db

    @staticmethod
    async def _apply_pragmas(db: aiosqlite.Connection) -> None:
        pragmas = [
            ("journal_mode", "WAL"),
            ("synchronous", "NORMAL"),
            ("temp_store", "MEMORY"),
            ("cache_size", "-16000"),
            ("mmap_size", "67108864"),
        ]
        for name, value in pragmas:
            try:
                await db.execute(f"PRAGMA {name} = {value};")
            except Exception as exc:
                log.warning(
                    "APICache: PRAGMA %s = %s failed (%s); "
                    "cache will fall back to SQLite defaults.",
                    name, value, type(exc).__name__,
                )

    async def get(self, key: str) -> Optional[dict]:
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
        if ttl_seconds == 0:
            ttl_seconds = TTL_METADATA
        db = await self._ensure_db()
        expires_at = time.time() + ttl_seconds
        await db.execute(
            "INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)",
            (key, json.dumps(value), expires_at),
        )
        await db.commit()

    async def set_title_index(self, normalized_title: str, ref_id: str) -> None:
        await self.set(f"title_idx:{normalized_title}", {"ref_id": ref_id}, TTL_METADATA)

    async def get_title_ref_id(self, normalized_title: str) -> Optional[str]:
        result = await self.get(f"title_idx:{normalized_title}")
        if result:
            return result.get("ref_id")
        return None

    async def get_all_title_keys(self) -> list[tuple[str, str]]:
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
        db = await self._ensure_db()
        cursor = await db.execute(
            "DELETE FROM cache WHERE key LIKE 'existence:%' AND value LIKE '%NOT_FOUND%'"
        )
        await db.commit()
        return cursor.rowcount

    async def purge_legacy_fulltext_keys(self) -> int:
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
