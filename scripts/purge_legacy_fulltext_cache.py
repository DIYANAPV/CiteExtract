"""Purge legacy ``fulltext:`` cache entries left over from the v1 keyspace.

The v1 cache used a 200K-char hard cap and stored over-cap papers as
``source: not_found``. After bumping to ``fulltext_v2`` keys (with soft
truncation), the v1 rows are dead weight: harmless but stale, and they
keep returning poisoned ``not_found`` snapshots if anything still reads
them by accident.

Run once after deploying the v2 fix:

    python scripts/purge_legacy_fulltext_cache.py

Idempotent — safe to run multiple times.
"""

import argparse
import sqlite3
import sys
from pathlib import Path


def _resolve_db_path(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)

    # Read the configured path from config.yaml without booting the full
    # config layer (which would import async deps for a sync utility).
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import config

    return Path(config.cache()["db_path"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        help="Path to the cache SQLite DB. Defaults to config.yaml -> cache.db_path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be deleted without modifying the DB.",
    )
    args = parser.parse_args()

    db_path = _resolve_db_path(args.db)
    if not db_path.exists():
        print(f"Cache DB not found at {db_path} — nothing to purge.")
        return 0

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Match keys produced by ``cited_paper_cache_key("fulltext", ...)``,
    # i.e. ``fulltext:doi:...``, ``fulltext:arxiv:...``, ``fulltext:title:...``.
    # The new keyspace is ``fulltext_v2:...`` so a literal LIKE on the
    # legacy prefix is enough.
    legacy_pattern = "fulltext:%"
    (count,) = cur.execute(
        "SELECT COUNT(*) FROM cache WHERE key LIKE ?", (legacy_pattern,)
    ).fetchone()

    if count == 0:
        print("No legacy fulltext: entries found.")
        return 0

    print(f"Found {count} legacy fulltext: entries.")
    if args.dry_run:
        print("--dry-run set — leaving the DB untouched.")
        return 0

    cur.execute("DELETE FROM cache WHERE key LIKE ?", (legacy_pattern,))
    conn.commit()
    print(f"Deleted {cur.rowcount} entries.")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
