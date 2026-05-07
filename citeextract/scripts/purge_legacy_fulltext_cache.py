
import argparse
import sqlite3
from pathlib import Path


def _resolve_db_path(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)

    from citeextract import config

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
