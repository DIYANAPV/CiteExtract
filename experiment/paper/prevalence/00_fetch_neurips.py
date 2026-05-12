"""Sample N accepted NeurIPS 2025 papers from the OpenReview API v2 and download"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import requests

_THIS_FILE = Path(__file__).resolve()
_PREV_DIR = _THIS_FILE.parent
_REPO_ROOT = _PREV_DIR.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

OPENREVIEW_API = "https://api2.openreview.net"
NEURIPS_2025_VENUEID = "NeurIPS.cc/2025/Conference"
ACCEPTED_VENUE_PREFIX = "NeurIPS 2025 "

DEFAULT_N = 20
DEFAULT_SEED = 42
DEFAULT_OUT_DIR = _PREV_DIR / "data" / "papers"
DEFAULT_MANIFEST = _PREV_DIR / "data" / "corpus_manifest.csv"
PAGE_SIZE = 1000
POLITE_DELAY_S = 1.0
PDF_429_BACKOFF_S = 8.0
PDF_MAX_RETRIES = 4

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fetch_neurips")


@dataclass
class PaperRecord:
    paper_id: str
    title: str
    decision: str
    openreview_url: str
    pdf_path: str


def _content_value(content: dict, field: str) -> str:
    raw = content.get(field)
    if isinstance(raw, dict):
        return str(raw.get("value", "") or "")
    if raw is None:
        return ""
    return str(raw)


def fetch_all_accepted(session: requests.Session) -> list[dict]:
    all_notes: list[dict] = []
    offset = 0
    while True:
        params = {
            "content.venueid": NEURIPS_2025_VENUEID,
            "limit": PAGE_SIZE,
            "offset": offset,
        }
        log.info("GET /notes offset=%d", offset)
        resp = session.get(f"{OPENREVIEW_API}/notes", params=params, timeout=60)
        resp.raise_for_status()
        page = resp.json().get("notes", [])
        if not page:
            break
        all_notes.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(POLITE_DELAY_S)

    log.info("fetched %d total notes for venueid=%s", len(all_notes), NEURIPS_2025_VENUEID)

    accepted: list[dict] = []
    for note in all_notes:
        venue = _content_value(note.get("content", {}), "venue")
        if venue.startswith(ACCEPTED_VENUE_PREFIX):
            accepted.append(note)
    log.info("%d notes are accepted (venue starts with %r)", len(accepted), ACCEPTED_VENUE_PREFIX)
    return accepted


def build_record(note: dict) -> PaperRecord | None:
    content = note.get("content", {})
    paper_id = note.get("id")
    title = _content_value(content, "title").strip()
    venue = _content_value(content, "venue").strip()
    pdf_field = _content_value(content, "pdf")

    if not paper_id or not title or not pdf_field:
        return None

    return PaperRecord(
        paper_id=paper_id,
        title=title,
        decision=venue,
        openreview_url=f"https://openreview.net/forum?id={paper_id}",
        pdf_path="",
    )


def download_pdf(session: requests.Session, paper_id: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 1024:
        log.debug("skip already-downloaded: %s", dest.name)
        return True

    url = f"https://openreview.net/pdf?id={paper_id}"
    tmp = dest.with_suffix(".pdf.tmp")

    for attempt in range(1, PDF_MAX_RETRIES + 1):
        try:
            with session.get(url, stream=True, timeout=120) as resp:
                if resp.status_code == 429:
                    backoff = PDF_429_BACKOFF_S * attempt
                    log.warning("429 for %s (attempt %d/%d) — sleeping %.1fs",
                                paper_id, attempt, PDF_MAX_RETRIES, backoff)
                    time.sleep(backoff)
                    continue
                resp.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            f.write(chunk)
            tmp.replace(dest)
            return True
        except Exception as exc:
            log.warning("download failed for %s (attempt %d/%d): %s",
                        paper_id, attempt, PDF_MAX_RETRIES, exc)
            if tmp.exists():
                tmp.unlink()
            time.sleep(PDF_429_BACKOFF_S * attempt)

    log.error("giving up on %s after %d attempts", paper_id, PDF_MAX_RETRIES)
    return False


def write_manifest(records: Iterable[PaperRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["paper_id", "title", "decision", "openreview_url", "pdf_path"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))
    log.info("manifest written: %s", path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N, help=f"number of papers to sample (default {DEFAULT_N})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"random seed (default {DEFAULT_SEED})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="PDF output directory")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="manifest CSV path")
    parser.add_argument("--list-only", action="store_true", help="list sampled papers but do not download")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "checkcitation-prevalence/1.0"})

    accepted = fetch_all_accepted(session)
    if not accepted:
        log.error("no accepted notes returned; aborting")
        return 1

    rng = random.Random(args.seed)
    sample_size = min(args.n, len(accepted))
    sampled = rng.sample(accepted, k=sample_size)
    log.info("sampled %d / %d accepted notes (seed=%d)", sample_size, len(accepted), args.seed)

    records: list[PaperRecord] = []
    for i, note in enumerate(sampled, start=1):
        rec = build_record(note)
        if rec is None:
            log.warning("[%d/%d] skipping note with missing fields: %s", i, sample_size, note.get("id"))
            continue

        dest = args.out / f"{rec.paper_id}.pdf"
        rec.pdf_path = str(dest.relative_to(_REPO_ROOT))

        if args.list_only:
            log.info("[%d/%d] %s — %s", i, sample_size, rec.paper_id, rec.title[:80])
            records.append(rec)
            continue

        log.info("[%d/%d] %s — %s", i, sample_size, rec.paper_id, rec.title[:80])
        if not download_pdf(session, rec.paper_id, dest):
            log.warning("dropping %s from manifest (download failed)", rec.paper_id)
            continue

        records.append(rec)
        time.sleep(POLITE_DELAY_S)

    write_manifest(records, args.manifest)
    log.info("done: %d papers in manifest", len(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
