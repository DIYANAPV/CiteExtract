
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from citeextract import config, paths
from citeextract.models.comprehension import ComprehensionReport
from citeextract.models.report import PaperReport

log = logging.getLogger(__name__)

CACHE_DIR = paths.data_dir() / "cache" / "reports"

SCHEMA_VERSION = 2


def _config_fingerprint() -> str:
    relevant = {
        "schema_version": SCHEMA_VERSION,
        "agentic": config.agentic() or {},
        "comprehension": config.comprehension(),
        "thresholds": config.thresholds(),
        "claim_verification": config.claim_verification(),
    }
    payload = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _file_hash(file_path: str) -> str:
    data = Path(file_path).read_bytes()
    return hashlib.sha256(data).hexdigest()[:16]


def make_key(
    file_path: str,
    mode: str,
    run_verification: bool,
    run_claim_verification: bool,
    run_comprehension: bool,
    ref_pdfs_dir: Optional[str] = None,
) -> str:
    parts = [
        _file_hash(file_path),
        mode,
        f"rv={int(run_verification)}",
        f"rcv={int(run_claim_verification)}",
        f"rc={int(run_comprehension)}",
        f"refpdfs={int(bool(ref_pdfs_dir))}",
        _config_fingerprint(),
    ]
    blob = "|".join(parts)
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def load(key: str) -> Optional[tuple[Optional[PaperReport], Optional[ComprehensionReport]]]:
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        paper = (
            PaperReport.model_validate(data["paper_report"])
            if data.get("paper_report") else None
        )
        comp = (
            ComprehensionReport.model_validate(data["comp_report"])
            if data.get("comp_report") else None
        )
        return (paper, comp)
    except Exception as e:
        log.warning(f"report_cache: failed to load {key}: {e}")
        return None


def save(
    key: str,
    paper_report: Optional[PaperReport],
    comp_report: Optional[ComprehensionReport],
) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "paper_report": paper_report.model_dump(mode="json") if paper_report else None,
        "comp_report": comp_report.model_dump(mode="json") if comp_report else None,
    }
    path = CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps(payload, default=str), encoding="utf-8")
    log.info(f"report_cache: saved {key}")
