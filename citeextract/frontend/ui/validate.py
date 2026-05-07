
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional

import gradio as gr
import requests

from citeextract import config


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
PDF_MAGIC = b"%PDF-"
ALLOWED_SUFFIXES = {".pdf", ".tex", ".bib", ".txt"}

_BATCH_ZIP_MAX_BYTES = 500 * 1024 * 1024


def check_prerequisites(file_path: str, mode: str) -> None:
    if file_path and file_path.lower().endswith(".pdf"):
        grobid_url = config.grobid()["service_url"]
        try:
            requests.get(f"{grobid_url}/api/isalive", timeout=3)
        except Exception:
            raise gr.Error(
                "GROBID is not running. PDF parsing requires GROBID.\n"
                "Run: docker run -d --name grobid -p 8070:8070 grobid/grobid:0.8.2-crf"
            )
    if mode == "agentic":
        if not config.openai_api_key():
            raise gr.Error(
                "Agentic mode requires an OpenAI API key.\n"
                "Add OPENAI_API_KEY to your .env file."
            )


def _validate_upload(file_path: str, label: str = "file") -> None:
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        raise gr.Error(f"Uploaded {label} is missing or not a regular file.")

    suffix = p.suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise gr.Error(
            f"{label.capitalize()} type '{suffix}' not allowed. "
            f"Accepted: {', '.join(sorted(ALLOWED_SUFFIXES))}"
        )

    size = p.stat().st_size
    if size > MAX_UPLOAD_BYTES:
        raise gr.Error(
            f"{label.capitalize()} is {size / 1024 / 1024:.1f} MB "
            f"(limit is {MAX_UPLOAD_BYTES // 1024 // 1024} MB)."
        )
    if size == 0:
        raise gr.Error(f"{label.capitalize()} is empty.")

    if suffix == ".pdf":
        with open(p, "rb") as f:
            head = f.read(len(PDF_MAGIC))
        if head != PDF_MAGIC:
            raise gr.Error(
                f"{label.capitalize()} has a .pdf extension but is not a valid PDF "
                "(missing %PDF- header)."
            )


def _unique_dest(dest_dir: Path, name: str) -> Path:
    candidate = dest_dir / name
    if not candidate.exists():
        return candidate
    stem = Path(name).stem
    suffix = Path(name).suffix
    i = 1
    while True:
        candidate = dest_dir / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def prepare_ref_pdfs_dir(pdf_paths: Optional[list[str]]) -> Optional[str]:
    if not pdf_paths:
        return None
    for p in pdf_paths:
        _validate_upload(p, label="reference PDF")
    tmp_dir = Path(tempfile.mkdtemp(prefix="citeextract_refs_"))
    for p in pdf_paths:
        src = Path(p)
        shutil.copy2(str(src), str(_unique_dest(tmp_dir, src.name)))
    return str(tmp_dir)
