
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import HTTPException, UploadFile

from citeextract.models.report import PaperReport


_ALLOWED_SUFFIXES = {".pdf", ".tex", ".bib", ".txt"}

_MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def _suffix_for_upload(upload: UploadFile) -> str:
    if upload.filename:
        suf = Path(upload.filename).suffix.lower()
        if suf in _ALLOWED_SUFFIXES:
            return suf
    ct = (upload.content_type or "").lower()
    if "pdf" in ct:
        return ".pdf"
    if "latex" in ct or "x-tex" in ct:
        return ".tex"
    if "bibtex" in ct:
        return ".bib"
    return ".txt"


def _build_summary(report: PaperReport) -> dict:
    counts: dict[str, int] = {}
    for v in report.verdicts:
        counts[v.verdict] = counts.get(v.verdict, 0) + 1
    return counts


def _persist_upload(upload: UploadFile, prefix: str) -> tuple[Path, Path, str]:
    suffix = _suffix_for_upload(upload)
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. "
                   f"Allowed: {sorted(_ALLOWED_SUFFIXES)}.",
        )
    tmpdir = Path(tempfile.mkdtemp(prefix=prefix))
    safe_name = Path(upload.filename or f"paper{suffix}").name or f"paper{suffix}"
    tmp_path = tmpdir / safe_name
    bytes_written = 0
    try:
        with open(tmp_path, "wb") as f:
            while True:
                chunk = upload.file.read(1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > _MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large (>{_MAX_UPLOAD_BYTES // (1024*1024)} MB).",
                    )
                f.write(chunk)
    except HTTPException:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise
    finally:
        upload.file.close()
    return tmp_path, tmpdir, suffix
