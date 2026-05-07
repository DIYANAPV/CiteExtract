
from __future__ import annotations

import json
import logging
import shutil
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import gradio as gr

from citeextract import rate_limit
from citeextract.pipeline import run_unified
from citeextract_ui.downloads.batch_export import _write_batch_csv, _write_batch_json
from citeextract_ui.downloads.bibtex import _write_batch_problem_bibtex, _write_problematic_bibtex
from citeextract_ui.downloads.pdf_annotate import _try_annotate_pdf
from citeextract_ui.render.batch import (
    _format_batch_per_paper,
    _format_batch_rollup,
    _format_batch_summary,
)
from citeextract_ui.render.cards import format_unified_cards
from citeextract_ui.render.coverage import format_comprehension_coverage
from citeextract_ui.render.dashboard import format_dashboard
from citeextract_ui.render.helpers import _safe_stem
from citeextract_ui.render.pipeline import (
    _FINALIZE_STAGE_ID,
    _STAGE_ALIASES,
    _STAGE_PROGRESS_RE,
    _STAGE_RE,
    _applicable_stages,
    _render_pipeline,
)
from citeextract_ui.ui.validate import (
    ALLOWED_SUFFIXES,
    MAX_UPLOAD_BYTES,
    _BATCH_ZIP_MAX_BYTES,
    _unique_dest,
    _validate_upload,
    check_prerequisites,
    prepare_ref_pdfs_dir,
)


def run_analyze(file, ref_pdfs, check_existence, check_claims, retry_failed,
                request: gr.Request = None):
    if file is None:
        raise gr.Error("Please upload a file.")
    try:
        rate_limit.check_single(request)
    except rate_limit.RateLimitExceeded as e:
        raise gr.Error(str(e)) from None
    if not check_existence and not check_claims:
        raise gr.Error("Select at least one analysis option.")

    file_path = file if isinstance(file, str) else file.name
    _validate_upload(file_path, label="paper")

    effective_mode = "agentic" if check_claims else "quick"
    check_prerequisites(file_path, effective_mode)

    ref_dir = prepare_ref_pdfs_dir(ref_pdfs) if check_claims else None

    applicable = _applicable_stages(check_existence, check_claims)
    seen_stages: set[str] = set()
    stage_progress: dict[str, tuple[int, int]] = {}

    def _progress_update(elapsed: float | None = None):
        return (
            gr.update(),
            gr.update(),
            _render_pipeline(
                applicable, seen_stages, elapsed, stage_progress,
            ),
            gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(),
            gr.update(), gr.update(),
            gr.update(visible=False),
        )

    yield (
        "",
        "",
        _render_pipeline(applicable, seen_stages, 0.0),
        "",
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=False),
        "",
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    )

    stage_events: list[str] = []
    stage_progress_updates: dict[str, tuple[int, int]] = {}
    event_lock = threading.Lock()

    class _StageCap(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.allowed_thread: Optional[int] = None

        def emit(self, record: logging.LogRecord) -> None:
            if self.allowed_thread is None or record.thread != self.allowed_thread:
                return
            msg = record.getMessage()
            pm = _STAGE_PROGRESS_RE.search(msg)
            if pm:
                with event_lock:
                    stage_progress_updates[pm.group(1)] = (
                        int(pm.group(2)), int(pm.group(3))
                    )
                return
            m = _STAGE_RE.search(msg)
            if m:
                with event_lock:
                    stage_events.append(m.group(1))

    timing_log = logging.getLogger("citeextract.timing")
    handler = _StageCap()
    timing_log.addHandler(handler)

    result_holder: dict = {}

    def _worker():
        try:
            result_holder["value"] = run_unified(
                file_path,
                mode=effective_mode,
                run_verification=check_existence,
                run_claim_verification=check_claims,
                run_comprehension=check_claims,
                ref_pdfs_dir=ref_dir,
                retry_failed=retry_failed,
            )
        except BaseException as e:
            result_holder["error"] = e

    start = time.time()
    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    handler.allowed_thread = worker.ident

    try:
        last_yield = 0.0
        applicable_ids = {sid for sid, _ in applicable}
        while worker.is_alive():
            worker.join(timeout=0.4)
            with event_lock:
                current = list(stage_events)
                progress_updates = dict(stage_progress_updates)
                stage_progress_updates.clear()
            new_seen = False
            for stage_name in current:
                expanded = _STAGE_ALIASES.get(stage_name, (stage_name,))
                for ui_id in expanded:
                    if ui_id in seen_stages or ui_id not in applicable_ids:
                        continue
                    seen_stages.add(ui_id)
                    new_seen = True
            new_progress = False
            for sid, (done_n, total_n) in progress_updates.items():
                old = stage_progress.get(sid)
                if old != (done_n, total_n):
                    stage_progress[sid] = (done_n, total_n)
                    new_progress = True
            now = time.time()
            if new_seen or new_progress or (now - last_yield) > 1.0:
                yield _progress_update(elapsed=now - start)
                last_yield = now
    finally:
        timing_log.removeHandler(handler)
        if ref_dir:
            if worker.is_alive():
                logging.getLogger(__name__).debug(
                    "Skipping ref_dir cleanup; worker still alive: %s", ref_dir,
                )
            else:
                shutil.rmtree(ref_dir, ignore_errors=True)

    if "error" in result_holder:
        yield (
            "",
            "",
            "",
            "",
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            "",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )
        err = result_holder["error"]
        if isinstance(err, gr.Error):
            raise err
        raise gr.Error(f"Analysis failed: {err}")

    paper_report, comp_report, parsed = result_holder["value"]
    elapsed = time.time() - start

    for sid, _ in applicable:
        if sid != _FINALIZE_STAGE_ID:
            seen_stages.add(sid)
    yield _progress_update(elapsed=elapsed)

    dashboard_html = ""
    if paper_report:
        dashboard_html = format_dashboard(paper_report, selected_mode=effective_mode, elapsed=elapsed)

    coverage_html = ""
    if comp_report and check_claims:
        coverage_html = format_comprehension_coverage(comp_report)

    cards_html = format_unified_cards(
        paper_report, comp_report, parsed.references,
        has_verification=bool(paper_report),
        has_passages=check_claims,
    )

    combined = {}
    if paper_report:
        combined["verification"] = paper_report.model_dump()
    if comp_report:
        combined["comprehension"] = comp_report.model_dump()
    report_json = json.dumps(combined, indent=2, default=str)

    paper_stem = _safe_stem(file_path)
    tmp = tempfile.NamedTemporaryFile(
        suffix=".json", prefix=f"{paper_stem}_report_", delete=False, mode="w"
    )
    tmp.write(report_json)
    tmp.close()

    bib_path = _write_problematic_bibtex(
        paper_report, parsed.references if parsed else [],
        source_label=Path(file_path).name,
        paper_stem=paper_stem,
    )

    annotated_pdf_path, annotate_status_html = _try_annotate_pdf(
        file_path, paper_report, parsed, comp_report,
        paper_stem=paper_stem,
    )

    has_retryable = bool(paper_report) and any(
        v.verdict in ("FABRICATED", "UNVERIFIABLE") for v in paper_report.verdicts
    )
    show_retry = has_retryable and not retry_failed

    yield (
        dashboard_html, coverage_html, cards_html, report_json,
        gr.update(value=tmp.name, visible=True),
        gr.update(value=bib_path, visible=bool(bib_path)),
        gr.update(value=annotated_pdf_path, visible=bool(annotated_pdf_path)),
        annotate_status_html,
        gr.update(visible=True),
        gr.update(visible=bool(bib_path or annotated_pdf_path)),
        gr.update(visible=show_retry),
    )


def _expand_zip_to_papers(zip_path: str) -> list[str]:
    import zipfile

    zip_size = Path(zip_path).stat().st_size
    if zip_size > _BATCH_ZIP_MAX_BYTES:
        raise gr.Error(
            f"ZIP archive is {zip_size / 1024 / 1024:.0f} MB "
            f"(limit is {_BATCH_ZIP_MAX_BYTES // 1024 // 1024} MB)."
        )

    out_dir = Path(tempfile.mkdtemp(prefix="citeextract_batch_"))
    extracted: list[str] = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                safe_name = Path(info.filename).name
                if not safe_name or safe_name in (".", ".."):
                    continue
                suffix = Path(safe_name).suffix.lower()
                if suffix not in ALLOWED_SUFFIXES:
                    continue
                dest = _unique_dest(out_dir, safe_name)
                bytes_written = 0
                with zf.open(info) as src, open(dest, "wb") as dst:
                    while True:
                        chunk = src.read(64 * 1024)
                        if not chunk:
                            break
                        bytes_written += len(chunk)
                        if bytes_written > MAX_UPLOAD_BYTES:
                            raise gr.Error(
                                f"ZIP entry '{safe_name}' exceeds "
                                f"{MAX_UPLOAD_BYTES // 1024 // 1024} MB "
                                "(possible zip bomb)."
                            )
                        dst.write(chunk)
                extracted.append(str(dest))
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    return extracted


def _collect_batch_inputs(files) -> list[str]:
    if not files:
        return []
    if not isinstance(files, list):
        files = [files]
    paper_paths: list[str] = []
    for f in files:
        p = f if isinstance(f, str) else getattr(f, "name", None)
        if not p:
            continue
        if Path(p).suffix.lower() == ".zip":
            paper_paths.extend(_expand_zip_to_papers(p))
        else:
            paper_paths.append(p)
    return paper_paths


def run_batch(files, check_existence, check_claims, retry_failed,
              request: gr.Request = None, progress=gr.Progress()):
    if not files:
        raise gr.Error("Please upload at least one paper (or a .zip of papers).")
    if not check_existence and not check_claims:
        raise gr.Error("Select at least one analysis option.")

    paper_paths = _collect_batch_inputs(files)
    if not paper_paths:
        raise gr.Error(
            "No valid paper files found in the upload "
            f"(accepted: {', '.join(sorted(ALLOWED_SUFFIXES))})."
        )

    try:
        rate_limit.check_batch(len(paper_paths), request)
    except rate_limit.RateLimitExceeded as e:
        raise gr.Error(str(e)) from None

    effective_mode = "agentic" if check_claims else "quick"

    for p in paper_paths:
        _validate_upload(p, label="paper")

    per_paper: list[dict] = []
    start = time.time()
    total = len(paper_paths)

    progress(0.02, desc=f"Starting batch of {total} papers...")

    for idx, p in enumerate(paper_paths, start=1):
        name = Path(p).name
        progress(idx / (total + 1), desc=f"Paper {idx}/{total}: {name}")
        try:
            check_prerequisites(p, effective_mode)
            paper_report, comp_report, parsed = run_unified(
                p,
                mode=effective_mode,
                run_verification=check_existence,
                run_claim_verification=check_claims,
                run_comprehension=check_claims,
                retry_failed=retry_failed,
            )
            counts = defaultdict(int)
            if paper_report:
                for v in paper_report.verdicts:
                    counts[v.metadata_verdict or v.verdict] += 1
            per_paper.append({
                "name": name,
                "path": p,
                "status": "ok",
                "paper_report": paper_report,
                "comp_report": comp_report,
                "parsed": parsed,
                "counts": dict(counts),
                "total_refs": len(parsed.references) if parsed else 0,
            })
        except BaseException as e:
            logging.getLogger(__name__).warning(f"Batch: {name} failed: {e}")
            per_paper.append({
                "name": name,
                "path": p,
                "status": "error",
                "error": str(e),
            })

    elapsed = time.time() - start
    progress(0.97, desc="Building aggregate report...")

    summary_html = _format_batch_summary(per_paper, elapsed, effective_mode)
    rollup_html = _format_batch_rollup(per_paper)
    per_paper_html = _format_batch_per_paper(per_paper, check_claims)

    csv_path = _write_batch_csv(per_paper)
    json_path = _write_batch_json(per_paper, effective_mode, elapsed)
    bib_path = _write_batch_problem_bibtex(per_paper)

    progress(1.0, desc="Done")
    return summary_html, rollup_html, per_paper_html, csv_path, json_path, bib_path
