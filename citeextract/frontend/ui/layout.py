
from __future__ import annotations

import gradio as gr

from citeextract import rate_limit
from citeextract_ui.ui.about import ABOUT_HTML
from citeextract_ui.ui.handlers import run_analyze, run_batch


def create_app() -> gr.Blocks:
    with gr.Blocks(title="CiteExtract", analytics_enabled=False) as demo:

        gr.HTML('''
            <div class="cc-hero">
                <div class="cc-hero-text">
                    <h1>Verify every citation in your paper, <span class="cc-hero-accent">with evidence</span>.</h1>
                    <p class="cc-hero-sub">
                        CiteExtract finds fabricated references, flags metadata mismatches, and checks whether each cited paper actually supports the claim you made for it. It returns the passage that grounds every verdict.
                    </p>
                </div>
                <div class="cc-hero-video">
                    <!-- autoplay + muted is the only browser-permitted combination
                         for unattended playback; playsinline keeps mobile Safari from
                         going fullscreen; preload=metadata starts loading enough to
                         lay out the frame so the page does not jump when it appears. -->
                    <video autoplay muted loop playsinline preload="metadata"
                           aria-label="CiteExtract demo">
                        <source src="/cc-static/citeextract.mp4" type="video/mp4">
                    </video>
                </div>
            </div>

            <div class="cc-features">
                <div class="cc-feature">
                    <div class="cc-feature-icon">
                        <!-- Duotone: filled silhouette (low-opacity wash) under the stroked outline -->
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                            <path class="duo-fill" fill="currentColor" stroke="none" d="M6 2h8l6 6v6h-3.5a3.5 3.5 0 1 0-2.5 5.95V22H6a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2z"/>
                            <circle class="duo-fill" fill="currentColor" stroke="none" cx="16.5" cy="16.5" r="3"/>
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h7"/>
                            <path d="M14 2v6h6"/>
                            <circle cx="16.5" cy="16.5" r="3"/>
                            <path d="M21 21l-2-2"/>
                        </svg>
                    </div>
                    <div class="cc-feature-step">STEP 1</div>
                    <div class="cc-feature-title">Does the paper exist?</div>
                    <div class="cc-feature-body">We query CrossRef, Semantic Scholar, OpenAlex and PubMed; flag references that resolve nowhere as fabricated.</div>
                </div>
                <div class="cc-feature">
                    <div class="cc-feature-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                            <rect class="duo-fill" fill="currentColor" stroke="none" x="3" y="4" width="18" height="16" rx="2"/>
                            <rect x="3" y="4" width="18" height="16" rx="2"/>
                            <path d="M7.2 9.2l.6.6 1.4-1.4"/>
                            <path d="M12 9h5"/>
                            <path d="M7.2 14.2l.6.6 1.4-1.4"/>
                            <path d="M12 14h5"/>
                        </svg>
                    </div>
                    <div class="cc-feature-step">STEP 2</div>
                    <div class="cc-feature-title">Does the metadata match?</div>
                    <div class="cc-feature-body">Field-level checks across title, authors, venue, year catch chimera citations: a real paper attached to the wrong attribution.</div>
                </div>
                <div class="cc-feature">
                    <div class="cc-feature-icon">
                        <!-- Comment bubble with three dots: the universal "passage / discussion" mark -->
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                            <path class="duo-fill" fill="currentColor" stroke="none" d="M5 4h14a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-7l-4 4v-4H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z"/>
                            <path d="M5 4h14a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-7l-4 4v-4H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z"/>
                            <circle cx="8" cy="10.5" r="1" fill="currentColor"/>
                            <circle cx="12" cy="10.5" r="1" fill="currentColor"/>
                            <circle cx="16" cy="10.5" r="1" fill="currentColor"/>
                        </svg>
                    </div>
                    <div class="cc-feature-step">STEP 3</div>
                    <div class="cc-feature-title">Does it support your claim?</div>
                    <div class="cc-feature-body">Multi-query retrieval pulls passages from the cited paper and an LLM checks semantic support, returning the quote it used.</div>
                </div>
            </div>
        ''')

        with gr.Tabs():
            with gr.TabItem("Analyze"):
                gr.Markdown(
                    "**Upload a paper** (PDF, LaTeX, BibTeX, or plain text), "
                    "pick what to check, then click **Analyze**."
                )
                with gr.Row(equal_height=True, elem_classes=["cc-panel-row"]):
                    with gr.Column(scale=3, elem_classes=["cc-panel", "cc-panel-upload"]):
                        gr.HTML('<div class="cc-panel-title">Paper</div>')
                        analyze_file = gr.File(
                            label="",
                            show_label=False,
                            file_types=[".pdf", ".tex", ".bib", ".txt"],
                            type="filepath",
                            elem_classes=["cc-filedrop"],
                        )
                        analyze_refs = gr.File(
                            label="Reference PDFs (optional, for non-open-access cited papers)",
                            file_types=[".pdf"],
                            file_count="multiple",
                            type="filepath",
                            visible=False,
                        )
                    with gr.Column(scale=2, elem_classes=["cc-panel", "cc-panel-options"]):
                        gr.HTML('<div class="cc-panel-title">Analysis options</div>')
                        chk_existence = gr.Checkbox(
                            label="Existence & Metadata",
                            value=True,
                        )
                        chk_claims = gr.Checkbox(
                            label="Claim Verification",
                            value=False,
                        )
                        analyze_btn = gr.Button("Analyze", variant="primary", size="lg", elem_classes=["cc-analyze-btn"])

                        chk_claims.change(
                            fn=lambda checked: gr.update(visible=checked),
                            inputs=[chk_claims],
                            outputs=[analyze_refs],
                        )

                gr.HTML('<div class="cc-section-label" style="margin-top:20px;">Results</div>')
                analyze_dashboard = gr.HTML()
                analyze_coverage = gr.HTML()
                with gr.Row(visible=False) as analyze_retry_row:
                    gr.HTML(
                        '<div class="cc-retry-hint">Some references didn\'t '
                        'resolve. This can happen on transient API errors — '
                        'try again to re-check just the failed ones.</div>'
                    )
                    analyze_retry_btn = gr.Button(
                        "Re-run with retries",
                        variant="secondary", size="sm",
                        elem_classes=["cc-retry-btn"],
                    )
                analyze_cards = gr.HTML(
                    value='''<div class="cc-empty">
                        <div class="cc-empty-icon">&#x1F50D;</div>
                        <div class="cc-empty-title">Ready to analyze</div>
                        <div class="cc-empty-sub">Upload a paper and click Analyze to begin</div>
                    </div>''',
                )

                with gr.Accordion("JSON Report", open=False, visible=False) as analyze_json_acc:
                    analyze_download = gr.File(label="Download JSON report", interactive=False, visible=False)
                    analyze_json = gr.Code(language="json", label="Report JSON")
                with gr.Row(visible=False) as analyze_dl_row:
                    analyze_bib = gr.File(
                        label="Download problematic refs (.bib)", interactive=False, visible=False,
                    )
                    analyze_annotated_pdf = gr.File(
                        label="Download annotated PDF", interactive=False, visible=False,
                    )
                analyze_annotated_status = gr.HTML()

                _analyze_inputs = [
                    analyze_file, analyze_refs,
                    chk_existence, chk_claims,
                ]
                _analyze_outputs = [
                    analyze_dashboard, analyze_coverage, analyze_cards,
                    analyze_json, analyze_download, analyze_bib,
                    analyze_annotated_pdf, analyze_annotated_status,
                    analyze_json_acc, analyze_dl_row, analyze_retry_row,
                ]

                def run_analyze_main(file, ref_pdfs, ce, cc, request: gr.Request = None):
                    yield from run_analyze(
                        file, ref_pdfs, ce, cc,
                        retry_failed=False, request=request,
                    )

                def run_analyze_retry(file, ref_pdfs, ce, cc, request: gr.Request = None):
                    yield from run_analyze(
                        file, ref_pdfs, ce, cc,
                        retry_failed=True, request=request,
                    )

                analyze_btn.click(
                    fn=run_analyze_main,
                    inputs=_analyze_inputs,
                    outputs=_analyze_outputs,
                    show_progress="hidden",
                )

                analyze_retry_btn.click(
                    fn=run_analyze_retry,
                    inputs=_analyze_inputs,
                    outputs=_analyze_outputs,
                    show_progress="hidden",
                )


            with gr.TabItem("Batch"):
                gr.Markdown(
                    f"**Upload multiple papers** (or a `.zip` of papers) to "
                    f"analyze them in one go. Each paper goes through the same "
                    f"pipeline and you get an aggregate report plus per-paper "
                    f"breakdowns.\n\n"
                    f"**Limits:** up to **{rate_limit.BATCH_MAX_PAPERS} papers per batch**, "
                    f"**{rate_limit.HOURLY_IP_BATCH_LIMIT} batches per hour** per IP."
                )
                with gr.Row(equal_height=True, elem_classes=["cc-panel-row"]):
                    with gr.Column(scale=3, elem_classes=["cc-panel", "cc-panel-upload"]):
                        gr.HTML('<div class="cc-panel-title">Papers</div>')
                        batch_files = gr.File(
                            label="",
                            show_label=False,
                            file_types=[".pdf", ".tex", ".bib", ".txt", ".zip"],
                            file_count="multiple",
                            type="filepath",
                            elem_classes=["cc-filedrop"],
                        )
                    with gr.Column(scale=2, elem_classes=["cc-panel", "cc-panel-options"]):
                        gr.HTML('<div class="cc-panel-title">Analysis options</div>')
                        batch_chk_existence = gr.Checkbox(
                            label="Existence & Metadata",
                            value=True,
                        )
                        batch_chk_claims = gr.Checkbox(
                            label="Claim Verification",
                            value=False,
                        )
                        batch_retry = gr.State(value=False)
                        batch_btn = gr.Button(
                            "Analyze batch", variant="primary", size="lg",
                            elem_classes=["cc-analyze-btn"],
                        )

                batch_summary = gr.HTML()
                batch_rollup = gr.HTML()
                batch_per_paper = gr.HTML()
                with gr.Row(visible=False) as batch_dl_row:
                    batch_csv = gr.File(label="Download CSV (one row per verdict)", interactive=False, visible=False)
                    batch_json = gr.File(label="Download JSON (full batch)", interactive=False, visible=False)
                    batch_bib = gr.File(
                        label="Download problematic refs (.bib)", interactive=False, visible=False,
                    )

                def _run_batch_with_visibility(*args, request: gr.Request = None):
                    yield (
                        "", "", "",
                        gr.update(value=None, visible=False),
                        gr.update(value=None, visible=False),
                        gr.update(value=None, visible=False),
                        gr.update(visible=False),
                    )
                    out = run_batch(*args, request=request)
                    summary_html, rollup_html, per_paper_html, csv_path, json_path, bib_path = out
                    yield (
                        summary_html, rollup_html, per_paper_html,
                        gr.update(value=csv_path, visible=bool(csv_path)),
                        gr.update(value=json_path, visible=bool(json_path)),
                        gr.update(value=bib_path, visible=bool(bib_path)),
                        gr.update(visible=True),
                    )

                batch_btn.click(
                    fn=_run_batch_with_visibility,
                    inputs=[batch_files, batch_chk_existence, batch_chk_claims, batch_retry],
                    outputs=[
                        batch_summary, batch_rollup, batch_per_paper,
                        batch_csv, batch_json, batch_bib, batch_dl_row,
                    ],
                    show_progress_on=batch_per_paper,
                )

            with gr.TabItem("About"):
                gr.HTML(ABOUT_HTML)

        gr.HTML(
            '<footer class="cc-page-footer">'
            '<code>v0.2.0</code>'
            '<span class="cc-footer-sep">·</span>'
            'open-source'
            '<span class="cc-footer-sep">·</span>'
            'anonymous (under review)'
            '</footer>'
        )

    return demo
