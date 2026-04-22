"""CLI entry point for CheckCitation.

Usage:
    python -m src parse paper.pdf                          # parse only (L1)
    python -m src verify paper.bib                         # quick mode (default)
    python -m src verify paper.pdf --quick                 # existence + metadata only
    python -m src verify paper.pdf --standard              # + passage retrieval + LLM claim verification
    python -m src verify paper.pdf --agentic               # smart triage + focused agents
    python -m src comprehend paper.pdf                     # passage retrieval from cited papers
    python -m src comprehend paper.pdf --ref-pdfs ./refs/  # with user PDFs
"""

import argparse
import json
import sys

from src.parsers.router import parse_file, UnsupportedFormatError, InputTooLargeError


def cmd_parse(args: argparse.Namespace) -> None:
    """Parse a file and print structured JSON."""
    try:
        result = parse_file(args.file)
    except (UnsupportedFormatError, InputTooLargeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    print(f"Format: {result.input_format}", file=sys.stderr)
    print(f"References: {len(result.references)}", file=sys.stderr)
    print(f"Citations: {len(result.citations)}", file=sys.stderr)
    print(f"Has body text: {result.has_body_text}", file=sys.stderr)
    for w in result.warnings:
        print(f"Warning: {w}", file=sys.stderr)
    print(json.dumps(result.model_dump(), indent=2))


def _print_table(report) -> None:
    """Print a human-readable table of results."""
    verdicts = report.verdicts
    if not verdicts:
        print("No references to display.")
        return

    # Header
    print(f"\n{'#':<5} {'Verdict':<16} {'Action':<18} {'Title / Explanation'}")
    print("-" * 90)

    for v in verdicts:
        # Get title from existence result if available
        title = ""
        if v.existence and v.existence.matched_title:
            title = v.existence.matched_title[:50]

        verdict_str = v.verdict

        print(f"{v.ref_id:<5} {verdict_str:<16} {v.action:<18} {title}")

        # Print explanation on next line, indented
        if v.explanation:
            expl = v.explanation[:100]
            print(f"{'':5} {'':16} {'':18} {expl}")

        # Print flags if any
        for flag in v.flags[:2]:  # max 2 flags per ref
            print(f"{'':5} {'':16} {'':18} ! {flag[:80]}")

    print("-" * 90)


def cmd_verify(args: argparse.Namespace) -> None:
    """Run full pipeline and save reports."""
    from src.pipeline import run_and_save

    mode = args.mode
    print(f"Verifying: {args.file} (mode={mode})", file=sys.stderr)
    report = run_and_save(args.file, mode=mode, retry_failed=args.retry_failed)

    # Print summary
    s = report.summary
    print(f"\n{'='*50}", file=sys.stderr)
    print(f"CheckCitation Report", file=sys.stderr)
    print(f"{'='*50}", file=sys.stderr)
    print(f"Mode: {report.mode}", file=sys.stderr)
    print(f"References: {s.total_checked}", file=sys.stderr)
    print(f"Integrity: {s.integrity_score*100:.1f}%", file=sys.stderr)
    print(f"Risk: {s.risk_level}", file=sys.stderr)
    print(f"Flagged for review: {s.flagged_for_review}", file=sys.stderr)
    print(f"", file=sys.stderr)
    for verdict, count in sorted(s.by_verdict.items()):
        print(f"  {verdict}: {count}", file=sys.stderr)

    print(f"\nReports saved to data/output/", file=sys.stderr)
    for w in report.warnings:
        print(f"Warning: {w}", file=sys.stderr)

    # Output
    if args.output_format == "table":
        _print_table(report)
    else:
        print(json.dumps(report.model_dump(), indent=2, default=str))


def cmd_comprehend(args: argparse.Namespace) -> None:
    """Run comprehension pipeline: retrieve passages from cited papers."""
    from src.pipeline import run_comprehension_and_save

    ref_pdfs_dir = args.ref_pdfs
    print(f"Comprehension check: {args.file}", file=sys.stderr)
    if ref_pdfs_dir:
        print(f"Reference PDFs: {ref_pdfs_dir}", file=sys.stderr)

    report = run_comprehension_and_save(args.file, ref_pdfs_dir=ref_pdfs_dir)

    # Print summary
    cov = report.coverage
    print(f"\n{'='*50}", file=sys.stderr)
    print(f"Comprehension Report", file=sys.stderr)
    print(f"{'='*50}", file=sys.stderr)
    print(f"Citations processed: {report.total_citations}", file=sys.stderr)
    print(f"Full text available: {cov.get('full_text', 0)}", file=sys.stderr)
    print(f"Abstract only:       {cov.get('abstract_only', 0)}", file=sys.stderr)
    print(f"Not found:           {cov.get('not_found', 0)}", file=sys.stderr)

    # Show top results
    for r in report.results:
        if not r.citing_sentence or not r.top_passages:
            continue
        title = r.paper_metadata.get("title", "?")[:60]
        p = r.top_passages[0]
        # Show best available score: rrf > dense > bm25
        score = p.rrf_score or p.dense_score or p.bm25_score or 0
        score_type = "rrf" if p.rrf_score else ("dense" if p.dense_score else "bm25")
        source = r.full_text_source or "?"
        print(f"\n  [{r.ref_id}] {title}", file=sys.stderr)
        print(f"  Claim: {r.citing_sentence[:80]}...", file=sys.stderr)
        print(f"  Top passage ({score_type}={score:.3f}, source={source}):", file=sys.stderr)
        print(f"    {p.chunk.text[:120]}...", file=sys.stderr)

    print(f"\nReport saved to data/output/", file=sys.stderr)

    # JSON to stdout
    print(json.dumps(report.model_dump(), indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="checkcitation",
        description="Citation hallucination detection for academic papers.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # parse
    sp_parse = subparsers.add_parser(
        "parse", help="Parse a file and extract references/citations (L1 only)"
    )
    sp_parse.add_argument("file", help="Path to input file (.pdf, .tex, .bib, .txt)")

    # verify
    sp_verify = subparsers.add_parser(
        "verify", help="Run full verification pipeline (L1-L6)"
    )
    sp_verify.add_argument("file", help="Path to input file (.pdf, .tex, .bib, .txt)")
    mode_group = sp_verify.add_mutually_exclusive_group()
    mode_group.add_argument("--quick", dest="mode", action="store_const", const="quick",
                            help="Quick (rule-based) mode: existence + metadata only (free, fast)")
    mode_group.add_argument("--agentic", dest="mode", action="store_const", const="agentic",
                            help="Agentic mode: LLM-driven triage + focused agents for ambiguous cases")
    sp_verify.add_argument("--retry-failed", dest="retry_failed", action="store_true",
                            help="Re-check references that were previously NOT_FOUND (clears failed cache entries)")
    sp_verify.add_argument("--format", dest="output_format", choices=["json", "table"],
                            default="json", help="Output format: json (default) or table (human-readable)")
    sp_verify.set_defaults(mode="quick")

    # comprehend
    sp_comp = subparsers.add_parser(
        "comprehend", help="Retrieve relevant passages from cited papers"
    )
    sp_comp.add_argument("file", help="Path to input file (.pdf, .tex, .bib, .txt)")
    sp_comp.add_argument("--ref-pdfs", dest="ref_pdfs", default=None,
                         help="Directory containing PDFs of cited papers")

    args = parser.parse_args()

    if args.command == "parse":
        cmd_parse(args)
    elif args.command == "verify":
        cmd_verify(args)
    elif args.command == "comprehend":
        cmd_comprehend(args)


if __name__ == "__main__":
    main()
