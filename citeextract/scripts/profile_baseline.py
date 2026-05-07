
import asyncio
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from citeextract import paths
from citeextract.pipeline import run_unified_pipeline


@dataclass
class RunResult:
    label: str
    paper: str
    mode: str
    refs: int = 0
    stages: dict[str, float] = field(default_factory=dict)
    total: float = 0.0
    cost_usd: float = 0.0
    note: str = ""


class StageCapture(logging.Handler):

    STAGE_RE = re.compile(r"STAGE (\S+) seconds=([\d.]+)")

    def __init__(self):
        super().__init__()
        self.stages: dict[str, float] = {}

    def reset(self):
        self.stages = {}

    def emit(self, record):
        msg = record.getMessage()
        m = self.STAGE_RE.search(msg)
        if m:
            self.stages[m.group(1)] = float(m.group(2))


def extract_cost(report) -> float:
    if report is None:
        return 0.0
    for w in report.warnings or []:
        m = re.search(r"\$([\d.]+)", w)
        if m and "cost" in w.lower():
            return float(m.group(1))
    return 0.0


async def run_scenario(
    label: str, paper: str, mode: str,
    run_claim_verification: bool, run_comprehension: bool,
    capture: StageCapture,
) -> RunResult:
    capture.reset()
    t0 = time.perf_counter()
    paper_report, comp_report, parsed = await run_unified_pipeline(
        paper, mode=mode,
        run_claim_verification=run_claim_verification,
        run_comprehension=run_comprehension,
        force_refresh=True,
    )
    wall = time.perf_counter() - t0

    result = RunResult(
        label=label,
        paper=Path(paper).name,
        mode=mode,
        refs=len(parsed.references) if parsed else 0,
        stages=dict(capture.stages),
        total=wall,
        cost_usd=extract_cost(paper_report) + extract_cost(comp_report),
    )
    return result


def fmt_row(r: RunResult) -> str:
    s = r.stages
    return (
        f"| {r.label} | {r.paper} | {r.mode} | {r.refs} | "
        f"{s.get('L1_parse', 0):.2f} | {s.get('L2_existence', 0):.2f} | "
        f"{s.get('L3_L5_quick', 0) + s.get('L5_agentic', 0):.2f} | "
        f"{s.get('L4_comprehension', 0):.2f} | "
        f"{r.total:.2f} | ${r.cost_usd:.4f} |"
    )


def write_report(results: list[RunResult], agentic_detail: dict):
    out = Path("PROFILING.md")
    lines = [
        "# CiteExtract Profiling Baseline",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}. "
        f"Each scenario run once with `force_refresh=True` (report cache bypassed). "
        f"APICache warms naturally across scenarios on the same paper._",
        "",
        "## Per-scenario timings (seconds)",
        "",
        "| Label | Paper | Mode | Refs | L1 Parse | L2 Exist | L3/L5 | L4 Comp | Total | LLM Cost |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        lines.append(fmt_row(r))
    lines.append("")

    if agentic_detail:
        lines.extend([
            "## Agentic dispatch breakdown",
            "",
            "| Label | pre_retrieve | triage | meta_dispatch | claim_dispatch | both_dispatch |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for label, d in agentic_detail.items():
            lines.append(
                f"| {label} | {d.get('agentic_pre_retrieve', 0):.2f} | "
                f"{d.get('agentic_triage', 0):.2f} | "
                f"{d.get('agentic_metadata_dispatch', 0):.2f} | "
                f"{d.get('agentic_claim_dispatch', 0):.2f} | "
                f"{d.get('agentic_both_dispatch', 0):.2f} |"
            )
        lines.append("")

    lines.extend([
        "## Notes",
        "",
        "- L1 reflects spaCy/parser load + GROBID/LaTeX parsing (parse cache on disk).",
        "- L2 existence checks use `APICache` (SQLite, TTL in config). Repeat DOIs hit cache.",
        "- L3/L5 in quick mode is near-instant (rule-based classification).",
        "- L5 in agentic mode = triage + agent LLM calls (semaphore cap 5, see config).",
        "- `paper2_attention.pdf`: skipped (GROBID not running at profile time).",
        "- Cost extracted from `report.warnings`; falls back to $0 if not reported.",
        "",
    ])
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out}")


async def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    capture = StageCapture()
    logging.getLogger("citeextract.timing").addHandler(capture)

    cache_root = paths.data_dir() / "cache"
    report_dir = cache_root / "reports"
    api_db = cache_root / "api_cache.db"
    if report_dir.exists():
        shutil.rmtree(report_dir)
    if api_db.exists():
        api_db.unlink()
    print("Cleared data/cache/reports and data/cache/api_cache.db")
    print()

    results: list[RunResult] = []
    agentic_detail: dict[str, dict] = {}

    P1 = "data/test_inputs/paper1_watermark_llm.pdf"
    CLEAN = "data/test_inputs/clean_paper.tex"
    HALL = "data/test_inputs/hallucinated_paper.tex"

    scenarios = [
        ("P1-quick",     P1,    "quick",    False, False),
        ("P1-standard",  P1,    "standard", True,  False),
        ("P1-agentic",   P1,    "agentic",  False, False),
        ("clean-quick",  CLEAN, "quick",    False, False),
        ("hall-agentic", HALL,  "agentic",  False, False),
    ]

    for label, paper, mode, rcv, rc in scenarios:
        print(f"=== Running {label} ({mode} on {Path(paper).name}) ===")
        try:
            r = await run_scenario(label, paper, mode, rcv, rc, capture)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append(RunResult(label=label, paper=Path(paper).name,
                                     mode=mode, note=f"ERROR: {e}"))
            continue

        results.append(r)
        if mode == "agentic":
            agentic_detail[label] = {
                k: v for k, v in r.stages.items() if k.startswith("agentic_")
            }

        print(f"  refs={r.refs} total={r.total:.2f}s cost=${r.cost_usd:.4f}")
        for name in ("L1_parse", "L2_existence", "L3_L5_quick",
                     "L5_agentic", "L4_comprehension"):
            if name in r.stages:
                print(f"  {name}: {r.stages[name]:.2f}s")
        print()

    write_report(results, agentic_detail)


if __name__ == "__main__":
    asyncio.run(main())
