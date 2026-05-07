
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PAPER_ROOT = _THIS_FILE.parent.parent
_REPO_ROOT = _PAPER_ROOT.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

RESULTS_DIR = _PAPER_ROOT / "results"

CELLS: list[tuple[str, str, str]] = [
    ("gpt-4o-mini",           "llm_only",        "gpt-4o-mini"),
    ("gpt-4o-mini",           "llm_with_search", "gpt-4o-mini + web search"),
    ("gpt-4o",                "llm_only",        "gpt-4o"),
    ("gpt-4o",                "llm_with_search", "gpt-4o + web search"),
    ("Qwen3-8B",              "llm_only",        "Qwen3-8B"),
    ("Llama-3.1-8B-Instruct", "llm_only",        "Llama-3.1-8B-Instruct"),
    ("our_system",            "production",      "CiteExtract pipeline (ours)"),
]


def _csv_path(model: str, condition: str) -> Path:
    return RESULTS_DIR / f"metadata_{model}_{condition}.csv"


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def cell_metrics(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    correct = sum(1 for r in rows if r["predicted_verdict"] == r["gold_label"])
    tp_f = sum(1 for r in rows if r["gold_label"] == "fabricated" and r["predicted_verdict"] == "fabricated")
    fp_f = sum(1 for r in rows if r["gold_label"] == "valid"      and r["predicted_verdict"] == "fabricated")
    fn_f = sum(1 for r in rows if r["gold_label"] == "fabricated" and r["predicted_verdict"] == "valid")
    tp_v = sum(1 for r in rows if r["gold_label"] == "valid"      and r["predicted_verdict"] == "valid")
    fp_v, fn_v = fn_f, fp_f
    fp, fr, ff = _f1(tp_f, fp_f, fn_f)
    vp, vr, vf = _f1(tp_v, fp_v, fn_v)
    macro_f1 = (vf + ff) / 2
    n_search = sum(int(r.get("n_search_invocations", 0) or 0) for r in rows)
    cost = sum(float(r.get("cost_usd", 0) or 0) for r in rows)
    lat = sum(float(r.get("latency_seconds", 0) or 0) for r in rows)
    err = sum(1 for r in rows if r.get("error"))
    return {
        "n": n,
        "accuracy": correct / n,
        "macro_f1": macro_f1,
        "fab_precision": fp, "fab_recall": fr, "fab_f1": ff,
        "valid_precision": vp, "valid_recall": vr, "valid_f1": vf,
        "errors": err,
        "total_cost_usd": round(cost, 4),
        "total_latency_seconds": round(lat, 1),
        "avg_searches_per_ref": round(n_search / n, 3),
    }


def render_latex(metrics: dict[tuple[str, str], dict]) -> str:
    lines = []
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append("Method & Accuracy & Precision (Fab) & Recall (Fab) & F1 (Fab) & Searches/ref \\\\")
    lines.append("\\midrule")
    for model, condition, label in CELLS:
        m = metrics.get((model, condition), {})
        if not m or m.get("n", 0) == 0:
            lines.append(f"{_safe(label)} & -- & -- & -- & -- & -- \\\\")
            continue
        acc = m["accuracy"] * 100
        p = m["fab_precision"] * 100
        r_ = m["fab_recall"] * 100
        f1 = m["fab_f1"] * 100
        searches = (
            f"{m['avg_searches_per_ref']:.2f}" if condition == "llm_with_search" else "--"
        )
        bold = (model == "our_system")
        if bold:
            cells = [f"\\textbf{{{v:.2f}}}" for v in (acc, p, r_, f1)] + ["--"]
        else:
            cells = [f"{v:.2f}" for v in (acc, p, r_, f1)] + [searches]
        lines.append(f"{_safe(label)} & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def render_latex_detailed(metrics: dict[tuple[str, str], dict]) -> str:
    lines = []
    lines.append("\\begin{tabular}{lrrrrrr}")
    lines.append("\\toprule")
    lines.append(
        "Method & Acc & Macro-F1 & V-F1 & F-F1 & Time (min) & Cost (\\$) \\\\"
    )
    lines.append("\\midrule")
    for model, condition, label in CELLS:
        m = metrics.get((model, condition), {})
        if not m or m.get("n", 0) == 0:
            lines.append(f"{_safe(label)} & -- & -- & -- & -- & -- & -- \\\\")
            continue
        acc = m["accuracy"] * 100
        macro = m["macro_f1"] * 100
        vf = m["valid_f1"] * 100
        ff = m["fab_f1"] * 100
        time_min = m["total_latency_seconds"] / 60.0
        cost = m["total_cost_usd"]
        cost_cell = f"{cost:.2f}" if cost > 0 else "--"
        if model == "our_system":
            row = (
                f"\\textbf{{{acc:.2f}}} & \\textbf{{{macro:.2f}}} "
                f"& \\textbf{{{vf:.2f}}} & \\textbf{{{ff:.2f}}} "
                f"& {time_min:.1f} & {cost_cell}"
            )
        else:
            row = (
                f"{acc:.2f} & {macro:.2f} & {vf:.2f} & {ff:.2f} "
                f"& {time_min:.1f} & {cost_cell}"
            )
        lines.append(f"{_safe(label)} & {row} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def _safe(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    metrics: dict[tuple[str, str], dict] = {}
    for model, condition, _ in CELLS:
        rows = _read(_csv_path(model, condition))
        metrics[(model, condition)] = cell_metrics(rows)

    suffix = "_smoke" if args.smoke else ""
    tex_path = RESULTS_DIR / f"tab_metadata_results{suffix}.tex"
    detailed_tex_path = RESULTS_DIR / f"tab_metadata_results_detailed{suffix}.tex"
    json_path = RESULTS_DIR / f"tab_metadata_results_full{suffix}.json"

    tex = render_latex(metrics)
    tex_detailed = render_latex_detailed(metrics)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(tex, encoding="utf-8")
    detailed_tex_path.write_text(tex_detailed, encoding="utf-8")

    json_path.write_text(json.dumps(
        {f"{m}__{c}": metrics[(m, c)] for m, c, _ in CELLS},
        indent=2, default=str,
    ), encoding="utf-8")

    print(f"wrote {tex_path}")
    print(f"wrote {detailed_tex_path}")
    print(f"wrote {json_path}")
    print()
    print(tex_detailed)


if __name__ == "__main__":
    main()
