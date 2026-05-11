
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

from experiment.paper.semantic.prompt_builder import CONDITIONS

RESULTS_DIR = _PAPER_ROOT / "results"
DEFAULT_MODELS = (
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-5-min",
    "gpt-5-med",
    "gpt-5.5-min",
    "gpt-5.5-med",
    "Qwen3-8B",
    "Llama-3.1-8B-Instruct",
)

CONDITION_LABELS = {
    "title_only":              "Title only",
    "title_abstract":          "Title + Abstract",
    "title_abstract_passages": "Title + Abstract + Passages",
}

PRODUCTION_CELL = ("gpt-4o-mini", "title_abstract_passages")


def _csv_path(model: str, condition: str, suffix: str = "") -> Path:
    safe = model.replace("/", "_")
    return RESULTS_DIR / f"semantic_{safe}_{condition}{suffix}.csv"


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cell_metrics(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    correct = sum(1 for r in rows if r["predicted_verdict"] == r["gold_label"])
    tp_v = sum(1 for r in rows if r["gold_label"] == "VALID" and r["predicted_verdict"] == "VALID")
    fp_v = sum(1 for r in rows if r["gold_label"] == "MISREPRESENTED" and r["predicted_verdict"] == "VALID")
    fn_v = sum(1 for r in rows if r["gold_label"] == "VALID" and r["predicted_verdict"] == "MISREPRESENTED")
    tp_m = sum(1 for r in rows if r["gold_label"] == "MISREPRESENTED" and r["predicted_verdict"] == "MISREPRESENTED")

    def _f1(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    vp, vr, vf = _f1(tp_v, fp_v, fn_v)
    mp, mr, mf = _f1(tp_m, fn_v, fp_v)
    cost = sum(float(r["cost_usd"] or 0) for r in rows)
    lat = sum(float(r["latency_seconds"] or 0) for r in rows)
    err = sum(1 for r in rows if r.get("error"))
    return {
        "n": n,
        "accuracy": correct / n,
        "macro_f1": (vf + mf) / 2,
        "valid":  {"precision": vp, "recall": vr, "f1": vf, "support": tp_v + fn_v},
        "misrep": {"precision": mp, "recall": mr, "f1": mf, "support": tp_m + fp_v},
        "errors": err,
        "total_cost_usd": round(cost, 4),
        "total_latency_seconds": round(lat, 1),
    }


def render_latex(
    metrics: dict[tuple[str, str], dict],
    models: tuple[str, ...] = DEFAULT_MODELS,
) -> str:
    lines = []
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append(
        "Model & "
        + " & ".join(CONDITION_LABELS[c] for c in CONDITIONS)
        + " \\\\"
    )
    lines.append("\\midrule")
    for m in models:
        cells: list[str] = []
        for c in CONDITIONS:
            mt = metrics.get((m, c), {})
            if not mt or mt.get("n", 0) == 0:
                cells.append("--")
                continue
            v = mt["accuracy"] * 100
            txt = f"{v:.2f}"
            if (m, c) == PRODUCTION_CELL:
                txt = f"\\textbf{{{txt}}}"
            cells.append(txt)
        lines.append(f"{_latex_safe(m)} & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def render_latex_detailed(
    metrics: dict[tuple[str, str], dict],
    models: tuple[str, ...] = DEFAULT_MODELS,
) -> str:
    lines = []
    lines.append("\\begin{tabular}{llrrrrrr}")
    lines.append("\\toprule")
    lines.append(
        "Model & Condition & Acc & Macro-F1 & V-F1 & M-F1 & Time (min) & Cost (\\$) \\\\"
    )
    lines.append("\\midrule")
    for m in models:
        first_row_for_model = True
        for c in CONDITIONS:
            mt = metrics.get((m, c), {})
            cond_label = CONDITION_LABELS[c]
            model_cell = _latex_safe(m) if first_row_for_model else ""
            if not mt or mt.get("n", 0) == 0:
                lines.append(
                    f"{model_cell} & {cond_label} & -- & -- & -- & -- & -- & -- \\\\"
                )
                first_row_for_model = False
                continue
            acc = mt["accuracy"] * 100
            macro = mt["macro_f1"] * 100
            vf = mt["valid"]["f1"] * 100
            mf = mt["misrep"]["f1"] * 100
            time_min = mt["total_latency_seconds"] / 60.0
            cost = mt["total_cost_usd"]
            cost_cell = f"{cost:.2f}" if cost > 0 else "--"
            row_metrics = (
                f"{acc:.2f} & {macro:.2f} & {vf:.2f} & {mf:.2f} "
                f"& {time_min:.1f} & {cost_cell}"
            )
            if (m, c) == PRODUCTION_CELL:
                row_metrics = (
                    f"\\textbf{{{acc:.2f}}} & \\textbf{{{macro:.2f}}} "
                    f"& \\textbf{{{vf:.2f}}} & \\textbf{{{mf:.2f}}} "
                    f"& {time_min:.1f} & {cost_cell}"
                )
            lines.append(
                f"{model_cell} & {cond_label} & {row_metrics} \\\\"
            )
            first_row_for_model = False
        if m != models[-1]:
            lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def _latex_safe(s: str) -> str:
    return s.replace("_", r"\_")


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--smoke", action="store_true",
                    help="emit smoke-suffixed artifacts (no overwrite of full results)")
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    args = ap.parse_args()

    metrics: dict[tuple[str, str], dict] = {}
    for m in args.models:
        for c in CONDITIONS:
            rows = _read_rows(_csv_path(m, c))
            metrics[(m, c)] = cell_metrics(rows)

    suffix = "_smoke" if args.smoke else ""
    tex_path = RESULTS_DIR / f"tab_semantic_results{suffix}.tex"
    detailed_tex_path = RESULTS_DIR / f"tab_semantic_results_detailed{suffix}.tex"
    json_path = RESULTS_DIR / f"tab_semantic_results_full{suffix}.json"

    tex = render_latex(metrics, models=tuple(args.models))
    tex_detailed = render_latex_detailed(metrics, models=tuple(args.models))
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(tex, encoding="utf-8")
    detailed_tex_path.write_text(tex_detailed, encoding="utf-8")

    full_dict = {f"{m}__{c}": metrics[(m, c)] for m in args.models for c in CONDITIONS}
    json_path.write_text(json.dumps(full_dict, indent=2), encoding="utf-8")

    print(f"wrote {tex_path}")
    print(f"wrote {detailed_tex_path}")
    print(f"wrote {json_path}")
    print()
    print(tex_detailed)


if __name__ == "__main__":
    main()
