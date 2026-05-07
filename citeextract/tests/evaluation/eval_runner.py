
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from citeextract.pipeline import run_pipeline
from citeextract.models.report import PaperReport


EVAL_DIR = Path(__file__).parent
DATASETS_DIR = EVAL_DIR / "datasets"
RESULTS_DIR = EVAL_DIR / "results"

ALL_VERDICTS = [
    "FABRICATED", "VALID", "UNVERIFIABLE",
]


def compute_metrics(
    ground_truth: dict[str, str],
    predictions: dict[str, str],
) -> dict:
    results: dict = {}

    for v in ALL_VERDICTS:
        tp = sum(1 for k in ground_truth if ground_truth[k] == v and predictions.get(k) == v)
        fp = sum(1 for k in predictions if predictions[k] == v and ground_truth.get(k) != v)
        fn = sum(1 for k in ground_truth if ground_truth[k] == v and predictions.get(k) != v)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        if tp + fp + fn > 0:
            results[v] = {
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
                "tp": tp, "fp": fp, "fn": fn,
            }

    correct = sum(1 for k in ground_truth if predictions.get(k) == ground_truth[k])
    total = len(ground_truth)
    results["_overall"] = {
        "accuracy": round(correct / total, 3) if total > 0 else 0.0,
        "correct": correct,
        "total": total,
    }

    return results


async def evaluate_controlled() -> dict:
    bib_path = DATASETS_DIR / "controlled.bib"
    gt_path = DATASETS_DIR / "controlled_ground_truth.json"

    if not bib_path.exists() or not gt_path.exists():
        print(f"Error: Dataset files not found in {DATASETS_DIR}")
        return {}

    with open(gt_path) as f:
        gt_data = json.load(f)
    annotations = gt_data["annotations"]
    ground_truth = {ref_id: ann["expected"] for ref_id, ann in annotations.items()}

    print(f"Running pipeline on {bib_path}...")
    report = await run_pipeline(str(bib_path), mode="quick")

    predictions: dict[str, str] = {}
    for v in report.verdicts:
        predictions[v.ref_id] = v.verdict

    print(f"\n{'Ref ID':<30} {'Expected':<20} {'Predicted':<20} {'Match'}")
    print("-" * 80)
    for ref_id in sorted(ground_truth.keys()):
        expected = ground_truth[ref_id]
        predicted = predictions.get(ref_id, "MISSING")
        match = "OK" if expected == predicted else "WRONG"
        print(f"{ref_id:<30} {expected:<20} {predicted:<20} {match}")

    metrics = compute_metrics(ground_truth, predictions)

    print(f"\n{'='*60}")
    print("METRICS")
    print(f"{'='*60}")
    overall = metrics.pop("_overall", {})
    print(f"Overall accuracy: {overall.get('accuracy', 0):.1%} ({overall.get('correct', 0)}/{overall.get('total', 0)})")
    print()
    for verdict, m in sorted(metrics.items()):
        print(f"  {verdict:<20} P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1']:.2f}  (TP={m['tp']} FP={m['fp']} FN={m['fn']})")

    metrics["_overall"] = overall
    return metrics


async def evaluate_llm_papers() -> dict:
    import os
    base = Path(os.environ.get(
        "LLM_PAPERS_DIR",
        Path(__file__).resolve().parents[2] / "data" / "llm_papers" / "latex",
    ))
    if not base.exists():
        print(f"LLM papers directory not found: {base}")
        return {}

    all_results: dict = {}

    for paper_dir in sorted(base.iterdir()):
        bib_files = list(paper_dir.glob("*.bib"))
        if not bib_files:
            continue

        bib_path = bib_files[0]
        paper_name = paper_dir.name
        print(f"\n{'='*60}")
        print(f"Paper: {paper_name}")
        print(f"{'='*60}")

        report = await run_pipeline(str(bib_path), mode="quick")

        s = report.summary
        print(f"  References: {s.total_checked}")
        print(f"  Integrity: {s.integrity_score:.0%}")
        print(f"  Verdicts: {s.by_verdict}")

        all_results[paper_name] = {
            "total": s.total_checked,
            "integrity": s.integrity_score,
            "by_verdict": s.by_verdict,
        }

    return all_results


def save_results(results: dict, name: str) -> str:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{name}_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")
    return str(path)


async def main():
    args = sys.argv[1:]

    if "--llm-papers" in args:
        results = await evaluate_llm_papers()
        if results:
            save_results(results, "llm_papers")
    else:
        results = await evaluate_controlled()
        if results:
            save_results(results, "controlled")


if __name__ == "__main__":
    asyncio.run(main())
