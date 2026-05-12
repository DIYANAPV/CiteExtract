"""Statistical significance + bootstrap CIs for the metadata and semantic
benchmark tables.

For every cell (system × condition) we:
  - load the per-instance CSV
  - convert each row to correct/incorrect (predicted_verdict == gold_label)
  - compute accuracy and a percentile bootstrap 95% CI (10k resamples)

Then we run McNemar's test (with continuity correction; exact binomial when
discordant pairs are < 25) for the targeted pairs:
  - metadata: every system vs CiteExtract+gpt-4o-mini AND vs CiteExtract+gpt-5.5(med)
  - semantic: title-only vs +passages within each model (retrieval effect)
  - semantic: gpt-4o vs gpt-4o-mini in the +passages condition (model-scale-with-retrieval)

Outputs:
  - results/significance/significance.json (all numbers)
  - results/significance/significance_summary.md (readable for the paper)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, chi2

_THIS = Path(__file__).resolve()
_PAPER_ROOT = _THIS.parent.parent
_RESULTS = _PAPER_ROOT / "results"
_METADATA_CSV_DIR = _RESULTS / "metadata" / "csv"
_SEMANTIC_CSV_DIR = _RESULTS / "semantic" / "csv"
_SIGNIFICANCE_DIR = _RESULTS / "significance"

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20260510

METADATA_CELLS: list[tuple[str, str, str]] = [
    ("llama_3_1_8B",                "metadata_Llama-3.1-8B-Instruct_llm_only.csv",      "Llama-3.1-8B"),
    ("qwen3_8B",                    "metadata_Qwen3-8B_llm_only.csv",                   "Qwen3-8B"),
    ("gpt-4o-mini",                 "metadata_gpt-4o-mini_llm_only.csv",                "GPT-4o-mini"),
    ("gpt-4o-mini+search",          "metadata_gpt-4o-mini_llm_with_search.csv",         "GPT-4o-mini + search"),
    ("gpt-4o",                      "metadata_gpt-4o_llm_only.csv",                     "GPT-4o"),
    ("gpt-4o+search",               "metadata_gpt-4o_llm_with_search.csv",              "GPT-4o + search"),
    ("gpt-5(min)",                  "metadata_gpt-5-min_llm_only.csv",                  "GPT-5 (min)"),
    ("gpt-5+search(min)",           "metadata_gpt-5-min_llm_with_search.csv",           "GPT-5 + search (min)"),
    ("gpt-5(med)",                  "metadata_gpt-5-med_llm_only.csv",                  "GPT-5 (med)"),
    ("gpt-5+search(med)",           "metadata_gpt-5-med_llm_with_search.csv",           "GPT-5 + search (med)"),
    ("gpt-5.5(min)",                "metadata_gpt-5.5-min_llm_only.csv",                "GPT-5.5 (min)"),
    ("gpt-5.5+search(min)",         "metadata_gpt-5.5-min_llm_with_search.csv",         "GPT-5.5 + search (min)"),
    ("gpt-5.5(med)",                "metadata_gpt-5.5-med_llm_only.csv",                "GPT-5.5 (med)"),
    ("gpt-5.5+search(med)",         "metadata_gpt-5.5-med_llm_with_search.csv",         "GPT-5.5 + search (med)"),
    ("citeextract+gpt-4o-mini",     "metadata_our_system_production.csv",               "CiteExtract + GPT-4o-mini"),
    ("citeextract+gpt-5-mini",      "metadata_our_system_production_gpt5mini.csv",      "CiteExtract + GPT-5-mini"),
    ("citeextract+gpt-5.5(med)",    "metadata_our_system_production_gpt55med.csv",      "CiteExtract + GPT-5.5 (med)"),
]

CITEEXTRACT_REFS = ["citeextract+gpt-4o-mini", "citeextract+gpt-5.5(med)"]

SEMANTIC_MODELS = [
    ("Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
    ("Qwen3-8B",              "Qwen3-8B"),
    ("gpt-4o-mini",           "GPT-4o-mini"),
    ("gpt-4o",                "GPT-4o"),
    ("gpt-5-min",             "GPT-5 (min)"),
    ("gpt-5-med",             "GPT-5 (med)"),
    ("gpt-5.5-min",           "GPT-5.5 (min)"),
    ("gpt-5.5-med",           "GPT-5.5 (med)"),
]
SEMANTIC_CONDITIONS = [
    ("title_only",              "Title only"),
    ("title_abstract",          "Title + Abstract"),
    ("title_abstract_passages", "Title + Abs + Passages"),
]


def _load_correct_vector(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (instance_ids, correct) for a per-instance CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    ids: list[int] = []
    correct: list[int] = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids.append(int(row["instance_id"]))
            correct.append(1 if row["predicted_verdict"] == row["gold_label"] else 0)
    order = np.argsort(ids)
    return np.array(ids)[order], np.array(correct)[order]


def bootstrap_ci(correct: np.ndarray, n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(correct)
    idx = rng.integers(0, n, size=(n_boot, n))
    acc = correct[idx].mean(axis=1)
    lo, hi = np.quantile(acc, [0.025, 0.975])
    return float(lo), float(hi)


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """McNemar's test on paired correctness vectors. Continuity-corrected
    chi-square when discordant pairs (b+c) >= 25; exact binomial otherwise."""
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
    # b: a correct, b wrong; c: a wrong, b correct
    b_count = int(((a == 1) & (b == 0)).sum())
    c_count = int(((a == 0) & (b == 1)).sum())
    n_disc = b_count + c_count
    if n_disc == 0:
        return {"b": b_count, "c": c_count, "n_discordant": 0, "p_value": 1.0, "method": "exact"}
    if n_disc < 25:
        # exact two-sided binomial test on min(b, c) successes out of n_disc trials, p=0.5
        res = binomtest(min(b_count, c_count), n=n_disc, p=0.5, alternative="two-sided")
        return {"b": b_count, "c": c_count, "n_discordant": n_disc, "p_value": float(res.pvalue), "method": "exact"}
    chi2_stat = (abs(b_count - c_count) - 1) ** 2 / n_disc
    p = float(1 - chi2.cdf(chi2_stat, df=1))
    return {"b": b_count, "c": c_count, "n_discordant": n_disc, "p_value": p, "method": "chi2_cc", "chi2": float(chi2_stat)}


def stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def holm_correct(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in
    the original input order. Adjusted values are clipped to <= 1.0."""
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    adj = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = (n - rank) * p_values[idx]
        running_max = max(running_max, candidate)
        adj[idx] = min(running_max, 1.0)
    return adj


def _annotate_holm(tests: list[dict], family: str) -> None:
    """Add `p_holm` and `stars_holm` fields in-place; tag with `family`."""
    if not tests:
        return
    p_raw = [t["p_value"] for t in tests]
    p_adj = holm_correct(p_raw)
    for t, p in zip(tests, p_adj):
        t["p_holm"] = float(p)
        t["stars_holm"] = stars(p)
        t["family"] = family
        t["family_size"] = len(tests)


def _load_metadata() -> dict[str, dict]:
    cells: dict[str, dict] = {}
    for cid, fname, label in METADATA_CELLS:
        csv_path = _METADATA_CSV_DIR / fname
        ids, correct = _load_correct_vector(csv_path)
        acc = float(correct.mean())
        ci = bootstrap_ci(correct)
        cells[cid] = {
            "label": label, "n": int(len(correct)),
            "accuracy": acc, "ci95": ci,
            "ids": ids, "correct": correct,
        }
    return cells


def _load_semantic() -> dict[tuple[str, str], dict]:
    cells: dict[tuple[str, str], dict] = {}
    for model_id, model_label in SEMANTIC_MODELS:
        for cond_id, cond_label in SEMANTIC_CONDITIONS:
            safe = model_id.replace("/", "_")
            csv_path = _SEMANTIC_CSV_DIR / f"semantic_{safe}_{cond_id}.csv"
            if not csv_path.exists():
                cells[(model_id, cond_id)] = {"label": f"{model_label} × {cond_label}", "n": 0,
                                               "accuracy": None, "ci95": None,
                                               "ids": np.array([]), "correct": np.array([])}
                continue
            ids, correct = _load_correct_vector(csv_path)
            acc = float(correct.mean())
            ci = bootstrap_ci(correct)
            cells[(model_id, cond_id)] = {
                "label": f"{model_label} × {cond_label}",
                "model_label": model_label, "condition_label": cond_label,
                "n": int(len(correct)), "accuracy": acc, "ci95": ci,
                "ids": ids, "correct": correct,
            }
    return cells


def _aligned(a: dict, b: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Inner-join two cells on instance_id and return (a_correct, b_correct,
    alignment_info). alignment_info exposes n_a, n_b, n_pair, and whether any
    instances were dropped from either side."""
    common = np.intersect1d(a["ids"], b["ids"])
    a_idx = np.searchsorted(a["ids"], common)
    b_idx = np.searchsorted(b["ids"], common)
    info = {
        "n_a": int(len(a["ids"])),
        "n_b": int(len(b["ids"])),
        "n_pair": int(len(common)),
        "dropped_from_a": int(len(a["ids"]) - len(common)),
        "dropped_from_b": int(len(b["ids"]) - len(common)),
    }
    return a["correct"][a_idx], b["correct"][b_idx], info


def main() -> None:
    md = _load_metadata()
    sm = _load_semantic()

    out: dict = {"metadata": {}, "semantic": {}, "metadata_pairwise": [], "semantic_targeted": []}

    for cid, c in md.items():
        out["metadata"][cid] = {"label": c["label"], "n": c["n"],
                                 "accuracy_pct": round(c["accuracy"]*100, 2),
                                 "ci95_pct": [round(c["ci95"][0]*100, 2), round(c["ci95"][1]*100, 2)]}

    # Metadata: per-reference families. Each reference (gpt-4o-mini variant,
    # gpt-5.5 med variant) gets its own family of pairwise comparisons against
    # all non-pipeline systems. Pipeline-vs-pipeline tests live in their own
    # family so the multiple-comparisons correction is applied within rather
    # than across logically distinct comparison sets.
    metadata_baselines = [cid for cid in md if cid not in {"citeextract+gpt-4o-mini", "citeextract+gpt-5-mini", "citeextract+gpt-5.5(med)"}]

    def _pair_md(other_cid: str, ref_cid: str) -> dict:
        ra, rb, info = _aligned(md[other_cid], md[ref_cid])
        mc = mcnemar(rb, ra)  # b = ref right, other wrong
        mc.update({
            "system": other_cid, "system_label": md[other_cid]["label"],
            "reference": ref_cid, "reference_label": md[ref_cid]["label"],
            "diff_pct": round((md[ref_cid]["accuracy"] - md[other_cid]["accuracy"]) * 100, 2),
            "stars": stars(mc["p_value"]),
            **info,
        })
        return mc

    for ref in CITEEXTRACT_REFS:
        family_label = f"metadata_vs_{ref}"
        family: list[dict] = [_pair_md(cid, ref) for cid in metadata_baselines]
        _annotate_holm(family, family_label)
        out["metadata_pairwise"].extend(family)

    pipeline_ids = ["citeextract+gpt-4o-mini", "citeextract+gpt-5-mini", "citeextract+gpt-5.5(med)"]
    pipeline_family: list[dict] = []
    for i in range(len(pipeline_ids)):
        for j in range(i + 1, len(pipeline_ids)):
            a_id, b_id = pipeline_ids[i], pipeline_ids[j]
            pipeline_family.append(_pair_md(a_id, b_id))  # ref = b_id
    _annotate_holm(pipeline_family, "metadata_pipeline_pipeline")
    out["metadata_pairwise"].extend(pipeline_family)

    for (mid, cid), c in sm.items():
        if c["n"] == 0:
            continue
        out["semantic"][f"{mid}__{cid}"] = {
            "label": c["label"], "model": c.get("model_label"), "condition": c.get("condition_label"),
            "n": c["n"], "accuracy_pct": round(c["accuracy"]*100, 2),
            "ci95_pct": [round(c["ci95"][0]*100, 2), round(c["ci95"][1]*100, 2)],
        }

    # Semantic family A: within each model, title_only vs +passages (retrieval effect)
    sem_retrieval: list[dict] = []
    for mid, mlab in SEMANTIC_MODELS:
        a = sm.get((mid, "title_only"))
        b = sm.get((mid, "title_abstract_passages"))
        if not a or not b or a["n"] == 0 or b["n"] == 0:
            continue
        ra, rb, info = _aligned(a, b)
        mc = mcnemar(rb, ra)  # b = +passages right, title-only wrong
        mc.update({"name": f"{mlab}: title-only vs +passages",
                   "model": mlab,
                   "diff_pct": round((b["accuracy"] - a["accuracy"]) * 100, 2),
                   "acc_a_pct": round(a["accuracy"] * 100, 2),
                   "acc_b_pct": round(b["accuracy"] * 100, 2),
                   "stars": stars(mc["p_value"]),
                   **info})
        sem_retrieval.append(mc)
    _annotate_holm(sem_retrieval, "semantic_retrieval_effect")
    out["semantic_targeted"].extend(sem_retrieval)

    # Semantic family B: cross-model in the +passages condition. Production
    # cell (gpt-4o-mini × passages, the bolded cell in Table 2) is the ref.
    # Tests every other model's +passages row against it.
    PROD_SEMANTIC = ("gpt-4o-mini", "title_abstract_passages")
    prod = sm.get(PROD_SEMANTIC)
    sem_prod_vs_others: list[dict] = []
    if prod and prod["n"]:
        for mid, mlab in SEMANTIC_MODELS:
            if mid == PROD_SEMANTIC[0]:
                continue
            other = sm.get((mid, "title_abstract_passages"))
            if not other or other["n"] == 0:
                continue
            ra, rb, info = _aligned(prod, other)
            mc = mcnemar(rb, ra)  # b = other right, prod wrong  (ref=other)
            mc.update({"name": f"{mlab} vs GPT-4o-mini in +passages",
                       "diff_pct": round((other["accuracy"] - prod["accuracy"]) * 100, 2),
                       "acc_a_pct": round(prod["accuracy"] * 100, 2),
                       "acc_b_pct": round(other["accuracy"] * 100, 2),
                       "stars": stars(mc["p_value"]),
                       **info})
            sem_prod_vs_others.append(mc)
    _annotate_holm(sem_prod_vs_others, "semantic_prod_vs_others_in_passages")
    out["semantic_targeted"].extend(sem_prod_vs_others)

    # Semantic family C: rhetorically-load-bearing cross-model pairs that
    # are NOT against the production cell. Each is a "scale doesn't matter"
    # narrative beat. Tested as its own small family.
    SEM_NARRATIVE_PAIRS = [
        (("gpt-4o-mini", "title_abstract_passages"), ("gpt-4o", "title_abstract_passages"),
         "GPT-4o vs GPT-4o-mini in +passages (scale-with-retrieval)"),
        (("gpt-4o", "title_abstract_passages"), ("gpt-5-min", "title_abstract_passages"),
         "GPT-5(min) vs GPT-4o in +passages (newer-frontier-doesnt-help)"),
        (("gpt-5-min", "title_abstract_passages"), ("gpt-5.5-min", "title_abstract_passages"),
         "GPT-5.5(min) vs GPT-5(min) in +passages"),
        (("Qwen3-8B", "title_abstract_passages"), ("gpt-4o", "title_abstract_passages"),
         "GPT-4o vs Qwen3-8B in +passages (open-weight-matches-frontier)"),
    ]
    sem_narrative: list[dict] = []
    for a_key, b_key, name in SEM_NARRATIVE_PAIRS:
        a = sm.get(a_key); b = sm.get(b_key)
        if not a or not b or a["n"] == 0 or b["n"] == 0:
            continue
        ra, rb, info = _aligned(a, b)
        mc = mcnemar(rb, ra)
        mc.update({"name": name,
                   "diff_pct": round((b["accuracy"] - a["accuracy"]) * 100, 2),
                   "acc_a_pct": round(a["accuracy"] * 100, 2),
                   "acc_b_pct": round(b["accuracy"] * 100, 2),
                   "stars": stars(mc["p_value"]),
                   **info})
        sem_narrative.append(mc)
    _annotate_holm(sem_narrative, "semantic_narrative_pairs")
    out["semantic_targeted"].extend(sem_narrative)

    json_path = _SIGNIFICANCE_DIR / "significance.json"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {json_path}")

    md_lines: list[str] = []
    md_lines.append("# Statistical significance summary\n")
    md_lines.append(
        f"_Bootstrap CIs: {BOOTSTRAP_N:,} resamples, percentile method. "
        f"McNemar's test: continuity-corrected chi-square (exact binomial when "
        f"discordant pairs < 25). Holm-Bonferroni correction is applied within "
        f"each comparison family (one family per natural rhetorical question). "
        f"Stars: \\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001, ns p>=0.05. "
        f"Stars-Holm column applies the same thresholds to the Holm-adjusted p-value._\n"
    )

    audit_rows: list[dict] = []
    for tests in (out["metadata_pairwise"], out["semantic_targeted"]):
        for t in tests:
            if t.get("dropped_from_a", 0) or t.get("dropped_from_b", 0):
                audit_rows.append(t)
    if audit_rows:
        md_lines.append("\n## ⚠ Instance-alignment audit")
        md_lines.append("\nThe following pairs had at least one instance present in one CSV but absent in the other. "
                        "McNemar uses only the paired intersection; non-paired predictions are excluded from the test.")
        md_lines.append("\n| Comparison | n_A | n_B | n_paired | dropped_A | dropped_B |")
        md_lines.append("|---|---|---|---|---|---|")
        for t in audit_rows:
            label = t.get("name") or f"{t.get('system_label','?')} vs {t.get('reference_label','?')}"
            md_lines.append(f"| {label} | {t['n_a']} | {t['n_b']} | {t['n_pair']} | {t['dropped_from_a']} | {t['dropped_from_b']} |")
    else:
        md_lines.append("\n## Instance-alignment audit\n")
        md_lines.append("All compared CSV pairs share identical instance ID sets — no instances dropped from any McNemar test.")

    md_lines.append("\n## Metadata table — accuracy with 95% CI")
    md_lines.append("\n| System | Acc % [95% CI] | n |")
    md_lines.append("|---|---|---|")
    for cid, c in md.items():
        lo, hi = c["ci95"]
        md_lines.append(f"| {c['label']} | {c['accuracy']*100:.2f} [{lo*100:.2f}, {hi*100:.2f}] | {c['n']} |")

    md_lines.append("\n## Metadata pairwise McNemar")
    by_family: dict[str, list[dict]] = {}
    for mc in out["metadata_pairwise"]:
        by_family.setdefault(mc["family"], []).append(mc)
    for fam, tests in by_family.items():
        ks = tests[0]["family_size"]
        md_lines.append(f"\n### {fam}  (K={ks} comparisons; Holm threshold for α=0.05)")
        md_lines.append("\n| System | vs Reference | n_pair | Δ (Ref − Sys) | b | c | p (raw) | p (Holm) | sig (raw) | sig (Holm) |")
        md_lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for mc in tests:
            md_lines.append(
                f"| {mc['system_label']} | {mc['reference_label']} | {mc['n_pair']} | "
                f"{mc['diff_pct']:+.2f} | {mc['b']} | {mc['c']} | "
                f"{mc['p_value']:.4g} | {mc['p_holm']:.4g} | {mc['stars']} | {mc['stars_holm']} |"
            )

    md_lines.append("\n## Semantic table — accuracy with 95% CI")
    md_lines.append("\n| Model × Condition | Acc % [95% CI] | n |")
    md_lines.append("|---|---|---|")
    for key in sorted(out["semantic"]):
        s = out["semantic"][key]
        lo, hi = s["ci95_pct"]
        md_lines.append(f"| {s['label']} | {s['accuracy_pct']:.2f} [{lo:.2f}, {hi:.2f}] | {s['n']} |")

    md_lines.append("\n## Semantic targeted pairs")
    sem_by_family: dict[str, list[dict]] = {}
    for mc in out["semantic_targeted"]:
        sem_by_family.setdefault(mc["family"], []).append(mc)
    for fam, tests in sem_by_family.items():
        ks = tests[0]["family_size"]
        md_lines.append(f"\n### {fam}  (K={ks} comparisons)")
        md_lines.append("\n| Comparison | n_pair | Acc A | Acc B | Δ | b | c | p (raw) | p (Holm) | sig (raw) | sig (Holm) |")
        md_lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for mc in tests:
            md_lines.append(
                f"| {mc['name']} | {mc['n_pair']} | {mc['acc_a_pct']:.2f} | {mc['acc_b_pct']:.2f} | "
                f"{mc['diff_pct']:+.2f} | {mc['b']} | {mc['c']} | "
                f"{mc['p_value']:.4g} | {mc['p_holm']:.4g} | {mc['stars']} | {mc['stars_holm']} |"
            )

    md_lines.append("\n## Suggested Evaluation Protocol paragraph (for the paper)\n")
    md_lines.append(
        "_For each pairwise comparison we report McNemar's test on per-instance "
        "correctness, with continuity correction for the chi-square approximation "
        "and an exact binomial test when fewer than 25 discordant pairs are "
        "observed. To control the family-wise error rate we apply the "
        "Holm-Bonferroni step-down correction within each comparison family "
        "(systems-vs-CiteExtract; pipeline-vs-pipeline; within-model retrieval "
        "effect; cross-model at fixed retrieval condition). We also report 95\\% "
        "bootstrap confidence intervals on each accuracy estimate from 10{,}000 "
        "resamples of the test set with replacement._\n"
    )

    md_path = _SIGNIFICANCE_DIR / "significance_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    sys.exit(main())
