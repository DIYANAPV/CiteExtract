"""Evaluate BM25 vs Dense vs Hybrid retrieval for citation-passage matching.

Compares all three retrieval strategies on the same labeled data and reports
MRR, Recall@1, Recall@3, and Recall@5 for each.

Usage:
    python scripts/eval_retrieval.py

Settings — edit the variables below:
"""

# ─── Settings ────────────────────────────────────────────────────────────────
LABELED_PATH = "data/eval/passage_labels.json"  # same format as eval_rrf_k.py
DENSE_MODEL = "all-MiniLM-L6-v2"
TOP_K_VALUES = [1, 3, 5]
# ─────────────────────────────────────────────────────────────────────────────

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.verification.comprehension import (
    chunk_text,
    retrieve_passages_bm25,
    retrieve_passages_dense,
    retrieve_passages_hybrid,
)
from src.models.comprehension import Chunk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_labeled_data(path: str) -> list[dict]:
    """Load labeled data. Same format as eval_rrf_k.py."""
    with open(path) as f:
        return json.load(f)


def find_gold_chunk_index(chunks: list[Chunk], gold_passage: str) -> int | None:
    """Find which chunk index contains the gold passage."""
    gold_lower = gold_passage.lower().strip()
    for i, chunk in enumerate(chunks):
        if gold_lower in chunk.text.lower():
            return i
    # Fuzzy fallback
    gold_words = set(gold_lower.split())
    best_idx = None
    best_overlap = 0
    for i, chunk in enumerate(chunks):
        chunk_words = set(chunk.text.lower().split())
        overlap = len(gold_words & chunk_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = i
    if best_overlap >= len(gold_words) * 0.5:
        return best_idx
    return None


def evaluate_retriever(
    name: str,
    data: list[dict],
    retrieve_fn,
) -> dict:
    """Run a retriever on all labeled pairs and compute metrics."""
    metrics = {k: {"rr_sum": 0.0, "recall_sum": 0.0} for k in TOP_K_VALUES}
    valid = 0

    for item in data:
        sections = item.get("sections")
        chunks = chunk_text(item["paper_text"], sections=sections)
        if not chunks:
            continue

        gold_idx = find_gold_chunk_index(chunks, item["gold_passage"])
        if gold_idx is None:
            log.warning(f"[{name}] Gold not found: {item['gold_passage'][:60]}...")
            continue

        # Retrieve
        max_k = max(TOP_K_VALUES)
        results = retrieve_fn(item["citing_sentence"], chunks, top_k=max_k)

        # Map results to chunk list indices
        chunk_idx_map = {c.paragraph_index: i for i, c in enumerate(chunks)}
        ranked = [chunk_idx_map[r.chunk.paragraph_index] for r in results]

        # Compute metrics at each k
        for k in TOP_K_VALUES:
            top_k_ranked = ranked[:k]
            # MRR (only matters for k >= rank of gold)
            rr = 0.0
            for rank, idx in enumerate(ranked, 1):
                if idx == gold_idx:
                    rr = 1.0 / rank
                    break
            metrics[k]["rr_sum"] += rr
            # Recall@k
            metrics[k]["recall_sum"] += (1.0 if gold_idx in top_k_ranked else 0.0)

        valid += 1

    if valid == 0:
        return {"name": name, "valid": 0, "results": {}}

    results = {}
    for k in TOP_K_VALUES:
        results[f"recall@{k}"] = round(metrics[k]["recall_sum"] / valid, 4)
    # MRR is the same regardless of k cutoff (it's rank of gold in full list)
    mrr = round(metrics[max(TOP_K_VALUES)]["rr_sum"] / valid, 4)
    results["mrr"] = mrr

    return {"name": name, "valid": valid, "results": results}


def main():
    path = Path(LABELED_PATH)
    if not path.exists():
        print(f"Labeled data not found at {path}")
        print()
        print("Create a JSON file with this format:")
        print(json.dumps([{
            "citing_sentence": "Smith et al. showed transformers outperform RNNs.",
            "paper_text": "Full text of the cited paper...",
            "sections": [{"name": "Introduction", "text": "..."}],
            "gold_passage": "substring that appears in one of the chunks",
        }], indent=2))
        print(f"\nSave it to {LABELED_PATH} and re-run.")
        sys.exit(1)

    data = load_labeled_data(str(path))
    log.info(f"Loaded {len(data)} labeled pairs from {path}")

    # Define retrievers
    def bm25_fn(query, chunks, top_k):
        return retrieve_passages_bm25(query, chunks, top_k=top_k)

    def dense_fn(query, chunks, top_k):
        return retrieve_passages_dense(query, chunks, top_k=top_k, model_name=DENSE_MODEL)

    def hybrid_fn(query, chunks, top_k):
        return retrieve_passages_hybrid(query, chunks, top_k=top_k, model_name=DENSE_MODEL)

    retrievers = [
        ("BM25", bm25_fn),
        ("Dense", dense_fn),
        ("Hybrid (BM25+Dense+RRF)", hybrid_fn),
    ]

    all_results = []
    for name, fn in retrievers:
        log.info(f"Evaluating {name}...")
        result = evaluate_retriever(name, data, fn)
        all_results.append(result)
        log.info(f"  {name}: {result['results']}")

    # Summary table
    print("\n" + "=" * 80)
    print(f"  RETRIEVAL COMPARISON — {len(data)} pairs")
    print("=" * 80)

    header = f"{'Method':<28}"
    for k in TOP_K_VALUES:
        header += f"{'Recall@' + str(k):<12}"
    header += f"{'MRR':<12} {'Valid'}"
    print(f"\n{header}")
    print("-" * 80)

    for r in all_results:
        row = f"{r['name']:<28}"
        for k in TOP_K_VALUES:
            val = r["results"].get(f"recall@{k}", 0)
            row += f"{val:<12.4f}"
        row += f"{r['results'].get('mrr', 0):<12.4f} {r['valid']}"
        print(row)

    # Winner
    best = max(all_results, key=lambda r: r["results"].get("mrr", 0))
    print(f"\nBest by MRR: {best['name']} (MRR={best['results'].get('mrr', 0):.4f})")

    # Recommendation
    bm25_mrr = all_results[0]["results"].get("mrr", 0)
    hybrid_mrr = all_results[2]["results"].get("mrr", 0)
    if hybrid_mrr > bm25_mrr + 0.05:
        print("Recommendation: Use hybrid (meaningful improvement over BM25)")
    elif hybrid_mrr > bm25_mrr:
        print("Recommendation: Hybrid is marginally better — keep for robustness")
    else:
        print("Recommendation: BM25-only is sufficient (set dense_model: null in config)")

    # Save
    out_dir = Path("data/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "retrieval_comparison.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
