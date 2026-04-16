"""Evaluate RRF k parameter for citation-passage matching.

Sweeps k over {10, 20, 30, 60, 100} and measures MRR and Recall@3
on a labeled set of (citing_sentence, gold_passage) pairs.

Usage:
    python scripts/eval_rrf_k.py

Settings — edit the variables below:
"""

# ─── Settings ────────────────────────────────────────────────────────────────
LABELED_PATH = "data/eval/passage_labels.json"  # see format below
K_VALUES = [10, 20, 30, 60, 100]
DENSE_MODEL = "all-MiniLM-L6-v2"
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
    reciprocal_rank_fusion,
)
from src.models.comprehension import Chunk, ScoredChunk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_labeled_data(path: str) -> list[dict]:
    """Load labeled passage retrieval data.

    Expected format:
    [
        {
            "citing_sentence": "Smith et al. showed transformers outperform RNNs.",
            "paper_text": "Full text of the cited paper...",
            "sections": [{"name": "Introduction", "text": "..."}],  # optional
            "gold_passage": "substring that appears in one of the chunks"
        },
        ...
    ]
    """
    with open(path) as f:
        return json.load(f)


def find_gold_chunk_index(chunks: list[Chunk], gold_passage: str) -> int | None:
    """Find which chunk index contains the gold passage (substring match)."""
    gold_lower = gold_passage.lower().strip()
    for i, chunk in enumerate(chunks):
        if gold_lower in chunk.text.lower():
            return i
    # Fallback: find best overlap
    best_idx = None
    best_overlap = 0
    gold_words = set(gold_lower.split())
    for i, chunk in enumerate(chunks):
        chunk_words = set(chunk.text.lower().split())
        overlap = len(gold_words & chunk_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = i
    if best_overlap >= len(gold_words) * 0.5:
        return best_idx
    return None


def reciprocal_rank(ranked_indices: list[int], gold_index: int) -> float:
    """Compute reciprocal rank: 1/rank of the gold result."""
    for rank, idx in enumerate(ranked_indices, 1):
        if idx == gold_index:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_indices: list[int], gold_index: int, k: int = 3) -> float:
    """1 if gold is in top-k, else 0."""
    return 1.0 if gold_index in ranked_indices[:k] else 0.0


def evaluate_k(data: list[dict], k_value: int) -> dict:
    """Run hybrid retrieval with a specific RRF k and compute metrics."""
    mrr_sum = 0.0
    recall3_sum = 0.0
    valid = 0

    for item in data:
        sections = item.get("sections")
        chunks = chunk_text(item["paper_text"], sections=sections)
        if not chunks:
            continue

        gold_idx = find_gold_chunk_index(chunks, item["gold_passage"])
        if gold_idx is None:
            log.warning(f"Gold passage not found in chunks: {item['gold_passage'][:60]}...")
            continue

        # Run BM25 and dense
        bm25_results = retrieve_passages_bm25(item["citing_sentence"], chunks, top_k=10)
        dense_results = retrieve_passages_dense(
            item["citing_sentence"], chunks, top_k=10, model_name=DENSE_MODEL
        )

        # Build index maps
        chunk_idx_map = {c.paragraph_index: i for i, c in enumerate(chunks)}
        bm25_ranking = [(chunk_idx_map[r.chunk.paragraph_index], r.bm25_score) for r in bm25_results]
        dense_ranking = [(chunk_idx_map[r.chunk.paragraph_index], r.dense_score or 0.0) for r in dense_results]

        # Fuse with this k value
        merged = reciprocal_rank_fusion([bm25_ranking, dense_ranking], k=k_value)
        ranked_indices = [idx for idx, _ in merged]

        mrr_sum += reciprocal_rank(ranked_indices, gold_idx)
        recall3_sum += recall_at_k(ranked_indices, gold_idx, k=3)
        valid += 1

    if valid == 0:
        return {"k": k_value, "mrr": 0, "recall_at_3": 0, "valid": 0}

    return {
        "k": k_value,
        "mrr": round(mrr_sum / valid, 4),
        "recall_at_3": round(recall3_sum / valid, 4),
        "valid": valid,
    }


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
        print()
        print(f"Save it to {LABELED_PATH} and re-run.")
        sys.exit(1)

    data = load_labeled_data(str(path))
    log.info(f"Loaded {len(data)} labeled pairs from {path}")

    # Run for each k
    results = []
    for k in K_VALUES:
        log.info(f"Evaluating k={k}...")
        result = evaluate_k(data, k)
        results.append(result)
        log.info(f"  k={k}: MRR={result['mrr']:.4f}  Recall@3={result['recall_at_3']:.4f}  ({result['valid']} valid)")

    # Summary
    print("\n" + "=" * 70)
    print(f"  RRF k-SWEEP EVALUATION — {len(data)} pairs")
    print("=" * 70)
    print(f"\n{'k':<8} {'MRR':<12} {'Recall@3':<12} {'Valid'}")
    print("-" * 50)
    for r in results:
        print(f"{r['k']:<8} {r['mrr']:<12.4f} {r['recall_at_3']:<12.4f} {r['valid']}")

    best = max(results, key=lambda r: r["mrr"])
    print(f"\nBest k by MRR: {best['k']} (MRR={best['mrr']:.4f})")

    # Save results
    out_dir = Path("data/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rrf_k_sweep.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
