"""
eval_reranker.py
──────────────────
Phase 3: evaluate the fine-tuned CrossEncoder reranker.

Metrics: NDCG@10, MAP on the validation set.

For each resume in the val set, collects all jobs paired with it (at various
scores), reranks them using the CrossEncoder, and computes ranking metrics
against the teacher's graded relevance scores.

Usage:
    python -u ml/scripts/eval_reranker.py
    python -u ml/scripts/eval_reranker.py --model ml/models/exported/reranker
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def iter_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def dcg_at_k(relevances: list[float], k: int) -> float:
    """Compute DCG@k."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += rel / math.log2(i + 2)
    return dcg


def ndcg_at_k(predicted_relevances: list[float], true_relevances: list[float], k: int) -> float:
    """Compute NDCG@k."""
    dcg = dcg_at_k(predicted_relevances, k)
    ideal = dcg_at_k(sorted(true_relevances, reverse=True), k)
    if ideal == 0:
        return 0.0
    return dcg / ideal


def average_precision(predicted_relevances: list[float], threshold: float = 0.5) -> float:
    """Compute Average Precision for binary relevance (score >= threshold = relevant)."""
    relevant = 0
    ap_sum = 0.0
    for i, rel in enumerate(predicted_relevances):
        if rel >= threshold:
            relevant += 1
            ap_sum += relevant / (i + 1)
    if relevant == 0:
        return 0.0
    return ap_sum / relevant


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CrossEncoder reranker quality")
    parser.add_argument("--model", default="ml/models/exported/reranker", help="Trained model path")
    parser.add_argument("--val-data", default="ml/data/splits/reranker_val.jsonl")
    parser.add_argument("--k", type=int, default=10, help="K for NDCG@K (default: 10)")
    args = parser.parse_args()

    from sentence_transformers.cross_encoder import CrossEncoder

    print(f"Loading model from {args.model} ...")
    model = CrossEncoder(args.model)

    print(f"Loading validation data from {args.val_data} ...")
    val_records = list(iter_jsonl(args.val_data))
    print(f"  {len(val_records):,} validation pairs")

    if not val_records:
        print("No validation data found.")
        return

    # Group by resume_id to evaluate per-query ranking
    resume_groups: dict[int, list[dict]] = defaultdict(list)
    for record in val_records:
        rid = record.get("resume_id")
        if rid is not None:
            resume_groups[rid].append(record)

    # Filter to resumes with at least 2 paired jobs (need something to rank)
    rankable = {rid: pairs for rid, pairs in resume_groups.items() if len(pairs) >= 2}
    print(f"  {len(rankable):,} resumes with 2+ paired jobs (rankable)")

    if not rankable:
        print("Not enough multi-pair resumes for ranking evaluation.")
        print("Falling back to correlation analysis ...")

        # Simple correlation: predict scores and compare to teacher scores
        pairs = [(r["text_a"], r["text_b"]) for r in val_records]
        teacher_scores = [r["score"] for r in val_records]
        predicted = model.predict(pairs, show_progress_bar=True)

        from scipy.stats import spearmanr, pearsonr
        spearman, _ = spearmanr(teacher_scores, predicted)
        pearson, _ = pearsonr(teacher_scores, predicted)
        print(f"\n  Spearman correlation: {spearman:.4f}")
        print(f"  Pearson correlation:  {pearson:.4f}")

        if spearman >= 0.6:
            print(f"\n  GATE PASSED: Spearman = {spearman:.4f} >= 0.6")
        else:
            print(f"\n  GATE FAILED: Spearman = {spearman:.4f} < 0.6")
        return

    # Score all pairs with the reranker
    all_pairs = []
    pair_indices = []  # (resume_id, index_in_group)
    for rid, pairs in rankable.items():
        for i, record in enumerate(pairs):
            all_pairs.append((record["text_a"], record["text_b"]))
            pair_indices.append((rid, i))

    print(f"  Scoring {len(all_pairs):,} pairs ...")
    predicted_scores = model.predict(all_pairs, show_progress_bar=True)

    # Assign predicted scores back to groups
    for idx, (rid, i) in enumerate(pair_indices):
        rankable[rid][i]["predicted_score"] = float(predicted_scores[idx])

    # Compute NDCG@K and MAP per resume
    ndcg_scores = []
    ap_scores = []

    for rid, pairs in rankable.items():
        # Sort by predicted score (descending) = reranker's ranking
        pairs_sorted = sorted(pairs, key=lambda x: -x.get("predicted_score", 0))
        predicted_relevances = [p["score"] for p in pairs_sorted]
        true_relevances = [p["score"] for p in pairs]

        ndcg = ndcg_at_k(predicted_relevances, true_relevances, args.k)
        ap = average_precision(predicted_relevances, threshold=0.5)
        ndcg_scores.append(ndcg)
        ap_scores.append(ap)

    mean_ndcg = np.mean(ndcg_scores)
    mean_ap = np.mean(ap_scores)

    print(f"\n{'='*60}")
    print(f"Results ({len(rankable):,} rankable queries)")
    print(f"{'='*60}")
    print(f"  NDCG@{args.k}:  {mean_ndcg:.4f}")
    print(f"  MAP:       {mean_ap:.4f}")

    if mean_ndcg >= 0.7:
        print(f"\n  GATE PASSED: NDCG@{args.k} = {mean_ndcg:.4f} >= 0.7")
    else:
        print(f"\n  GATE FAILED: NDCG@{args.k} = {mean_ndcg:.4f} < 0.7")

    print("\nDone.")


if __name__ == "__main__":
    main()
