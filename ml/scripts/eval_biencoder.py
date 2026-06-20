"""
eval_biencoder.py
───────────────────
Phase 2: evaluate the fine-tuned bi-encoder on the validation set.

Metrics: Recall@10, Recall@100, MRR — broken out per country and language.

Encodes all val anchors (resumes) and positives (jobs), then for each anchor
finds the top-K nearest neighbors and checks if the true positive is among them.

Usage:
    python -u ml/scripts/eval_biencoder.py
    python -u ml/scripts/eval_biencoder.py --model ml/models/exported/biencoder
    python -u ml/scripts/eval_biencoder.py --top-k 100
"""

import argparse
import json
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
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate bi-encoder retrieval quality")
    parser.add_argument("--model", default="ml/models/exported/biencoder", help="Trained model path")
    parser.add_argument("--val-data", default="ml/data/splits/biencoder_val.jsonl")
    parser.add_argument("--top-k", type=int, default=100, help="Top-K for recall (default: 100)")
    parser.add_argument("--batch-size", type=int, default=64, help="Encoding batch size (default: 64)")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    print(f"Loading model from {args.model} ...")
    model = SentenceTransformer(args.model, trust_remote_code=True)

    print(f"Loading validation data from {args.val_data} ...")
    val_records = list(iter_jsonl(args.val_data))
    print(f"  {len(val_records):,} validation triples")

    if not val_records:
        print("No validation data found.")
        return

    # Extract texts
    anchors = [r["anchor"] for r in val_records]
    positives = [r["positive"] for r in val_records]

    # Also collect all negatives as additional candidates to search against
    negatives = [r["negative"] for r in val_records]

    # Build candidate pool: all positives + all negatives (deduplicated)
    all_candidates = positives + negatives
    candidate_texts = list(set(all_candidates))
    print(f"  Candidate pool: {len(candidate_texts):,} unique texts")

    # Map each positive to its index in candidate_texts
    candidate_to_idx = {text: idx for idx, text in enumerate(candidate_texts)}
    true_positive_indices = [candidate_to_idx[p] for p in positives]

    # Encode
    print(f"Encoding {len(anchors):,} anchors ...")
    anchor_embs = model.encode(anchors, batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True)

    print(f"Encoding {len(candidate_texts):,} candidates ...")
    candidate_embs = model.encode(candidate_texts, batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True)

    # Compute similarity and retrieval metrics
    print(f"Computing Recall@10, Recall@{args.top_k}, MRR ...")
    recall_at_10 = 0
    recall_at_k = 0
    mrr_sum = 0.0

    # Per-country/language tracking
    country_metrics: dict[str, list] = defaultdict(list)

    for i in range(len(anchors)):
        # Cosine similarity (embeddings are normalized, so dot product = cosine)
        sims = anchor_embs[i] @ candidate_embs.T
        # Get top-K indices
        top_indices = np.argsort(-sims)[:args.top_k]

        true_idx = true_positive_indices[i]
        rank = np.where(top_indices == true_idx)[0]

        if len(rank) > 0:
            rank_pos = int(rank[0]) + 1  # 1-indexed
            mrr_sum += 1.0 / rank_pos
            if rank_pos <= 10:
                recall_at_10 += 1
            recall_at_k += 1
        # else: true positive not in top-K

        # Track per record for country breakdown
        record = val_records[i]
        # Extract country from anchor text prefix if available
        anchor_text = record.get("anchor", "")
        country = "XX"
        if "[country=" in anchor_text:
            start = anchor_text.index("[country=") + 9
            end = anchor_text.index("]", start)
            country = anchor_text[start:end]
        country_metrics[country].append(1 if len(rank) > 0 and int(rank[0]) + 1 <= args.top_k else 0)

    n = len(anchors)
    print(f"\n{'='*60}")
    print(f"Results ({n:,} queries, {len(candidate_texts):,} candidates)")
    print(f"{'='*60}")
    print(f"  Recall@10:  {recall_at_10/n:.4f} ({recall_at_10}/{n})")
    print(f"  Recall@{args.top_k}: {recall_at_k/n:.4f} ({recall_at_k}/{n})")
    print(f"  MRR:        {mrr_sum/n:.4f}")

    # Gate check
    recall_100 = recall_at_k / n
    if recall_100 >= 0.85:
        print(f"\n  GATE PASSED: Recall@{args.top_k} = {recall_100:.4f} >= 0.85")
    else:
        print(f"\n  GATE FAILED: Recall@{args.top_k} = {recall_100:.4f} < 0.85")

    # Per-country breakdown
    if len(country_metrics) > 1:
        print(f"\nRecall@{args.top_k} by country:")
        sorted_countries = sorted(country_metrics.items(), key=lambda x: -len(x[1]))
        for country, hits in sorted_countries[:20]:
            r = sum(hits) / len(hits) if hits else 0
            print(f"  {country:>5}: {r:.4f} ({sum(hits)}/{len(hits)})")

    print("\nDone.")


if __name__ == "__main__":
    main()
