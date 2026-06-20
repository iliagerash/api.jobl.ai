"""
build_reranker_data.py
───────────────────────
Phase 3 prep: convert teacher match scores into CrossEncoder training format.

Each example is a (resume_text, job_text, score) tuple where score is the
teacher's graded relevance: 0.0, 0.3, 0.7, or 1.0.

Input:  ml/data/teacher_labels/match_scores.jsonl
Output: ml/data/splits/reranker_train.jsonl
        ml/data/splits/reranker_val.jsonl

Usage:
    python -u ml/scripts/build_reranker_data.py
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def iter_jsonl(path: str):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reranker training data from teacher match scores")
    parser.add_argument("--scores-input", default="ml/data/teacher_labels/match_scores.jsonl")
    parser.add_argument("--output-dir", default="ml/data/splits")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading scored pairs from {args.scores_input} ...")
    examples = []
    score_dist = {0.0: 0, 0.3: 0, 0.7: 0, 1.0: 0}
    skipped = 0

    for record in iter_jsonl(args.scores_input):
        score = record.get("score")
        if score is None:
            skipped += 1
            continue

        resume_title = record.get("resume_title", "")
        job_title = record.get("job_title", "")

        # We need the full text, but match_scores only has titles.
        # Build text from what we have — the reranker will learn from
        # title-level signals. For richer text, the triples file has
        # resume_text and job_text.
        text_a = f"[type=resume] {resume_title}"
        text_b = f"[type=job] {job_title}"

        examples.append({
            "text_a": text_a,
            "text_b": text_b,
            "score": float(score),
            "resume_id": record.get("resume_id"),
            "job_id": record.get("job_id"),
        })

        # Track score distribution
        nearest = min(score_dist.keys(), key=lambda v: abs(v - score))
        score_dist[nearest] += 1

    print(f"  {len(examples):,} valid pairs loaded, {skipped:,} skipped")
    print(f"  Score distribution: {score_dist}")

    # Also load triples which have full text
    triples_path = args.scores_input.replace("match_scores", "match_triples")
    if os.path.exists(triples_path):
        print(f"Enriching with full text from {triples_path} ...")
        enriched = 0
        triple_map = {}
        for record in iter_jsonl(triples_path):
            key = (record.get("resume_id"), record.get("job_id"))
            triple_map[key] = record

        for ex in examples:
            key = (ex["resume_id"], ex["job_id"])
            if key in triple_map:
                triple = triple_map[key]
                ex["text_a"] = f"[type=resume] {triple.get('resume_title', '')} — {triple.get('resume_text', '')}"[:1000]
                ex["text_b"] = f"[type=job] {triple.get('job_title', '')} — {triple.get('job_text', '')}"[:1000]
                enriched += 1
        print(f"  Enriched {enriched:,} pairs with full text")

    # Split
    random.shuffle(examples)
    split_idx = int(len(examples) * (1 - args.val_fraction))
    train = examples[:split_idx]
    val = examples[split_idx:]

    for path, data, label in [
        (os.path.join(args.output_dir, "reranker_train.jsonl"), train, "train"),
        (os.path.join(args.output_dir, "reranker_val.jsonl"), val, "val"),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False))
                f.write("\n")
        print(f"  {label}: {len(data):,} pairs -> {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
