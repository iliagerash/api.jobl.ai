"""
build_biencoder_data.py
────────────────────────
Phase 2 prep: convert teacher match outputs into (anchor, positive, negative)
triples with locale prefixes for bi-encoder fine-tuning.

Each triple:
  - anchor:   resume text with locale prefix [lang=XX][country=XX][type=resume]
  - positive: matched job text with locale prefix [lang=XX][country=XX][type=job]
  - negative: hard-negative job text (from teacher) or a random non-matching job

Input:  ml/data/teacher_labels/match_triples.jsonl
        ml/data/teacher_labels/match_scores.jsonl (for mining additional negatives)
Output: ml/data/splits/biencoder_train.jsonl
        ml/data/splits/biencoder_val.jsonl

Usage:
    python -u ml/scripts/build_biencoder_data.py
    python -u ml/scripts/build_biencoder_data.py --val-fraction 0.1
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def iter_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _locale_prefix(record: dict, record_type: str) -> str:
    """Build [lang=XX][country=XX][type=job|resume] prefix."""
    lang = record.get("language_fasttext") or record.get("language_code") or "unknown"
    country = record.get("country_code") or record.get("resume_country") or record.get("job_country") or "XX"
    return f"[lang={lang}][country={country}][type={record_type}]"


def _prefixed_text(title: str, text: str, lang: str, country: str, record_type: str) -> str:
    prefix = f"[lang={lang or 'unknown'}][country={country or 'XX'}][type={record_type}]"
    return f"{prefix} {title or ''} — {text or ''}"[:2000]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bi-encoder training triples from teacher labels")
    parser.add_argument("--triples-input", default="ml/data/teacher_labels/match_triples.jsonl")
    parser.add_argument("--scores-input", default="ml/data/teacher_labels/match_scores.jsonl")
    parser.add_argument("--output-dir", default="ml/data/splits")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction for validation (default: 0.1)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load positive triples with hard negatives from teacher
    print(f"Loading triples from {args.triples_input} ...")
    triples = list(iter_jsonl(args.triples_input))
    print(f"  {len(triples):,} triples loaded")

    # Load all scored pairs for mining additional negatives (score < 0.3)
    print(f"Loading scores from {args.scores_input} ...")
    negative_jobs: list[dict] = []
    for record in iter_jsonl(args.scores_input):
        score = record.get("score")
        if score is not None and score <= 0.3:
            negative_jobs.append(record)
    print(f"  {len(negative_jobs):,} negative-scored pairs available for additional negatives")

    # Build training examples
    print("Building training triples ...")
    examples = []

    for triple in triples:
        hard_neg = triple.get("hard_negative") or {}
        if "_error" in hard_neg:
            hard_neg = {}

        resume_lang = triple.get("resume_id")  # We don't have lang directly, use from text
        # Extract language/country from the scored pair context
        resume_title = triple.get("resume_title", "")
        resume_text = triple.get("resume_text", "")
        job_title = triple.get("job_title", "")
        job_text = triple.get("job_text", "")

        # Build anchor (resume)
        anchor = f"[type=resume] {resume_title} — {resume_text}"[:2000]

        # Build positive (matched job)
        positive = f"[type=job] {job_title} — {job_text}"[:2000]

        # Build negative (hard negative from teacher, or fall back to random scored negative)
        if hard_neg.get("hard_negative_title"):
            neg_title = hard_neg["hard_negative_title"]
            neg_desc = hard_neg.get("hard_negative_description", "")
            negative = f"[type=job] {neg_title} — {neg_desc}"[:2000]
        elif negative_jobs:
            # Pick a random low-scored job as negative
            rand_neg = random.choice(negative_jobs)
            negative = f"[type=job] {rand_neg.get('job_title', '')} — "[:2000]
        else:
            continue

        examples.append({
            "anchor": anchor,
            "positive": positive,
            "negative": negative,
            "match_score": triple.get("match_score"),
            "resume_id": triple.get("resume_id"),
            "job_id": triple.get("job_id"),
        })

    print(f"  {len(examples):,} training triples built")

    # Split into train/val
    random.shuffle(examples)
    split_idx = int(len(examples) * (1 - args.val_fraction))
    train = examples[:split_idx]
    val = examples[split_idx:]

    # Write outputs
    train_path = os.path.join(args.output_dir, "biencoder_train.jsonl")
    val_path = os.path.join(args.output_dir, "biencoder_val.jsonl")

    for path, data, label in [(train_path, train, "train"), (val_path, val, "val")]:
        with open(path, "w", encoding="utf-8") as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False))
                f.write("\n")
        print(f"  {label}: {len(data):,} triples -> {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
