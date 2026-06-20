"""
train_reranker.py
───────────────────
Phase 3: fine-tune XLM-RoBERTa-large as a CrossEncoder reranker.

The CrossEncoder takes a (resume, job) pair concatenated as a single input
and outputs a relevance score 0.0–1.0. It reranks the bi-encoder's top-100
candidates to produce the final ranked results.

Input:  ml/data/splits/reranker_train.jsonl (text_a, text_b, score)
Output: ml/models/exported/reranker/ (CrossEncoder format)

Usage:
    python -u ml/scripts/train_reranker.py
    python -u ml/scripts/train_reranker.py --epochs 5 --batch-size 32
"""

import argparse
import json
import os
import sys

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


def load_examples(path: str):
    """Load as CrossEncoder InputExample list."""
    from sentence_transformers.cross_encoder import InputExample
    examples = []
    for record in iter_jsonl(path):
        examples.append(InputExample(
            texts=[record["text_a"], record["text_b"]],
            label=float(record["score"]),
        ))
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune CrossEncoder reranker")
    parser.add_argument("--base-model", default="ml/models/base/reranker", help="Base model path")
    parser.add_argument("--train-data", default="ml/data/splits/reranker_train.jsonl")
    parser.add_argument("--val-data", default="ml/data/splits/reranker_val.jsonl")
    parser.add_argument("--output-dir", default="ml/models/exported/reranker")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length for concatenated pair (default: 512)")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio (default: 0.1)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from sentence_transformers.cross_encoder import CrossEncoder

    # Load model
    print(f"Loading base model from {args.base_model} ...")
    model = CrossEncoder(args.base_model, num_labels=1, max_length=args.max_length)
    print(f"  Model loaded, max_length={args.max_length}")

    # Load training data
    print(f"Loading training data from {args.train_data} ...")
    train_examples = load_examples(args.train_data)
    print(f"  {len(train_examples):,} training pairs")

    # Load val data
    val_examples = None
    if os.path.exists(args.val_data):
        print(f"Loading validation data from {args.val_data} ...")
        val_examples = load_examples(args.val_data)
        print(f"  {len(val_examples):,} validation pairs")

    # Calculate steps
    steps_per_epoch = max(1, len(train_examples) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    print(f"  Total steps: {total_steps:,}, warmup: {warmup_steps:,}")

    # Train
    print(f"\nStarting training: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    model.fit(
        train_dataloader=None,
        train_examples=train_examples,
        evaluator=None,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        output_path=args.output_dir,
        save_best_model=True,
        show_progress_bar=True,
        use_amp=True,
    )

    print(f"\nModel saved to {args.output_dir}")

    # Sanity check
    print("\nSanity check: scoring a test pair ...")
    scores = model.predict([
        ["Senior Python Developer with 5 years experience", "Senior Software Engineer - Python, AWS, 5+ years required"],
        ["Junior Barista looking for part-time work", "Senior Software Engineer - Python, AWS, 5+ years required"],
    ])
    print(f"  Good match score: {scores[0]:.4f}")
    print(f"  Bad match score:  {scores[1]:.4f}")

    if scores[0] > scores[1]:
        print("  Sanity check PASSED: good match scores higher than bad match")
    else:
        print("  Sanity check WARNING: good match scored lower than bad match")

    print("\nDone. Run eval_reranker.py next to evaluate NDCG@10.")


if __name__ == "__main__":
    main()
