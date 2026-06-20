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


def load_dataset(path: str):
    """Load as a HuggingFace Dataset for CrossEncoder training."""
    from datasets import Dataset
    texts_a, texts_b, scores = [], [], []
    for record in iter_jsonl(path):
        texts_a.append(record["text_a"])
        texts_b.append(record["text_b"])
        scores.append(float(record["score"]))
    return Dataset.from_dict({
        "sentence1": texts_a,
        "sentence2": texts_b,
        "label": scores,
    })


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
    train_dataset = load_dataset(args.train_data)
    print(f"  {len(train_dataset):,} training pairs")

    # Load val data
    val_dataset = None
    if os.path.exists(args.val_data):
        print(f"Loading validation data from {args.val_data} ...")
        val_dataset = load_dataset(args.val_data)
        print(f"  {len(val_dataset):,} validation pairs")

    # Calculate steps
    steps_per_epoch = max(1, len(train_dataset) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    print(f"  Total steps: {total_steps:,}, warmup: {warmup_steps:,}")

    # Training arguments
    from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer

    training_args = CrossEncoderTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        bf16=True,
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=50,
        report_to="none",
    )

    # Train
    print(f"\nStarting training: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()

    # Save the final model properly
    model.save(args.output_dir)
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
