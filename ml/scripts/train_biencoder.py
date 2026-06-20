"""
train_biencoder.py
────────────────────
Phase 2, Day 3-4: fine-tune Harrier-OSS-v1-0.6B as a bi-encoder embedding model
using Sentence Transformers with MultipleNegativesRankingLoss.

Input:  ml/data/splits/biencoder_train.jsonl (anchor, positive, negative triples)
Output: ml/models/exported/biencoder/ (Sentence Transformers format)

Usage:
    python -u ml/scripts/train_biencoder.py
    python -u ml/scripts/train_biencoder.py --epochs 3 --batch-size 128
    python -u ml/scripts/train_biencoder.py --base-model ml/models/base/harrier
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
                yield json.loads(line)


def load_training_examples(path: str):
    """Load triples as InputExample objects for Sentence Transformers."""
    from sentence_transformers import InputExample
    examples = []
    for record in iter_jsonl(path):
        examples.append(InputExample(texts=[
            record["anchor"],
            record["positive"],
            record["negative"],
        ]))
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune bi-encoder embedding model")
    parser.add_argument("--base-model", default="ml/models/base/harrier", help="Base model path or HF repo")
    parser.add_argument("--train-data", default="ml/data/splits/biencoder_train.jsonl")
    parser.add_argument("--val-data", default="ml/data/splits/biencoder_val.jsonl")
    parser.add_argument("--output-dir", default="ml/models/exported/biencoder")
    parser.add_argument("--checkpoint-dir", default="ml/models/checkpoints/biencoder")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Max sequence length (default: 512)")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio (default: 0.1)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    from sentence_transformers import SentenceTransformer, losses
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments
    from sentence_transformers.trainer import SentenceTransformerTrainer
    from torch.utils.data import DataLoader

    # Load model
    print(f"Loading base model from {args.base_model} ...")
    model = SentenceTransformer(args.base_model, trust_remote_code=True)
    model.max_seq_length = args.max_seq_length
    print(f"  Model loaded, max_seq_length={model.max_seq_length}")

    # Load training data
    print(f"Loading training data from {args.train_data} ...")
    train_examples = load_training_examples(args.train_data)
    print(f"  {len(train_examples):,} training triples")

    # Create dataloader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)

    # Loss function
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Calculate training steps
    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    print(f"  Total steps: {total_steps:,}, warmup: {warmup_steps:,}")

    # Training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.checkpoint_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        fp16=True,
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=50,
        report_to="none",
    )

    # Load val data if available
    val_examples = None
    if os.path.exists(args.val_data):
        print(f"Loading validation data from {args.val_data} ...")
        val_examples = load_training_examples(args.val_data)
        print(f"  {len(val_examples):,} validation triples")

    # Train
    print(f"\nStarting training: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_examples,
        eval_dataset=val_examples,
        loss=train_loss,
    )
    trainer.train()

    # Save final model
    print(f"\nSaving model to {args.output_dir} ...")
    model.save(args.output_dir)
    print(f"  Model saved")

    # Quick sanity check
    print("\nSanity check: encoding a test sentence ...")
    emb = model.encode(["Senior Software Engineer with 5 years of Python experience"])
    print(f"  Embedding shape: {emb.shape}, norm: {(emb**2).sum()**0.5:.4f}")

    print("\nDone. Run eval_biencoder.py next to evaluate retrieval quality.")


if __name__ == "__main__":
    main()
