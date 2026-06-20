"""
train_extractor_lora.py
────────────────────────
Phase 4: fine-tune Qwen2.5-7B-Instruct with LoRA for structured extraction.

Uses TRL SFTTrainer with PEFT/LoRA on chat-formatted examples.

LoRA config (from spec):
  - Rank: 32
  - Alpha: 64
  - Target: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  - Dropout: 0.05

Input:  ml/data/splits/extractor_train.jsonl (chat-format SFT examples)
Output: ml/models/checkpoints/extractor-lora/ (LoRA adapter weights)

Usage:
    python -u ml/scripts/train_extractor_lora.py
    python -u ml/scripts/train_extractor_lora.py --epochs 3 --batch-size 4
"""

import argparse
import json
import os
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


def load_dataset(path: str):
    """Load chat-format SFT examples as a HuggingFace Dataset."""
    from datasets import Dataset
    records = []
    for record in iter_jsonl(path):
        records.append({"messages": record["messages"]})
    return Dataset.from_list(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-7B-Instruct with LoRA for extraction")
    parser.add_argument("--base-model", default="ml/models/base/extractor")
    parser.add_argument("--train-data", default="ml/data/splits/extractor_train.jsonl")
    parser.add_argument("--val-data", default="ml/data/splits/extractor_val.jsonl")
    parser.add_argument("--output-dir", default="ml/models/checkpoints/extractor-lora")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size (default: 4)")
    parser.add_argument("--gradient-accumulation", type=int, default=8, help="Gradient accumulation steps (default: 8, effective batch = 32)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length (default: 2048)")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank (default: 32)")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha (default: 64)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer

    # Load tokenizer
    print(f"Loading tokenizer from {args.base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading base model from {args.base_model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    print(f"  Model loaded: {model.config.model_type}, {model.config.hidden_size}d")

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load datasets
    print(f"Loading training data from {args.train_data} ...")
    train_dataset = load_dataset(args.train_data)
    print(f"  {len(train_dataset):,} training examples")

    val_dataset = None
    if os.path.exists(args.val_data):
        print(f"Loading validation data from {args.val_data} ...")
        val_dataset = load_dataset(args.val_data)
        print(f"  {len(val_dataset):,} validation examples")

    # Training arguments
    effective_batch = args.batch_size * args.gradient_accumulation
    steps_per_epoch = max(1, len(train_dataset) // effective_batch)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * 0.1)
    print(f"  Effective batch size: {effective_batch}, steps/epoch: {steps_per_epoch}, total: {total_steps}, warmup: {warmup_steps}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        bf16=True,
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=10,
        report_to="none",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
    )

    # Trainer
    print(f"\nStarting LoRA training: {args.epochs} epochs, batch={args.batch_size}x{args.gradient_accumulation}={effective_batch}, lr={args.lr}")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        max_seq_length=args.max_seq_length,
    )
    trainer.train()

    # Save LoRA adapter
    print(f"\nSaving LoRA adapter to {args.output_dir} ...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("  LoRA adapter saved")

    print("\nDone. Run merge_lora.py next to merge into the base model.")


if __name__ == "__main__":
    main()
