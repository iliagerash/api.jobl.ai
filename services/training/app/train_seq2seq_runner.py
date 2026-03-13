import argparse
import inspect
import logging
from pathlib import Path

from app.logging import configure_logging


logger = logging.getLogger("jobl.training.train_seq2seq")
DEFAULT_BASE_MODEL = "google/flan-t5-large"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seq2Seq fine-tuning for normalization task")
    parser.add_argument(
        "--train-jsonl",
        default="data/sft/train.jsonl",
        help="Instruction train JSONL path",
    )
    parser.add_argument(
        "--val-jsonl",
        default="data/sft/val.jsonl",
        help="Instruction val JSONL path",
    )
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL, help=f"Base HF model id (default: {DEFAULT_BASE_MODEL})")
    parser.add_argument("--out-dir", default="artifacts/flan-t5-normalize-v1", help="Output directory")
    parser.add_argument("--epochs", type=float, default=5.0, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Per-device train batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--memory-safe",
        action="store_true",
        help="Use CPU-safe defaults (smaller batches and fewer workers)",
    )
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    configure_logging("INFO")

    try:
        _train(args)
    except KeyboardInterrupt:
        logger.warning("interrupted by user (Ctrl+C), exiting gracefully")
        return 130
    return 0


def _train(args: argparse.Namespace) -> None:
    try:
        import torch
        from datasets import load_dataset
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            EarlyStoppingCallback,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Install with: pip install -e '.[train]'"
        ) from exc

    has_cuda = torch.cuda.is_available()
    args, dataloader_num_workers = _apply_memory_safe_profile(args, has_cuda=has_cuda)

    train_path = Path(args.train_jsonl)
    val_path = Path(args.val_jsonl)
    if not train_path.exists() or not val_path.exists():
        raise SystemExit(
            "Training JSONL not found. Build datasets first with "
            "jobl-training-build-jsonl for train/val splits."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if has_cuda else torch.float32,
        low_cpu_mem_usage=True,
    )

    train_ds = load_dataset("json", data_files=args.train_jsonl, split="train")
    val_ds = load_dataset("json", data_files=args.val_jsonl, split="train")

    def tokenize_fn(batch):
        model_inputs = tokenizer(
            batch["input"],
            max_length=128,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=32,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    remove_train_columns = [col for col in ["input", "target"] if col in train_ds.column_names]
    remove_val_columns = [col for col in ["input", "target"] if col in val_ds.column_names]
    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=remove_train_columns)
    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=remove_val_columns)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    training_kwargs = {
        "output_dir": args.out_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": max(1, args.batch_size),
        "learning_rate": args.lr,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.01,
        "dataloader_num_workers": dataloader_num_workers,
        "predict_with_generate": True,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "bf16": has_cuda,
        "fp16": False,
        "save_total_limit": 2,
        "report_to": "none",
        "logging_steps": 20,
    }
    ta_params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    if "evaluation_strategy" in ta_params:
        training_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in ta_params:
        training_kwargs["eval_strategy"] = "epoch"

    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": data_collator,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=2)],
    }
    trainer_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    logger.info(
        "training started model=%s train_rows=%s val_rows=%s out=%s batch_size=%s epochs=%s lr=%s memory_safe=%s",
        args.model,
        len(train_ds),
        len(val_ds),
        args.out_dir,
        args.batch_size,
        args.epochs,
        args.lr,
        args.memory_safe,
    )
    trainer.train()

    best_dir = Path(args.out_dir) / "best"
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    logger.info("training completed best_model_dir=%s", best_dir)


def _apply_memory_safe_profile(args: argparse.Namespace, *, has_cuda: bool):
    if not args.memory_safe:
        if not has_cuda:
            logger.warning(
                "No CUDA device detected. Training on CPU may run out of memory with model=%s; "
                "retry with --memory-safe.",
                args.model,
            )
        return args, 4

    if has_cuda:
        logger.info("memory-safe profile requested on CUDA host; default training profile retained")
        return args, 4

    args.batch_size = 4
    args.epochs = 2.0
    logger.info(
        "memory-safe profile applied model=%s batch_size=%s epochs=%s dataloader_num_workers=%s",
        args.model,
        args.batch_size,
        args.epochs,
        0,
    )
    return args, 0


if __name__ == "__main__":
    raise SystemExit(run())
