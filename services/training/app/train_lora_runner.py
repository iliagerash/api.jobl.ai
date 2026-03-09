import argparse
import inspect
import logging
from pathlib import Path

from app.logging import configure_logging


logger = logging.getLogger("jobl.training.train_lora")
DEFAULT_BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT pilot training for normalization task")
    parser.add_argument(
        "--train-jsonl",
        default="data/sft_chunks/train.jsonl",
        help="Instruction train JSONL path",
    )
    parser.add_argument(
        "--val-jsonl",
        default="data/sft_chunks/val.jsonl",
        help="Instruction val JSONL path",
    )
    parser.add_argument("--model", default=None, help=f"Base HF model id (default: {DEFAULT_BASE_MODEL})")
    parser.add_argument("--out-dir", default="artifacts/lora-normalize-v1", help="Output directory")
    parser.add_argument("--epochs", type=float, default=2.0, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device train batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length")
    parser.add_argument(
        "--memory-safe",
        action="store_true",
        help="Use CPU-safe defaults (smaller model, shorter seq len, smaller batches)",
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
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Install with: pip install -e '.[train]'"
        ) from exc

    has_cuda = torch.cuda.is_available()
    model_explicit = bool(args.model)
    if not args.model:
        args.model = DEFAULT_BASE_MODEL
    args = _apply_memory_safe_profile(args, has_cuda=has_cuda, model_explicit=model_explicit)

    train_path = Path(args.train_jsonl)
    val_path = Path(args.val_jsonl)
    if not train_path.exists() or not val_path.exists():
        raise SystemExit(
            "Training JSONL not found. Build chunk datasets first: "
            "jobl-training-build-chunks --in=data/sft/train.jsonl --out=data/sft_chunks/train.jsonl "
            "and --in=data/sft/val.jsonl --out=data/sft_chunks/val.jsonl"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if has_cuda else torch.float32,
        device_map="auto" if has_cuda else None,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    train_ds = load_dataset("json", data_files=args.train_jsonl, split="train")
    val_ds = load_dataset("json", data_files=args.val_jsonl, split="train")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    training_kwargs = {
        "output_dir": args.out_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": max(1, args.batch_size),
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "logging_steps": 20,
        "eval_steps": 100,
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 2,
        "bf16": has_cuda,
        "fp16": False,
        "report_to": [],
        "remove_unused_columns": False,
        "dataloader_pin_memory": has_cuda,
        "gradient_checkpointing": True,
    }
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in ta_params:
        training_kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in ta_params:
        training_kwargs["eval_strategy"] = "steps"

    training_args = TrainingArguments(**training_kwargs)

    sft_params = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs = {
        "model": model,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "peft_config": peft_config,
        "args": training_args,
    }

    if "tokenizer" in sft_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in sft_params:
        trainer_kwargs["processing_class"] = tokenizer

    if "max_seq_length" in sft_params:
        trainer_kwargs["max_seq_length"] = args.max_seq_len

    if "formatting_func" in sft_params:
        trainer_kwargs["formatting_func"] = _formatting_func
    else:
        train_ds = _with_text_column(train_ds)
        val_ds = _with_text_column(val_ds)
        trainer_kwargs["train_dataset"] = train_ds
        trainer_kwargs["eval_dataset"] = val_ds
        if "dataset_text_field" in sft_params:
            trainer_kwargs["dataset_text_field"] = "text"

    trainer = SFTTrainer(**trainer_kwargs)

    logger.info(
        "training started model=%s train_rows=%s val_rows=%s out=%s batch_size=%s grad_accum=%s max_seq_len=%s memory_safe=%s",
        args.model,
        len(train_ds),
        len(val_ds),
        args.out_dir,
        args.batch_size,
        args.grad_accum,
        args.max_seq_len,
        args.memory_safe,
    )
    trainer.train()
    trainer.model.save_pretrained(Path(args.out_dir) / "adapter")
    tokenizer.save_pretrained(Path(args.out_dir) / "adapter")
    logger.info("training completed adapter_dir=%s", Path(args.out_dir) / "adapter")


def _formatting_func(batch: dict):
    messages_obj = batch["messages"]

    # Some TRL versions call formatting_func with a single row (list[dict]),
    # others pass batched rows (list[list[dict]]).
    if isinstance(messages_obj, list) and messages_obj and isinstance(messages_obj[0], dict):
        return _format_messages(messages_obj)

    texts: list[str] = []
    if isinstance(messages_obj, list):
        for messages in messages_obj:
            texts.append(_format_messages(messages if isinstance(messages, list) else []))
    return texts


def _format_messages(messages: list[dict]) -> str:
    system = messages[0].get("content", "") if len(messages) > 0 else ""
    user = messages[1].get("content", "") if len(messages) > 1 else ""
    assistant = messages[2].get("content", "") if len(messages) > 2 else ""
    return (
        f"<|system|>\n{system}\n"
        f"<|user|>\n{user}\n"
        f"<|assistant|>\n{assistant}"
    )


def _with_text_column(ds):
    return ds.map(
        lambda batch: {"text": _formatting_func({"messages": batch["messages"]})},
        batched=True,
    )


def _apply_memory_safe_profile(args: argparse.Namespace, *, has_cuda: bool, model_explicit: bool) -> argparse.Namespace:
    if not args.memory_safe:
        if not has_cuda:
            logger.warning(
                "No CUDA device detected. Training on CPU may run out of memory with model=%s; "
                "retry with --memory-safe.",
                args.model,
            )
        return args

    # Conservative defaults for 32GB RAM CPU hosts.
    if not model_explicit and not has_cuda and args.model == "microsoft/Phi-3-mini-4k-instruct":
        args.model = "Qwen/Qwen2.5-0.5B-Instruct"
    elif not model_explicit and not has_cuda and args.model == "Qwen/Qwen2.5-3B-Instruct":
        args.model = "Qwen/Qwen2.5-0.5B-Instruct"
    args.batch_size = 1
    args.grad_accum = max(args.grad_accum, 16)
    args.max_seq_len = min(args.max_seq_len, 1024)
    args.epochs = min(args.epochs, 1.0)
    logger.info(
        "memory-safe profile applied model=%s batch_size=%s grad_accum=%s max_seq_len=%s epochs=%s",
        args.model,
        args.batch_size,
        args.grad_accum,
        args.max_seq_len,
        args.epochs,
    )
    return args


if __name__ == "__main__":
    raise SystemExit(run())
