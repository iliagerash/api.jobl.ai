import argparse
import logging

from app.logging import configure_logging
from app.train_lora_runner import _train


logger = logging.getLogger("jobl.training.train_lora_deepseek")

# User-facing alias "deepseek-r1:7b" mapped to Hugging Face model id used by transformers.
DEFAULT_DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA SFT training for normalization task (DeepSeek R1 7B variant)"
    )
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
    parser.add_argument(
        "--model",
        default=DEFAULT_DEEPSEEK_MODEL,
        help=f"Base HF model id (default: {DEFAULT_DEEPSEEK_MODEL})",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/lora-normalize-deepseek-r1-7b",
        help="Output directory",
    )
    parser.add_argument("--epochs", type=float, default=2.0, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device train batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length")
    parser.add_argument(
        "--memory-safe",
        action="store_true",
        help="Use CPU-safe defaults (smaller seq len, smaller batches)",
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


if __name__ == "__main__":
    raise SystemExit(run())
