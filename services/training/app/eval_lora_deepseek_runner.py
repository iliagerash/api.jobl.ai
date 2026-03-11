import argparse
import logging

from app.eval_lora_runner import _run_eval
from app.logging import configure_logging


logger = logging.getLogger("jobl.training.eval_lora_deepseek")

DEFAULT_DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DEFAULT_DEEPSEEK_ADAPTER_DIR = "artifacts/lora-normalize-deepseek-r1-7b/adapter"
DEFAULT_DEEPSEEK_OUT_DIR = "artifacts/lora-normalize-deepseek-r1-7b/eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DeepSeek LoRA adapter on SFT test set")
    parser.add_argument("--test-jsonl", default="data/sft/test.jsonl", help="Instruction test JSONL path")
    parser.add_argument("--model", default=DEFAULT_DEEPSEEK_MODEL, help=f"Base HF model id (default: {DEFAULT_DEEPSEEK_MODEL})")
    parser.add_argument("--adapter-dir", default=DEFAULT_DEEPSEEK_ADAPTER_DIR, help="LoRA adapter directory")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to evaluate, 0 means all")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--max-new-tokens", type=int, default=768, help="Generation max_new_tokens")
    parser.add_argument(
        "--chunked",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Chunk description_raw during eval inference (default: true; use --no-chunked to disable)",
    )
    parser.add_argument("--chunk-max-chars", type=int, default=3500, help="Max chars per chunk when --chunked")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--progress-every", type=int, default=10, help="Log progress every N rows")
    parser.add_argument("--out-dir", default=DEFAULT_DEEPSEEK_OUT_DIR, help="Directory for evaluation artifacts")
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    configure_logging("INFO")

    try:
        _run_eval(args)
    except KeyboardInterrupt:
        logger.warning("interrupted by user (Ctrl+C), exiting gracefully")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
