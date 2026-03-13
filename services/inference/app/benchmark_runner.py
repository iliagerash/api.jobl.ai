import argparse
import json
import logging
import time
from pathlib import Path

from app.config import settings
from app.io_utils import read_jsonl
from app.logging import configure_logging


logger = logging.getLogger("jobl.inference.benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark local inference latency/throughput")
    parser.add_argument("--input-jsonl", default="../training/data/sft/test.jsonl", help="Input SFT JSONL (flat input field)")
    parser.add_argument("--model-dir", default="../training/artifacts/flan-t5-normalize-v1/best", help="Fine-tuned seq2seq model directory")
    parser.add_argument("--limit", type=int, default=50, help="Rows to benchmark")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup rows not counted in metrics")
    parser.add_argument("--max-new-tokens", type=int, default=0, help="Generation max_new_tokens; 0 uses default")
    parser.add_argument("--num-beams", type=int, default=4, help="Beam search width")
    parser.add_argument("--progress-every", type=int, default=5, help="Progress log interval")
    parser.add_argument("--out", default="artifacts/benchmark_summary.json", help="Output summary JSON path")
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    configure_logging(settings.log_level)

    try:
        _run(args)
    except KeyboardInterrupt:
        logger.warning("interrupted by user (Ctrl+C), exiting gracefully")
        return 130
    return 0


def _run(args: argparse.Namespace) -> None:
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Dependencies missing. Install with: pip install -e .") from exc

    input_path = Path(args.input_jsonl)
    rows = read_jsonl(input_path)
    if not rows:
        raise SystemExit(f"No rows found in {input_path}")

    limit = max(1, args.limit)
    rows = rows[:limit]
    warmup = max(0, min(args.warmup, len(rows) - 1))
    progress_every = max(1, args.progress_every)

    model_dir = str(args.model_dir)
    max_new_tokens = args.max_new_tokens if args.max_new_tokens > 0 else 32

    logger.info(
        "benchmark started model_dir=%s rows=%s warmup=%s max_new_tokens=%s num_beams=%s",
        model_dir,
        len(rows),
        warmup,
        max_new_tokens,
        args.num_beams,
    )

    has_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if has_cuda else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map="auto" if has_cuda else None,
        low_cpu_mem_usage=True,
    )
    model.eval()

    device = "cuda" if has_cuda else "cpu"
    if not has_cuda:
        model.to(device)

    latencies: list[float] = []
    out_tokens: list[int] = []
    started = time.time()
    measured_started = None

    for idx, row in enumerate(rows, start=1):
        prompt = _build_prompt(row=row)
        tokenizer.padding_side = "right"
        prompt_ids = tokenizer([prompt], return_tensors="pt", padding=True)
        input_ids = prompt_ids["input_ids"].to(device)
        attention_mask = prompt_ids.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=args.num_beams,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        dt = time.perf_counter() - t0

        token_count = int(out[0].shape[0])

        if idx > warmup:
            if measured_started is None:
                measured_started = time.time()
            latencies.append(dt)
            out_tokens.append(token_count)

        if idx % progress_every == 0 or idx == len(rows):
            elapsed = max(1e-6, time.time() - started)
            logger.info(
                "benchmark progress processed=%s/%s elapsed=%s",
                idx,
                len(rows),
                _fmt_seconds(int(elapsed)),
            )

    if not latencies:
        raise SystemExit("No measured rows after warmup; increase --limit or reduce --warmup")

    measured_elapsed = max(1e-6, time.time() - (measured_started or time.time()))
    measured_rows = len(latencies)
    summary = {
        "device": device,
        "model_dir": model_dir,
        "input_jsonl": str(input_path),
        "rows_total": len(rows),
        "rows_measured": measured_rows,
        "warmup_rows": warmup,
        "max_new_tokens": max_new_tokens,
        "num_beams": args.num_beams,
        "latency_seconds": {
            "min": round(min(latencies), 4),
            "p50": round(_percentile(latencies, 50), 4),
            "p95": round(_percentile(latencies, 95), 4),
            "max": round(max(latencies), 4),
            "avg": round(sum(latencies) / measured_rows, 4),
        },
        "throughput": {
            "rows_per_sec": round(measured_rows / measured_elapsed, 4),
            "output_tokens_per_sec": round(sum(out_tokens) / measured_elapsed, 2),
            "avg_output_tokens_per_row": round(sum(out_tokens) / measured_rows, 2),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "benchmark completed rows=%s p50=%ss p95=%ss rows_per_sec=%s tokens_per_sec=%s out=%s",
        measured_rows,
        summary["latency_seconds"]["p50"],
        summary["latency_seconds"]["p95"],
        summary["throughput"]["rows_per_sec"],
        summary["throughput"]["output_tokens_per_sec"],
        out_path,
    )


def _build_prompt(*, row: dict[str, object]) -> str:
    value = str(row.get("input") or "").strip()
    if not value:
        return ""
    if value.lower().startswith("normalize job title:"):
        return value
    return f"normalize job title: {value}"


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    ordered = sorted(values)
    k = (len(ordered) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


def _fmt_seconds(total_seconds: int) -> str:
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


if __name__ == "__main__":
    raise SystemExit(run())
