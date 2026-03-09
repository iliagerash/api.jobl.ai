import argparse
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any

from app.config import settings
from app.io_utils import read_jsonl
from app.logging import configure_logging


logger = logging.getLogger("jobl.inference.benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark local inference latency/throughput")
    parser.add_argument("--input-jsonl", default="../training/data/sft/test.jsonl", help="Input SFT JSONL (messages format)")
    parser.add_argument("--adapter-dir", default="../training/artifacts/lora-normalize-v1/adapter", help="LoRA adapter directory")
    parser.add_argument("--model", default=None, help="Base model override; defaults to adapter metadata")
    parser.add_argument("--task", choices=["full", "title"], default="full", help="Benchmark full normalization or title-only output")
    parser.add_argument("--limit", type=int, default=50, help="Rows to benchmark")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup rows not counted in metrics")
    parser.add_argument("--max-new-tokens", type=int, default=0, help="Generation max_new_tokens; 0 uses task default")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
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
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
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

    model_name = _resolve_base_model_name(Path(args.adapter_dir), args.model)
    max_new_tokens = args.max_new_tokens if args.max_new_tokens > 0 else (512 if args.task == "full" else 64)

    logger.info(
        "benchmark started model=%s adapter=%s rows=%s warmup=%s task=%s max_new_tokens=%s",
        model_name,
        args.adapter_dir,
        len(rows),
        warmup,
        args.task,
        max_new_tokens,
    )

    has_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if has_cuda else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map="auto" if has_cuda else None,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    device = "cuda" if has_cuda else "cpu"
    if not has_cuda:
        model.to(device)

    latencies: list[float] = []
    out_tokens: list[int] = []
    started = time.time()
    measured_started = None

    for idx, row in enumerate(rows, start=1):
        prompt = _build_prompt(row=row, task=args.task)
        prompt_ids = tokenizer(prompt, return_tensors="pt")
        input_ids = prompt_ids["input_ids"].to(device)
        attention_mask = prompt_ids.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        do_sample = args.temperature > 0
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=args.temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        dt = time.perf_counter() - t0

        generated = out[0][input_ids.shape[1] :]
        token_count = int(generated.shape[0])

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
        "task": args.task,
        "device": device,
        "model": model_name,
        "adapter_dir": args.adapter_dir,
        "input_jsonl": str(input_path),
        "rows_total": len(rows),
        "rows_measured": measured_rows,
        "warmup_rows": warmup,
        "max_new_tokens": max_new_tokens,
        "temperature": args.temperature,
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


def _resolve_base_model_name(adapter_dir: Path, explicit_model: str | None) -> str:
    if explicit_model:
        return explicit_model

    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise SystemExit(f"Cannot auto-detect model; {cfg_path} not found. Pass --model.")
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Cannot parse {cfg_path}. Pass --model.") from exc

    model_name = str(cfg.get("base_model_name_or_path") or "").strip()
    if not model_name:
        raise SystemExit(f"No base_model_name_or_path in {cfg_path}. Pass --model.")
    return model_name


def _build_prompt(*, row: dict[str, Any], task: str) -> str:
    messages = row.get("messages") or []
    system = _message_content(messages, 0)
    user = _message_content(messages, 1)

    if task == "title":
        system = (
            f"{system}\n"
            "Task mode: title-only. Return strict JSON with one field: "
            '{"title_normalized": "..."}. Do not include other fields.'
        )
    return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"


def _message_content(messages: Any, idx: int) -> str:
    if not isinstance(messages, list) or len(messages) <= idx:
        return ""
    msg = messages[idx]
    if isinstance(msg, dict):
        return str(msg.get("content") or "")
    return ""


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
