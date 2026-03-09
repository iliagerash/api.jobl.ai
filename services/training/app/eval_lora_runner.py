import argparse
import csv
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from app.io_utils import read_jsonl, write_jsonl
from app.logging import configure_logging


logger = logging.getLogger("jobl.training.eval_lora")

ALLOWED_TAGS = {
    "p",
    "ul",
    "ol",
    "li",
    "i",
    "b",
    "em",
    "strong",
    "u",
    "h2",
    "h3",
    "h4",
    "br",
    "a",
}
TAG_RE = re.compile(r"<\s*/?\s*([a-zA-Z0-9]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LoRA adapter on SFT test set")
    parser.add_argument("--test-jsonl", default="data/sft/test.jsonl", help="Instruction test JSONL path")
    parser.add_argument(
        "--model",
        default=None,
        help="Base HF model id. If omitted, auto-detected from adapter_config.json",
    )
    parser.add_argument("--adapter-dir", default="artifacts/lora-normalize-v1/adapter", help="LoRA adapter directory")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to evaluate, 0 means all")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--max-new-tokens", type=int, default=768, help="Generation max_new_tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--progress-every", type=int, default=10, help="Log progress every N rows")
    parser.add_argument("--out-dir", default="artifacts/lora-normalize-v1/eval", help="Directory for evaluation artifacts")
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


def _run_eval(args: argparse.Namespace) -> None:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Dependencies missing. Install with: pip install -e '.[train]'") from exc

    rows = read_jsonl(Path(args.test_jsonl))
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No test rows found")

    has_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if has_cuda else torch.float32

    model_name = _resolve_base_model_name(adapter_dir=Path(args.adapter_dir), explicit_model=args.model)
    logger.info("eval base model resolved model=%s", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if has_cuda else None,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    device = "cuda" if has_cuda else "cpu"
    if not has_cuda:
        model.to("cpu")

    metrics = {
        "total": len(rows),
        "valid_json": 0,
        "title_exact": 0,
        "html_exact": 0,
        "title_non_empty": 0,
        "html_non_empty": 0,
        "html_allowed_tags_only": 0,
    }
    mismatches: list[dict[str, Any]] = []
    started_at = time.time()
    progress_every = max(1, args.progress_every)
    batch_size = max(1, args.batch_size)
    processed = 0

    for offset in range(0, len(rows), batch_size):
        batch_rows = rows[offset : offset + batch_size]
        prompts = [_build_inference_prompt(row) for row in batch_rows]
        preds = _generate_predictions_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        if len(preds) != len(batch_rows):
            raise SystemExit("batch generation returned unexpected number of predictions")

        for row, pred in zip(batch_rows, preds):
            expected = _parse_assistant_target(row)
            pred_obj = _parse_json_loose(pred)

            valid_json = isinstance(pred_obj, dict)
            if valid_json:
                metrics["valid_json"] += 1

            pred_title = str((pred_obj or {}).get("title_normalized") or "").strip()
            pred_html = str((pred_obj or {}).get("description_html") or "").strip()
            exp_title = str(expected.get("title_normalized") or "").strip()
            exp_html = str(expected.get("description_html") or "").strip()

            if pred_title:
                metrics["title_non_empty"] += 1
            if pred_html:
                metrics["html_non_empty"] += 1
            if _uses_only_allowed_tags(pred_html):
                metrics["html_allowed_tags_only"] += 1

            title_exact = pred_title == exp_title
            html_exact = pred_html == exp_html
            if title_exact:
                metrics["title_exact"] += 1
            if html_exact:
                metrics["html_exact"] += 1

            if not (title_exact and html_exact):
                mismatches.append(
                    {
                        "id": row.get("id"),
                        "language_code": row.get("language_code"),
                        "expected_title_normalized": exp_title,
                        "predicted_title_normalized": pred_title,
                        "expected_description_html": exp_html,
                        "predicted_description_html": pred_html,
                        "raw_prediction": pred,
                        "valid_json": valid_json,
                        "title_exact": title_exact,
                        "html_exact": html_exact,
                        "html_allowed_tags_only": _uses_only_allowed_tags(pred_html),
                    }
                )

        processed += len(batch_rows)
        if processed % progress_every == 0 or processed == len(rows):
            elapsed = max(1e-6, time.time() - started_at)
            rate = processed / elapsed
            remaining = max(0, len(rows) - processed)
            eta_seconds = int(remaining / rate) if rate > 0 else 0
            logger.info(
                "eval progress processed=%s/%s (%.1f%%) elapsed=%s eta=%s rows_per_sec=%.3f",
                processed,
                len(rows),
                (processed / len(rows)) * 100.0,
                _format_seconds(int(elapsed)),
                _format_seconds(eta_seconds),
                rate,
            )

    summary = {
        **metrics,
        "model": model_name,
        "adapter_dir": args.adapter_dir,
        "valid_json_rate": round(metrics["valid_json"] / metrics["total"], 4),
        "title_exact_rate": round(metrics["title_exact"] / metrics["total"], 4),
        "html_exact_rate": round(metrics["html_exact"] / metrics["total"], 4),
        "title_non_empty_rate": round(metrics["title_non_empty"] / metrics["total"], 4),
        "html_non_empty_rate": round(metrics["html_non_empty"] / metrics["total"], 4),
        "html_allowed_tags_only_rate": round(metrics["html_allowed_tags_only"] / metrics["total"], 4),
        "mismatch_count": len(mismatches),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(out_dir / "mismatches.jsonl", mismatches)
    _write_mismatches_csv(out_dir / "mismatches.csv", mismatches)

    logger.info("evaluation completed total=%s valid_json=%s title_exact=%s html_exact=%s mismatches=%s", metrics["total"], metrics["valid_json"], metrics["title_exact"], metrics["html_exact"], len(mismatches))
    logger.info("artifacts summary=%s mismatches_jsonl=%s mismatches_csv=%s", out_dir / "summary.json", out_dir / "mismatches.jsonl", out_dir / "mismatches.csv")


def _build_inference_prompt(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    system = _message_content(messages, 0)
    user = _message_content(messages, 1)
    return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"


def _resolve_base_model_name(*, adapter_dir: Path, explicit_model: str | None) -> str:
    if explicit_model:
        return explicit_model

    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise SystemExit(
            f"Cannot auto-detect base model: {cfg_path} not found. "
            "Pass --model explicitly."
        )

    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Cannot parse {cfg_path}. Pass --model explicitly."
        ) from exc

    model_name = str(cfg.get("base_model_name_or_path") or "").strip()
    if not model_name:
        raise SystemExit(
            f"No base_model_name_or_path in {cfg_path}. Pass --model explicitly."
        )
    return model_name


def _message_content(messages: Any, index: int) -> str:
    if not isinstance(messages, list) or len(messages) <= index:
        return ""
    msg = messages[index]
    if isinstance(msg, dict):
        return str(msg.get("content") or "")
    return ""


def _generate_predictions_batch(
    *, model, tokenizer, prompts: list[str], device: str, max_new_tokens: int, temperature: float
) -> list[str]:
    import torch

    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    do_sample = temperature > 0
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_lengths = attention_mask.sum(dim=1).tolist() if attention_mask is not None else [input_ids.shape[1]] * len(prompts)
    predictions: list[str] = []
    for idx, in_len in enumerate(input_lengths):
        gen_ids = out[idx][int(in_len) :]
        predictions.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
    return predictions


def _parse_assistant_target(row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("messages") or []
    content = _message_content(messages, 2)
    return _parse_json_loose(content) or {}


def _parse_json_loose(content: str) -> dict[str, Any] | None:
    text = (content or "").strip()
    if not text:
        return None

    decoder = json.JSONDecoder()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start >= 0:
        candidate = text[start:]
        try:
            obj, _end = decoder.raw_decode(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def _uses_only_allowed_tags(html: str) -> bool:
    if not html:
        return True
    tags = {m.group(1).lower() for m in TAG_RE.finditer(html)}
    return tags.issubset(ALLOWED_TAGS)


def _write_mismatches_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "language_code",
        "valid_json",
        "title_exact",
        "html_exact",
        "html_allowed_tags_only",
        "expected_title_normalized",
        "predicted_title_normalized",
        "expected_description_html",
        "predicted_description_html",
        "raw_prediction",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _format_seconds(total_seconds: int) -> str:
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


if __name__ == "__main__":
    raise SystemExit(run())
