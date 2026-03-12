import argparse
import csv
import json
import logging
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from app.io_utils import read_jsonl, write_jsonl
from app.logging import configure_logging


logger = logging.getLogger("jobl.training.eval_lora")

FIELD_RE = {
    "title_normalized": re.compile(r'"title_normalized"\s*:\s*"((?:\\.|[^"\\])*)"', re.DOTALL),
    "job_title": re.compile(r'"job_title"\s*:\s*"((?:\\.|[^"\\])*)"', re.DOTALL),
    "title": re.compile(r'"title"\s*:\s*"((?:\\.|[^"\\])*)"', re.DOTALL),
    "title_clean": re.compile(r'"title_clean"\s*:\s*"((?:\\.|[^"\\])*)"', re.DOTALL),
}


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
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Generation max_new_tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--progress-every", type=int, default=10, help="Log progress every N rows")
    parser.add_argument("--out-dir", default="artifacts/lora-normalize-v1/eval", help="Directory for evaluation artifacts")
    parser.add_argument(
        "--changed-titles-only",
        dest="changed_titles_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate only rows where expected title differs from raw title (default: true; use --all-titles or --no-changed-titles-only to disable)",
    )
    parser.add_argument(
        "--all-titles",
        dest="changed_titles_only",
        action="store_false",
        help="Evaluate all rows, including unchanged titles",
    )
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

    all_rows = read_jsonl(Path(args.test_jsonl))
    skipped_unchanged_titles = 0
    if args.changed_titles_only:
        rows = [row for row in all_rows if _row_has_changed_title(row)]
        skipped_unchanged_titles = len(all_rows) - len(rows)
    else:
        rows = all_rows
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No test rows found")
    if args.changed_titles_only:
        logger.info(
            "eval row selection changed_titles_only=true selected=%s skipped_unchanged=%s source_total=%s",
            len(rows),
            skipped_unchanged_titles,
            len(all_rows),
        )

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
        "title_non_empty": 0,
        "title_similarity_sum": 0.0,
        "title_similarity_ge_0_8": 0,
    }
    mismatches: list[dict[str, Any]] = []
    started_at = time.time()
    progress_every = max(1, args.progress_every)
    batch_size = max(1, args.batch_size)
    processed = 0
    logger.info(
        "eval inference started rows=%s batch_size=%s max_new_tokens=%s progress_every=%s",
        len(rows),
        batch_size,
        args.max_new_tokens,
        progress_every,
    )

    for offset in range(0, len(rows), batch_size):
        batch_rows = rows[offset : offset + batch_size]
        batch_end = offset + len(batch_rows)
        logger.info("eval batch started rows=%s-%s/%s", offset + 1, batch_end, len(rows))

        t_prompt = time.perf_counter()
        prompts = [_build_inference_prompt(row) for row in batch_rows]
        prompt_build_sec = time.perf_counter() - t_prompt

        preds, stage_timings = _generate_predictions_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            return_timings=True,
        )
        if len(preds) != len(batch_rows):
            raise SystemExit("batch generation returned unexpected number of predictions")

        t_parse = time.perf_counter()
        for row, raw_prediction in zip(batch_rows, preds):
            expected = _parse_assistant_target(row)
            pred_obj = _parse_json_loose(raw_prediction)

            valid_json = isinstance(pred_obj, dict) and _is_strict_prediction_object(pred_obj)
            if valid_json:
                metrics["valid_json"] += 1

            pred_title = str((pred_obj or {}).get("title_normalized") or "").strip()
            exp_title = str(expected.get("title_normalized") or "").strip()

            if pred_title:
                metrics["title_non_empty"] += 1

            title_exact = pred_title == exp_title
            if title_exact:
                metrics["title_exact"] += 1

            title_similarity = _text_similarity(exp_title, pred_title)
            metrics["title_similarity_sum"] += title_similarity
            if title_similarity >= 0.8:
                metrics["title_similarity_ge_0_8"] += 1

            if not title_exact:
                mismatches.append(
                    {
                        "id": row.get("id"),
                        "language_code": row.get("language_code"),
                        "title_original": _extract_original_title(row),
                        "expected_title_normalized": exp_title,
                        "predicted_title_normalized": pred_title,
                        "raw_prediction": raw_prediction,
                        "valid_json": valid_json,
                        "title_exact": title_exact,
                        "title_similarity": round(title_similarity, 4),
                    }
                )
        parse_sec = time.perf_counter() - t_parse

        processed += len(batch_rows)
        logger.info(
            "eval batch timing rows=%s-%s/%s prompt_build=%.3fs tokenize=%.3fs generate=%.3fs decode=%.3fs parse=%.3fs",
            offset + 1,
            batch_end,
            len(rows),
            prompt_build_sec,
            stage_timings["tokenize_sec"],
            stage_timings["generate_sec"],
            stage_timings["decode_sec"],
            parse_sec,
        )

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
        "changed_titles_only": bool(args.changed_titles_only),
        "source_total_rows": len(all_rows),
        "selected_rows": len(rows),
        "skipped_unchanged_title_rows": skipped_unchanged_titles,
        "valid_json_rate": round(metrics["valid_json"] / metrics["total"], 4),
        "title_exact_rate": round(metrics["title_exact"] / metrics["total"], 4),
        "title_non_empty_rate": round(metrics["title_non_empty"] / metrics["total"], 4),
        "title_similarity_avg": round(metrics["title_similarity_sum"] / metrics["total"], 4),
        "title_similarity_ge_0_8_rate": round(metrics["title_similarity_ge_0_8"] / metrics["total"], 4),
        "mismatch_count": len(mismatches),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(out_dir / "mismatches.jsonl", mismatches)
    _write_mismatches_csv(out_dir / "mismatches.csv", mismatches)

    logger.info(
        "evaluation completed total=%s valid_json=%s title_exact=%s mismatches=%s",
        metrics["total"],
        metrics["valid_json"],
        metrics["title_exact"],
        len(mismatches),
    )
    logger.info(
        "artifacts summary=%s mismatches_jsonl=%s mismatches_csv=%s",
        out_dir / "summary.json",
        out_dir / "mismatches.jsonl",
        out_dir / "mismatches.csv",
    )


def _build_inference_prompt(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    system = _message_content(messages, 0)
    user = _message_content(messages, 1)
    user = _build_user_payload_for_inference(user)
    return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"


def _build_user_payload_for_inference(user_content: str) -> str:
    try:
        payload = json.loads(user_content)
    except json.JSONDecodeError:
        return user_content
    if not isinstance(payload, dict):
        return user_content

    payload["response_schema"] = {"title_normalized": "string"}
    payload["response_rules"] = [
        "Return valid JSON only.",
        "Return exactly one key: title_normalized.",
        "Do not include prompt_version, job_title, title_raw, description_raw, or any other keys.",
    ]
    return json.dumps(payload, ensure_ascii=False)


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
    *,
    model,
    tokenizer,
    prompts: list[str],
    device: str,
    max_new_tokens: int,
    temperature: float,
    return_timings: bool = False,
) -> list[str] | tuple[list[str], dict[str, float]]:
    import torch

    tokenizer.padding_side = "left"
    t0 = time.perf_counter()
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    t1 = time.perf_counter()

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
    t2 = time.perf_counter()

    prompt_width = int(input_ids.shape[1])
    predictions: list[str] = []
    for idx in range(len(prompts)):
        gen_ids = out[idx][prompt_width:]
        predictions.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
    t3 = time.perf_counter()
    timings = {
        "tokenize_sec": t1 - t0,
        "generate_sec": t2 - t1,
        "decode_sec": t3 - t2,
    }
    if return_timings:
        return predictions, timings
    return predictions


def _parse_assistant_target(row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("messages") or []
    content = _message_content(messages, 2)
    return _parse_json_loose(content) or {}


def _extract_original_title(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    user_content = _message_content(messages, 1)
    try:
        user_payload = json.loads(user_content)
    except json.JSONDecodeError:
        return ""
    if not isinstance(user_payload, dict):
        return ""
    title = user_payload.get("title_raw")
    if title is None:
        title = user_payload.get("title")
    return str(title or "").strip()


def _parse_json_loose(content: str) -> dict[str, Any] | None:
    text = (content or "").strip()
    if not text:
        return None
    text = _strip_wrappers(text)

    for obj in _iter_json_objects(text):
        canonical = _canonicalize_prediction_obj(obj)
        if _looks_like_prediction_object(canonical):
            return canonical

    for obj in _iter_json_objects(text):
        return _canonicalize_prediction_obj(obj)

    recovered = _recover_fields_with_regex(text)
    if recovered:
        return recovered
    return None


def _iter_json_objects(text: str):
    decoder = json.JSONDecoder()
    starts = [idx for idx, ch in enumerate(text) if ch == "{"]
    for start in starts:
        try:
            obj, _end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


def _looks_like_prediction_object(obj: dict[str, Any]) -> bool:
    return "title_normalized" in obj


def _is_strict_prediction_object(obj: dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    if set(obj.keys()) != {"title_normalized"}:
        return False
    return isinstance(obj.get("title_normalized"), str)


def _strip_wrappers(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    marker = "<|output_json_schema|>"
    pos = cleaned.find(marker)
    if pos >= 0:
        cleaned = cleaned[:pos].strip()
    return cleaned


def _recover_fields_with_regex(text: str) -> dict[str, str] | None:
    recovered: dict[str, str] = {}
    for key, regex in FIELD_RE.items():
        match = regex.search(text)
        if not match:
            continue
        raw = match.group(1)
        try:
            decoded = json.loads(f'"{raw}"')
        except json.JSONDecodeError:
            decoded = raw
        recovered[key] = str(decoded).strip()
    if recovered:
        return _canonicalize_prediction_obj(recovered)
    return None


def _canonicalize_prediction_obj(obj: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return {}
    out = dict(obj)
    if "title_normalized" in out:
        return out
    for alias in ("job_title", "title", "title_clean"):
        value = str(out.get(alias) or "").strip()
        if value:
            out["title_normalized"] = value
            break
    return out


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _text_similarity(expected: str, predicted: str) -> float:
    exp = _normalize_text(expected)
    pred = _normalize_text(predicted)
    if not exp and not pred:
        return 1.0
    return SequenceMatcher(a=exp, b=pred).ratio()


def _row_has_changed_title(row: dict[str, Any]) -> bool:
    messages = row.get("messages") or []
    user_content = _message_content(messages, 1)
    expected = _parse_assistant_target(row)
    expected_title = _canonical_title(str(expected.get("title_normalized") or ""))
    if not expected_title:
        return False
    try:
        user_obj = json.loads(user_content)
    except json.JSONDecodeError:
        return True
    raw_title = _canonical_title(str((user_obj or {}).get("title_raw") or ""))
    return raw_title != expected_title


def _canonical_title(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).casefold()


def _write_mismatches_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "title_original",
        "title_expected",
        "title_generated",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "title_original": row.get("title_original") or "",
                    "title_expected": row.get("expected_title_normalized") or "",
                    "title_generated": row.get("predicted_title_normalized") or "",
                }
            )


def _format_seconds(total_seconds: int) -> str:
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


if __name__ == "__main__":
    raise SystemExit(run())
