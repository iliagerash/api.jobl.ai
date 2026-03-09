import argparse
import csv
import json
import logging
import re
import time
from difflib import SequenceMatcher
from html import unescape
from pathlib import Path
from typing import Any

from app.chunking import split_text_chunks
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
FIELD_RE = {
    "title_normalized": re.compile(r'"title_normalized"\s*:\s*"((?:\\.|[^"\\])*)"', re.DOTALL),
    "description_html": re.compile(r'"description_html"\s*:\s*"((?:\\.|[^"\\])*)"', re.DOTALL),
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
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--max-new-tokens", type=int, default=768, help="Generation max_new_tokens")
    parser.add_argument("--chunked", action="store_true", help="Chunk description_raw during eval inference")
    parser.add_argument("--chunk-max-chars", type=int, default=3500, help="Max chars per chunk when --chunked")
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
        "title_similarity_sum": 0.0,
        "html_text_similarity_sum": 0.0,
        "title_similarity_ge_0_8": 0,
        "html_text_similarity_ge_0_8": 0,
    }
    mismatches: list[dict[str, Any]] = []
    started_at = time.time()
    progress_every = max(1, args.progress_every)
    batch_size = max(1, args.batch_size)
    processed = 0
    logger.info(
        "eval inference started rows=%s batch_size=%s chunked=%s max_new_tokens=%s progress_every=%s",
        len(rows),
        batch_size,
        args.chunked,
        args.max_new_tokens,
        progress_every,
    )

    for offset in range(0, len(rows), batch_size):
        batch_rows = rows[offset : offset + batch_size]
        batch_end = offset + len(batch_rows)
        logger.info("eval batch started rows=%s-%s/%s", offset + 1, batch_end, len(rows))
        batch_prompt_build_sec = 0.0
        batch_tokenize_sec = 0.0
        batch_generate_sec = 0.0
        batch_decode_sec = 0.0
        batch_parse_sec = 0.0
        batch_postprocess_sec = 0.0
        if args.chunked:
            batch_preds = []
            for row in batch_rows:
                pred = _predict_row_chunked(
                    row=row,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    chunk_max_chars=args.chunk_max_chars,
                    batch_size=batch_size,
                )
                timings = pred.get("_timings") or {}
                batch_prompt_build_sec += float(timings.get("prompt_build_sec") or 0.0)
                batch_tokenize_sec += float(timings.get("tokenize_sec") or 0.0)
                batch_generate_sec += float(timings.get("generate_sec") or 0.0)
                batch_decode_sec += float(timings.get("decode_sec") or 0.0)
                batch_parse_sec += float(timings.get("parse_sec") or 0.0)
                batch_preds.append(pred)
        else:
            t_prompt = time.perf_counter()
            prompts = [_build_inference_prompt(row) for row in batch_rows]
            batch_prompt_build_sec += time.perf_counter() - t_prompt
            preds, stage_timings = _generate_predictions_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                return_timings=True,
            )
            batch_tokenize_sec += stage_timings["tokenize_sec"]
            batch_generate_sec += stage_timings["generate_sec"]
            batch_decode_sec += stage_timings["decode_sec"]
            if len(preds) != len(batch_rows):
                raise SystemExit("batch generation returned unexpected number of predictions")
            t_parse = time.perf_counter()
            batch_preds = []
            for pred in preds:
                batch_preds.append({"raw_prediction": pred, "pred_obj": _parse_json_loose(pred)})
            batch_parse_sec += time.perf_counter() - t_parse

        t_post = time.perf_counter()
        for row, pred_bundle in zip(batch_rows, batch_preds):
            expected = _parse_assistant_target(row)
            pred_obj = pred_bundle.get("pred_obj")
            raw_prediction = str(pred_bundle.get("raw_prediction") or "")

            valid_json = isinstance(pred_obj, dict) and _looks_like_prediction_object(pred_obj)
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
            title_similarity = _text_similarity(exp_title, pred_title)
            html_text_similarity = _html_text_similarity(exp_html, pred_html)
            metrics["title_similarity_sum"] += title_similarity
            metrics["html_text_similarity_sum"] += html_text_similarity
            if title_similarity >= 0.8:
                metrics["title_similarity_ge_0_8"] += 1
            if html_text_similarity >= 0.8:
                metrics["html_text_similarity_ge_0_8"] += 1

            if not (title_exact and html_exact):
                mismatches.append(
                    {
                        "id": row.get("id"),
                        "language_code": row.get("language_code"),
                        "expected_title_normalized": exp_title,
                        "predicted_title_normalized": pred_title,
                        "expected_description_html": exp_html,
                        "predicted_description_html": pred_html,
                        "raw_prediction": raw_prediction,
                        "valid_json": valid_json,
                        "title_exact": title_exact,
                        "html_exact": html_exact,
                        "html_allowed_tags_only": _uses_only_allowed_tags(pred_html),
                        "title_similarity": round(title_similarity, 4),
                        "html_text_similarity": round(html_text_similarity, 4),
                    }
                )
        batch_postprocess_sec += time.perf_counter() - t_post

        processed += len(batch_rows)
        logger.info(
            "eval batch timing rows=%s-%s/%s prompt_build=%.3fs tokenize=%.3fs generate=%.3fs decode=%.3fs parse=%.3fs postprocess=%.3fs",
            offset + 1,
            batch_end,
            len(rows),
            batch_prompt_build_sec,
            batch_tokenize_sec,
            batch_generate_sec,
            batch_decode_sec,
            batch_parse_sec,
            batch_postprocess_sec,
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
        "chunked": args.chunked,
        "chunk_max_chars": args.chunk_max_chars if args.chunked else None,
        "valid_json_rate": round(metrics["valid_json"] / metrics["total"], 4),
        "title_exact_rate": round(metrics["title_exact"] / metrics["total"], 4),
        "html_exact_rate": round(metrics["html_exact"] / metrics["total"], 4),
        "title_non_empty_rate": round(metrics["title_non_empty"] / metrics["total"], 4),
        "html_non_empty_rate": round(metrics["html_non_empty"] / metrics["total"], 4),
        "html_allowed_tags_only_rate": round(metrics["html_allowed_tags_only"] / metrics["total"], 4),
        "title_similarity_avg": round(metrics["title_similarity_sum"] / metrics["total"], 4),
        "html_text_similarity_avg": round(metrics["html_text_similarity_sum"] / metrics["total"], 4),
        "title_similarity_ge_0_8_rate": round(metrics["title_similarity_ge_0_8"] / metrics["total"], 4),
        "html_text_similarity_ge_0_8_rate": round(metrics["html_text_similarity_ge_0_8"] / metrics["total"], 4),
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


def _predict_row_chunked(
    *,
    row: dict[str, Any],
    model,
    tokenizer,
    device: str,
    max_new_tokens: int,
    temperature: float,
    chunk_max_chars: int,
    batch_size: int,
) -> dict[str, Any]:
    t_prompt_build = 0.0
    t_tokenize = 0.0
    t_generate = 0.0
    t_decode = 0.0
    t_parse = 0.0
    messages = row.get("messages") or []
    system = _message_content(messages, 0)
    user_content = _message_content(messages, 1)

    try:
        user_payload = json.loads(user_content)
    except json.JSONDecodeError:
        prompt = f"<|system|>\n{system}\n<|user|>\n{user_content}\n<|assistant|>\n"
        t_prompt = time.perf_counter()
        pred_list, stage_timings = _generate_predictions_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=[prompt],
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            return_timings=True,
        )
        t_prompt_build += time.perf_counter() - t_prompt
        t_tokenize += stage_timings["tokenize_sec"]
        t_generate += stage_timings["generate_sec"]
        t_decode += stage_timings["decode_sec"]
        pred = pred_list[0]
        t_parse0 = time.perf_counter()
        pred_obj = _parse_json_loose(pred)
        t_parse += time.perf_counter() - t_parse0
        return {
            "raw_prediction": pred,
            "pred_obj": pred_obj,
            "_timings": {
                "prompt_build_sec": t_prompt_build,
                "tokenize_sec": t_tokenize,
                "generate_sec": t_generate,
                "decode_sec": t_decode,
                "parse_sec": t_parse,
            },
        }

    desc = str(user_payload.get("description_raw") or "")
    chunks = split_text_chunks(desc, max_chars=chunk_max_chars)
    if not chunks:
        chunks = [""]

    prompts: list[str] = []
    t_prompt = time.perf_counter()
    for idx, chunk in enumerate(chunks, start=1):
        p = dict(user_payload)
        p["description_raw"] = chunk
        p["chunk_context"] = {"index": idx, "total": len(chunks)}
        prompts.append(f"<|system|>\n{system}\n<|user|>\n{json.dumps(p, ensure_ascii=False)}\n<|assistant|>\n")
    t_prompt_build += time.perf_counter() - t_prompt

    raw_parts: list[str] = []
    title = ""
    html_parts: list[str] = []

    for offset in range(0, len(prompts), max(1, batch_size)):
        sub_prompts = prompts[offset : offset + max(1, batch_size)]
        sub_preds, stage_timings = _generate_predictions_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=sub_prompts,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            return_timings=True,
        )
        t_tokenize += stage_timings["tokenize_sec"]
        t_generate += stage_timings["generate_sec"]
        t_decode += stage_timings["decode_sec"]
        raw_parts.extend(sub_preds)
        t_parse0 = time.perf_counter()
        for pred in sub_preds:
            obj = _parse_json_loose(pred) or {}
            if not title:
                maybe_title = str(obj.get("title_normalized") or "").strip()
                if maybe_title:
                    title = maybe_title
            maybe_html = str(obj.get("description_html") or "").strip()
            if maybe_html:
                html_parts.append(maybe_html)
        t_parse += time.perf_counter() - t_parse0

    merged = {
        "title_normalized": title,
        "description_html": "\n".join(html_parts).strip(),
    }
    return {
        "raw_prediction": "\n<chunk>\n".join(raw_parts),
        "pred_obj": merged,
        "_timings": {
            "prompt_build_sec": t_prompt_build,
            "tokenize_sec": t_tokenize,
            "generate_sec": t_generate,
            "decode_sec": t_decode,
            "parse_sec": t_parse,
        },
    }


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

    # Decoder-only models should use left padding for correct batched generation behavior.
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

    # In batched generation, decoded continuation starts after the padded prompt width.
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


def _parse_json_loose(content: str) -> dict[str, Any] | None:
    text = (content or "").strip()
    if not text:
        return None
    text = _strip_wrappers(text)

    # Prefer JSON objects that look like our target schema.
    for obj in _iter_json_objects(text):
        if _looks_like_prediction_object(obj):
            return obj

    # Fallback: first JSON object if schema-specific object is not found.
    for obj in _iter_json_objects(text):
        return obj

    # Last-resort fallback for near-JSON outputs with trailing/garbled text.
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
    # Accept if at least one expected key is present; both is preferred.
    return "title_normalized" in obj or "description_html" in obj


def _strip_wrappers(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    # Drop common trailing schema/control marker if present.
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
        # Decode JSON string escapes safely.
        try:
            decoded = json.loads(f'"{raw}"')
        except json.JSONDecodeError:
            decoded = raw
        recovered[key] = str(decoded).strip()
    if recovered:
        return recovered
    return None


def _uses_only_allowed_tags(html: str) -> bool:
    if not html:
        return True
    tags = {m.group(1).lower() for m in TAG_RE.finditer(html)}
    return tags.issubset(ALLOWED_TAGS)


def _html_text_similarity(expected_html: str, predicted_html: str) -> float:
    exp = _normalize_text(_strip_tags(expected_html))
    pred = _normalize_text(_strip_tags(predicted_html))
    return _text_similarity(exp, pred)


def _strip_tags(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value or "")
    return unescape(text)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _text_similarity(expected: str, predicted: str) -> float:
    exp = _normalize_text(expected)
    pred = _normalize_text(predicted)
    if not exp and not pred:
        return 1.0
    return SequenceMatcher(a=exp, b=pred).ratio()


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
        "title_similarity",
        "html_text_similarity",
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
