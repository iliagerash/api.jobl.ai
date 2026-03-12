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


logger = logging.getLogger("jobl.training.eval_seq2seq")
INPUT_PREFIX = "normalize job title: "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate seq2seq model on SFT test set")
    parser.add_argument("--test-jsonl", default="data/sft/test.jsonl", help="Instruction test JSONL path")
    parser.add_argument("--model-dir", default="artifacts/flan-t5-normalize-v1/best", help="Seq2Seq model directory")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to evaluate, 0 means all")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--progress-every", type=int, default=10, help="Log progress every N rows")
    parser.add_argument("--out-dir", default="artifacts/flan-t5-normalize-v1/eval", help="Directory for evaluation artifacts")
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
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        device_map="auto" if has_cuda else None,
        low_cpu_mem_usage=True,
    )
    model.eval()

    device = "cuda" if has_cuda else "cpu"
    if not has_cuda:
        model.to("cpu")

    metrics = {
        "total": len(rows),
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
        32,
        progress_every,
    )

    for offset in range(0, len(rows), batch_size):
        batch_rows = rows[offset : offset + batch_size]
        batch_end = offset + len(batch_rows)
        logger.info("eval batch started rows=%s-%s/%s", offset + 1, batch_end, len(rows))

        t_prompt = time.perf_counter()
        prompts = [_build_inference_input(row) for row in batch_rows]
        prompt_build_sec = time.perf_counter() - t_prompt

        preds, stage_timings = _generate_predictions_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            device=device,
            max_new_tokens=32,
            return_timings=True,
        )
        if len(preds) != len(batch_rows):
            raise SystemExit("batch generation returned unexpected number of predictions")

        t_parse = time.perf_counter()
        for row, raw_prediction in zip(batch_rows, preds):
            expected = _parse_assistant_target(row)
            pred_title = _parse_plain_title(raw_prediction)
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
                        "valid_json": None,
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
        "model": str(args.model_dir),
        "model_dir": args.model_dir,
        "changed_titles_only": bool(args.changed_titles_only),
        "source_total_rows": len(all_rows),
        "selected_rows": len(rows),
        "skipped_unchanged_title_rows": skipped_unchanged_titles,
        # Not applicable for seq2seq plain-text output; key is kept for downstream compatibility.
        "valid_json": None,
        # Not applicable for seq2seq plain-text output; key is kept for downstream compatibility.
        "valid_json_rate": None,
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
        None,
        metrics["title_exact"],
        len(mismatches),
    )
    logger.info(
        "artifacts summary=%s mismatches_jsonl=%s mismatches_csv=%s",
        out_dir / "summary.json",
        out_dir / "mismatches.jsonl",
        out_dir / "mismatches.csv",
    )


def _build_inference_input(row: dict[str, Any]) -> str:
    return str(row.get("input") or "")


def _generate_predictions_batch(
    *,
    model,
    tokenizer,
    prompts: list[str],
    device: str,
    max_new_tokens: int,
    return_timings: bool = False,
) -> list[str] | tuple[list[str], dict[str, float]]:
    import torch

    tokenizer.padding_side = "right"
    t0 = time.perf_counter()
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    t1 = time.perf_counter()

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    t2 = time.perf_counter()

    predictions = [text.strip() for text in tokenizer.batch_decode(out, skip_special_tokens=True)]
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
    return {"title_normalized": str(row.get("target") or "").strip()}


def _extract_original_title(row: dict[str, Any]) -> str:
    input_text = str(row.get("input") or "")
    if input_text.startswith(INPUT_PREFIX):
        return input_text[len(INPUT_PREFIX) :].strip()
    return input_text.strip()


def _parse_plain_title(raw: str) -> str:
    return str(raw or "").strip()


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _text_similarity(expected: str, predicted: str) -> float:
    exp = _normalize_text(expected)
    pred = _normalize_text(predicted)
    if not exp and not pred:
        return 1.0
    return SequenceMatcher(a=exp, b=pred).ratio()


def _row_has_changed_title(row: dict[str, Any]) -> bool:
    expected = _parse_assistant_target(row)
    expected_title = _canonical_title(str(expected.get("title_normalized") or ""))
    if not expected_title:
        return False
    raw_title = _canonical_title(_extract_original_title(row))
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
