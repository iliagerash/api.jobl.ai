"""
teacher_extract.py
────────────────────
Phase 1, Day 1-2: run the locked extraction prompt against job postings via
the vLLM-served teacher model (Qwen2.5-72B-Instruct-AWQ).

Connects to the local vLLM server's OpenAI-compatible API. Processes jobs
concurrently for throughput. Supports resumability — skips IDs already
present in the output file.

Target: 100K–200K extractions.

Usage (on the A100, with vLLM running):
    # Start vLLM in a separate terminal/screen first:
    # vllm serve ml/models/base/teacher --quantization awq --max-model-len 4096

    python -u ml/scripts/teacher_extract.py
    python -u ml/scripts/teacher_extract.py --limit 1000               # test run
    python -u ml/scripts/teacher_extract.py --workers 8                # more concurrency
    python -u ml/scripts/teacher_extract.py --input ml/data/splits/val_pool_jobs.jsonl
    python -u ml/scripts/teacher_extract.py --resume                   # skip already processed

Outputs:
    ml/data/teacher_labels/extractions.jsonl
"""

import argparse
import html
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ml.scripts.validate_extraction_prompt import (
    SYSTEM_PROMPT,
    VALID_ENUMS,
    CATEGORIES,
    REQUIRED_FIELDS,
)

HEARTBEAT_EVERY = 500
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _clean_description(desc: str | None) -> str:
    text = html.unescape(desc or "")
    text = re.sub(r"<\s*script\b[^>]*>.*?</\s*script\s*>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<\s*style\b[^>]*>.*?</\s*style\s*>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = _HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:3000]


def _build_user_message(record: dict) -> str:
    lang = record.get("language_fasttext") or record.get("language_code") or "unknown"
    country = record.get("country_code") or "unknown"
    title = record.get("title") or ""
    desc = _clean_description(record.get("description"))
    return f"[lang={lang}][country={country}][type=job] {title} — {desc}"


def _validate_extraction(result: dict) -> list[str]:
    errors = []
    missing = REQUIRED_FIELDS - set(result.keys())
    if missing:
        errors.append(f"missing fields: {missing}")
    for field, valid_values in VALID_ENUMS.items():
        value = result.get(field)
        if value is not None and value not in valid_values:
            errors.append(f"{field}={value!r} invalid")
    cat = result.get("occupation_category")
    if cat and cat not in CATEGORIES:
        errors.append(f"occupation_category={cat!r} invalid")
    return errors


def iter_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _json_default(v):
    from datetime import date, datetime
    from decimal import Decimal
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    raise TypeError(type(v).__name__)


def load_processed_ids(output_path: str) -> set:
    ids = set()
    if not os.path.exists(output_path):
        return ids
    for record in iter_jsonl(output_path):
        job_id = record.get("job_id")
        if job_id is not None:
            ids.add(job_id)
    return ids


def extract_one(client, model: str, record: dict, max_retries: int = 3) -> dict | None:
    """Send a single job to the teacher model, return extraction result or None."""
    user_msg = _build_user_message(record)

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=1000,
                temperature=0.0,
            )
            content = resp.choices[0].message.content or ""
            # Parse JSON from raw output (no constrained decoding)
            content = content.strip()
            if content.startswith("```"):
                content = content.strip("`").replace("json\n", "", 1).strip()
            result = json.loads(content)
            errors = _validate_extraction(result)
            if errors:
                if attempt >= max_retries:
                    return {"_error": f"validation failed: {errors}", "_raw": content[:500]}
                time.sleep(1)
                continue
            return result
        except json.JSONDecodeError as e:
            if attempt >= max_retries:
                return {"_error": f"invalid JSON: {e}", "_raw": content[:500]}
            time.sleep(1)
        except Exception as e:
            if attempt >= max_retries:
                return {"_error": str(e)}
            time.sleep(2 ** attempt)

    return {"_error": "max retries exceeded"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run teacher extraction on job postings via vLLM")
    parser.add_argument("--input", default="ml/data/splits/train_pool_jobs.jsonl")
    parser.add_argument("--output", default="ml/data/teacher_labels/extractions.jsonl")
    parser.add_argument("--api-base", default="http://localhost:8000/v1", help="vLLM server URL")
    parser.add_argument("--model", default="ml/models/base/teacher", help="Model name as registered in vLLM")
    parser.add_argument("--limit", type=int, default=None, help="Max records to process")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent request workers (default: 4)")
    parser.add_argument("--resume", action="store_true", help="Skip IDs already in the output file")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    from openai import OpenAI
    client = OpenAI(base_url=args.api_base, api_key="not-needed")

    # Load already-processed IDs for resumability
    processed_ids = set()
    if args.resume:
        processed_ids = load_processed_ids(args.output)
        print(f"Resume mode: {len(processed_ids):,} already processed, will skip")

    # Collect jobs to process
    print(f"Loading jobs from {args.input} ...")
    jobs = []
    for record in iter_jsonl(args.input):
        if args.resume and record.get("id") in processed_ids:
            continue
        jobs.append(record)
        if args.limit and len(jobs) >= args.limit:
            break
    print(f"Processing {len(jobs):,} jobs with {args.workers} workers")

    # Open output file in append mode for resumability
    mode = "a" if args.resume and os.path.exists(args.output) else "w"
    total = 0
    succeeded = 0
    failed = 0

    with open(args.output, mode, encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for record in jobs:
                future = executor.submit(extract_one, client, args.model, record)
                futures[future] = record

            for future in as_completed(futures):
                record = futures[future]
                result = future.result()
                total += 1

                if result is None or "_error" in result:
                    failed += 1
                    error_msg = result.get("_error", "unknown") if result else "None returned"
                    if total <= 10 or failed % 100 == 0:
                        print(f"  FAIL id={record.get('id')}: {error_msg}")
                else:
                    succeeded += 1

                # Write output: original job metadata + extraction
                output_record = {
                    "job_id": record.get("id"),
                    "country_code": record.get("country_code"),
                    "language_code": record.get("language_code"),
                    "language_fasttext": record.get("language_fasttext"),
                    "title_raw": record.get("title"),
                    "extraction": result,
                }
                out.write(json.dumps(output_record, default=_json_default, ensure_ascii=False))
                out.write("\n")
                out.flush()

                if total % HEARTBEAT_EVERY == 0:
                    print(f"  ... {total:,} processed ({succeeded:,} ok, {failed:,} failed)")

    print(f"\nDone. {total:,} processed -> {args.output}")
    print(f"  Succeeded: {succeeded:,} ({succeeded/total:.1%})" if total else "")
    print(f"  Failed:    {failed:,} ({failed/total:.1%})" if total else "")


if __name__ == "__main__":
    main()
