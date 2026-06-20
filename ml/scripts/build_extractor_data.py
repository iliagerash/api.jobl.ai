"""
build_extractor_data.py
────────────────────────
Phase 4 prep: convert teacher extractions into chat-format SFT examples
for LoRA fine-tuning of Qwen2.5-7B-Instruct.

Each example is a conversation:
  system: extraction instruction
  user:   raw job posting with locale prefix
  assistant: the teacher's extraction JSON output

Input:  ml/data/teacher_labels/extractions.jsonl
        ml/data/splits/train_pool_jobs.jsonl (for full job text)
Output: ml/data/splits/extractor_train.jsonl
        ml/data/splits/extractor_val.jsonl

Usage:
    python -u ml/scripts/build_extractor_data.py
"""

import argparse
import html
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ml.scripts.validate_extraction_prompt import SYSTEM_PROMPT

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def iter_jsonl(path: str):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _clean_description(desc: str | None) -> str:
    text = html.unescape(desc or "")
    text = re.sub(r"<\s*script\b[^>]*>.*?</\s*script\s*>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<\s*style\b[^>]*>.*?</\s*style\s*>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = _HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:3000]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build extractor SFT dataset from teacher extractions")
    parser.add_argument("--extractions-input", default="ml/data/teacher_labels/extractions.jsonl")
    parser.add_argument("--jobs-input", default="ml/data/splits/train_pool_jobs.jsonl")
    parser.add_argument("--output-dir", default="ml/data/splits")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load extractions (only successful ones)
    print(f"Loading extractions from {args.extractions_input} ...")
    extractions = {}
    skipped = 0
    for record in iter_jsonl(args.extractions_input):
        ext = record.get("extraction") or {}
        if "_error" in ext:
            skipped += 1
            continue
        job_id = record.get("job_id")
        if job_id is not None:
            extractions[job_id] = {
                "extraction": ext,
                "language_fasttext": record.get("language_fasttext"),
                "language_code": record.get("language_code"),
                "country_code": record.get("country_code"),
            }
    print(f"  {len(extractions):,} successful extractions loaded, {skipped:,} skipped")

    # Load job descriptions to pair with extractions
    print(f"Loading job descriptions from {args.jobs_input} ...")
    examples = []
    matched = 0
    for record in iter_jsonl(args.jobs_input):
        job_id = record.get("id")
        if job_id not in extractions:
            continue

        ext_data = extractions[job_id]
        lang = ext_data.get("language_fasttext") or ext_data.get("language_code") or "unknown"
        country = ext_data.get("country_code") or "unknown"
        title = record.get("title", "")
        desc = _clean_description(record.get("description"))

        user_msg = f"[lang={lang}][country={country}][type=job] {title} — {desc}"
        assistant_msg = json.dumps(ext_data["extraction"], ensure_ascii=False)

        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ],
            "job_id": job_id,
        })
        matched += 1

        if matched % 5000 == 0:
            print(f"  ... {matched:,} examples built")

    print(f"  {len(examples):,} SFT examples built")

    # Split
    random.shuffle(examples)
    split_idx = int(len(examples) * (1 - args.val_fraction))
    train = examples[:split_idx]
    val = examples[split_idx:]

    for path, data, label in [
        (os.path.join(args.output_dir, "extractor_train.jsonl"), train, "train"),
        (os.path.join(args.output_dir, "extractor_val.jsonl"), val, "val"),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False))
                f.write("\n")
        print(f"  {label}: {len(data):,} examples -> {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
