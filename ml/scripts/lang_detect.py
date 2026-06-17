"""
lang_detect.py
───────────────
Phase 0, step 4: run fastText language detection (lid.176.bin) over all jobs
and resumes in ml/data/interim/, adding `language_fasttext` (ISO 639-1) and
`language_confidence` (0.0–1.0) fields to each record.

The existing `language_code` field (from the sync pipeline's langid detection,
covering ~10 languages) is preserved unchanged — it's the baseline for the
Phase 4 Extractor cutover gate.  The new fastText fields provide full
176-language coverage for ML pipeline use (locale prefixes, stratified splits).

Setup (one-time, on the server):
    pip install -e ".[ml]"
    # Download the fastText language ID model (~126MB):
    wget -P ml/models/ https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

Usage:
    python -u ml/scripts/lang_detect.py
    python -u ml/scripts/lang_detect.py --input ml/data/interim/jobs.jsonl --table jobs
    python -u ml/scripts/lang_detect.py --table resumes

Outputs:
    Updates ml/data/interim/jobs.jsonl and resumes.jsonl in place, adding
    language_fasttext and language_confidence fields to each record.
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

HEARTBEAT_EVERY = 100_000
DEFAULT_MODEL_PATH = "ml/models/lid.176.bin"

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _clean_for_detection(title: str, description: str | None, max_chars: int = 1000) -> str:
    """Strip HTML tags and collapse whitespace for language detection input."""
    desc = _HTML_TAG_RE.sub(" ", description or "")
    combined = f"{title or ''} {desc}"
    combined = _WHITESPACE_RE.sub(" ", combined).strip()
    return combined[:max_chars]


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


def process_file(model, input_path: str) -> dict:
    """Add language_fasttext + language_confidence to each record, in place."""
    tmp_path = input_path + ".tmp"
    total = 0
    lang_counts: dict[str, int] = {}
    low_confidence = 0
    confidence_buckets = {"<0.3": 0, "0.3-0.5": 0, "0.5-0.8": 0, "0.8-1.0": 0}

    with open(tmp_path, "w", encoding="utf-8") as out:
        for record in iter_jsonl(input_path):
            text = _clean_for_detection(record.get("title", ""), record.get("description"))

            if len(text.strip()) < 3:
                lang = "unknown"
                confidence = 0.0
            else:
                # fastText predict returns (labels, scores) — single prediction
                labels, scores = model.predict(text.replace("\n", " "), k=1)
                lang = labels[0].replace("__label__", "")
                confidence = float(scores[0])

            record["language_fasttext"] = lang
            record["language_confidence"] = round(confidence, 4)

            lang_counts[lang] = lang_counts.get(lang, 0) + 1
            if confidence < 0.5:
                low_confidence += 1

            if confidence < 0.3:
                confidence_buckets["<0.3"] += 1
            elif confidence < 0.5:
                confidence_buckets["0.3-0.5"] += 1
            elif confidence < 0.8:
                confidence_buckets["0.5-0.8"] += 1
            else:
                confidence_buckets["0.8-1.0"] += 1

            out.write(json.dumps(record, default=_json_default, ensure_ascii=False))
            out.write("\n")

            total += 1
            if total % HEARTBEAT_EVERY == 0:
                print(f"  ... {total:,} records processed")

    os.replace(tmp_path, input_path)

    return {
        "total": total,
        "lang_counts": lang_counts,
        "low_confidence": low_confidence,
        "confidence_buckets": confidence_buckets,
    }


def print_stats(label: str, stats: dict) -> None:
    total = stats["total"]
    lang_counts = stats["lang_counts"]
    low_confidence = stats["low_confidence"]
    buckets = stats["confidence_buckets"]

    print(f"\n  {label}: {total:,} records")
    print(f"  Low confidence (<0.5): {low_confidence:,} ({low_confidence / total:.1%})")
    print(f"  Confidence distribution:")
    for bucket, count in buckets.items():
        print(f"    {bucket:>7}: {count:>10,} ({count / total:.1%})")

    print(f"  Top 20 languages:")
    sorted_langs = sorted(lang_counts.items(), key=lambda x: -x[1])
    for lang, count in sorted_langs[:20]:
        print(f"    {lang:>7}: {count:>10,} ({count / total:.1%})")
    if len(sorted_langs) > 20:
        others = sum(c for _, c in sorted_langs[20:])
        print(f"    {'other':>7}: {others:>10,} ({others / total:.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="FastText language detection for ML pipeline data")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help=f"Path to lid.176.bin (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument(
        "--table",
        choices=["jobs", "resumes", "both"],
        default="both",
        help="Which file(s) to process (default: both)",
    )
    parser.add_argument("--input-dir", default="ml/data/interim", help="Directory containing jobs.jsonl / resumes.jsonl")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(
            f"ERROR: fastText model not found at {args.model_path}\n"
            f"Download it with:\n"
            f"  wget -P ml/models/ https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        )
        sys.exit(1)

    import fasttext
    # Suppress fastText's warning about deprecated loading method
    fasttext.FastText.eprint = lambda x: None
    model = fasttext.load_model(args.model_path)
    print(f"Loaded fastText model from {args.model_path}")

    if args.table in ("jobs", "both"):
        jobs_path = os.path.join(args.input_dir, "jobs.jsonl")
        if os.path.exists(jobs_path):
            print(f"Processing {jobs_path} ...")
            stats = process_file(model, jobs_path)
            print_stats("Jobs", stats)
        else:
            print(f"Skipping jobs — {jobs_path} not found")

    if args.table in ("resumes", "both"):
        resumes_path = os.path.join(args.input_dir, "resumes.jsonl")
        if os.path.exists(resumes_path):
            print(f"Processing {resumes_path} ...")
            stats = process_file(model, resumes_path)
            print_stats("Resumes", stats)
        else:
            print(f"Skipping resumes — {resumes_path} not found")

    print("\nDone. Records updated in place with language_fasttext + language_confidence fields.")


if __name__ == "__main__":
    main()
