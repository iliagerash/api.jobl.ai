"""
make_splits.py
───────────────
Phase 0, step 5: split the interim job and resume corpora into train/val pools
by time (not random), stratified to ensure the val set covers ≥15 countries.

Uses a two-pass streaming approach to avoid loading the full corpus into memory:
  Pass 1: scan all timestamps to find the percentile cutoff date
  Pass 2: stream records into train or val output files based on the cutoff

Usage:
    python -u ml/scripts/make_splits.py
    python -u ml/scripts/make_splits.py --val-fraction 0.1 --min-countries 15

Outputs:
    ml/data/splits/train_pool_jobs.jsonl
    ml/data/splits/train_pool_resumes.jsonl
    ml/data/splits/val_pool_jobs.jsonl
    ml/data/splits/val_pool_resumes.jsonl
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

HEARTBEAT_EVERY = 100_000

_EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)


def _parse_dt(value: str | None) -> datetime:
    if not value:
        return _EPOCH
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return _EPOCH


def iter_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _json_default(v):
    from datetime import date
    from decimal import Decimal
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    raise TypeError(type(v).__name__)


def pass1_find_cutoff(input_path: str, val_fraction: float) -> tuple[datetime, int]:
    """Scan all timestamps, sort just the timestamps (not full records), find cutoff."""
    print(f"  Pass 1: scanning timestamps in {input_path} ...")
    timestamps: list[datetime] = []
    count = 0
    for record in iter_jsonl(input_path):
        timestamps.append(_parse_dt(record.get("published_at")))
        count += 1
        if count % HEARTBEAT_EVERY == 0:
            print(f"    ... {count:,} records scanned")

    print(f"    {count:,} records total")
    timestamps.sort()

    split_idx = int(count * (1 - val_fraction))
    cutoff = timestamps[split_idx] if split_idx < count else timestamps[-1]
    print(f"    Cutoff date: {cutoff.isoformat()} (record {split_idx:,} of {count:,})")
    return cutoff, count


def pass2_split(
    input_path: str,
    train_path: str,
    val_path: str,
    cutoff: datetime,
    min_countries: int,
) -> dict:
    """Stream records into train/val files based on cutoff date."""
    print(f"  Pass 2: splitting records ...")
    train_count = 0
    val_count = 0
    val_countries: dict[str, int] = {}
    count = 0

    with open(train_path, "w", encoding="utf-8") as train_f, \
         open(val_path, "w", encoding="utf-8") as val_f:
        for record in iter_jsonl(input_path):
            dt = _parse_dt(record.get("published_at"))
            if dt > cutoff:
                val_f.write(json.dumps(record, default=_json_default, ensure_ascii=False))
                val_f.write("\n")
                val_count += 1
                cc = record.get("country_code") or "NULL"
                val_countries[cc] = val_countries.get(cc, 0) + 1
            else:
                train_f.write(json.dumps(record, default=_json_default, ensure_ascii=False))
                train_f.write("\n")
                train_count += 1

            count += 1
            if count % HEARTBEAT_EVERY == 0:
                print(f"    ... {count:,} records processed")

    print(f"    {train_count:,} train / {val_count:,} val")

    # Check if we need to augment val with underrepresented countries.
    # If val already has enough countries, skip the expensive augmentation.
    distinct_countries = len([c for c in val_countries if c != "NULL"])
    if distinct_countries < min_countries:
        print(f"    Val covers {distinct_countries} countries, need {min_countries} — augmenting ...")
        _augment_countries(train_path, val_path, val_countries, min_countries)

    return {
        "train_count": train_count,
        "val_count": val_count,
        "val_countries": val_countries,
    }


def _augment_countries(
    train_path: str,
    val_path: str,
    val_countries: dict[str, int],
    min_countries: int,
) -> None:
    """Move recent records from underrepresented countries into the val set.

    Reads the train file to find countries missing from val, collects up to 50
    records per missing country (from the end of the file = most recent), then
    rewrites the train file without those records and appends them to val.
    """
    existing_val_countries = set(val_countries.keys()) - {"NULL"}

    # Find all countries in train
    train_countries: set[str] = set()
    for record in iter_jsonl(train_path):
        cc = record.get("country_code")
        if cc:
            train_countries.add(cc)

    missing = train_countries - existing_val_countries
    if not missing:
        return

    # Collect IDs to move (up to 50 per missing country)
    ids_to_move: set = set()
    country_collected: dict[str, int] = {}
    # Read train in reverse order (most recent first) — but since we can't
    # efficiently reverse a file, just scan forward and keep the LAST 50 per country
    country_records: dict[str, list[int]] = {c: [] for c in missing}
    for record in iter_jsonl(train_path):
        cc = record.get("country_code")
        if cc in country_records:
            country_records[cc].append(record.get("id"))

    for cc in sorted(missing):
        # Take the last 50 (most recent since file is roughly time-ordered after pass2)
        to_move = country_records[cc][-50:]
        ids_to_move.update(to_move)
        val_countries[cc] = val_countries.get(cc, 0) + len(to_move)
        if len([c for c in val_countries if c != "NULL"]) >= min_countries:
            break

    # Rewrite train (excluding moved records) and append to val
    tmp_train = train_path + ".tmp"
    moved = 0
    with open(tmp_train, "w", encoding="utf-8") as train_f, \
         open(val_path, "a", encoding="utf-8") as val_f:
        for record in iter_jsonl(train_path):
            if record.get("id") in ids_to_move:
                val_f.write(json.dumps(record, default=_json_default, ensure_ascii=False))
                val_f.write("\n")
                moved += 1
            else:
                train_f.write(json.dumps(record, default=_json_default, ensure_ascii=False))
                train_f.write("\n")

    os.replace(tmp_train, train_path)
    distinct = len([c for c in val_countries if c != "NULL"])
    print(f"    Augmented {moved} records from {len(ids_to_move)} IDs; val now covers {distinct} countries")


def print_country_stats(label: str, country_counts: dict[str, int]) -> None:
    sorted_countries = sorted(country_counts.items(), key=lambda x: -x[1])
    print(f"  {label} val set — {len(sorted_countries)} countries:")
    for cc, count in sorted_countries[:30]:
        print(f"    {cc:>5}: {count:>8,}")
    if len(sorted_countries) > 30:
        others = sum(c for _, c in sorted_countries[30:])
        print(f"    other: {others:>8,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-based stratified train/val split for ML pipeline")
    parser.add_argument("--input-dir", default="ml/data/interim")
    parser.add_argument("--output-dir", default="ml/data/splits")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of records for val set (default: 0.1)")
    parser.add_argument("--min-countries", type=int, default=15, help="Minimum distinct countries in val set (default: 15)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    jobs_path = os.path.join(args.input_dir, "jobs.jsonl")
    resumes_path = os.path.join(args.input_dir, "resumes.jsonl")

    if os.path.exists(jobs_path):
        print("Splitting jobs ...")
        cutoff, total = pass1_find_cutoff(jobs_path, args.val_fraction)
        stats = pass2_split(
            jobs_path,
            os.path.join(args.output_dir, "train_pool_jobs.jsonl"),
            os.path.join(args.output_dir, "val_pool_jobs.jsonl"),
            cutoff,
            args.min_countries,
        )
        print_country_stats("Jobs", stats["val_countries"])
        n_countries = len([c for c in stats["val_countries"] if c != "NULL"])
        if n_countries >= args.min_countries:
            print(f"  GATE PASSED: {n_countries} countries >= {args.min_countries}")
        else:
            print(f"  GATE FAILED: {n_countries} countries < {args.min_countries} — review data coverage")
    else:
        print(f"Skipping jobs — {jobs_path} not found")

    if os.path.exists(resumes_path):
        print("\nSplitting resumes ...")
        cutoff, total = pass1_find_cutoff(resumes_path, args.val_fraction)
        stats = pass2_split(
            resumes_path,
            os.path.join(args.output_dir, "train_pool_resumes.jsonl"),
            os.path.join(args.output_dir, "val_pool_resumes.jsonl"),
            cutoff,
            args.min_countries,
        )
        print_country_stats("Resumes", stats["val_countries"])
        n_countries = len([c for c in stats["val_countries"] if c != "NULL"])
        if n_countries >= args.min_countries:
            print(f"  GATE PASSED: {n_countries} countries >= {args.min_countries}")
        else:
            print(f"  GATE FAILED: {n_countries} countries < {args.min_countries} — review data coverage")
    else:
        print(f"Skipping resumes — {resumes_path} not found")

    print("\nDone.")


if __name__ == "__main__":
    main()
