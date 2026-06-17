"""
make_splits.py
───────────────
Phase 0, step 5: split the interim job and resume corpora into train/val pools
by time (not random), stratified to ensure the val set covers ≥15 countries.

Time-based splitting prevents template-pattern leakage: the val pool contains
the most recent records, and the train pool contains older ones — mimicking
how the production system would encounter new data after training.

The val pools are where Phase 1's teacher labeling will draw validation pairs
from; the actual (resume, job, score) triples are constructed during teacher
labeling, not here.

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

# Records without a parseable published_at get this sentinel (very old) so
# they land in the train pool, not val.
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


def _write_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=_json_default, ensure_ascii=False))
            f.write("\n")


def split_by_time(
    input_path: str,
    val_fraction: float,
    min_countries: int,
    label: str,
) -> tuple[list[dict], list[dict], dict]:
    """Load, sort by published_at, split into train/val pools.

    If the initial time-based val set has fewer than min_countries distinct
    country codes, augment it by pulling the most recent records from
    underrepresented countries in the train set.
    """
    print(f"  Loading {input_path} ...")
    records = list(iter_jsonl(input_path))
    total = len(records)
    print(f"  {total:,} records loaded")

    # Sort by published_at ascending (oldest first)
    records.sort(key=lambda r: _parse_dt(r.get("published_at")))

    # Split at the time boundary
    split_idx = int(total * (1 - val_fraction))
    train = records[:split_idx]
    val = records[split_idx:]

    print(f"  Time split: {len(train):,} train / {len(val):,} val (cutoff at record {split_idx:,})")
    if train:
        cutoff_date = _parse_dt(train[-1].get("published_at"))
        print(f"  Train ends at: {cutoff_date.isoformat()}")
    if val:
        val_start = _parse_dt(val[0].get("published_at"))
        print(f"  Val starts at: {val_start.isoformat()}")

    # Check country coverage in val set
    val_countries = set(r.get("country_code") for r in val if r.get("country_code"))
    print(f"  Val set covers {len(val_countries)} countries")

    if len(val_countries) < min_countries:
        # Find countries missing from val but present in train
        train_countries = set(r.get("country_code") for r in train if r.get("country_code"))
        missing = train_countries - val_countries
        print(f"  Augmenting val set with records from {len(missing)} underrepresented countries ...")

        # For each missing country, move the most recent records (from the end
        # of train, since it's sorted by time) into val
        augmented = 0
        for country in sorted(missing):
            # Find up to 50 most-recent records for this country in train
            country_records = [r for r in reversed(train) if r.get("country_code") == country][:50]
            for r in country_records:
                train.remove(r)
                val.append(r)
                augmented += 1
            val_countries.add(country)
            if len(val_countries) >= min_countries:
                break

        print(f"  Augmented {augmented} records; val now covers {len(val_countries)} countries")

    # Compute per-country stats for val
    country_counts: dict[str, int] = {}
    for r in val:
        cc = r.get("country_code") or "NULL"
        country_counts[cc] = country_counts.get(cc, 0) + 1

    return train, val, country_counts


def print_country_stats(label: str, country_counts: dict[str, int]) -> None:
    sorted_countries = sorted(country_counts.items(), key=lambda x: -x[1])
    print(f"  {label} val set — {len(sorted_countries)} countries:")
    for cc, count in sorted_countries:
        print(f"    {cc:>5}: {count:>8,}")


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
        train_jobs, val_jobs, job_country_counts = split_by_time(
            jobs_path, args.val_fraction, args.min_countries, "jobs"
        )
        _write_jsonl(os.path.join(args.output_dir, "train_pool_jobs.jsonl"), train_jobs)
        _write_jsonl(os.path.join(args.output_dir, "val_pool_jobs.jsonl"), val_jobs)
        print(f"  Wrote {len(train_jobs):,} train + {len(val_jobs):,} val jobs")
        print_country_stats("Jobs", job_country_counts)
        # Gate check
        n_countries = len([c for c in job_country_counts if c != "NULL"])
        if n_countries >= args.min_countries:
            print(f"  GATE PASSED: {n_countries} countries ≥ {args.min_countries}")
        else:
            print(f"  GATE FAILED: {n_countries} countries < {args.min_countries} — review data coverage")
    else:
        print(f"Skipping jobs — {jobs_path} not found")

    if os.path.exists(resumes_path):
        print("\nSplitting resumes ...")
        train_resumes, val_resumes, resume_country_counts = split_by_time(
            resumes_path, args.val_fraction, args.min_countries, "resumes"
        )
        _write_jsonl(os.path.join(args.output_dir, "train_pool_resumes.jsonl"), train_resumes)
        _write_jsonl(os.path.join(args.output_dir, "val_pool_resumes.jsonl"), val_resumes)
        print(f"  Wrote {len(train_resumes):,} train + {len(val_resumes):,} val resumes")
        print_country_stats("Resumes", resume_country_counts)
        n_countries = len([c for c in resume_country_counts if c != "NULL"])
        if n_countries >= args.min_countries:
            print(f"  GATE PASSED: {n_countries} countries ≥ {args.min_countries}")
        else:
            print(f"  GATE FAILED: {n_countries} countries < {args.min_countries} — review data coverage")
    else:
        print(f"Skipping resumes — {resumes_path} not found")

    print("\nDone.")


if __name__ == "__main__":
    main()
