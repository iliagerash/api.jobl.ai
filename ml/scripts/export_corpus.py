"""
export_corpus.py
─────────────────
Phase 0, step 1 of the ML pipeline implementation plan: export the `jobs` and
`resumes` tables into JSONL for offline training data prep.

Reuses the existing app.db.session / app.models (no separate DB layer for the
ML pipeline) so this stays in sync with schema changes made via Alembic.

Streams rows with a server-side cursor (`yield_per`) instead of loading the
full ~1.5M jobs / ~225K resumes into memory at once.

Usage:
    python ml/scripts/export_corpus.py
    python ml/scripts/export_corpus.py --table jobs --limit 1000
    python ml/scripts/export_corpus.py --output-dir ml/data/raw --sample 20

Outputs:
    ml/data/raw/jobs.jsonl
    ml/data/raw/resumes.jsonl
"""

import argparse
import json
import os
import random
import sys
from datetime import date, datetime
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.job import Job
from app.models.resume import Resume


# Columns pulled per table. `category`/`language_code` on jobs are carried
# through as-is (not recomputed) — they're the baseline the Phase 4 Extractor
# cutover gate compares against, per the implementation plan.
JOB_FIELDS = [
    "id",
    "source_db",
    "external_id",
    "title",
    "description",
    "company_name",
    "city_title",
    "region_title",
    "country_code",
    "language_code",
    "salary_min",
    "salary_max",
    "salary_period",
    "salary_currency",
    "contract",
    "experience",
    "education",
    "published_at",
    "expires_at",
    "is_remote",
    "is_active",
    "category",
]

RESUME_FIELDS = [
    "id",
    "source_website",
    "external_id",
    "title",
    "description",
    "city_title",
    "region_title",
    "country_code",
    "salary",
    "salary_period",
    "salary_currency",
    "contract",
    "published_at",
    "is_remote",
    "language_code",
]


def _json_default(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _row_to_dict(row, fields: list[str]) -> dict:
    return {field: getattr(row, field) for field in fields}


def export_table(db, model, fields: list[str], out_path: str, *, limit: int | None, batch_size: int) -> int:
    stmt = select(model).execution_options(yield_per=batch_size)
    if limit is not None:
        stmt = stmt.limit(limit)

    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        result = db.execute(stmt)
        for row in result.scalars():
            record = _row_to_dict(row, fields)
            f.write(json.dumps(record, default=_json_default, ensure_ascii=False))
            f.write("\n")
            count += 1
            if count % 100_000 == 0:
                print(f"  ... {count:,} rows written to {out_path}")
    return count


def print_sample(out_path: str, n: int, *, description_chars: int = 300) -> None:
    with open(out_path, encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        print(f"  (no rows in {out_path})")
        return
    sample = random.sample(lines, min(n, len(lines)))
    print(f"\n--- {n} random sample rows from {out_path} (spot-check these) ---")
    for line in sample:
        record = json.loads(line)
        description = record.get("description")
        if description and len(description) > description_chars:
            record["description"] = description[:description_chars] + f"... [{len(description)} chars total]"
        print(json.dumps(record, indent=2, ensure_ascii=False))
        print("---")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export jobs/resumes tables to JSONL for ML pipeline data prep")
    parser.add_argument("--output-dir", default="ml/data/raw", help="Output directory (default: ml/data/raw)")
    parser.add_argument(
        "--table",
        choices=["jobs", "resumes", "both"],
        default="both",
        help="Which table(s) to export (default: both)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap rows per table (for testing)")
    parser.add_argument("--batch-size", type=int, default=5000, help="Server-side cursor batch size (default: 5000)")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="After export, print N random rows per table to stdout for manual spot-check (gate requires 20)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    db = SessionLocal()
    try:
        if args.table in ("jobs", "both"):
            jobs_path = os.path.join(args.output_dir, "jobs.jsonl")
            print(f"Exporting jobs -> {jobs_path} ...")
            n_jobs = export_table(db, Job, JOB_FIELDS, jobs_path, limit=args.limit, batch_size=args.batch_size)
            print(f"Wrote {n_jobs:,} rows to {jobs_path}")
            if args.sample:
                print_sample(jobs_path, args.sample)

        if args.table in ("resumes", "both"):
            resumes_path = os.path.join(args.output_dir, "resumes.jsonl")
            print(f"Exporting resumes -> {resumes_path} ...")
            n_resumes = export_table(
                db, Resume, RESUME_FIELDS, resumes_path, limit=args.limit, batch_size=args.batch_size
            )
            print(f"Wrote {n_resumes:,} rows to {resumes_path}")
            if args.sample:
                print_sample(resumes_path, args.sample)
    finally:
        db.close()


if __name__ == "__main__":
    main()
