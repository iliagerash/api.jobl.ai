"""
backfill_destination_from_removals.py
──────────────────────────────────────
Backfill destination and destination_job_id on jobl jobs table using
job_removals dumps from target job boards.

Matches by position = title, city_title, region_title, and expires_at
within a 24-hour window of expired_at.

CSV files in data/job_removals/ are named as {destination}.csv
(e.g. newzealand.jobradars.com.csv).

Usage:
    python -u scripts/backfill_destination_from_removals.py
    python -u scripts/backfill_destination_from_removals.py --dry-run
    python -u scripts/backfill_destination_from_removals.py --file data/job_removals/jobradars.com.csv
"""

import argparse
import csv
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

BATCH_SIZE = 500
REMOVALS_DIR = "data/job_removals"


def process_file(engine, csv_path: str, dry_run: bool) -> tuple[int, int]:
    destination = os.path.basename(csv_path).replace(".csv", "")
    print(f"\n--- {destination} ---")

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"  {len(rows):,} removals loaded")

    if not rows:
        return 0, 0

    query = text("""
        UPDATE jobs
        SET destination = :destination,
            destination_job_id = :destination_job_id
        WHERE title = :position
          AND city_title = :city_title
          AND region_title = :region_title
          AND expires_at BETWEEN CAST(:expired_at AS timestamptz) - interval '24 hours'
                             AND CAST(:expired_at AS timestamptz) + interval '24 hours'
          AND destination IS NULL
    """)

    updated = 0
    scanned = 0
    start = time.time()

    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        params = [
            {
                "destination": destination,
                "destination_job_id": int(row["id"]),
                "position": row["position"],
                "city_title": row["city_title"],
                "region_title": row["region_title"],
                "expired_at": row["expired_at"],
            }
            for row in batch
            if row.get("expired_at") and row.get("position") and row["expired_at"] >= "2026-"
        ]

        if params and not dry_run:
            with engine.begin() as conn:
                result = conn.execute(query, params)
                updated += result.rowcount

        scanned += len(batch)
        if scanned % 10000 < BATCH_SIZE:
            elapsed = time.time() - start
            rate = scanned / elapsed if elapsed > 0 else 0
            print(f"  scanned={scanned:,}  updated={updated:,}  ({rate:.0f} rows/sec)")

    elapsed = time.time() - start
    print(f"  Done: {updated:,} updated out of {scanned:,} scanned in {elapsed:.1f}s")
    return scanned, updated


def main():
    parser = argparse.ArgumentParser(description="Backfill destination from job_removals CSV dumps")
    parser.add_argument("--file", default=None, help="Process a single CSV file")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("Missing DATABASE_URL in environment")

    engine = create_engine(db_url, pool_pre_ping=True)

    if args.file:
        files = [args.file]
    else:
        files = sorted(glob.glob(os.path.join(REMOVALS_DIR, "*.csv")))

    if not files:
        print(f"No CSV files found in {REMOVALS_DIR}/")
        return

    print(f"Processing {len(files)} dump files")
    if args.dry_run:
        print("DRY RUN — no writes")

    total_scanned = 0
    total_updated = 0

    try:
        for csv_path in files:
            scanned, updated = process_file(engine, csv_path, args.dry_run)
            total_scanned += scanned
            total_updated += updated
    finally:
        engine.dispose()

    print(f"\nAll done. {total_updated:,} updated out of {total_scanned:,} total removals across {len(files)} files.")


if __name__ == "__main__":
    main()
