"""
assign_keywords.py
───────────────────
Assign jobs to their nearest keyword using vector similarity.
Matches by language_code and category_id, inserts into job_keywords table.

Usage:
    python -u scripts/assign_keywords.py
    python -u scripts/assign_keywords.py --batch-size 1000
    python -u scripts/assign_keywords.py --max-distance 0.5
    python -u scripts/assign_keywords.py --country AU
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

BATCH_SIZE = 1000


def main():
    parser = argparse.ArgumentParser(description="Assign jobs to nearest keywords")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-distance", type=float, default=0.5)
    parser.add_argument("--country", default=None, help="Filter by country code")
    parser.add_argument("--db-url", default=None)
    args = parser.parse_args()

    db_url = args.db_url or os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("No DATABASE_URL")

    engine = create_engine(db_url, pool_pre_ping=True)

    # Count jobs to process
    country_filter = "AND j.country_code = :country" if args.country else ""
    country_param = {"country": args.country} if args.country else {}

    with engine.connect() as conn:
        total = conn.execute(text(f"""
            SELECT COUNT(*)
            FROM jobs j
            WHERE j.embedding IS NOT NULL
              AND j.category_id IS NOT NULL
              AND j.language_code IN ('en', 'fr')
              AND j.expires_at > NOW()
              AND j.country_code IN ('AU', 'CA', 'GB', 'NZ', 'SG', 'US')
              AND NOT EXISTS (SELECT 1 FROM job_keywords jk WHERE jk.job_id = j.id)
              {country_filter}
        """), country_param).scalar()

    print(f"Jobs to assign: {total:,}")
    if total == 0:
        print("Nothing to do.")
        engine.dispose()
        return

    assigned = 0
    skipped = 0
    start = time.time()
    last_id = 0

    while True:
        with engine.begin() as conn:
            rows = conn.execute(text(f"""
                SELECT j.id, j.language_code, j.category_id
                FROM jobs j
                WHERE j.embedding IS NOT NULL
                  AND j.category_id IS NOT NULL
                  AND j.language_code IN ('en', 'fr')
                  AND j.expires_at > NOW()
                  AND j.country_code IN ('AU', 'CA', 'GB', 'NZ', 'SG', 'US')
                  AND j.id > :last_id
                  AND NOT EXISTS (SELECT 1 FROM job_keywords jk WHERE jk.job_id = j.id)
                  {country_filter}
                ORDER BY j.id
                LIMIT :batch_size
            """), {"last_id": last_id, "batch_size": args.batch_size, **country_param}).fetchall()

            if not rows:
                break

            for row in rows:
                result = conn.execute(text("""
                    SELECT k.id, j.embedding <=> k.embedding AS distance
                    FROM keywords k, jobs j
                    WHERE j.id = :job_id
                      AND k.language_code = :lang
                      AND k.category_id = :cat_id
                      AND k.embedding IS NOT NULL
                    ORDER BY j.embedding <=> k.embedding
                    LIMIT 1
                """), {"job_id": row.id, "lang": row.language_code, "cat_id": row.category_id}).first()

                if result and result.distance <= args.max_distance:
                    conn.execute(text("""
                        INSERT INTO job_keywords (job_id, keyword_id, distance)
                        VALUES (:job_id, :keyword_id, :distance)
                        ON CONFLICT (job_id, keyword_id) DO NOTHING
                    """), {"job_id": row.id, "keyword_id": result.id, "distance": float(result.distance)})
                    assigned += 1
                else:
                    skipped += 1

            last_id = rows[-1].id

        elapsed = time.time() - start
        rate = (assigned + skipped) / elapsed if elapsed > 0 else 0
        print(f"  processed={assigned + skipped:,}  assigned={assigned:,}  skipped={skipped:,}  last_id={last_id}  ({rate:.0f}/sec)", flush=True)

    elapsed = time.time() - start
    print(f"\nDone. {assigned:,} assigned, {skipped:,} skipped (distance > {args.max_distance}) in {elapsed:.1f}s")

    engine.dispose()


if __name__ == "__main__":
    main()
