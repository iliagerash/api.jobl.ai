"""
sync_predictions_from_scraper.py
─────────────────────────────────
Sync prediction columns (salary, profitability, keyword) from scraper MySQL
DBs into jobl Postgres jobs table. Updates matching jobl rows where any
prediction column is NULL.

Reads the same .env vars as jobl-sync (SOURCE_DB_*, DATABASE_URL).

Usage:
    python -u scripts/sync_predictions_from_scraper.py
    python -u scripts/sync_predictions_from_scraper.py --country=sg,gb
    python -u scripts/sync_predictions_from_scraper.py --batch-size=1000
    python -u scripts/sync_predictions_from_scraper.py --dry-run
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

load_dotenv()

COUNTRIES = {"AU", "CA", "GB", "NZ", "SG", "ZA", "US"}


def env(name: str, default: str | None = None) -> str:
    val = os.environ.get(name, default)
    if val is None:
        raise SystemExit(f"Missing env var: {name}")
    return val


def create_source_engine(db_name: str):
    ssl_disabled = env("SOURCE_DB_SSL_DISABLED", "true").lower() in ("true", "1")
    connect_args: dict[str, object] = {}
    if ssl_disabled:
        connect_args["ssl_disabled"] = True
    return create_engine(
        URL.create(
            drivername=env("SOURCE_DB_DRIVER", "mysql+pymysql"),
            username=env("SOURCE_DB_USER"),
            password=env("SOURCE_DB_PASSWORD"),
            host=env("SOURCE_DB_HOST"),
            port=int(env("SOURCE_DB_PORT", "3306")),
            database=db_name,
        ),
        pool_pre_ping=True,
        connect_args=connect_args,
    )


def load_source_dbs(target_engine, countries: set[str]) -> dict[str, str]:
    with target_engine.connect() as conn:
        rows = conn.execute(
            text("SELECT db_name, country_code FROM source_countries ORDER BY db_name")
        ).mappings()
        return {
            row["db_name"]: row["country_code"]
            for row in rows
            if (row.get("country_code") or "").strip().upper() in countries
        }


def fetch_source_batch(source_engine, from_id: int, batch_size: int) -> list[dict]:
    query = text("""
        SELECT id, predicted_salary_min, predicted_salary_max, predicted_salary_period,
               predicted_revenue, predicted_rps, priority_tier,
               keyword_id, keyword_title, keyword_distance
        FROM job
        WHERE id > :from_id
          AND (keyword_title IS NOT NULL OR predicted_salary_min IS NOT NULL OR predicted_revenue IS NOT NULL)
        ORDER BY id
        LIMIT :batch_size
    """)
    with source_engine.connect() as conn:
        rows = conn.execute(query, {"from_id": from_id, "batch_size": batch_size}).mappings()
        return [dict(r) for r in rows]


def update_jobs(target_engine, db_name: str, payload: list[dict]) -> int:
    query = text("""
        UPDATE jobs
        SET predicted_salary_min = COALESCE(:predicted_salary_min, predicted_salary_min),
            predicted_salary_max = COALESCE(:predicted_salary_max, predicted_salary_max),
            predicted_salary_period = COALESCE(:predicted_salary_period, predicted_salary_period),
            predicted_revenue = COALESCE(:predicted_revenue, predicted_revenue),
            predicted_rps = COALESCE(:predicted_rps, predicted_rps),
            priority_tier = COALESCE(:priority_tier, priority_tier),
            keyword_id = COALESCE(:keyword_id, keyword_id),
            keyword_title = COALESCE(:keyword_title, keyword_title),
            keyword_distance = COALESCE(:keyword_distance, keyword_distance)
        WHERE source_db = :source_db
          AND source_job_id = :source_job_id
          AND (predicted_salary_min IS NULL OR predicted_revenue IS NULL OR keyword_title IS NULL)
    """)
    params = [
        {
            "source_db": db_name,
            "source_job_id": row["id"],
            "predicted_salary_min": float(row["predicted_salary_min"]) if row.get("predicted_salary_min") is not None else None,
            "predicted_salary_max": float(row["predicted_salary_max"]) if row.get("predicted_salary_max") is not None else None,
            "predicted_salary_period": row.get("predicted_salary_period"),
            "predicted_revenue": float(row["predicted_revenue"]) if row.get("predicted_revenue") is not None else None,
            "predicted_rps": float(row["predicted_rps"]) if row.get("predicted_rps") is not None else None,
            "priority_tier": row.get("priority_tier"),
            "keyword_id": int(row["keyword_id"]) if row.get("keyword_id") else None,
            "keyword_title": row.get("keyword_title"),
            "keyword_distance": float(row["keyword_distance"]) if row.get("keyword_distance") is not None else None,
        }
        for row in payload
    ]
    with target_engine.begin() as conn:
        result = conn.execute(query, params)
        return result.rowcount


def backfill_db(target_engine, db_name: str, batch_size: int, dry_run: bool) -> int:
    source_engine = create_source_engine(db_name)
    updated = 0
    scanned = 0
    cursor = 0
    start = time.time()

    try:
        while True:
            rows = fetch_source_batch(source_engine, cursor, batch_size)
            if not rows:
                break

            if not dry_run:
                matched = update_jobs(target_engine, db_name, rows)
                updated += matched
            else:
                updated += len(rows)

            scanned += len(rows)
            cursor = rows[-1]["id"]
            elapsed = time.time() - start
            rate = scanned / elapsed if elapsed > 0 else 0
            print(f"  db={db_name}  scanned={scanned}  updated={updated}  last_source_id={cursor}  ({rate:.0f} rows/sec)")
    finally:
        source_engine.dispose()

    return updated


def main():
    parser = argparse.ArgumentParser(description="Sync predictions from scraper MySQL into jobl Postgres")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--country", default=None, help="Comma-separated country codes (default: all)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    countries = {c.strip().upper() for c in args.country.split(",")} if args.country else COUNTRIES
    target_engine = create_engine(env("DATABASE_URL"), pool_pre_ping=True)

    try:
        source_dbs = load_source_dbs(target_engine, countries)
        if not source_dbs:
            print(f"No source databases found for countries: {sorted(countries)}")
            return

        print(f"Syncing predictions from {len(source_dbs)} source DBs: {sorted(source_dbs.keys())}")
        if args.dry_run:
            print("DRY RUN — no writes")

        total = 0
        for db_name in sorted(source_dbs.keys()):
            print(f"\n--- {db_name} ({source_dbs[db_name]}) ---")
            updated = backfill_db(target_engine, db_name, args.batch_size, args.dry_run)
            total += updated
            print(f"  {db_name} done: {updated} rows updated")

        print(f"\nAll done. {total} total rows updated.")
    finally:
        target_engine.dispose()


if __name__ == "__main__":
    main()
