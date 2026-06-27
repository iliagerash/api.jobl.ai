"""
backfill_embeddings_from_scraper.py
────────────────────────────────────
One-time script: scans scraper MySQL for jobs that have embedding/skills,
then updates the matching jobl Postgres rows where those columns are NULL.

Reads the same .env vars as jobl-sync (SOURCE_DB_*, DATABASE_URL).

Usage:
    python -u scripts/backfill_embeddings_from_scraper.py
    python -u scripts/backfill_embeddings_from_scraper.py --country=us,ca
    python -u scripts/backfill_embeddings_from_scraper.py --batch-size=1000
    python -u scripts/backfill_embeddings_from_scraper.py --dry-run
"""

import argparse
import json
import os
import time

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

load_dotenv()

COUNTRIES = {"AU", "CA", "NZ", "SG", "ZA", "UK", "US"}


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
        SELECT id, embedding, skills
        FROM job
        WHERE id > :from_id
          AND (embedding IS NOT NULL OR skills IS NOT NULL)
        ORDER BY id
        LIMIT :batch_size
    """)
    with source_engine.connect() as conn:
        rows = conn.execute(query, {"from_id": from_id, "batch_size": batch_size}).mappings()
        return [dict(r) for r in rows]


def update_jobs(target_engine, db_name: str, payload: list[dict]) -> int:
    query = text("""
        UPDATE jobs
        SET embedding = COALESCE(CAST(:embedding AS vector), embedding),
            skills = COALESCE(:skills, skills)
        WHERE source_db = :source_db
          AND source_job_id = :source_job_id
          AND (embedding IS NULL OR skills IS NULL)
    """)
    params = [{"source_db": db_name, **row} for row in payload]
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

            payload = []
            for row in rows:
                embedding_val = None
                skills_val = None

                if row.get("embedding"):
                    try:
                        parsed = json.loads(row["embedding"])
                        if isinstance(parsed, list) and len(parsed) == 1024:
                            embedding_val = str(parsed)
                    except (json.JSONDecodeError, TypeError):
                        pass

                if row.get("skills"):
                    try:
                        parsed = json.loads(row["skills"])
                        if isinstance(parsed, list):
                            skills_val = json.dumps(parsed)
                    except (json.JSONDecodeError, TypeError):
                        pass

                if embedding_val or skills_val:
                    payload.append({
                        "source_job_id": row["id"],
                        "embedding": embedding_val,
                        "skills": skills_val,
                    })

            if payload and not dry_run:
                matched = update_jobs(target_engine, db_name, payload)
                updated += matched
            elif payload:
                updated += len(payload)

            scanned += len(rows)
            cursor = rows[-1]["id"]
            elapsed = time.time() - start
            rate = scanned / elapsed if elapsed > 0 else 0
            print(f"  db={db_name}  scanned={scanned}  updated={updated}  last_source_id={cursor}  ({rate:.0f} rows/sec)")
    finally:
        source_engine.dispose()

    return updated


def main():
    parser = argparse.ArgumentParser(description="Backfill embedding/skills from scraper MySQL into jobl Postgres")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--country", default=None, help="Comma-separated country codes (default: au,ca,nz,sg,za,uk,us)")
    parser.add_argument("--dry-run", action="store_true", help="Log what would be updated without writing")
    args = parser.parse_args()

    countries = {c.strip().upper() for c in args.country.split(",")} if args.country else COUNTRIES
    target_engine = create_engine(env("DATABASE_URL"), pool_pre_ping=True)

    try:
        source_dbs = load_source_dbs(target_engine, countries)
        if not source_dbs:
            print(f"No source databases found for countries: {sorted(countries)}")
            return

        print(f"Backfilling from {len(source_dbs)} source DBs: {sorted(source_dbs.keys())}")
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
