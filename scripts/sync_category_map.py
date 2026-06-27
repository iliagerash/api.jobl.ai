"""
sync_category_map.py
─────────────────────
Sync category_map tables from scraper MySQL DBs into jobl Postgres.
Pulls all (original_category, category_id) pairs from each scraper DB
and upserts into the jobl category_map table.

Reads the same .env vars as jobl-sync (SOURCE_DB_*, DATABASE_URL).

Usage:
    python -u scripts/sync_category_map.py
    python -u scripts/sync_category_map.py --country=sg,uk
    python -u scripts/sync_category_map.py --dry-run
"""

import argparse
import os

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


def fetch_source_mappings(source_engine) -> list[dict]:
    query = text("""
        SELECT original_category, category_id
        FROM category_map
        WHERE category_id > 0
    """)
    with source_engine.connect() as conn:
        rows = conn.execute(query).mappings()
        return [dict(r) for r in rows]


def upsert_mappings(target_engine, mappings: list[dict]) -> int:
    query = text("""
        INSERT INTO category_map (original_category, category_id)
        VALUES (:original_category, :category_id)
        ON CONFLICT (original_category) DO UPDATE SET category_id = EXCLUDED.category_id
    """)
    with target_engine.begin() as conn:
        conn.execute(query, mappings)
    return len(mappings)


def main():
    parser = argparse.ArgumentParser(description="Sync category_map from scraper MySQL DBs into jobl Postgres")
    parser.add_argument("--country", default=None, help="Comma-separated country codes (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Log what would be synced without writing")
    args = parser.parse_args()

    countries = {c.strip().upper() for c in args.country.split(",")} if args.country else COUNTRIES
    target_engine = create_engine(env("DATABASE_URL"), pool_pre_ping=True)

    try:
        source_dbs = load_source_dbs(target_engine, countries)
        if not source_dbs:
            print(f"No source databases found for countries: {sorted(countries)}")
            return

        print(f"Syncing category_map from {len(source_dbs)} source DBs: {sorted(source_dbs.keys())}")
        if args.dry_run:
            print("DRY RUN — no writes")

        total = 0
        for db_name in sorted(source_dbs.keys()):
            country = source_dbs[db_name]
            source_engine = create_source_engine(db_name)
            try:
                mappings = fetch_source_mappings(source_engine)
                print(f"  {db_name} ({country}): {len(mappings)} mappings")

                if mappings and not args.dry_run:
                    upserted = upsert_mappings(target_engine, mappings)
                    total += upserted
                elif mappings:
                    total += len(mappings)
            finally:
                source_engine.dispose()

        print(f"\nDone. {total} total mappings upserted.")
    finally:
        target_engine.dispose()


if __name__ == "__main__":
    main()
