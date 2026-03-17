"""
seed_category_map.py
────────────────────
Loads a CSV file into the category_map table. Each row maps an original
category string (as it appears in the source DB) to a local category_id.

CSV format (with header):
    original_category,category_id

Usage:
    python scripts/seed_category_map.py --input data/category_map.csv
    python scripts/seed_category_map.py --input data/category_map.csv --truncate
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sqlalchemy import text
from app.db.session import SessionLocal


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed category_map from CSV")
    parser.add_argument("--input", required=True, help="Path to CSV file (original_category,category_id)")
    parser.add_argument(
        "--truncate",
        action="store_true",
        default=False,
        help="Truncate category_map before inserting (default: upsert)",
    )
    args = parser.parse_args()

    rows: list[dict] = []
    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            original = row.get("original_category", "").strip()
            cat_id = row.get("category_id", "").strip()
            if not original or not cat_id:
                continue
            rows.append({"original_category": original, "category_id": int(cat_id)})

    if not rows:
        print("No rows to insert.")
        return

    db = SessionLocal()
    try:
        if args.truncate:
            db.execute(text("TRUNCATE TABLE category_map"))
            print("Truncated category_map.")

        db.execute(
            text(
                """
                INSERT INTO category_map (original_category, category_id)
                VALUES (:original_category, :category_id)
                ON CONFLICT (original_category)
                DO UPDATE SET category_id = EXCLUDED.category_id
                """
            ),
            rows,
        )
        db.commit()
        print(f"Upserted {len(rows)} rows into category_map.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
