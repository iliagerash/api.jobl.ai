"""
generate_training_data_labelled.py
───────────────────────────────────
Generates categorizer training data from the manually reviewed job_labelling
table, producing a CSV in the same format as generate_training_data.py.

Usage:
    python scripts/generate_training_data_labelled.py --output data/
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from bs4 import BeautifulSoup
from sqlalchemy import text
from app.db.session import SessionLocal


def _strip_html(html: str) -> str:
    if not html:
        return ""
    return BeautifulSoup(html, "lxml").get_text(separator=" ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training data from job_labelling table")
    parser.add_argument("--output", default="data/", help="Output directory (default: data/)")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        rows = db.execute(
            text("""
                SELECT title, title_clean, original_category,
                       description_clean, category_id
                FROM job_labelling
                ORDER BY category_id, id
            """)
        ).fetchall()
    finally:
        db.close()

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "categorizer_training.csv")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "original_category", "description_plaintext", "category_id"])
        for title, title_clean, original_category, description_clean, category_id in rows:
            effective_title = title_clean or title or ""
            desc_plain = _strip_html(description_clean or "")
            writer.writerow([effective_title, original_category or "", desc_plain[:1000], category_id])

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
