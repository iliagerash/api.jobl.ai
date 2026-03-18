"""
generate_training_data_labelled.py
───────────────────────────────────
Generates categorizer training data from the manually reviewed job_labelling
table, producing a CSV in the same format as generate_training_data.py.

Only rows where category_id != 26 (Other) are included.
Rows that have been manually reviewed (labelled_at IS NOT NULL) take
precedence — the script always includes all rows regardless, but you can
pass --reviewed-only to restrict to manually corrected rows only.

Usage:
    python scripts/generate_training_data_labelled.py --output data/
    python scripts/generate_training_data_labelled.py --output data/ --reviewed-only
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
    parser.add_argument(
        "--reviewed-only",
        action="store_true",
        default=False,
        help="Only include rows that were manually reviewed (labelled_at IS NOT NULL).",
    )
    args = parser.parse_args()

    reviewed_filter = "AND labelled_at IS NOT NULL" if args.reviewed_only else ""

    db = SessionLocal()
    try:
        rows = db.execute(
            text(f"""
                SELECT title, title_clean, original_category,
                       description_clean, category_id
                FROM job_labelling
                WHERE category_id != 26
                  {reviewed_filter}
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

    reviewed = sum(1 for r in rows if r[4])  # category_id always set, so just count all
    print(f"Wrote {len(rows)} rows to {out_path}")
    if args.reviewed_only:
        print("  (manually reviewed rows only)")


if __name__ == "__main__":
    main()
