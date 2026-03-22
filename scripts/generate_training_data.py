"""
generate_training_data.py
─────────────────────────
Generates categorizer training data from the manually reviewed job_labelling
table.

Usage:
    python scripts/generate_training_data.py --output data/

Outputs:
    data/categorizer_training.csv — title,original_category,description_plaintext,category_id
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from bs4 import BeautifulSoup
from sqlalchemy import text
from app.db.session import SessionLocal


CATEGORIES = [
    (1,  "Manufacturing & Engineering"),
    (2,  "Automotive"),
    (3,  "Food & Beverage Manufacturing"),
    (4,  "IT & Telecommunications"),
    (5,  "Construction & Infrastructure"),
    (6,  "Consulting & Advisory"),
    (7,  "Human Resources"),
    (8,  "Transportation & Logistics"),
    (9,  "Healthcare & Medical Services"),
    (10, "Aerospace & Defense"),
    (11, "Financial Services & Banking"),
    (12, "Real Estate & Architecture"),
    (13, "Marketing, Advertising & Media"),
    (14, "Hospitality & Restaurants"),
    (15, "Retail, Wholesale & Customer Service"),
    (16, "Education & Science"),
    (17, "Energy & Natural Resources"),
    (18, "Nonprofit & Government"),
    (19, "Arts, Entertainment & Recreation"),
    (20, "Legal Services"),
    (21, "Security & Surveillance"),
    (22, "Other"),
]


def _strip_html(html: str) -> str:
    if not html:
        return ""
    return BeautifulSoup(html, "lxml").get_text(separator=" ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate categorizer training data from job_labelling table")
    parser.add_argument("--output", default="data/", help="Output directory (default: data/)")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        rows = db.execute(
            text("""
                SELECT title, original_category,
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
        for title, original_category, description_clean, category_id in rows:
            effective_title = title or ""
            desc_plain = _strip_html(description_clean or "")
            writer.writerow([effective_title, original_category or "", desc_plain, category_id])

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
