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
    (1,  "Manufacturing & Industrial Production"),
    (2,  "Automotive"),
    (3,  "Food & Beverage Manufacturing"),
    (4,  "Information Technology"),
    (5,  "Telecommunications & Internet"),
    (6,  "Construction & Infrastructure"),
    (7,  "Professional Services"),
    (8,  "Human Resources"),
    (9,  "Transportation & Logistics"),
    (10, "Healthcare & Medical Services"),
    (11, "Aerospace & Defense"),
    (12, "Financial Services & Banking"),
    (13, "Real Estate & Architecture"),
    (14, "Marketing, Advertising & Media"),
    (15, "Hospitality & Restaurants"),
    (16, "Retail & Wholesale"),
    (17, "Education & Training"),
    (18, "Energy & Natural Resources"),
    (19, "Engineering Services"),
    (20, "Nonprofit & Government"),
    (21, "Arts, Entertainment & Recreation"),
    (22, "Legal Services"),
    (23, "Science & Research"),
    (24, "Customer Service & Support"),
    (25, "Security & Surveillance"),
    (26, "Other"),
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
