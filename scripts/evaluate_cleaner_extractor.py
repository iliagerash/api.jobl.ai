"""
evaluate_cleaner_extractor.py
──────────────────────────────
Re-runs clean_job_description() and email/expiry extraction for every row in
job_labelling where verified = true, then updates description_clean, email,
and expiry_date in the DB.

Workflow:
  1. Run this script after fixing extraction bugs.
  2. Open the labelling web page and review the updated results.
  3. For each job that looks correct, click "Verify".
  4. For jobs that still have issues, note the pattern and iterate.
  5. Once satisfied with a batch, click "Pending" to reset verified → false.

Usage:
    python scripts/evaluate_cleaner_extractor.py [--dry-run] [--category ID]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from bs4 import BeautifulSoup
from sqlalchemy import text

from app.db.session import SessionLocal
from app.services.cleaner import clean_job_description, extract_expiry_raw
from app.api.v1.process import (
    _EMAIL_RE,
    _EXCLUDE_KEYWORDS_RE,
    _EXCLUDE_LOCAL_PART_RE,
    _extract_application_email,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate cleaner/extractor for verified labelling rows"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print changes without writing to the DB",
    )
    parser.add_argument(
        "--category",
        type=int,
        metavar="ID",
        help="Limit to a specific category_id",
    )
    args = parser.parse_args()

    db = SessionLocal()
    try:
        category_filter = "AND category_id = :cat_id" if args.category else ""
        rows = db.execute(
            text(f"""
                SELECT id, description
                FROM job_labelling
                WHERE verified = true {category_filter}
                ORDER BY id
            """),
            {"cat_id": args.category} if args.category else {},
        ).fetchall()

        total = len(rows)
        cat_info = f" in category {args.category}" if args.category else ""
        print(f"Found {total} verified row(s){cat_info} to process.\n")
        if not total:
            return

        updated = 0
        errors = 0

        for row_id, description in rows:
            desc = description or ""

            try:
                clean_result = clean_job_description(desc)
                new_desc_clean = clean_result.html

                plain_text = BeautifulSoup(new_desc_clean, "lxml").get_text(separator=" ")
                # Mirror the endpoint: prefer hl-email spans from the raw description,
                # fall back to keyword-based text extraction on the cleaned plain text.
                new_email: str | None = None
                raw_soup = BeautifulSoup(desc, "lxml")
                for hl_tag in raw_soup.find_all("span", class_="hl-email"):
                    candidate = hl_tag.get_text(strip=True)
                    if not _EMAIL_RE.fullmatch(candidate):
                        continue
                    local_part = candidate.split("@")[0]
                    if _EXCLUDE_LOCAL_PART_RE.search(local_part):
                        continue
                    container_text = (hl_tag.parent or hl_tag).get_text()
                    if not _EXCLUDE_KEYWORDS_RE.search(container_text):
                        new_email = candidate
                        break
                if new_email is None:
                    new_email = _extract_application_email(plain_text)

                raw_expiry = extract_expiry_raw(desc)
                new_expiry = raw_expiry.isoformat() if raw_expiry else None
            except Exception as exc:
                print(f"[{row_id}] ERROR during extraction: {exc}")
                errors += 1
                continue

            print(f"[{row_id}] email={new_email!r}  expiry={new_expiry!r}")

            if not args.dry_run:
                try:
                    db.execute(
                        text("""
                            UPDATE job_labelling
                            SET description_clean = :desc_clean,
                                email             = :email,
                                expiry_date       = :expiry
                            WHERE id = :id
                        """),
                        {
                            "desc_clean": new_desc_clean,
                            "email": new_email,
                            "expiry": new_expiry,
                            "id": row_id,
                        },
                    )
                    db.commit()
                    updated += 1
                except Exception as exc:
                    db.rollback()
                    print(f"  → DB write error: {exc}")
                    errors += 1
            else:
                updated += 1  # count as "would update" in dry-run

        suffix = " (dry run — no changes written)" if args.dry_run else ""
        print(
            f"\nDone. Processed: {total} | "
            f"{'Would update' if args.dry_run else 'Updated'}: {updated} | "
            f"Errors: {errors}"
            + suffix
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()
