"""
extract_labelling_data.py
─────────────────────────
Populates the job_labelling table with a balanced sample of jobs across all
categories (1–25, skipping 26/Other), up to --limit rows per class.

Category assignment priority:
  1. category_map match (unless original_category is in _AMBIGUOUS_CATEGORIES)
  2. Keyword heuristics (same rules as generate_training_data.py)

Jobs already present in job_labelling are skipped (deduplication via UNIQUE
constraint on job_id).

Usage:
    python scripts/extract_labelling_data.py --limit 200 --countries=us,ca
"""

import argparse
import os
import sys
from datetime import timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sqlalchemy import text
from app.db.session import SessionLocal
from app.services.cleaner import clean_job_description
from app.api.v1.process import _extract_application_email

# Re-use shared constants from generate_training_data
from scripts.generate_training_data import _AMBIGUOUS_CATEGORIES, _RULES, _assign_category

_SKIP_CATEGORY = 26  # Other — not trained on explicitly


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract balanced labelling data into job_labelling table")
    parser.add_argument("--limit", type=int, default=200, help="Max rows per category class (default: 200)")
    parser.add_argument(
        "--countries",
        default=None,
        help="Comma-separated country codes to include (e.g. --countries=us,ca). Default: all.",
    )
    args = parser.parse_args()

    country_list = [c.strip().upper() for c in args.countries.split(",")] if args.countries else []

    country_filter = ""
    country_params: dict = {}
    if country_list:
        placeholders = ", ".join(f":country_{i}" for i in range(len(country_list)))
        country_filter = f"AND j.country_code IN ({placeholders})"
        country_params = {f"country_{i}": c for i, c in enumerate(country_list)}

    db = SessionLocal()
    try:
        # How many rows already exist per category
        existing = db.execute(
            text("SELECT category_id, COUNT(*) FROM job_labelling GROUP BY category_id")
        ).fetchall()
        existing_counts: dict[int, int] = {int(r[0]): int(r[1]) for r in existing}

        already_labelled = set(
            r[0] for r in db.execute(text("SELECT job_id FROM job_labelling")).fetchall()
        )

        total_inserted = 0

        for cat_id in range(1, 27):
            if cat_id == _SKIP_CATEGORY:
                continue

            have = existing_counts.get(cat_id, 0)
            need = args.limit - have
            if need <= 0:
                print(f"  [{cat_id:>2}] already at limit ({have}), skipping")
                continue

            BATCH = 2000
            inserted = 0
            # Reserve 20% of slots for heuristic rows so the training set always
            # includes authentic "no original_category" examples. Unused map slots
            # roll over to heuristics if category_map can't fill its quota.
            map_cap = int(need * 0.8)

            passes = [
                # (label, cap, query)
                ("category_map", map_cap, text(f"""
                    SELECT j.id, j.title, j.title_clean, j.description,
                           j.company_name, j.country_code, j.language_code,
                           j.category AS original_category,
                           cm.category_id AS mapped_category_id
                    FROM jobs j
                    JOIN category_map cm
                        ON LOWER(cm.original_category) = LOWER(j.category)
                    WHERE j.language_code IN ('en', 'fr')
                      AND j.title IS NOT NULL
                      AND cm.category_id = :cat_id
                      AND j.id NOT IN (SELECT job_id FROM job_labelling)
                      {country_filter}
                    ORDER BY RANDOM()
                    LIMIT :limit
                """)),
                ("heuristics", need, text(f"""
                    SELECT j.id, j.title, j.title_clean, j.description,
                           j.company_name, j.country_code, j.language_code,
                           j.category AS original_category,
                           cm.category_id AS mapped_category_id
                    FROM jobs j
                    LEFT JOIN category_map cm
                        ON LOWER(cm.original_category) = LOWER(j.category)
                    WHERE j.language_code IN ('en', 'fr')
                      AND j.title IS NOT NULL
                      AND j.id NOT IN (SELECT job_id FROM job_labelling)
                      {country_filter}
                    ORDER BY RANDOM()
                    LIMIT :limit
                """)),
            ]

            for pass_num, (label, cap, query) in enumerate(passes, 1):
                if inserted >= need:
                    break
                seen_ids: set[int] = set()
                iteration = 0

                while inserted < cap:
                    iteration += 1
                    print(f"  [{cat_id:>2}] pass {pass_num} ({label}), iter {iteration}: {inserted}/{need} inserted ...", flush=True)

                    rows = db.execute(query, {"limit": BATCH, "cat_id": cat_id, **country_params}).fetchall()
                    if not rows:
                        break

                    batch_inserted = 0
                    for (
                        job_id, title, title_clean, description,
                        company_name, country_code, language_code,
                        original_category, mapped_category_id,
                    ) in rows:
                        if inserted >= cap:
                            break
                        if job_id in already_labelled or job_id in seen_ids:
                            continue
                        seen_ids.add(job_id)

                        effective_title = title_clean or title or ""
                        lang = language_code or "en"

                        if mapped_category_id is not None and original_category not in _AMBIGUOUS_CATEGORIES:
                            assigned_cat = int(mapped_category_id)
                        elif mapped_category_id is not None and original_category in _AMBIGUOUS_CATEGORIES:
                            h = _assign_category(effective_title, "", lang)
                            assigned_cat = h if h != 26 else int(mapped_category_id)
                        else:
                            assigned_cat = _assign_category(effective_title, description or "", lang)

                        if assigned_cat != cat_id:
                            continue

                        clean_result = clean_job_description(description or "")
                        desc_clean = clean_result.html
                        email = _extract_application_email(description or "")
                        expiry = clean_result.expiry
                        expiry_date = expiry if expiry and expiry != "expired" else None

                        try:
                            db.execute(
                                text("""
                                    INSERT INTO job_labelling
                                        (job_id, title, title_clean, description, description_clean,
                                         company_name, country_code, language_code, original_category,
                                         email, expiry_date, category_id)
                                    VALUES
                                        (:job_id, :title, :title_clean, :description, :description_clean,
                                         :company_name, :country_code, :language_code, :original_category,
                                         :email, :expiry_date, :category_id)
                                    ON CONFLICT (job_id) DO NOTHING
                                """),
                                {
                                    "job_id": job_id,
                                    "title": effective_title,
                                    "title_clean": title_clean,
                                    "description": description,
                                    "description_clean": desc_clean,
                                    "company_name": company_name,
                                    "country_code": country_code,
                                    "language_code": language_code,
                                    "original_category": original_category,
                                    "email": email,
                                    "expiry_date": expiry_date,
                                    "category_id": cat_id,
                                },
                            )
                            db.commit()
                            already_labelled.add(job_id)
                            inserted += 1
                            batch_inserted += 1
                        except Exception as exc:
                            db.rollback()
                            print(f"    insert error job_id={job_id}: {exc}")

                    if batch_inserted == 0:
                        print(f"  [{cat_id:>2}] no new matches in pass {pass_num} ({label})", flush=True)
                        break

            total_inserted += inserted
            print(f"  [{cat_id:>2}] done: {have + inserted}/{args.limit}", flush=True)

        print(f"\nDone. Total inserted: {total_inserted}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
