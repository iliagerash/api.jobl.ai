"""
seed_skills.py
────────────────
Load skill taxonomy into the skills + skill_labels tables.

Downloads the ESCO skills CSV from the EU portal, parses multilingual labels,
and bulk-inserts into Postgres.

Usage:
    python sql/seed_skills.py
    python sql/seed_skills.py --csv-path /path/to/skills_en.csv   # use local file
"""

import argparse
import csv
import io
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ESCO_SKILLS_URL = "https://ec.europa.eu/esco/api/resource/skill?language={lang}&offset={offset}&limit={limit}&full=true"

# Supported languages matching our corpus
LANGUAGES = ["en", "es", "de", "fr", "pt", "uk", "el", "nl", "it", "da", "ar"]


def download_esco_skills() -> list[dict]:
    """Download skills from ESCO API."""
    import httpx

    skills = {}
    print("Downloading skills from ESCO API ...")

    for lang in LANGUAGES:
        offset = 0
        limit = 100
        lang_count = 0
        while True:
            url = ESCO_SKILLS_URL.format(lang=lang, offset=offset, limit=limit)
            try:
                resp = httpx.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"  Error fetching {lang} offset={offset}: {e}")
                break

            items = data.get("_embedded", {}).get("results", [])
            if not items:
                break

            for item in items:
                uri = item.get("uri", "")
                title = item.get("title", "")
                if not uri or not title:
                    continue

                if uri not in skills:
                    skills[uri] = {
                        "uri": uri,
                        "preferred_label_en": "",
                        "skill_type": item.get("skillType", "skill"),
                        "labels": {},
                    }

                skills[uri]["labels"].setdefault(lang, set()).add(title)
                if lang == "en":
                    skills[uri]["preferred_label_en"] = title

                # Also add alternative labels
                for alt in item.get("alternativeLabel", {}).get(lang, []):
                    if alt:
                        skills[uri]["labels"].setdefault(lang, set()).add(alt)

                lang_count += 1

            offset += limit
            if offset >= data.get("total", 0):
                break

        print(f"  {lang}: {lang_count} labels")

    print(f"  Total skills: {len(skills)}")
    return list(skills.values())


def load_from_csv(csv_path: str) -> list[dict]:
    """Load skills from a local CSV file (ESCO download format)."""
    print(f"Loading skills from {csv_path} ...")
    skills = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uri = row.get("conceptUri", row.get("uri", ""))
            label = row.get("preferredLabel", row.get("label", ""))
            lang = row.get("language", "en")
            skill_type = row.get("skillType", "skill")

            if not uri or not label:
                continue

            if uri not in skills:
                skills[uri] = {
                    "uri": uri,
                    "preferred_label_en": "",
                    "skill_type": skill_type,
                    "labels": {},
                }

            skills[uri]["labels"].setdefault(lang, set()).add(label)
            if lang == "en" and not skills[uri]["preferred_label_en"]:
                skills[uri]["preferred_label_en"] = label

    print(f"  {len(skills)} skills loaded")
    return list(skills.values())


def seed_database(skills: list[dict]) -> None:
    """Insert skills into Postgres."""
    from sqlalchemy import text
    from app.db.session import SessionLocal

    db = SessionLocal()
    try:
        # Clear existing data
        db.execute(text("DELETE FROM skill_labels"))
        db.execute(text("DELETE FROM skills"))
        db.commit()

        # Insert skills
        skill_count = 0
        label_count = 0
        for skill in skills:
            if not skill.get("preferred_label_en"):
                # Use first available label as fallback
                for labels in skill["labels"].values():
                    if labels:
                        skill["preferred_label_en"] = next(iter(labels))
                        break

            db.execute(
                text("INSERT INTO skills (uri, preferred_label_en, skill_type) VALUES (:uri, :label, :type) ON CONFLICT DO NOTHING"),
                {"uri": skill["uri"], "label": skill["preferred_label_en"], "type": skill["skill_type"]},
            )
            skill_count += 1

            for lang, labels in skill.get("labels", {}).items():
                for label in labels:
                    db.execute(
                        text("INSERT INTO skill_labels (skill_uri, language_code, label) VALUES (:uri, :lang, :label) ON CONFLICT DO NOTHING"),
                        {"uri": skill["uri"], "lang": lang, "label": label},
                    )
                    label_count += 1

            if skill_count % 1000 == 0:
                db.commit()
                print(f"  ... {skill_count} skills, {label_count} labels inserted")

        db.commit()
        print(f"\nDone. {skill_count} skills, {label_count} labels inserted")
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="Seed skill taxonomy into Postgres")
    parser.add_argument("--csv-path", default=None, help="Local ESCO CSV file (skips API download)")
    args = parser.parse_args()

    if args.csv_path:
        skills = load_from_csv(args.csv_path)
    else:
        skills = download_esco_skills()

    if not skills:
        print("No skills to insert")
        return

    seed_database(skills)


if __name__ == "__main__":
    main()
