"""
seed_skills.py
────────────────
Load skill taxonomy into the skills + skill_labels tables from ESCO CSV files.

Download CSVs manually from https://esco.ec.europa.eu/en/use-esco/download
Select: Version=v1.2.1, Content=Skills, Format=CSV
Download one file per language needed.

Expected CSV format (ESCO v1.2):
  conceptUri, skillType, preferredLabel, altLabels, description, ...

Usage:
    # Single language:
    python sql/seed_skills.py --csv data/esco/skills_en.csv

    # Multiple languages:
    python sql/seed_skills.py --csv data/esco/skills_en.csv data/esco/skills_es.csv data/esco/skills_de.csv

    # All CSVs in a directory:
    python sql/seed_skills.py --dir data/esco/
"""

import argparse
import csv
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_language_from_filename(path: str) -> str | None:
    """Extract language code from filename like 'skills_en.csv'."""
    basename = os.path.basename(path)
    match = re.search(r"skills_(\w{2})\.csv", basename)
    return match.group(1) if match else None


def extract_uuid(concept_uri: str) -> str:
    """Extract UUID from ESCO conceptUri like 'http://data.europa.eu/esco/skill/UUID'."""
    return concept_uri.rsplit("/", 1)[-1]


def load_csv(path: str, language: str | None = None) -> tuple[dict, dict]:
    """Parse an ESCO skills CSV. Returns (skills_dict, labels_dict)."""
    lang = language or parse_language_from_filename(path) or "en"
    skills = {}
    labels = {}

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_uri = row.get("conceptUri", "").strip()
            if not raw_uri:
                continue
            uri = extract_uuid(raw_uri)
            if not uri:
                continue

            preferred = row.get("preferredLabel", "").strip()
            skill_type = row.get("skillType", "skill/competence").strip()
            alt_labels_raw = row.get("altLabels", "")

            if uri not in skills:
                skills[uri] = {
                    "uri": uri,
                    "preferred_label_en": preferred if lang == "en" else "",
                    "skill_type": skill_type,
                }
            elif lang == "en" and preferred:
                skills[uri]["preferred_label_en"] = preferred

            # Add preferred label (skip overly long labels — they cause false matches)
            if preferred and len(preferred) <= 60:
                key = (uri, lang, preferred)
                labels[key] = True

            # Add alternative labels (newline-separated in ESCO CSV)
            if alt_labels_raw:
                for alt in alt_labels_raw.split("\n"):
                    alt = alt.strip()
                    if alt and len(alt) <= 60:
                        key = (uri, lang, alt)
                        labels[key] = True

    return skills, labels


def main():
    parser = argparse.ArgumentParser(description="Seed skill taxonomy from ESCO CSV files")
    parser.add_argument("--csv", nargs="*", help="One or more ESCO CSV files (e.g. skills_en.csv skills_es.csv)")
    parser.add_argument("--dir", default=None, help="Directory containing skills_*.csv files")
    parser.add_argument("--clear", action="store_true", default=True, help="Clear existing data before seeding (default: true)")
    args = parser.parse_args()

    # Collect CSV files
    csv_files = []
    if args.dir:
        csv_files = sorted(glob.glob(os.path.join(args.dir, "skills_*.csv")))
    if args.csv:
        csv_files.extend(args.csv)

    if not csv_files:
        print("ERROR: no CSV files specified.")
        print("Download from: https://esco.ec.europa.eu/en/use-esco/download")
        print("Usage: python sql/seed_skills.py --csv data/esco/skills_en.csv")
        return

    # Parse all CSVs
    all_skills = {}
    all_labels = {}
    for path in csv_files:
        lang = parse_language_from_filename(path)
        print(f"Parsing {path} (lang={lang}) ...")
        skills, labels = load_csv(path, lang)
        all_skills.update(skills)
        all_labels.update(labels)
        print(f"  {len(skills)} skills, {len(labels)} labels")

    # Ensure all skills have an English preferred label
    # Priority: English label > first available label
    for uri, skill in all_skills.items():
        if not skill["preferred_label_en"]:
            # Try English labels first
            for key in all_labels:
                if key[0] == uri and key[1] == "en":
                    skill["preferred_label_en"] = key[2]
                    break
            # Fallback to any label
            if not skill["preferred_label_en"]:
                for key in all_labels:
                    if key[0] == uri:
                        skill["preferred_label_en"] = key[2]
                        break

    print(f"\nTotal: {len(all_skills)} unique skills, {len(all_labels)} labels across {len(csv_files)} files")

    # Insert into database
    from sqlalchemy import text
    from app.db.session import SessionLocal

    db = SessionLocal()
    try:
        if args.clear:
            print("Clearing existing skill data ...")
            db.execute(text("DELETE FROM skill_labels"))
            db.execute(text("DELETE FROM skills"))
            db.commit()

        print("Inserting skills ...")
        count = 0
        for skill in all_skills.values():
            db.execute(
                text("INSERT INTO skills (uri, preferred_label_en, skill_type) VALUES (:uri, :label, :type) ON CONFLICT DO NOTHING"),
                {"uri": skill["uri"], "label": skill["preferred_label_en"], "type": skill["skill_type"]},
            )
            count += 1
            if count % 2000 == 0:
                db.commit()
                print(f"  ... {count} skills")
        db.commit()

        print("Inserting labels ...")
        label_count = 0
        for (uri, lang, label) in all_labels:
            db.execute(
                text("INSERT INTO skill_labels (skill_uri, language_code, label) VALUES (:uri, :lang, :label) ON CONFLICT DO NOTHING"),
                {"uri": uri, "lang": lang, "label": label},
            )
            label_count += 1
            if label_count % 5000 == 0:
                db.commit()
                print(f"  ... {label_count} labels")
        db.commit()

        print(f"\nDone. {count} skills, {label_count} labels inserted")
    finally:
        db.close()


if __name__ == "__main__":
    main()
