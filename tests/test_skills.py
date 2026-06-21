"""Test skill extractor service against seeded skill taxonomy.

Run:
    # First run migration and seed:
    alembic upgrade head
    python sql/seed_skills.py

    # Then test:
    python -u tests/test_skills.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from sqlalchemy import text as sql_text
    from app.db.session import SessionLocal
    from app.services.skill_extractor import SkillExtractor

    # Check that skills are seeded
    db = SessionLocal()
    count = db.execute(sql_text("SELECT COUNT(*) FROM skills")).scalar()
    db.close()
    if count == 0:
        print("ERROR: skills table is empty. Run seed first:")
        print("  python sql/seed_skills.py")
        return
    print(f"Skills in database: {count}")

    # Load skill extractor
    print("Loading skill extractor ...")
    extractor = SkillExtractor(SessionLocal)
    print(f"  Ready: {extractor.is_ready()}")

    # Test 1: English job posting (use skills known to be in ESCO)
    text_en = "Senior Software Engineer with Python programming experience. Must have project management and data analysis skills. Good communication required."
    skills_en = extractor.extract_skills(text_en, language="en")
    print(f"\n  EN: '{text_en[:60]}...'")
    print(f"  Skills found: {skills_en}")
    assert len(skills_en) > 0, "Should find at least one skill"

    # Test 2: Spanish job posting
    text_es = "Desarrollador con experiencia en gestión de proyectos y análisis de datos. Se requiere comunicación efectiva."
    skills_es = extractor.extract_skills(text_es, language="es")
    print(f"\n  ES: '{text_es[:60]}...'")
    print(f"  Skills found: {skills_es}")
    assert len(skills_es) > 0, "Should find at least one skill in Spanish text"

    # Test 3: German job posting
    text_de = "Projektmanagement und Datenanalyse Kenntnisse erforderlich. Kommunikation und Teamarbeit sind wichtig."
    skills_de = extractor.extract_skills(text_de, language="de")
    print(f"\n  DE: '{text_de[:60]}...'")
    print(f"  Skills found: {skills_de}")
    assert len(skills_de) > 0, "Should find at least one skill in German text"

    # Test 4: Minimal text (should find few or no skills)
    text_empty = "We are a great company."
    skills_empty = extractor.extract_skills(text_empty, language="en")
    print(f"\n  Minimal: '{text_empty}'")
    print(f"  Skills found: {skills_empty}")

    print("\n  ALL TESTS PASSED")


if __name__ == "__main__":
    main()
