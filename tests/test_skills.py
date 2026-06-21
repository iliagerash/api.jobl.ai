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

    # Test 1: English job posting
    text_en = "Senior Software Engineer with 5+ years of Python and AWS experience. Must know Docker and React."
    skills_en = extractor.extract_skills(text_en, language="en")
    print(f"\n  EN: '{text_en[:60]}...'")
    print(f"  Skills found: {skills_en}")
    assert len(skills_en) > 0, "Should find at least one skill"

    # Test 2: Spanish job posting (tech skills are usually in English even in Spanish postings)
    text_es = "Desarrollador Full Stack con experiencia en React, Node.js y Docker."
    skills_es = extractor.extract_skills(text_es, language="es")
    print(f"\n  ES: '{text_es[:60]}...'")
    print(f"  Skills found: {skills_es}")
    assert len(skills_es) > 0, "Should find at least one skill in Spanish text"

    # Test 3: German job posting
    text_de = "Java-Entwickler mit Spring Boot Kenntnissen und Docker Erfahrung."
    skills_de = extractor.extract_skills(text_de, language="de")
    print(f"\n  DE: '{text_de[:60]}...'")
    print(f"  Skills found: {skills_de}")
    assert len(skills_de) > 0, "Should find at least one skill in German text"

    # Test 4: Minimal text
    text_empty = "We are a great company."
    skills_empty = extractor.extract_skills(text_empty, language="en")
    print(f"\n  Minimal: '{text_empty}'")
    print(f"  Skills found: {skills_empty}")

    print("\n  ALL TESTS PASSED")


if __name__ == "__main__":
    main()
