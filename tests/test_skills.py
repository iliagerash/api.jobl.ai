"""Test skill tables migration and skill extractor service.

Run:
    # First run migration:
    alembic upgrade head

    # Then seed with test data and run test:
    python -u tests/test_skills.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def seed_test_data():
    """Insert a small set of test skills for validation."""
    from sqlalchemy import text
    from app.db.session import SessionLocal

    db = SessionLocal()
    try:
        # Clear existing test data
        db.execute(text("DELETE FROM skill_labels"))
        db.execute(text("DELETE FROM skills"))

        # Insert test skills
        test_skills = [
            ("test:python", "Python", "skill"),
            ("test:java", "Java", "skill"),
            ("test:aws", "Amazon Web Services", "skill"),
            ("test:docker", "Docker", "skill"),
            ("test:react", "React", "skill"),
            ("test:sql", "SQL", "skill"),
            ("test:leadership", "leadership", "competence"),
            ("test:nodejs", "Node.js", "skill"),
            ("test:spring-boot", "Spring Boot", "skill"),
        ]

        for uri, label, skill_type in test_skills:
            db.execute(
                text("INSERT INTO skills (uri, preferred_label_en, skill_type) VALUES (:uri, :label, :type)"),
                {"uri": uri, "label": label, "type": skill_type},
            )

        # Insert multilingual labels
        test_labels = [
            ("test:python", "en", "Python"),
            ("test:python", "es", "Python"),
            ("test:python", "de", "Python"),
            ("test:java", "en", "Java"),
            ("test:java", "es", "Java"),
            ("test:aws", "en", "AWS"),
            ("test:aws", "en", "Amazon Web Services"),
            ("test:docker", "en", "Docker"),
            ("test:docker", "es", "Docker"),
            ("test:react", "en", "React"),
            ("test:react", "es", "React"),
            ("test:sql", "en", "SQL"),
            ("test:leadership", "en", "leadership"),
            ("test:leadership", "es", "liderazgo"),
            ("test:leadership", "de", "Führung"),
            ("test:nodejs", "en", "Node.js"),
            ("test:spring-boot", "en", "Spring Boot"),
            ("test:spring-boot", "de", "Spring Boot"),
        ]

        for uri, lang, label in test_labels:
            db.execute(
                text("INSERT INTO skill_labels (skill_uri, language_code, label) VALUES (:uri, :lang, :label)"),
                {"uri": uri, "lang": lang, "label": label},
            )

        db.commit()
        print(f"  Seeded {len(test_skills)} skills, {len(test_labels)} labels")
    finally:
        db.close()


def main():
    from app.db.session import SessionLocal
    from app.services.skill_extractor import SkillExtractor

    # Seed test data
    print("Seeding test skill data ...")
    seed_test_data()

    # Load skill extractor
    print("Loading skill extractor ...")
    extractor = SkillExtractor(SessionLocal)
    print(f"  Ready: {extractor.is_ready()}")

    # Test 1: English job posting
    text_en = "Senior Software Engineer with 5+ years of Python and AWS experience. Must know Docker and React."
    skills_en = extractor.extract_skills(text_en, language="en")
    print(f"\n  EN: '{text_en[:60]}...'")
    print(f"  Skills: {skills_en}")
    assert "Python" in skills_en, "Should find Python"
    assert "Amazon Web Services" in skills_en, "Should find AWS → Amazon Web Services"
    assert "Docker" in skills_en, "Should find Docker"
    assert "React" in skills_en, "Should find React"

    # Test 2: Spanish job posting (tech skills are usually in English)
    text_es = "Desarrollador Full Stack con experiencia en React, Node.js y Docker. Se valora liderazgo."
    skills_es = extractor.extract_skills(text_es, language="es")
    print(f"\n  ES: '{text_es[:60]}...'")
    print(f"  Skills: {skills_es}")
    assert "React" in skills_es, "Should find React in Spanish text"
    assert "Docker" in skills_es, "Should find Docker in Spanish text"
    assert "leadership" in skills_es, "Should find liderazgo → leadership"

    # Test 3: German job posting
    text_de = "Java-Entwickler mit Spring Boot Kenntnissen. Führung von Teams."
    skills_de = extractor.extract_skills(text_de, language="de")
    print(f"\n  DE: '{text_de[:60]}...'")
    print(f"  Skills: {skills_de}")
    assert "Java" in skills_de, "Should find Java in German text"
    assert "Spring Boot" in skills_de, "Should find Spring Boot in German text"
    assert "leadership" in skills_de, "Should find Führung → leadership"

    # Test 4: No skills text
    text_empty = "We are a great company with a wonderful culture."
    skills_empty = extractor.extract_skills(text_empty, language="en")
    print(f"\n  Empty: '{text_empty}'")
    print(f"  Skills: {skills_empty}")
    assert len(skills_empty) == 0, "Should find no skills"

    print("\n  ALL TESTS PASSED")


if __name__ == "__main__":
    main()
