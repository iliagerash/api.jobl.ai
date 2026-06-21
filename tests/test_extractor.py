"""Quick test for the extractor service.

Run: python -u tests/test_extractor.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.environ.get("EXTRACTOR_MODEL_PATH", "models/extractor-gguf/model-Q4_K_M.gguf")


def main():
    from app.services.extractor import JobExtractor

    print(f"Loading extractor from {MODEL_PATH} ...")
    start = time.time()
    model = JobExtractor(MODEL_PATH)
    print(f"  Loaded in {time.time() - start:.1f}s, ready={model.is_ready()}")

    # Test extraction
    title = "Senior Software Engineer"
    description = """
    We are looking for a Senior Software Engineer with 5+ years of Python experience.
    Must have experience with AWS, Docker, and CI/CD pipelines.
    Salary: $120,000 - $160,000 per year. Remote work available.
    """

    print(f"  Extracting: '{title}' ...")
    start = time.time()
    result = model.extract(title, description, language="en", country="US")
    ms = (time.time() - start) * 1000
    print(f"  Extraction time: {ms:.0f}ms")

    if "_error" in result:
        print(f"  FAIL: {result}")
        return

    print(f"  normalized_title: {result.get('normalized_title')}")
    print(f"  seniority: {result.get('seniority')}")
    print(f"  occupation_category: {result.get('occupation_category')}")
    print(f"  employment_type: {result.get('employment_type')}")
    print(f"  work_mode: {result.get('work_mode')}")
    print(f"  salary_present: {result.get('salary_present')}")
    print(f"  salary_min: {result.get('salary_min')}")
    print(f"  salary_max: {result.get('salary_max')}")
    print(f"  skills: {result.get('skills', [])[:5]}")
    print(f"  experience_years_min: {result.get('experience_years_min')}")

    # Assertions
    assert model.is_ready(), "Model not ready"
    assert "_error" not in result, f"Extraction failed: {result}"
    assert result.get("normalized_title"), "Missing normalized_title"
    assert result.get("seniority") in ("intern", "junior", "mid", "senior", "lead", "executive"), f"Bad seniority: {result.get('seniority')}"
    assert result.get("salary_present") is True, "Should detect salary"
    assert isinstance(result.get("skills"), list), "Skills should be a list"
    assert len(result.get("skills", [])) > 0, "Should extract at least one skill"

    print("\n  ALL TESTS PASSED")


if __name__ == "__main__":
    main()
