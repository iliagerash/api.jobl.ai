"""
validate_extraction_prompt.py
──────────────────────────────
Phase 0, step 6: test the Component 3 extraction prompt against ~20 sample job
postings via OpenAI API.  Validates that the output is valid JSON with correct
enum values and all required fields present.

Once validated, the SYSTEM_PROMPT and EXTRACTION_SCHEMA constants in this file
become the locked prompt template that ml/scripts/teacher_extract.py will use
during Phase 1 (Day 1-2) on the A100 via vLLM.

Setup:
    pip install -e ".[ml]"                # adds openai
    export OPENAI_API_KEY="sk-..."        # or add to .env

Usage:
    python -u ml/scripts/validate_extraction_prompt.py
    python -u ml/scripts/validate_extraction_prompt.py --limit 30 --model gpt-4o-mini
    python -u ml/scripts/validate_extraction_prompt.py --input ml/data/splits/val_pool_jobs.jsonl
"""

import argparse
import html
import json
import os
import random
import re
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ── Extraction schema (Component 3) ─────────────────────────────────────────
# Uses the existing 26 categories from sql/seed_categories.sql.

CATEGORIES = [
    "Manufacturing & Engineering",
    "Automotive",
    "Food & Beverage Manufacturing",
    "IT & Telecommunications",
    "Construction & Infrastructure",
    "Consulting & Advisory",
    "Human Resources",
    "Transportation & Logistics",
    "Healthcare & Medical Services",
    "Aerospace & Defense",
    "Financial Services & Banking",
    "Real Estate & Architecture",
    "Marketing, Advertising & Media",
    "Hospitality & Restaurants",
    "Retail, Wholesale & Customer Service",
    "Education & Science",
    "Energy & Natural Resources",
    "Nonprofit & Government",
    "Arts, Entertainment & Recreation",
    "Legal Services",
    "Security & Surveillance",
    "Other",
]

VALID_ENUMS = {
    "seniority": {"intern", "junior", "mid", "senior", "lead", "executive"},
    "employment_type": {"full_time", "part_time", "contract", "freelance", "internship"},
    "contract_type": {"permanent", "temporary", "fixed_term", "freelance"},
    "work_mode": {"onsite", "hybrid", "remote"},
}

SYSTEM_PROMPT = """You extract structured fields from job postings for a multilingual jobs platform covering 39 countries.

Input: a raw job posting prefixed with locale tags [lang=XX][country=XX].

Output: a JSON object with these fields:

- normalized_title (string): English-normalized canonical job title, concise (1-5 words). Remove company names, salaries, locations, employment type qualifiers. Preserve legal markers like (m/w/d).
- original_title (string): the original title exactly as provided.
- seniority (enum): one of: intern, junior, mid, senior, lead, executive. Infer from title, description, and experience requirements. Default to "mid" if unclear.
- occupation_category (string): one of the following categories:
  {categories}
  Choose the single best match. Use "Other" only if no category fits.
- employment_type (enum): one of: full_time, part_time, contract, freelance, internship.
- contract_type (enum): one of: permanent, temporary, fixed_term, freelance.
- work_mode (enum): one of: onsite, hybrid, remote. Default to "onsite" if not stated.
- location_city (string or null): city name extracted from the posting.
- location_country (string): ISO 3166-1 alpha-2 country code.
- language (string): ISO 639-1 language code of the posting text.
- salary_present (boolean): true if any salary information is mentioned.
- salary_min (number or null): minimum salary if present, as a number.
- salary_max (number or null): maximum salary if present, as a number.
- salary_currency (string or null): ISO 4217 currency code if salary is present.
- skills (array of strings): extracted skill keywords, normalized to English. Include technical skills, tools, certifications, and soft skills explicitly mentioned.
- experience_years_min (number or null): minimum years of experience if stated.
- is_expired_signal (boolean): true if there are heuristic clues the posting is stale (past dates, "closed", "filled", etc.).
- is_duplicate_signal (boolean): always false for single-posting extraction.

Rules:
- Work from the provided text only. Do not hallucinate information not present.
- If a field cannot be determined, use null (for nullable fields) or the stated default.
- Skills should be normalized to English even if the posting is in another language.
- Return ONLY valid JSON, no markdown fencing, no explanation.
""".format(categories="\n  ".join(f'"{c}"' for c in CATEGORIES)).strip()

RESPONSE_JSON_SCHEMA = {
    "type": "json_schema",
    "name": "job_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "normalized_title": {"type": "string"},
            "original_title": {"type": "string"},
            "seniority": {"type": "string", "enum": list(VALID_ENUMS["seniority"])},
            "occupation_category": {"type": "string"},
            "employment_type": {"type": "string", "enum": list(VALID_ENUMS["employment_type"])},
            "contract_type": {"type": "string", "enum": list(VALID_ENUMS["contract_type"])},
            "work_mode": {"type": "string", "enum": list(VALID_ENUMS["work_mode"])},
            "location_city": {"type": ["string", "null"]},
            "location_country": {"type": "string"},
            "language": {"type": "string"},
            "salary_present": {"type": "boolean"},
            "salary_min": {"type": ["number", "null"]},
            "salary_max": {"type": ["number", "null"]},
            "salary_currency": {"type": ["string", "null"]},
            "skills": {"type": "array", "items": {"type": "string"}},
            "experience_years_min": {"type": ["number", "null"]},
            "is_expired_signal": {"type": "boolean"},
            "is_duplicate_signal": {"type": "boolean"},
        },
        "required": [
            "normalized_title", "original_title", "seniority", "occupation_category",
            "employment_type", "contract_type", "work_mode", "location_city",
            "location_country", "language", "salary_present", "salary_min",
            "salary_max", "salary_currency", "skills", "experience_years_min",
            "is_expired_signal", "is_duplicate_signal",
        ],
        "additionalProperties": False,
    },
}

REQUIRED_FIELDS = set(RESPONSE_JSON_SCHEMA["schema"]["required"])


# ── Helpers ──────────────────────────────────────────────────────────────────

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _clean_description(desc: str | None) -> str:
    text = html.unescape(desc or "")
    text = re.sub(r"<\s*script\b[^>]*>.*?</\s*script\s*>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<\s*style\b[^>]*>.*?</\s*style\s*>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = _HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:3000]


def _build_user_message(record: dict) -> str:
    lang = record.get("language_fasttext") or record.get("language_code") or "unknown"
    country = record.get("country_code") or "unknown"
    title = record.get("title") or ""
    desc = _clean_description(record.get("description"))
    return f"[lang={lang}][country={country}][type=job] {title} — {desc}"


def iter_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _validate_extraction(result: dict) -> list[str]:
    """Return a list of validation errors (empty = valid)."""
    errors = []

    # Check required fields
    missing = REQUIRED_FIELDS - set(result.keys())
    if missing:
        errors.append(f"missing fields: {missing}")

    # Check enums
    for field, valid_values in VALID_ENUMS.items():
        value = result.get(field)
        if value is not None and value not in valid_values:
            errors.append(f"{field}={value!r} not in {valid_values}")

    # Check occupation_category
    cat = result.get("occupation_category")
    if cat and cat not in CATEGORIES:
        errors.append(f"occupation_category={cat!r} not in valid categories")

    # Check types
    if "skills" in result and not isinstance(result["skills"], list):
        errors.append(f"skills should be array, got {type(result['skills']).__name__}")

    if "salary_present" in result and not isinstance(result["salary_present"], bool):
        errors.append(f"salary_present should be boolean, got {type(result['salary_present']).__name__}")

    # Check country code format
    cc = result.get("location_country")
    if cc and (len(cc) != 2 or not cc.isalpha()):
        errors.append(f"location_country={cc!r} not a valid ISO 3166-1 alpha-2 code")

    return errors


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the Component 3 extraction prompt against sample jobs via OpenAI")
    parser.add_argument("--input", default="ml/data/interim/jobs.jsonl", help="JSONL file to sample from")
    parser.add_argument("--limit", type=int, default=20, help="Number of samples to test (default: 20)")
    parser.add_argument("--model", default="gpt-5-nano", help="OpenAI model to use (default: gpt-5-nano)")
    parser.add_argument("--api-key", default=None, help="OpenAI API key (default: OPENAI_API_KEY env var)")
    parser.add_argument("--debug", action="store_true", help="Print full prompts and responses")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY env var or pass --api-key")
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai")
        sys.exit(1)

    client = OpenAI(api_key=api_key, timeout=60)

    # Reservoir sampling — picks limit random records without loading the full file
    print(f"Sampling {args.limit} records from {args.input} ...")
    samples: list[dict] = []
    for i, record in enumerate(iter_jsonl(args.input)):
        if i < args.limit:
            samples.append(record)
        else:
            j = random.randint(0, i)
            if j < args.limit:
                samples[j] = record
    print(f"Testing {len(samples)} samples with model={args.model}")

    total = 0
    passed = 0
    failed = 0
    all_errors: list[tuple[int, str, list[str]]] = []

    for i, record in enumerate(samples, 1):
        record_id = record.get("id", "?")
        title = record.get("title", "")[:80]
        user_msg = _build_user_message(record)

        print(f"\n[{i}/{len(samples)}] id={record_id} title={title!r}")

        if args.debug:
            print(f"  USER: {user_msg[:200]}...")

        try:
            resp = client.responses.create(
                model=args.model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                text={"format": {"type": "json_object"}},
            )
            content = resp.output_text or ""

            if args.debug:
                print(f"  RESPONSE: {content[:500]}")

            result = json.loads(content)
            errors = _validate_extraction(result)

            if errors:
                failed += 1
                all_errors.append((record_id, title, errors))
                print(f"  FAIL: {errors}")
            else:
                passed += 1
                cat = result.get("occupation_category", "?")
                seniority = result.get("seniority", "?")
                skills = result.get("skills", [])[:5]
                norm_title = result.get("normalized_title", "?")
                print(f"  OK: {norm_title} | {seniority} | {cat} | skills={skills}")

        except json.JSONDecodeError as e:
            failed += 1
            all_errors.append((record_id, title, [f"invalid JSON: {e}"]))
            print(f"  FAIL: invalid JSON response")
        except Exception as e:
            failed += 1
            all_errors.append((record_id, title, [str(e)]))
            print(f"  FAIL: {e}")

        total += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed, {failed}/{total} failed")

    if all_errors:
        print(f"\nFailures:")
        for record_id, title, errors in all_errors:
            print(f"  id={record_id} title={title!r}")
            for err in errors:
                print(f"    - {err}")

    if failed == 0:
        print(f"\nGATE PASSED: all {total} samples produced valid extractions")
        print("Prompt template is locked — ready for Phase 1 teacher labeling.")
    else:
        print(f"\nGATE FAILED: {failed}/{total} samples had errors — review and fix the prompt before proceeding.")


if __name__ == "__main__":
    main()
