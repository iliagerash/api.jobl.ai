"""
classify_resumes.py
────────────────────
Phase 0, step 4b: classify resumes by type (full_cv / cover_letter / minimal)
based on text length and structural heuristics. No ML needed.

Runs after lang_detect.py and before make_splits.py so the resume_type field
is available for stratified splitting and downstream training decisions:
  - Bi-encoder/reranker: use all types (model should handle both)
  - Extractor: weight/stratify by type so the teacher doesn't waste labeling
    budget on minimal-text resumes that produce empty extractions
  - Validation: ensure a representative mix of types

Usage:
    python -u ml/scripts/classify_resumes.py
    python -u ml/scripts/classify_resumes.py --input ml/data/interim/resumes.jsonl

Updates ml/data/interim/resumes.jsonl in place, adding a `resume_type` field.
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

HEARTBEAT_EVERY = 10_000

# Section headers that indicate a structured CV (case-insensitive).
# Covers EN, ES, PT, FR, DE — the top languages in the resume corpus.
_CV_SECTION_RE = re.compile(
    r"(?:^|\n)\s*(?:"
    # English
    r"(?:work|professional)\s+(?:experience|history|summary)"
    r"|education(?:\s+and\s+training)?"
    r"|skills?\s*(?:&|and)?\s*(?:qualifications?|competencies)?"
    r"|key\s+skills"
    r"|core\s+competencies"
    r"|certifications?"
    r"|qualifications?"
    r"|references?"
    r"|career\s+(?:objective|summary|profile)"
    r"|professional\s+(?:profile|summary)"
    r"|personal\s+(?:profile|details|information)"
    # Spanish
    r"|experiencia\s+(?:laboral|profesional)"
    r"|educaci[oó]n"
    r"|habilidades"
    r"|formaci[oó]n\s+acad[eé]mica"
    r"|perfil\s+profesional"
    r"|objetivo\s+(?:laboral|profesional)"
    # Portuguese
    r"|experi[eê]ncia\s+profissional"
    r"|educa[cç][aã]o"
    r"|habilidades\s+e\s+compet[eê]ncias"
    r"|objetivo\s+profissional"
    # French
    r"|exp[eé]rience\s+professionnelle"
    r"|formation"
    r"|comp[eé]tences"
    r"|profil\s+professionnel"
    # German
    r"|berufserfahrung"
    r"|ausbildung"
    r"|kenntnisse"
    r"|berufliches?\s+profil"
    r")\s*[:\-]?\s*(?:\n|$)",
    re.IGNORECASE,
)

# Bullet/list patterns that indicate structured content
_BULLET_RE = re.compile(r"(?:^|\n)\s*[•◦▪▸✓✔●○■□·\-–—]\s+\S", re.MULTILINE)


def classify_resume(title: str, description: str) -> str:
    """Classify a resume as full_cv, cover_letter, or minimal."""
    text = description or ""
    text_len = len(text.strip())

    if text_len < 100:
        return "minimal"

    section_matches = len(_CV_SECTION_RE.findall(text))
    bullet_matches = len(_BULLET_RE.findall(text))
    line_count = len([l for l in text.split("\n") if l.strip()])

    # Full CV: has section headers and/or substantial structured content
    if section_matches >= 2:
        return "full_cv"
    if section_matches >= 1 and (bullet_matches >= 3 or text_len > 1000):
        return "full_cv"
    if bullet_matches >= 5 and text_len > 500:
        return "full_cv"
    if text_len > 2000 and line_count > 15:
        return "full_cv"

    # Cover letter: short-to-medium text, reads as prose, no/few section headers
    if text_len < 800 and section_matches == 0:
        return "cover_letter"
    if text_len < 1500 and section_matches <= 1 and bullet_matches <= 2:
        return "cover_letter"

    # Default: if it's long-ish but doesn't have clear CV structure,
    # still likely a full CV (just poorly formatted)
    if text_len > 800:
        return "full_cv"

    return "cover_letter"


def iter_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _json_default(v):
    from datetime import date, datetime
    from decimal import Decimal
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    raise TypeError(type(v).__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify resumes by type (full_cv / cover_letter / minimal)")
    parser.add_argument("--input", default="ml/data/interim/resumes.jsonl")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found")
        sys.exit(1)

    tmp_path = args.input + ".tmp"
    total = 0
    type_counts: dict[str, int] = {}

    print(f"Classifying resumes in {args.input} ...")
    with open(tmp_path, "w", encoding="utf-8") as out:
        for record in iter_jsonl(args.input):
            resume_type = classify_resume(
                record.get("title", ""),
                record.get("description", ""),
            )
            record["resume_type"] = resume_type
            type_counts[resume_type] = type_counts.get(resume_type, 0) + 1

            out.write(json.dumps(record, default=_json_default, ensure_ascii=False))
            out.write("\n")

            total += 1
            if total % HEARTBEAT_EVERY == 0:
                print(f"  ... {total:,} resumes classified")

    os.replace(tmp_path, args.input)

    print(f"\nDone. {total:,} resumes classified -> {args.input}")
    print(f"  Distribution:")
    for rtype in ("full_cv", "cover_letter", "minimal"):
        count = type_counts.get(rtype, 0)
        pct = count / total if total else 0
        print(f"    {rtype:>14}: {count:>10,} ({pct:.1%})")


if __name__ == "__main__":
    main()
