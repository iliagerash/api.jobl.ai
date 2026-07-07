"""
doc_classifier.py
─────────────────
Classifies a text document (title + description) into one of four types:

  resume        — structured CV with clear sections / work history
  cover_letter  — short prose letter accompanying an application
  stub          — too thin to be useful (< 100 chars of description)
  other         — doesn't look like a job-seeker document at all

Called by POST /v1/embed when type=resume to populate doc_type in the
response so callers can persist it on the resumes row.

The heuristic mirrors ml/scripts/classify_resumes.py but maps to the
API enum values instead of the offline ml labels (full_cv → resume,
minimal → stub).  No model loading — pure regex + length rules.
"""

import re
from typing import Literal

DocType = Literal["resume", "cover_letter", "stub", "other"]

# Structural section headers indicating a real CV.
# EN / ES / PT / FR / DE — top languages in the resume corpus.
_CV_SECTION_RE = re.compile(
    r"(?:^|\n)\s*(?:"
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
    r"|experiencia\s+(?:laboral|profesional)"
    r"|educaci[oó]n"
    r"|habilidades"
    r"|formaci[oó]n\s+acad[eé]mica"
    r"|perfil\s+profesional"
    r"|objetivo\s+(?:laboral|profesional)"
    r"|experi[eê]ncia\s+profissional"
    r"|educa[cç][aã]o"
    r"|habilidades\s+e\s+compet[eê]ncias"
    r"|objetivo\s+profissional"
    r"|exp[eé]rience\s+professionnelle"
    r"|formation"
    r"|comp[eé]tences"
    r"|profil\s+professionnel"
    r"|berufserfahrung"
    r"|ausbildung"
    r"|kenntnisse"
    r"|berufliches?\s+profil"
    r")\s*[:\-]?\s*(?:\n|$)",
    re.IGNORECASE,
)

# Bullet / list patterns indicating structured content.
_BULLET_RE = re.compile(r"(?:^|\n)\s*[•◦▪▸✓✔●○■□·\-–—]\s+\S", re.MULTILINE)

# Patterns that suggest a job posting rather than a candidate document.
_JOB_POSTING_RE = re.compile(
    r"(?:we(?:'re|\s+are)\s+(?:looking|hiring|seeking)"
    r"|(?:join|grow)\s+(?:our|the)\s+team"
    r"|(?:apply\s+now|send\s+(?:your\s+)?(?:cv|resume))"
    r"|(?:job\s+description|position\s+overview|role\s+summary)"
    r"|(?:about\s+the\s+(?:role|position|job)))",
    re.IGNORECASE,
)


def classify_doc(title: str, description: str | None) -> DocType:
    """Return the document type for a resume-feed entry."""
    text = (description or "").strip()
    text_len = len(text)

    if text_len < 100:
        return "stub"

    # If it looks like a job posting rather than a candidate document → other
    if _JOB_POSTING_RE.search(text[:500]):
        return "other"

    section_matches = len(_CV_SECTION_RE.findall(text))
    bullet_matches = len(_BULLET_RE.findall(text))
    line_count = len([ln for ln in text.split("\n") if ln.strip()])

    # Structured CV: two or more recognised section headers, or one header with
    # substantial bullet content, or large structured text block.
    if section_matches >= 2:
        return "resume"
    if section_matches >= 1 and (bullet_matches >= 3 or text_len > 1000):
        return "resume"
    if bullet_matches >= 5 and text_len > 500:
        return "resume"
    if text_len > 2000 and line_count > 15:
        return "resume"

    # Cover letter: prose-style, no/few section headers.
    if text_len < 800 and section_matches == 0:
        return "cover_letter"
    if text_len < 1500 and section_matches <= 1 and bullet_matches <= 2:
        return "cover_letter"

    # Longer unstructured text — still likely a resume, just poorly formatted.
    return "resume"
