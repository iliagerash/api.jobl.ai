import logging
import re

from bs4 import BeautifulSoup
from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from app.services.cleaner import clean_job_description, extract_expiry_raw
from app.services.language import detect_language_code

logger = logging.getLogger("jobl.api.process")

router = APIRouter()

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_EN_FR = {"en", "fr"}

# Keywords that suggest an email is for submitting an application
_APPLY_KEYWORDS_RE = re.compile(
    r"\b("
    r"apply|applying|application|applicants?"
    r"|submit|submission"
    r"|send.{0,30}(resume|cv|application|candidature)"
    r"|email.{0,30}(resume|cv|application|candidature)"
    r"|resume|curriculum vitae|\bcv\b|cover.?letter"
    r"|contact|staffing|recruit\w*|inquir\w*|emailing"
    r"|postuler|appliquer|candidature|soumettre|envoyer.{0,30}(cv|candidature|courriel)"
    r"|faire.{0,10}demande"
    r")\b",
    re.IGNORECASE,
)
_EXCLUDE_KEYWORDS_RE = re.compile(
    r"\b("
    r"accommodations?|accessibilit\w*"
    r"|reasonable.{0,20}accommodations?"
    r"|disability|disabilities|handicap"
    r"|mesures?.{0,20}d.adaptation|adaptation"
    r"|accessibilité|personnes?.{0,20}handicapées?"
    r"|aboriginal|torres\s+strait|indigenous|first\s+nations|koori\w*"
    r"|fraud|scam|legitimacy|phishing|spoofing|impersonat"
    r")\b",
    re.IGNORECASE,
)
_EXCLUDE_LOCAL_PART_RE = re.compile(
    r"accommodat|accessibl|disability|disabilities|noreply|no.reply|donotreply|do.not.reply|support|helpdesk|help.desk|fraud|scam",
    re.IGNORECASE,
)
_CONTEXT_WINDOW = 350  # characters around the email to search for keywords

# Keywords that may appear in the local part of an application email address
_LOCAL_PART_RE = re.compile(
    r"apply|application|careers?|recruit\w*|recrut\w*|hiring|jobs?|emploi|candidat\w*|rh|hr|human.?resources?|people.?culture|contact",
    re.IGNORECASE,
)


def _extract_application_email(text: str) -> str | None:
    """Return the first email that looks like an application address.

    Two strategies (both require absence of accommodation/disability keywords):
    1. Email appears near apply/submit/cv keywords in a 300-char context window.
    2. Email local part (left of @) itself contains apply/careers/hr/recruitment
       keywords — e.g. careers@company.com, hr@acme.org, recruitment@firm.com.
    """
    for m in _EMAIL_RE.finditer(text):
        local_part = m.group(0).split("@")[0]
        if _EXCLUDE_LOCAL_PART_RE.search(local_part):
            continue
        # Strong local-part signal (careers@, hr@, recruiter@, etc.): return immediately,
        # but only if accommodation/disability keywords aren't within 200 chars — which
        # would indicate the email itself is an accommodation contact, not an apply address
        # (e.g. "please contact myworkdayrecruitment@gflenv.com for accommodation").
        if _LOCAL_PART_RE.search(local_part):
            narrow_start = max(0, m.start() - 200)
            narrow_end = min(len(text), m.end() + 200)
            if not _EXCLUDE_KEYWORDS_RE.search(text[narrow_start:narrow_end]):
                return m.group(0)
        start = max(0, m.start() - _CONTEXT_WINDOW)
        end = min(len(text), m.end() + _CONTEXT_WINDOW)
        context = text[start:end]
        if _EXCLUDE_KEYWORDS_RE.search(context):
            continue
        if _APPLY_KEYWORDS_RE.search(context):
            return m.group(0)
    return None


class ProcessRequest(BaseModel):
    title: str = Field(min_length=1, max_length=512)
    description: str
    original_category: str | None = None

    model_config = ConfigDict(str_strip_whitespace=True)


class CategoryOut(BaseModel):
    id: int | None
    title: str


class ProcessResponse(BaseModel):
    title_normalized: str
    description_clean: str
    application_email: str | None
    expiry_date: str | None
    category: CategoryOut | None


@router.post("/process", response_model=ProcessResponse)
def process(body: ProcessRequest, request: Request) -> ProcessResponse:
    # 1. Detect language from title + description text
    lang_result = detect_language_code(
        title=body.title,
        description=body.description,
        country_code=None,
        source_db=None,
    )
    lang = lang_result.language_code

    # 2. Clean description (always, all languages)
    clean_result = clean_job_description(body.description)

    if lang not in _EN_FR:
        # Non-EN/FR: description cleanup only, pass original_category through unchanged
        if body.original_category:
            category = CategoryOut(id=None, title=body.original_category)
        else:
            category = CategoryOut(id=26, title="Other")
        return ProcessResponse(
            title_normalized=body.title,
            description_clean=clean_result.html,
            application_email=None,
            expiry_date=None,
            category=category,
        )

    # 3. Normalize title (EN/FR)
    normalizer = getattr(request.app.state, "normalizer", None)
    if normalizer and normalizer.is_ready():
        title_normalized = normalizer.normalize(body.title, lang)
        if not title_normalized.strip():
            from app.services.normalizer import pre_strip
            title_normalized = pre_strip(body.title) or body.title.strip()
    else:
        from app.services.normalizer import _normalize_rules_only
        title_normalized = _normalize_rules_only(body.title)

    # 4. Expiry date — use raw extraction so past dates are still returned
    raw_expiry = extract_expiry_raw(body.description)
    expiry_date: str | None = raw_expiry.isoformat() if raw_expiry else None

    # 5. Extract application email and mask it in HTML
    plain_text = BeautifulSoup(clean_result.html, "lxml").get_text(separator=" ")
    application_email: str | None = _extract_application_email(plain_text)
    description_clean = clean_result.html
    if application_email:
        description_clean = description_clean.replace(application_email, "***email_hidden***")

    # 6. Categorize
    category: CategoryOut | None = None
    categorizer = getattr(request.app.state, "categorizer", None)
    if categorizer and categorizer.is_ready():
        try:
            desc_plain = BeautifulSoup(description_clean, "lxml").get_text()
            cat = categorizer.predict(title_normalized, desc_plain)
            category = CategoryOut(id=cat["id"], title=cat["title"])
        except Exception:
            logger.exception("categorizer.predict failed")

    return ProcessResponse(
        title_normalized=title_normalized,
        description_clean=description_clean,
        application_email=application_email,
        expiry_date=expiry_date,
        category=category,
    )
