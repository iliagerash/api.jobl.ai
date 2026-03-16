import logging
import re
from datetime import date

from bs4 import BeautifulSoup
from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from app.services.cleaner import clean_job_description
from app.services.language import detect_language_code

logger = logging.getLogger("jobl.api.process")

router = APIRouter()

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_EN_FR = {"en", "fr"}


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
        category = CategoryOut(id=None, title=body.original_category) if body.original_category else None
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

    # 4. Expiry date
    expiry_date: str | None = None
    if isinstance(clean_result.expiry, date):
        expiry_date = clean_result.expiry.isoformat()

    # 5. Extract email and mask in HTML
    plain_text = BeautifulSoup(clean_result.html, "lxml").get_text()
    email_match = _EMAIL_RE.search(plain_text)
    application_email: str | None = email_match.group(0) if email_match else None
    description_clean = clean_result.html
    if application_email:
        description_clean = description_clean.replace(application_email, "***email_hidden***")

    # 6. Categorize
    category: CategoryOut | None = None
    categorizer = getattr(request.app.state, "categorizer", None)
    if categorizer and categorizer.is_ready():
        try:
            desc_plain = BeautifulSoup(description_clean, "lxml").get_text()
            cat = categorizer.predict(title_normalized, body.original_category, desc_plain)
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
