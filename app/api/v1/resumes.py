from datetime import datetime
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.auth import require_resumes_ingest_token
from app.db.session import get_db
from app.models.resume import Resume
from app.services.language import detect_language_code

router = APIRouter()


class ResumeIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_website: str = Field(..., max_length=128)
    external_id: int
    title: str = Field(..., max_length=255)
    description: str | None = None
    city_title: str | None = Field(default=None, max_length=255)
    region_title: str | None = Field(default=None, max_length=255)
    country_code: str | None = Field(default=None, min_length=2, max_length=2)
    salary: Decimal | None = None
    salary_period: str | None = Field(default=None, max_length=20)
    salary_currency: str | None = Field(default=None, min_length=3, max_length=3)
    contract: str | None = Field(default=None, max_length=24)
    published_at: datetime | None = None
    is_remote: bool = False


class ResumeIngestResponse(BaseModel):
    id: int
    language_code: str | None


@router.post(
    "/resumes",
    response_model=ResumeIngestResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_resumes_ingest_token)],
)
def ingest_resume(body: ResumeIngestRequest, db: Session = Depends(get_db)) -> ResumeIngestResponse:
    lang = detect_language_code(
        title=body.title,
        description=body.description,
        country_code=body.country_code,
        source_db=body.source_website,
    ).language_code

    row = Resume(
        source_website=body.source_website.strip(),
        external_id=body.external_id,
        title=body.title,
        description=body.description,
        city_title=body.city_title,
        region_title=body.region_title,
        country_code=body.country_code.upper() if body.country_code else None,
        salary=body.salary,
        salary_period=body.salary_period,
        salary_currency=body.salary_currency.upper() if body.salary_currency else None,
        contract=body.contract,
        published_at=body.published_at,
        is_remote=body.is_remote,
        language_code=lang,
    )

    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Resume with this source_website and external_id already exists",
        ) from None

    db.refresh(row)
    return ResumeIngestResponse(id=row.id, language_code=row.language_code)
