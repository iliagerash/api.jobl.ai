import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("jobl.api.extract")

router = APIRouter()


class ExtractRequest(BaseModel):
    title: str = Field(min_length=1, max_length=512)
    description: str
    type: str = Field(default="job", pattern="^(job|resume)$")
    language: str | None = None
    country: str | None = None

    model_config = ConfigDict(str_strip_whitespace=True)


class ExtractResponse(BaseModel):
    extraction: dict


@router.post("/extract", response_model=ExtractResponse)
def extract(body: ExtractRequest, request: Request) -> ExtractResponse:
    extractor = getattr(request.app.state, "extractor", None)
    if not extractor or not extractor.is_ready():
        raise HTTPException(status_code=503, detail="Extractor model not loaded")

    result = extractor.extract(
        title=body.title,
        description=body.description,
        language=body.language,
        country=body.country,
        record_type=body.type,
    )

    if "_error" in result:
        raise HTTPException(status_code=422, detail=result["_error"])

    return ExtractResponse(extraction=result)
