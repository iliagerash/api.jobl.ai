import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("jobl.api.embed")

router = APIRouter()


class EmbedRequest(BaseModel):
    text: str = Field(min_length=1)
    type: str = Field(default="job", pattern="^(job|resume)$")
    language: str | None = None
    country: str | None = None

    model_config = ConfigDict(str_strip_whitespace=True)


class EmbedResponse(BaseModel):
    embedding: list[float]


@router.post("/embed", response_model=EmbedResponse)
def embed(body: EmbedRequest, request: Request) -> EmbedResponse:
    biencoder = getattr(request.app.state, "biencoder", None)
    if not biencoder or not biencoder.is_ready():
        raise HTTPException(status_code=503, detail="Bi-encoder model not loaded")

    lang = body.language or "unknown"
    country = body.country or "XX"
    prefixed = f"[lang={lang}][country={country}][type={body.type}] {body.text}"

    embedding = biencoder.encode(prefixed)
    return EmbedResponse(embedding=embedding)
