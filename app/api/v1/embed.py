import logging

from bs4 import BeautifulSoup
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from app.services.language import detect_language_code

logger = logging.getLogger("jobl.api.embed")

router = APIRouter()


class EmbedRequest(BaseModel):
    title: str = Field(min_length=1, max_length=512)
    description: str | None = None
    type: str = Field(default="job", pattern="^(job|resume)$")
    language: str | None = None
    country: str | None = None
    category_id: int | None = None

    model_config = ConfigDict(str_strip_whitespace=True)


class KeywordOut(BaseModel):
    canonical_id: int
    title: str
    distance: float


class EmbedResponse(BaseModel):
    embedding: list[float]
    language: str | None = None
    keyword: KeywordOut | None = None


@router.post("/embed", response_model=EmbedResponse)
def embed(body: EmbedRequest, request: Request) -> EmbedResponse:
    biencoder = getattr(request.app.state, "biencoder", None)
    if not biencoder or not biencoder.is_ready():
        raise HTTPException(status_code=503, detail="Bi-encoder model not loaded")

    lang = body.language
    if not lang:
        result = detect_language_code(
            title=body.title, description=body.description,
            country_code=body.country, source_db=None,
        )
        lang = result.language_code

    country = body.country or "XX"
    plain_text = BeautifulSoup(body.description, "lxml").get_text(separator=" ") if body.description else None
    text = f"{body.title} — {plain_text[:2000]}" if plain_text else body.title
    prefixed = f"[lang={lang or 'unknown'}][country={country}][type={body.type}] {text}"

    embedding = biencoder.encode(prefixed)

    keyword: KeywordOut | None = None
    if lang:
        try:
            from sqlalchemy import text as sa_text
            from app.db.session import SessionLocal
            db = SessionLocal()
            vec_str = "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"
            if body.category_id:
                row = db.execute(sa_text("""
                    SELECT k.canonical_id, k.title, CAST(:vec AS vector) <=> k.embedding AS distance
                    FROM keywords k
                    WHERE k.language_code = :lang
                      AND k.category_id = :cat_id
                      AND k.embedding IS NOT NULL
                    ORDER BY CAST(:vec AS vector) <=> k.embedding
                    LIMIT 1
                """), {"vec": vec_str, "lang": lang, "cat_id": body.category_id}).first()
            else:
                row = db.execute(sa_text("""
                    SELECT k.canonical_id, k.title, CAST(:vec AS vector) <=> k.embedding AS distance
                    FROM keywords k
                    WHERE k.language_code = :lang
                      AND k.embedding IS NOT NULL
                    ORDER BY CAST(:vec AS vector) <=> k.embedding
                    LIMIT 1
                """), {"vec": vec_str, "lang": lang}).first()
            if row and row.distance <= 0.5:
                keyword = KeywordOut(canonical_id=row.canonical_id, title=row.title, distance=round(float(row.distance), 4))
            db.close()
        except Exception:
            logger.exception("keyword matching failed")

    return EmbedResponse(embedding=embedding, language=lang, keyword=keyword)
