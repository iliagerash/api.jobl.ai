from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from app.services.doc_classifier import classify_doc

router = APIRouter()


class ClassifyRequest(BaseModel):
    title: str = Field(min_length=1, max_length=512)
    description: str | None = None

    model_config = ConfigDict(str_strip_whitespace=True)


class ClassifyResponse(BaseModel):
    doc_type: str


@router.post("/classify", response_model=ClassifyResponse)
def classify(body: ClassifyRequest) -> ClassifyResponse:
    return ClassifyResponse(doc_type=classify_doc(title=body.title, description=body.description))
