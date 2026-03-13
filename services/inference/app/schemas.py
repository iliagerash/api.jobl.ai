from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config import settings


class NormalizeRequest(BaseModel):
    title_raw: str = Field(min_length=1, max_length=512)
    language_code: str | None = Field(default=None, min_length=2, max_length=16)

    model_config = ConfigDict(str_strip_whitespace=True)


class NormalizeBatchRequest(BaseModel):
    items: list[NormalizeRequest] = Field(min_length=1)

    model_config = ConfigDict(str_strip_whitespace=True)

    @model_validator(mode="after")
    def validate_batch_size(self) -> "NormalizeBatchRequest":
        if len(self.items) > settings.batch_size_limit:
            raise ValueError(f"items size exceeds batch_size_limit={settings.batch_size_limit}")
        return self


class NormalizeResponse(BaseModel):
    title_normalized: str

    model_config = ConfigDict(str_strip_whitespace=True)


class NormalizeBatchResponse(BaseModel):
    results: list[NormalizeResponse]

    model_config = ConfigDict(str_strip_whitespace=True)
