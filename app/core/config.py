from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Jobl API"
    app_version: str = "0.1.0"
    api_prefix: str = "/v1"
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/jobl"

    # Normalizer (seq2seq model — optional, falls back to rules-only if absent)
    model_dir: str | None = None
    num_beams: int = 4
    max_new_tokens: int = 32
    max_input_length: int = 128

    # Categorizer (LightGBM — optional, category field is null if absent)
    categorizer_model_path: str | None = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @field_validator("model_dir", "categorizer_model_path", mode="after")
    @classmethod
    def _resolve_path(cls, v: str | None) -> str | None:
        if v is None:
            return None
        p = Path(v)
        return str(p if p.is_absolute() else _PROJECT_ROOT / p)


settings = Settings()
