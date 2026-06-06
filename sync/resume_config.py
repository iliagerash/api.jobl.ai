from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ResumeXmlSettings(BaseSettings):
    log_level: str = "INFO"
    database_url: str = Field(alias="DATABASE_URL")
    resume_xml_feeds_dir: str = Field(alias="RESUME_XML_FEEDS_DIR")
    resume_xml_lookback_hours: int = Field(default=24, alias="RESUME_XML_LOOKBACK_HOURS")
    resume_xml_batch_size: int = Field(default=500, alias="RESUME_XML_BATCH_SIZE")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @field_validator("resume_xml_feeds_dir", mode="after")
    @classmethod
    def _resolve_feeds_dir(cls, value: str) -> str:
        path = Path(value)
        return str(path if path.is_absolute() else _PROJECT_ROOT / path)


settings = ResumeXmlSettings()
