from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class NormalizeSettings(BaseSettings):
    log_level: str = "INFO"
    target_database_url: str = Field(alias="TARGET_DATABASE_URL")
    normalize_batch_size: int = Field(default=1000, alias="NORMALIZE_BATCH_SIZE")
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = NormalizeSettings()
