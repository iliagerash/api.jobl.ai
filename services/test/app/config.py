from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TestSettings(BaseSettings):
    log_level: str = "INFO"
    target_database_url: str = Field(alias="TARGET_DATABASE_URL")
    inference_api_base_url: str = Field(default="http://localhost:8000", alias="INFERENCE_API_BASE_URL")
    inference_timeout_seconds: int = Field(default=15, alias="INFERENCE_TIMEOUT_SECONDS")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = TestSettings()
