from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainingSettings(BaseSettings):
    log_level: str = "INFO"
    target_database_url: str = Field(alias="TARGET_DATABASE_URL")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = TrainingSettings()
