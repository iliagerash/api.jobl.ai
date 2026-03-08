from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class NormalizeSettings(BaseSettings):
    log_level: str = "INFO"
    target_database_url: str = Field(alias="TARGET_DATABASE_URL")
    normalize_batch_size: int = Field(default=1000, alias="NORMALIZE_BATCH_SIZE")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-5-nano", alias="OPENAI_MODEL")
    openai_timeout_seconds: int = Field(default=60, alias="OPENAI_TIMEOUT_SECONDS")
    openai_max_retries: int = Field(default=3, alias="OPENAI_MAX_RETRIES")
    openai_batch_completion_window: str = Field(default="24h", alias="OPENAI_BATCH_COMPLETION_WINDOW")
    openai_batch_poll_seconds: int = Field(default=15, alias="OPENAI_BATCH_POLL_SECONDS")
    llm_prompt_version: str = Field(default="v1", alias="LLM_PROMPT_VERSION")
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = NormalizeSettings()
