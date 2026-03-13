from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_dir: str
    workers: int = 1
    num_beams: int = 4
    max_new_tokens: int = 32
    max_input_length: int = 128
    batch_size_limit: int = 64
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
