from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SyncSettings(BaseSettings):
    log_level: str = "INFO"
    source_db_host: str = Field(alias="SOURCE_DB_HOST")
    source_db_port: int = Field(default=3306, alias="SOURCE_DB_PORT")
    source_db_user: str = Field(alias="SOURCE_DB_USER")
    source_db_password: str = Field(alias="SOURCE_DB_PASSWORD")
    source_db_driver: str = Field(default="mysql+pymysql", alias="SOURCE_DB_DRIVER")
    source_db_ssl_disabled: bool = Field(default=True, alias="SOURCE_DB_SSL_DISABLED")
    database_url: str = Field(alias="DATABASE_URL")
    export_destination: str = Field(default="jobl.ai", alias="EXPORT_DESTINATION")
    sync_batch_size: int = Field(default=1000, alias="SYNC_BATCH_SIZE")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = SyncSettings()
