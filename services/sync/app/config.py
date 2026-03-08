from pydantic import Field

from common.settings import BaseServiceSettings


class SyncSettings(BaseServiceSettings):
    source_db_host: str = Field(alias="SOURCE_DB_HOST")
    source_db_port: int = Field(default=3306, alias="SOURCE_DB_PORT")
    source_db_user: str = Field(alias="SOURCE_DB_USER")
    source_db_password: str = Field(alias="SOURCE_DB_PASSWORD")
    source_db_driver: str = Field(default="mysql+pymysql", alias="SOURCE_DB_DRIVER")
    target_database_url: str = Field(alias="TARGET_DATABASE_URL")
    export_destination: str = Field(default="jobl.ai", alias="EXPORT_DESTINATION")
    sync_batch_size: int = Field(default=1000, alias="SYNC_BATCH_SIZE")
    sync_interval_seconds: int = Field(default=3600, alias="SYNC_INTERVAL_SECONDS")


settings = SyncSettings()
