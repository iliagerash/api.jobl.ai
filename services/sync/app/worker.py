import logging
from dataclasses import dataclass

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

logger = logging.getLogger("jobl.sync")


@dataclass
class SyncResult:
    fetched: int
    marked_exported: int
    upserted: int


class SyncWorker:
    def __init__(
        self,
        source_db_driver: str,
        source_db_host: str,
        source_db_port: int,
        source_db_user: str,
        source_db_password: str,
        target_database_url: str,
        export_destination: str,
    ) -> None:
        self.source_db_driver = source_db_driver
        self.source_db_host = source_db_host
        self.source_db_port = source_db_port
        self.source_db_user = source_db_user
        self.source_db_password = source_db_password
        self.target_database_url = target_database_url
        self.export_destination = export_destination

    def run_once(self, batch_size: int) -> SyncResult:
        logger.info(
            "sync iteration started batch_size=%s destination=%s",
            batch_size,
            self.export_destination,
        )
        logger.info(
            "source_db_host=%s source_db_port=%s source_db_user=%s",
            self.source_db_host,
            self.source_db_port,
            self.source_db_user,
        )
        logger.info("target_db=%s", self._redact_url(self.target_database_url))
        source_countries = self._load_source_countries()
        logger.info("source countries loaded count=%s", len(source_countries))
        if source_countries:
            logger.info("source dbs=%s", ", ".join(c["db_name"] for c in source_countries))

        # TODO: implement real pipeline:
        # 1) for each source country db, fetch non-exported jobs from source DB
        # 2) upsert into AI Postgres
        # 3) mark exported in source `export` table for destination='jobl.ai'
        return SyncResult(fetched=0, marked_exported=0, upserted=0)

    def _load_source_countries(self) -> list[dict[str, object | None]]:
        engine = create_engine(self.target_database_url, pool_pre_ping=True)
        try:
            with engine.connect() as conn:
                rows = conn.execute(
                    text(
                        """
                        SELECT db_name, country_code, config
                        FROM source_countries
                        ORDER BY db_name
                        """
                    )
                ).mappings()
                return [
                    {
                        "db_name": row["db_name"],
                        "country_code": row["country_code"],
                        "config": row["config"],
                    }
                    for row in rows
                ]
        finally:
            engine.dispose()

    @staticmethod
    def country_code_in_job_enabled(config: object | None) -> bool:
        if not isinstance(config, dict):
            return False
        value = config.get("country_code_in_job", 0)
        return value in (1, True, "1", "true", "True")

    def source_db_url_for(self, db_name: str) -> str:
        url = URL.create(
            drivername=self.source_db_driver,
            username=self.source_db_user,
            password=self.source_db_password,
            host=self.source_db_host,
            port=self.source_db_port,
            database=db_name,
        )
        return str(url)

    @staticmethod
    def _redact_url(url: str) -> str:
        if "@" not in url:
            return url
        left, right = url.split("@", 1)
        if "://" not in left:
            return f"***@{right}"
        scheme, _ = left.split("://", 1)
        return f"{scheme}://***@{right}"
