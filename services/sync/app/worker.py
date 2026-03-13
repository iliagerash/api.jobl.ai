import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from app.language import detect_language_code

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
        source_db_ssl_disabled: bool,
        target_database_url: str,
        export_destination: str,
    ) -> None:
        self.source_db_driver = source_db_driver
        self.source_db_host = source_db_host
        self.source_db_port = source_db_port
        self.source_db_user = source_db_user
        self.source_db_password = source_db_password
        self.source_db_ssl_disabled = source_db_ssl_disabled
        self.target_database_url = target_database_url
        self.export_destination = export_destination

    def run_once(self, batch_size: int, only_dbs: set[str] | None = None) -> SyncResult:
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
        if only_dbs:
            source_countries = [c for c in source_countries if str(c["db_name"]) in only_dbs]
        logger.info("source countries loaded count=%s", len(source_countries))
        if source_countries:
            logger.info("source dbs=%s", ", ".join(c["db_name"] for c in source_countries))

        total_fetched = 0
        total_upserted = 0
        total_marked_exported = 0

        target_engine = create_engine(self.target_database_url, pool_pre_ping=True)
        try:
            self._ensure_sync_state_table(target_engine)
            for country in source_countries:
                db_name = str(country["db_name"])
                default_country_code = self._normalize_country_code(country.get("country_code"))
                default_currency = self._normalize_currency(country.get("currency"))
                config = country.get("config")
                last_job_id = self._get_last_job_id(target_engine=target_engine, source_db=db_name)

                source_engine = self._create_source_engine(db_name)
                try:
                    require_country_code_from_city = self.country_code_in_city_enabled(config)
                    has_city_country_code_column = self._city_has_country_code_column(source_engine, db_name)
                    use_country_code_from_city = require_country_code_from_city and has_city_country_code_column
                    if require_country_code_from_city and not has_city_country_code_column:
                        logger.warning(
                            "country_code_in_city enabled but `city.country_code` column not found db=%s; skipping country",
                            db_name,
                        )
                        continue

                    prefer_currency_from_job = self.currency_in_job_enabled(config)
                    has_job_salary_currency_column = self._job_has_salary_currency_column(source_engine, db_name)
                    use_currency_from_job = prefer_currency_from_job and has_job_salary_currency_column
                    if prefer_currency_from_job and not has_job_salary_currency_column:
                        logger.warning(
                            "currency_in_job enabled but `job.salary_currency` column not found db=%s; using source_countries.currency",
                            db_name,
                        )

                    region_city_column = self.region_in_city_column(config)
                    use_region_from_city = False
                    if region_city_column:
                        has_city_region_column = self._column_exists(source_engine, db_name, "city", region_city_column)
                        use_region_from_city = has_city_region_column
                        if not has_city_region_column:
                            logger.warning(
                                "region_in_city column not found db=%s column=%s; falling back to region table",
                                db_name,
                                region_city_column,
                            )
                            region_city_column = None
                    if not region_city_column:
                        region_city_column = "region"
                        use_region_from_city = self._column_exists(source_engine, db_name, "city", region_city_column)

                    if config and isinstance(config, dict) and "region_in_city" in config and not use_region_from_city:
                        logger.warning(
                            "region_in_city configured but city column unavailable db=%s; falling back to region table",
                            db_name,
                        )
                    has_region_table = self._table_exists(source_engine, db_name, "region")
                    has_city_region_id = self._column_exists(source_engine, db_name, "city", "region_id")
                    use_region_join = (not use_region_from_city) and has_region_table and has_city_region_id

                    if not use_region_from_city and not use_region_join:
                        logger.warning(
                            "no region source available db=%s (no city.region and no region table join path)",
                            db_name,
                        )

                    country_fetched = 0
                    country_upserted = 0
                    country_marked = 0
                    country_skipped = 0
                    batch_no = 0

                    while True:
                        raw_rows = self._fetch_source_rows(
                            source_engine=source_engine,
                            batch_size=batch_size,
                            use_country_code_from_city=use_country_code_from_city,
                            use_currency_from_job=use_currency_from_job,
                            use_region_from_city=use_region_from_city,
                            region_city_column=region_city_column,
                            use_region_join=use_region_join,
                            last_job_id=last_job_id,
                        )
                        if not raw_rows:
                            break

                        process_rows, skipped_count = self._filter_rows_for_country_code_requirement(
                            raw_rows=raw_rows,
                            require_country_code_from_city=require_country_code_from_city,
                        )

                        jobs_payload = self._build_jobs_payload(
                            raw_rows=process_rows,
                            source_db=db_name,
                            default_country_code=default_country_code,
                            default_currency=default_currency,
                            use_country_code_from_city=use_country_code_from_city,
                            use_currency_from_job=use_currency_from_job,
                        )

                        self._upsert_jobs(target_engine=target_engine, payload=jobs_payload)
                        self._mark_exported(source_engine=source_engine, raw_rows=process_rows)
                        last_job_id = max(int(row["id"]) for row in raw_rows)
                        self._set_last_job_id(
                            target_engine=target_engine,
                            source_db=db_name,
                            last_job_id=last_job_id,
                        )

                        batch_no += 1
                        fetched_count = len(raw_rows)
                        processed_count = len(process_rows)
                        country_fetched += fetched_count
                        country_upserted += processed_count
                        country_marked += processed_count
                        country_skipped += skipped_count
                        logger.info(
                            "batch progress db=%s batch=%s fetched=%s processed=%s skipped=%s totals(fetched=%s, upserted=%s, marked=%s, skipped=%s)",
                            db_name,
                            batch_no,
                            fetched_count,
                            processed_count,
                            skipped_count,
                            country_fetched,
                            country_upserted,
                            country_marked,
                            country_skipped,
                        )

                    logger.info(
                        "country sync completed db=%s fetched=%s upserted=%s marked_exported=%s skipped=%s",
                        db_name,
                        country_fetched,
                        country_upserted,
                        country_marked,
                        country_skipped,
                    )
                    total_fetched += country_fetched
                    total_upserted += country_upserted
                    total_marked_exported += country_marked
                except Exception:
                    logger.exception("country sync failed db=%s", db_name)
                finally:
                    source_engine.dispose()
        finally:
            target_engine.dispose()

        return SyncResult(
            fetched=total_fetched,
            marked_exported=total_marked_exported,
            upserted=total_upserted,
        )

    def _load_source_countries(self) -> list[dict[str, object | None]]:
        engine = create_engine(self.target_database_url, pool_pre_ping=True)
        try:
            with engine.connect() as conn:
                rows = conn.execute(
                    text(
                        """
                        SELECT db_name, country_code, currency, config
                        FROM source_countries
                        ORDER BY db_name
                        """
                    )
                ).mappings()
                return [
                    {
                        "db_name": row["db_name"],
                        "country_code": row["country_code"],
                        "currency": row["currency"],
                        "config": row["config"],
                    }
                    for row in rows
                ]
        finally:
            engine.dispose()

    def _ensure_sync_state_table(self, target_engine) -> None:
        query = text(
            """
            CREATE TABLE IF NOT EXISTS sync_state (
                source_db VARCHAR(128) NOT NULL,
                destination VARCHAR(100) NOT NULL,
                last_job_id BIGINT NOT NULL DEFAULT 0,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (source_db, destination)
            )
            """
        )
        with target_engine.begin() as conn:
            conn.execute(query)

    def _get_last_job_id(self, target_engine, source_db: str) -> int:
        query = text(
            """
            SELECT last_job_id
            FROM sync_state
            WHERE source_db = :source_db
              AND destination = :destination
            """
        )
        with target_engine.connect() as conn:
            row = conn.execute(
                query,
                {"source_db": source_db, "destination": self.export_destination},
            ).first()
            if not row:
                return 0
            return int(row[0] or 0)

    def _set_last_job_id(self, target_engine, source_db: str, last_job_id: int) -> None:
        query = text(
            """
            INSERT INTO sync_state (source_db, destination, last_job_id)
            VALUES (:source_db, :destination, :last_job_id)
            ON CONFLICT (source_db, destination)
            DO UPDATE SET
                last_job_id = EXCLUDED.last_job_id,
                updated_at = NOW()
            """
        )
        with target_engine.begin() as conn:
            conn.execute(
                query,
                {
                    "source_db": source_db,
                    "destination": self.export_destination,
                    "last_job_id": last_job_id,
                },
            )

    def _city_has_country_code_column(self, source_engine, db_name: str) -> bool:
        with source_engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = :table_schema
                      AND table_name = 'city'
                      AND column_name = 'country_code'
                    LIMIT 1
                    """
                ),
                {"table_schema": db_name},
            ).first()
            return result is not None

    def _table_exists(self, source_engine, db_name: str, table_name: str) -> bool:
        with source_engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = :table_schema
                      AND table_name = :table_name
                    LIMIT 1
                    """
                ),
                {"table_schema": db_name, "table_name": table_name},
            ).first()
            return result is not None

    def _column_exists(self, source_engine, db_name: str, table_name: str, column_name: str) -> bool:
        with source_engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = :table_schema
                      AND table_name = :table_name
                      AND column_name = :column_name
                    LIMIT 1
                    """
                ),
                {"table_schema": db_name, "table_name": table_name, "column_name": column_name},
            ).first()
            return result is not None

    def _job_has_salary_currency_column(self, source_engine, db_name: str) -> bool:
        with source_engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = :table_schema
                      AND table_name = 'job'
                      AND column_name = 'salary_currency'
                    LIMIT 1
                    """
                ),
                {"table_schema": db_name},
            ).first()
            return result is not None

    def _fetch_source_rows(
        self,
        source_engine,
        batch_size: int,
        use_country_code_from_city: bool,
        use_currency_from_job: bool,
        use_region_from_city: bool,
        region_city_column: str,
        use_region_join: bool,
        last_job_id: int,
    ) -> list[dict[str, object | None]]:
        country_code_sql = "c.country_code AS city_country_code," if use_country_code_from_city else "NULL AS city_country_code,"
        currency_sql = "j.salary_currency AS job_salary_currency," if use_currency_from_job else "NULL AS job_salary_currency,"
        region_sql, region_join_sql = self._build_region_sql(
            use_region_from_city=use_region_from_city,
            region_city_column=region_city_column,
            use_region_join=use_region_join,
        )
        query = text(
            f"""
            SELECT
                j.id,
                j.site_id,
                j.external_id,
                j.position,
                j.description,
                j.company_id,
                j.company_name,
                s.name AS site_title,
                j.url,
                j.city_id,
                c.title AS city_title,
                {region_sql}
                {country_code_sql}
                {currency_sql}
                j.salary_min,
                j.salary_max,
                j.salary_period,
                j.contract,
                j.experience,
                j.education,
                j.published,
                j.expires,
                j.subcategory
            FROM job j
            LEFT JOIN city c ON c.id = j.city_id
            {region_join_sql}
            LEFT JOIN site s ON s.id = j.site_id
            WHERE j.id > :last_job_id
              AND NOT EXISTS (
                SELECT 1
                FROM export_ai e
                WHERE e.job_id = j.id
                  AND e.destination = :destination
            )
            ORDER BY j.id
            LIMIT :batch_size
            """
        )
        with source_engine.connect() as conn:
            rows = conn.execute(
                query,
                {
                    "last_job_id": last_job_id,
                    "destination": self.export_destination,
                    "batch_size": batch_size,
                },
            ).mappings()
            return [dict(row) for row in rows]

    def _build_region_sql(
        self,
        use_region_from_city: bool,
        region_city_column: str,
        use_region_join: bool,
    ) -> tuple[str, str]:
        if use_region_from_city:
            return f"NULL AS region_id, c.`{region_city_column}` AS region_title,", ""
        if use_region_join:
            return "c.region_id, r.title AS region_title,", "LEFT JOIN region r ON r.id = c.region_id"
        return "NULL AS region_id, NULL AS region_title,", ""

    def _filter_rows_for_country_code_requirement(
        self,
        raw_rows: list[dict[str, object | None]],
        require_country_code_from_city: bool,
    ) -> tuple[list[dict[str, object | None]], int]:
        if not require_country_code_from_city:
            return raw_rows, 0

        filtered: list[dict[str, object | None]] = []
        skipped = 0
        for row in raw_rows:
            if self._normalize_country_code(row.get("city_country_code")) is None:
                skipped += 1
                continue
            filtered.append(row)
        return filtered, skipped

    def _build_jobs_payload(
        self,
        raw_rows: list[dict[str, object | None]],
        source_db: str,
        default_country_code: str | None,
        default_currency: str | None,
        use_country_code_from_city: bool,
        use_currency_from_job: bool,
    ) -> list[dict[str, object | None]]:
        payload: list[dict[str, object | None]] = []
        now = datetime.now(timezone.utc)
        for row in raw_rows:
            city_country_code = self._normalize_country_code(row.get("city_country_code")) if use_country_code_from_city else None
            country_code = city_country_code or default_country_code
            job_currency = self._normalize_currency(row.get("job_salary_currency")) if use_currency_from_job else None
            salary_currency = job_currency or default_currency

            expires_at = row.get("expires")
            is_active = True
            if isinstance(expires_at, datetime):
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=timezone.utc)
                is_active = expires_at >= now

            payload.append(
                {
                    "source_db": source_db,
                    "source_job_id": row["id"],
                    "site_id": row["site_id"],
                    "external_id": row["external_id"],
                    "title": row["position"],
                    "description": row["description"],
                    "company_id": row["company_id"],
                    "company_name": row["company_name"],
                    "site_title": row["site_title"],
                    "url": row["url"],
                    "city_id": row["city_id"],
                    "city_title": row["city_title"],
                    "region_id": row["region_id"],
                    "region_title": row["region_title"],
                    "country_code": country_code,
                    "language_code": detect_language_code(
                        title=row.get("position"),
                        description=row.get("description"),
                        country_code=country_code,
                        source_db=source_db,
                    ).language_code,
                    "salary_min": row["salary_min"],
                    "salary_max": row["salary_max"],
                    "salary_period": row["salary_period"],
                    "salary_currency": salary_currency,
                    "contract": row["contract"],
                    "experience": row["experience"],
                    "education": row["education"],
                    "published_at": row["published"],
                    "expires_at": expires_at,
                    "is_remote": self._is_remote(row.get("subcategory")),
                    "is_active": is_active,
                }
            )
        return payload

    def _upsert_jobs(self, target_engine, payload: list[dict[str, object | None]]) -> None:
        if not payload:
            return
        query = text(
            """
            WITH cleanup AS (
                DELETE FROM jobs
                WHERE source_db = :source_db
                  AND source_job_id = :source_job_id
                  AND (site_id <> :site_id OR external_id <> :external_id)
            )
            INSERT INTO jobs (
                source_db,
                source_job_id,
                site_id,
                external_id,
                title,
                description,
                company_id,
                company_name,
                site_title,
                url,
                city_id,
                city_title,
                region_id,
                region_title,
                country_code,
                language_code,
                salary_min,
                salary_max,
                salary_period,
                salary_currency,
                contract,
                experience,
                education,
                published_at,
                expires_at,
                is_remote,
                is_active
            ) VALUES (
                :source_db,
                :source_job_id,
                :site_id,
                :external_id,
                :title,
                :description,
                :company_id,
                :company_name,
                :site_title,
                :url,
                :city_id,
                :city_title,
                :region_id,
                :region_title,
                :country_code,
                :language_code,
                :salary_min,
                :salary_max,
                :salary_period,
                :salary_currency,
                :contract,
                :experience,
                :education,
                :published_at,
                :expires_at,
                :is_remote,
                :is_active
            )
            ON CONFLICT ON CONSTRAINT uq_jobs_source_external
            DO UPDATE SET
                source_job_id = EXCLUDED.source_job_id,
                site_id = EXCLUDED.site_id,
                external_id = EXCLUDED.external_id,
                title = EXCLUDED.title,
                description = EXCLUDED.description,
                company_id = EXCLUDED.company_id,
                company_name = EXCLUDED.company_name,
                site_title = EXCLUDED.site_title,
                url = EXCLUDED.url,
                city_id = EXCLUDED.city_id,
                city_title = EXCLUDED.city_title,
                region_id = EXCLUDED.region_id,
                region_title = EXCLUDED.region_title,
                country_code = EXCLUDED.country_code,
                language_code = EXCLUDED.language_code,
                salary_min = EXCLUDED.salary_min,
                salary_max = EXCLUDED.salary_max,
                salary_period = EXCLUDED.salary_period,
                salary_currency = EXCLUDED.salary_currency,
                contract = EXCLUDED.contract,
                experience = EXCLUDED.experience,
                education = EXCLUDED.education,
                published_at = EXCLUDED.published_at,
                expires_at = EXCLUDED.expires_at,
                is_remote = EXCLUDED.is_remote,
                is_active = EXCLUDED.is_active
            """
        )
        with target_engine.begin() as conn:
            conn.execute(query, payload)

    def _mark_exported(self, source_engine, raw_rows: list[dict[str, object | None]]) -> None:
        if not raw_rows:
            return
        query = text(
            """
            INSERT INTO export_ai (job_id, destination, site_id)
            VALUES (:job_id, :destination, :site_id)
            ON DUPLICATE KEY UPDATE
                site_id = VALUES(site_id),
                dt = CURRENT_TIMESTAMP
            """
        )
        params = [
            {
                "job_id": row["id"],
                "destination": self.export_destination,
                "site_id": row["site_id"],
            }
            for row in raw_rows
        ]
        with source_engine.begin() as conn:
            conn.execute(query, params)

    @staticmethod
    def _normalize_country_code(value: object | None) -> str | None:
        if value is None:
            return None
        text_value = str(value).strip().upper()
        if len(text_value) != 2:
            return None
        return text_value

    @staticmethod
    def _is_remote(subcategory: object | None) -> bool:
        if subcategory is None:
            return False
        return "remote" in str(subcategory).lower()

    @staticmethod
    def country_code_in_city_enabled(config: object | None) -> bool:
        if not isinstance(config, dict):
            return False
        value = config.get("country_code_in_city", 0)
        return value in (1, True, "1", "true", "True")

    @staticmethod
    def currency_in_job_enabled(config: object | None) -> bool:
        if not isinstance(config, dict):
            return False
        value = config.get("currency_in_job", 0)
        return value in (1, True, "1", "true", "True")

    @staticmethod
    def region_in_city_column(config: object | None) -> str | None:
        if not isinstance(config, dict):
            return None
        value = config.get("region_in_city")
        if not isinstance(value, str):
            return None
        value = value.strip()
        if not value:
            return None
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", value):
            return None
        return value

    @staticmethod
    def _normalize_currency(value: object | None) -> str | None:
        if value is None:
            return None
        text_value = str(value).strip().upper()
        if len(text_value) != 3:
            return None
        return text_value

    def source_db_url_for(self, db_name: str) -> URL:
        return URL.create(
            drivername=self.source_db_driver,
            username=self.source_db_user,
            password=self.source_db_password,
            host=self.source_db_host,
            port=self.source_db_port,
            database=db_name,
        )

    def _create_source_engine(self, db_name: str):
        connect_args: dict[str, object] = {}
        if self.source_db_ssl_disabled:
            # Equivalent of mysql CLI `--skip-ssl` for PyMySQL connections.
            connect_args["ssl_disabled"] = True
        return create_engine(self.source_db_url_for(db_name), pool_pre_ping=True, connect_args=connect_args)

    @staticmethod
    def _redact_url(url: str) -> str:
        if "@" not in url:
            return url
        left, right = url.split("@", 1)
        if "://" not in left:
            return f"***@{right}"
        scheme, _ = left.split("://", 1)
        return f"{scheme}://***@{right}"
