import argparse
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import create_engine, text

from app.config import settings
from app.logging import configure_logging


logger = logging.getLogger("jobl.normalize.extract_samples")

COUNTRY_CODES = ["US", "CA", "GB", "AU", "NZ", "SG"]
LANGUAGE_CODES = ["en", "fr"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract normalization samples from jobs table")
    parser.add_argument("--limit", type=int, default=1000, help="Total number of rows to extract")
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    configure_logging(settings.log_level)

    limit = max(1, int(args.limit))
    country_targets = _build_country_targets(limit)
    batch_tag = f"extract_samples_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    logger.info(
        "extract samples started limit=%s batch_tag=%s countries=%s languages=%s",
        limit,
        batch_tag,
        ",".join(COUNTRY_CODES),
        ",".join(LANGUAGE_CODES),
    )

    engine = create_engine(settings.target_database_url, pool_pre_ping=True)
    try:
        try:
            inserted = _extract_and_insert(engine=engine, batch_tag=batch_tag, country_targets=country_targets)
            logger.info("extract samples completed inserted=%s batch_tag=%s", inserted, batch_tag)
        except KeyboardInterrupt:
            logger.warning("interrupted by user (Ctrl+C), exiting gracefully")
            return 130
    finally:
        engine.dispose()

    return 0


def _extract_and_insert(*, engine, batch_tag: str, country_targets: dict[str, int]) -> int:
    selected_rows: list[dict[str, Any]] = []

    with engine.connect() as conn:
        for country_code in COUNTRY_CODES:
            country_limit = int(country_targets.get(country_code, 0))
            if country_limit <= 0:
                continue

            with_at_target, without_at_target = _split_at_targets(country_limit)
            country_rows: list[dict[str, Any]] = []

            if with_at_target > 0:
                with_at_rows = _fetch_country_rows(
                    conn=conn,
                    country_code=country_code,
                    limit=with_at_target,
                    has_at=True,
                    exclude_ids=set(),
                )
                country_rows.extend(with_at_rows)

            if without_at_target > 0:
                without_at_rows = _fetch_country_rows(
                    conn=conn,
                    country_code=country_code,
                    limit=without_at_target,
                    has_at=False,
                    exclude_ids={int(row["id"]) for row in country_rows},
                )
                country_rows.extend(without_at_rows)

            missing = country_limit - len(country_rows)
            if missing > 0:
                topup_rows = _fetch_country_rows(
                    conn=conn,
                    country_code=country_code,
                    limit=missing,
                    has_at=None,
                    exclude_ids={int(row["id"]) for row in country_rows},
                )
                country_rows.extend(topup_rows)

            at_count = sum(1 for row in country_rows if "@" in str(row.get("description") or ""))
            logger.info(
                "country sampled country=%s target=%s selected=%s with_at=%s without_at=%s",
                country_code,
                country_limit,
                len(country_rows),
                at_count,
                len(country_rows) - at_count,
            )

            for row in country_rows:
                row["batch_tag"] = batch_tag
            selected_rows.extend(country_rows)

        total_target = sum(int(v) for v in country_targets.values())
        missing = total_target - len(selected_rows)
        if missing > 0:
            topup_rows = _fetch_topup_rows(
                conn=conn,
                limit=missing,
                exclude_ids={int(row["id"]) for row in selected_rows},
            )
            for row in topup_rows:
                row["batch_tag"] = batch_tag
            selected_rows.extend(topup_rows)
            logger.info(
                "global top-up requested=%s selected=%s",
                missing,
                len(topup_rows),
            )

    if not selected_rows:
        logger.info("no eligible rows matched extraction criteria")
        return 0

    insert_query = text(
        """
        INSERT INTO normalization_samples (
            source_db,
            country_code,
            language_code,
            country_name,
            city_title,
            region_title,
            site_id,
            source_job_id,
            url,
            company_name,
            title,
            description,
            batch_tag
        )
        VALUES (
            :source_db,
            :country_code,
            :language_code,
            :country_name,
            :city_title,
            :region_title,
            :site_id,
            :source_job_id,
            :url,
            :company_name,
            :title,
            :description,
            :batch_tag
        )
        """
    )

    with engine.begin() as conn:
        conn.execute(insert_query, selected_rows)

    return len(selected_rows)


def _fetch_country_rows(
    *,
    conn,
    country_code: str,
    limit: int,
    has_at: bool | None,
    exclude_ids: set[int],
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    where = [
        "j.country_code = :country_code",
        "j.language_code IN ('en', 'fr')",
        "COALESCE(BTRIM(j.title), '') <> ''",
        "COALESCE(BTRIM(j.description), '') <> ''",
    ]
    params: dict[str, Any] = {
        "country_code": country_code,
        "limit_rows": limit,
    }

    if has_at is True:
        where.append("POSITION('@' IN j.description) > 0")
    elif has_at is False:
        where.append("POSITION('@' IN j.description) = 0")

    if exclude_ids:
        exclude_sql = ",".join(str(int(v)) for v in sorted(exclude_ids))
        where.append(f"j.id NOT IN ({exclude_sql})")

    query = text(
        f"""
        SELECT
            j.id,
            j.source_db,
            j.country_code,
            j.language_code,
            c.name AS country_name,
            j.city_title,
            j.region_title,
            j.site_id,
            j.source_job_id,
            j.url,
            j.company_name,
            j.title AS title,
            j.description AS description
        FROM jobs j
        LEFT JOIN countries c ON c.code = j.country_code
        WHERE {' AND '.join(where)}
        ORDER BY RANDOM()
        LIMIT :limit_rows
        """
    )

    rows = conn.execute(query, params).mappings()
    return [dict(row) for row in rows]


def _build_country_targets(total: int) -> dict[str, int]:
    country_count = len(COUNTRY_CODES)
    base = total // country_count
    remainder = total % country_count

    targets: dict[str, int] = {}
    for idx, code in enumerate(COUNTRY_CODES):
        targets[code] = base + (1 if idx < remainder else 0)
    return targets


def _fetch_topup_rows(
    *,
    conn,
    limit: int,
    exclude_ids: set[int],
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    where = [
        "j.country_code IN ('US', 'CA', 'GB', 'AU', 'NZ', 'SG')",
        "j.language_code IN ('en', 'fr')",
        "COALESCE(BTRIM(j.title), '') <> ''",
        "COALESCE(BTRIM(j.description), '') <> ''",
    ]
    params: dict[str, Any] = {
        "limit_rows": limit,
    }

    if exclude_ids:
        exclude_sql = ",".join(str(int(v)) for v in sorted(exclude_ids))
        where.append(f"j.id NOT IN ({exclude_sql})")

    query = text(
        f"""
        SELECT
            j.id,
            j.source_db,
            j.country_code,
            j.language_code,
            c.name AS country_name,
            j.city_title,
            j.region_title,
            j.site_id,
            j.source_job_id,
            j.url,
            j.company_name,
            j.title AS title,
            j.description AS description
        FROM jobs j
        LEFT JOIN countries c ON c.code = j.country_code
        WHERE {' AND '.join(where)}
        ORDER BY RANDOM()
        LIMIT :limit_rows
        """
    )
    rows = conn.execute(query, params).mappings()
    return [dict(row) for row in rows]


def _split_at_targets(country_limit: int) -> tuple[int, int]:
    with_at = int(round(country_limit * 0.95))
    with_at = min(max(with_at, 0), country_limit)
    without_at = country_limit - with_at
    return with_at, without_at


if __name__ == "__main__":
    raise SystemExit(run())
