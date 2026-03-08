import argparse
import logging
from datetime import datetime, timezone

from sqlalchemy import create_engine, text

from app.config import settings
from app.logging import configure_logging


logger = logging.getLogger("jobl.normalize.sample")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build normalization samples from jobs table")
    parser.add_argument(
        "--per-country",
        type=int,
        default=200,
        help="Number of random rows per distinct country_code from jobs",
    )
    parser.add_argument(
        "--batch-tag",
        type=str,
        default=None,
        help="Batch tag to label inserted samples",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete existing rows for the same batch_tag before insert",
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    configure_logging(settings.log_level)

    batch_tag = args.batch_tag or f"sample_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    per_country = args.per_country

    logger.info("sample builder started per_country=%s batch_tag=%s replace=%s", per_country, batch_tag, args.replace)

    engine = create_engine(settings.target_database_url, pool_pre_ping=True)
    try:
        with engine.begin() as conn:
            if args.replace:
                conn.execute(
                    text("DELETE FROM normalization_samples WHERE batch_tag = :batch_tag"),
                    {"batch_tag": batch_tag},
                )

            conn.execute(
                text(
                    """
                    WITH ranked AS (
                        SELECT
                            j.id,
                            ROW_NUMBER() OVER (PARTITION BY j.country_code ORDER BY RANDOM()) AS rn
                        FROM jobs j
                        WHERE j.country_code IS NOT NULL
                          AND COALESCE(BTRIM(j.title), '') <> ''
                          AND COALESCE(BTRIM(j.description), '') <> ''
                    )
                    INSERT INTO normalization_samples (
                        source_db,
                        country_code,
                        country_name,
                        city_title,
                        region_title,
                        site_id,
                        source_job_id,
                        url,
                        title_raw,
                        description_raw,
                        batch_tag
                    )
                    SELECT
                        j.source_db,
                        j.country_code,
                        c.name AS country_name,
                        j.city_title,
                        j.region_title,
                        j.site_id,
                        j.source_job_id,
                        j.url,
                        j.title,
                        j.description,
                        :batch_tag
                    FROM ranked r
                    JOIN jobs j ON j.id = r.id
                    LEFT JOIN countries c ON c.code = j.country_code
                    WHERE r.rn <= :per_country
                    """
                ),
                {"batch_tag": batch_tag, "per_country": per_country},
            )

        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT country_code, COUNT(*) AS n
                    FROM normalization_samples
                    WHERE batch_tag = :batch_tag
                    GROUP BY country_code
                    ORDER BY country_code
                    """
                ),
                {"batch_tag": batch_tag},
            ).mappings()
            counts = [f"{row['country_code']}={row['n']}" for row in rows]

        logger.info("sample builder completed batch_tag=%s countries=%s", batch_tag, ", ".join(counts))
    finally:
        engine.dispose()
