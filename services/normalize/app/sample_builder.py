import argparse
import logging
import math
from datetime import datetime, timezone

from sqlalchemy import create_engine, text

from app.config import settings
from app.logging import configure_logging


logger = logging.getLogger("jobl.normalize.sample")

LANGUAGE_ORDER = ["en", "es", "de", "fr", "pt", "it", "gr", "nl", "da", "uk"]
DEFAULT_LANGUAGE_WEIGHTS = {
    "en": 0.45,
    "es": 0.20,
    "de": 0.10,
    "fr": 0.08,
    "pt": 0.08,
    "it": 0.03,
    "gr": 0.02,
    "nl": 0.015,
    "da": 0.01,
    "uk": 0.015,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build normalization samples from jobs table by language proportions")
    parser.add_argument(
        "--total",
        type=int,
        default=50000,
        help="Total target sample size across languages",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete all existing rows from normalization_samples before insert",
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    configure_logging(settings.log_level)

    batch_tag = f"sample_lang_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    total = max(1, args.total)
    targets = _build_language_targets(total)

    logger.info(
        "sample builder started total=%s batch_tag=%s replace=%s targets=%s",
        total,
        batch_tag,
        args.replace,
        ", ".join(f"{k}:{v}" for k, v in targets.items()),
    )

    engine = create_engine(settings.target_database_url, pool_pre_ping=True)
    try:
        with engine.begin() as conn:
            if args.replace:
                conn.execute(text("DELETE FROM normalization_samples"))

            conn.execute(
                text(
                    """
                    WITH targets(language_code, target_n) AS (
                        VALUES
                            ('en', :target_en),
                            ('es', :target_es),
                            ('de', :target_de),
                            ('fr', :target_fr),
                            ('pt', :target_pt),
                            ('it', :target_it),
                            ('gr', :target_gr),
                            ('nl', :target_nl),
                            ('da', :target_da),
                            ('uk', :target_uk)
                    ),
                    source_lang AS (
                        SELECT
                            j.id,
                            j.language_code AS language_code
                        FROM jobs j
                        WHERE j.country_code IS NOT NULL
                          AND COALESCE(BTRIM(j.title), '') <> ''
                          AND COALESCE(BTRIM(j.description), '') <> ''
                    ),
                    ranked AS (
                        SELECT
                            s.id,
                            s.language_code,
                            ROW_NUMBER() OVER (PARTITION BY s.language_code ORDER BY RANDOM()) AS rn
                        FROM source_lang s
                        WHERE s.language_code IN ('en', 'es', 'de', 'fr', 'pt', 'it', 'gr', 'nl', 'da', 'uk')
                    )
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
                        title_raw,
                        description_raw,
                        batch_tag
                    )
                    SELECT
                        j.source_db,
                        j.country_code,
                        r.language_code,
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
                    JOIN targets t ON t.language_code = r.language_code
                    JOIN jobs j ON j.id = r.id
                    LEFT JOIN countries c ON c.code = j.country_code
                    WHERE r.rn <= t.target_n
                    """
                ),
                {
                    "batch_tag": batch_tag,
                    "target_en": targets["en"],
                    "target_es": targets["es"],
                    "target_de": targets["de"],
                    "target_fr": targets["fr"],
                    "target_pt": targets["pt"],
                    "target_it": targets["it"],
                    "target_gr": targets["gr"],
                    "target_nl": targets["nl"],
                    "target_da": targets["da"],
                    "target_uk": targets["uk"],
                },
            )

        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT language_code, COUNT(*) AS n
                    FROM normalization_samples
                    WHERE batch_tag = :batch_tag
                    GROUP BY language_code
                    ORDER BY language_code
                    """
                ),
                {"batch_tag": batch_tag},
            ).mappings()
            counts = [f"{row['language_code']}={row['n']}" for row in rows]

        logger.info("sample builder completed batch_tag=%s languages=%s", batch_tag, ", ".join(counts))
    finally:
        engine.dispose()


def _build_language_targets(total: int) -> dict[str, int]:
    weighted = {k: total * DEFAULT_LANGUAGE_WEIGHTS[k] for k in LANGUAGE_ORDER}
    base = {k: int(math.floor(v)) for k, v in weighted.items()}
    remainder = total - sum(base.values())
    if remainder > 0:
        frac_sorted = sorted(
            LANGUAGE_ORDER,
            key=lambda code: weighted[code] - base[code],
            reverse=True,
        )
        for code in frac_sorted[:remainder]:
            base[code] += 1
    return base
