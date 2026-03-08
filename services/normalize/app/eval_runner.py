import argparse
import logging

from sqlalchemy import create_engine, text

from app.config import settings
from app.logging import configure_logging
from app.worker import NormalizeWorker


logger = logging.getLogger("jobl.normalize.eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate normalization on normalization_samples")
    parser.add_argument("--batch-tag", type=str, default=None, help="Process only one batch_tag")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows to process")
    parser.add_argument(
        "--only-pending",
        action="store_true",
        help="Process only rows where generated fields are still NULL",
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    configure_logging(settings.log_level)

    normalizer = NormalizeWorker(target_database_url=settings.target_database_url)
    engine = create_engine(settings.target_database_url, pool_pre_ping=True)
    try:
        rows = _fetch_rows(engine=engine, batch_tag=args.batch_tag, limit=args.limit, only_pending=args.only_pending)
        if not rows:
            logger.info("no rows selected for evaluation")
            return

        payload = []
        title_matches = 0
        description_clean_matches = 0
        description_html_matches = 0

        for row in rows:
            title_raw = row.get("title_raw") or ""
            description_raw = row.get("description_raw") or ""

            generated_title = normalizer._normalize_title(
                title_raw,
                city_title=row.get("city_title"),
                region_title=row.get("region_title"),
                country_code=row.get("country_code"),
                country_name=row.get("country_name"),
                country_alternate_names=row.get("country_alternate_names"),
            )
            generated_clean = normalizer._clean_description(description_raw)
            generated_html = normalizer._to_safe_html(generated_clean)

            title_match = _compute_match(row.get("expected_title_normalized"), generated_title)
            description_clean_match = _compute_match(row.get("expected_description_clean"), generated_clean)
            description_html_match = _compute_match(row.get("expected_description_html"), generated_html)

            if title_match is True:
                title_matches += 1
            if description_clean_match is True:
                description_clean_matches += 1
            if description_html_match is True:
                description_html_matches += 1

            payload.append(
                {
                    "id": row["id"],
                    "generated_title_normalized": generated_title,
                    "generated_description_clean": generated_clean,
                    "generated_description_html": generated_html,
                    "title_match": title_match,
                    "description_clean_match": description_clean_match,
                    "description_html_match": description_html_match,
                }
            )

        _update_rows(engine=engine, payload=payload)

        logger.info(
            "evaluation completed rows=%s title_matches=%s description_clean_matches=%s description_html_matches=%s",
            len(payload),
            title_matches,
            description_clean_matches,
            description_html_matches,
        )
    finally:
        engine.dispose()


def _fetch_rows(engine, batch_tag: str | None, limit: int | None, only_pending: bool) -> list[dict[str, object | None]]:
    where = ["1=1"]
    params: dict[str, object] = {}
    if batch_tag:
        where.append("ns.batch_tag = :batch_tag")
        params["batch_tag"] = batch_tag
    if only_pending:
        where.append(
            "(ns.generated_title_normalized IS NULL OR ns.generated_description_clean IS NULL OR ns.generated_description_html IS NULL)"
        )

    limit_sql = ""
    if limit is not None and limit > 0:
        limit_sql = "LIMIT :limit_rows"
        params["limit_rows"] = limit

    query = text(
        f"""
        SELECT
            ns.id,
            ns.title_raw,
            ns.description_raw,
            ns.expected_title_normalized,
            ns.expected_description_clean,
            ns.expected_description_html,
            ns.city_title,
            ns.region_title,
            ns.country_code,
            ns.country_name,
            c.alternate_names AS country_alternate_names
        FROM normalization_samples ns
        LEFT JOIN countries c ON c.code = ns.country_code
        WHERE {' AND '.join(where)}
        ORDER BY ns.id
        {limit_sql}
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, params).mappings()
        return [dict(row) for row in rows]


def _update_rows(engine, payload: list[dict[str, object | None]]) -> None:
    if not payload:
        return
    query = text(
        """
        UPDATE normalization_samples
        SET generated_title_normalized = :generated_title_normalized,
            generated_description_clean = :generated_description_clean,
            generated_description_html = :generated_description_html,
            title_match = :title_match,
            description_clean_match = :description_clean_match,
            description_html_match = :description_html_match,
            updated_at = NOW()
        WHERE id = :id
        """
    )
    with engine.begin() as conn:
        conn.execute(query, payload)


def _compute_match(expected: object | None, generated: str) -> bool | None:
    if expected is None:
        return None
    expected_value = str(expected).strip()
    return expected_value == (generated or "").strip()
