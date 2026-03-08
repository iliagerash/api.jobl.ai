import argparse
import logging

from sqlalchemy import create_engine, text

from app.config import settings
from app.language import detect_language_code
from app.logging import configure_logging


logger = logging.getLogger("jobl.sync.language_backfill")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill jobs.language_code from existing records")
    parser.add_argument("--batch-size", type=int, default=2000, help="Rows per batch")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to process; 0 means no limit")
    parser.add_argument("--from-id", type=int, default=0, help="Start with jobs.id > value")
    parser.add_argument("--overwrite", action="store_true", help="Recompute language_code even if already set")
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    configure_logging(settings.log_level)

    engine = create_engine(settings.target_database_url, pool_pre_ping=True)
    processed = 0
    updated = 0
    cursor = args.from_id
    limit = args.limit if args.limit and args.limit > 0 else None

    logger.info(
        "language backfill started batch_size=%s limit=%s from_id=%s overwrite=%s",
        args.batch_size,
        limit,
        args.from_id,
        args.overwrite,
    )

    try:
        try:
            while True:
                if limit is not None and processed >= limit:
                    break

                batch_size = args.batch_size
                if limit is not None:
                    batch_size = min(batch_size, limit - processed)
                    if batch_size <= 0:
                        break

                rows = _fetch_rows(engine=engine, from_id=cursor, batch_size=batch_size, overwrite=args.overwrite)
                if not rows:
                    break

                payload = []
                for row in rows:
                    detected = detect_language_code(
                        title=row.get("title"),
                        description=row.get("description"),
                        country_code=row.get("country_code"),
                        source_db=row.get("source_db"),
                    )
                    payload.append(
                        {
                            "id": row["id"],
                            "language_code": detected.language_code,
                        }
                    )

                _update_rows(engine=engine, payload=payload)

                processed += len(rows)
                updated += len(payload)
                cursor = int(rows[-1]["id"])
                logger.info(
                    "language backfill progress batch=%s processed=%s updated=%s last_id=%s",
                    len(rows),
                    processed,
                    updated,
                    cursor,
                )
        except KeyboardInterrupt:
            logger.warning("interrupted by user (Ctrl+C), exiting gracefully")
            return 130
    finally:
        engine.dispose()

    logger.info("language backfill completed processed=%s updated=%s", processed, updated)
    return 0


def _fetch_rows(engine, from_id: int, batch_size: int, overwrite: bool) -> list[dict[str, object | None]]:
    where = ["id > :from_id"]
    params: dict[str, object] = {"from_id": from_id, "batch_size": batch_size}
    if not overwrite:
        where.append("language_code IS NULL")

    query = text(
        f"""
        SELECT id, source_db, country_code, title, description
        FROM jobs
        WHERE {' AND '.join(where)}
        ORDER BY id
        LIMIT :batch_size
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
        UPDATE jobs
        SET language_code = :language_code
        WHERE id = :id
        """
    )
    with engine.begin() as conn:
        conn.execute(query, payload)


if __name__ == "__main__":
    run()
