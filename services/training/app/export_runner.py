import argparse
import logging
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text

from app.config import settings
from app.io_utils import write_jsonl
from app.logging import configure_logging


logger = logging.getLogger("jobl.training.export")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export labeled normalization_samples to JSONL")
    parser.add_argument("--out", type=str, default="data/raw/labeled.jsonl", help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to export, 0 means all")
    parser.add_argument("--batch-tag", type=str, default=None, help="Filter only one batch_tag")
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    configure_logging(settings.log_level)

    engine = create_engine(settings.target_database_url, pool_pre_ping=True)
    out_path = Path(args.out)
    limit = args.limit if args.limit > 0 else None

    try:
        rows = _fetch_rows(engine=engine, batch_tag=args.batch_tag, limit=limit)
        write_jsonl(out_path, rows)
    except KeyboardInterrupt:
        logger.warning("interrupted by user (Ctrl+C), exiting gracefully")
        return 130
    finally:
        engine.dispose()

    logger.info("export completed rows=%s output=%s", len(rows), out_path)
    return 0


def _fetch_rows(engine, batch_tag: str | None, limit: int | None) -> list[dict[str, Any]]:
    where = [
        "COALESCE(BTRIM(ns.title), '') <> ''",
        "COALESCE(BTRIM(ns.expected_title_normalized), '') <> ''",
        "ns.language_code IN ('en', 'fr')",
    ]
    params: dict[str, Any] = {}
    if batch_tag:
        where.append("ns.batch_tag = :batch_tag")
        params["batch_tag"] = batch_tag

    limit_sql = ""
    if limit is not None:
        limit_sql = "LIMIT :limit_rows"
        params["limit_rows"] = limit

    query = text(
        f"""
        SELECT
            ns.id,
            ns.language_code,
            ns.title,
            ns.expected_title_normalized
        FROM normalization_samples ns
        WHERE {' AND '.join(where)}
        ORDER BY ns.id
        {limit_sql}
        """
    )

    with engine.connect() as conn:
        result = conn.execute(query, params).mappings()
        return [dict(row) for row in result]


if __name__ == "__main__":
    raise SystemExit(run())
