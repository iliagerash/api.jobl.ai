import argparse
import logging
from typing import Any

from openai import OpenAI
from sqlalchemy import create_engine, text

from app.config import settings
from app.llm_label_runner import (
    _fetch_rows_by_ids,
    _file_content_to_text,
    _parse_batch_output_lines,
    _result_to_db_payload,
)
from app.logging import configure_logging


logger = logging.getLogger("jobl.normalize.llm_label_debug")


UPDATE_QUERY = text(
    """
    UPDATE normalization_samples
    SET expected_title_normalized = :expected_title_normalized,
        expected_description_html = :expected_description_html,
        batch_tag = COALESCE(:batch_tag, batch_tag),
        review_notes = :review_notes,
        updated_at = NOW()
    WHERE id = :id
    """
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume-only debug tool for failed ids in OpenAI batch output")
    parser.add_argument("--batch-id", required=True, help="Existing OpenAI batch id")
    parser.add_argument("--batch-tag", default=None, help="Optional batch_tag to write while applying")
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Analyze only (skip DB updates). Useful if you only need parse/missing ids.",
    )
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    configure_logging(settings.log_level)

    if not settings.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is required")

    client = OpenAI(api_key=settings.openai_api_key, timeout=settings.openai_timeout_seconds)
    engine = create_engine(settings.target_database_url, pool_pre_ping=True)

    try:
        summary = _run_debug(
            engine=engine,
            client=client,
            batch_id=args.batch_id,
            write_batch_tag=args.batch_tag,
            no_write=args.no_write,
        )
    except KeyboardInterrupt:
        logger.warning("interrupted by user (Ctrl+C), exiting gracefully")
        return 130
    finally:
        engine.dispose()

    logger.info(
        "debug completed fetched=%s ok=%s missing=%s parse_failed=%s db_failed=%s",
        summary["fetched"],
        summary["ok"],
        summary["missing"],
        summary["parse_failed"],
        summary["db_failed"],
    )
    if summary["missing_ids"]:
        logger.warning("missing_ids=%s", ",".join(summary["missing_ids"]))
    if summary["parse_failed_ids"]:
        logger.warning("parse_failed_ids=%s", ",".join(summary["parse_failed_ids"]))
    if summary["db_failed_ids"]:
        logger.warning("db_failed_ids=%s", ",".join(summary["db_failed_ids"]))
    return 0


def _run_debug(
    *,
    engine,
    client: OpenAI,
    batch_id: str,
    write_batch_tag: str | None,
    no_write: bool,
) -> dict[str, Any]:
    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed" or not batch.output_file_id:
        raise SystemExit(f"Batch not completed or missing output_file_id: status={batch.status}")

    output_resp = client.files.content(batch.output_file_id)
    output_text = _file_content_to_text(output_resp)
    results = _parse_batch_output_lines(output_text)

    result_ids = sorted(int(row_id) for row_id in results if str(row_id).isdigit())
    rows = _fetch_rows_by_ids(engine=engine, ids=result_ids)
    row_map = {str(row["id"]): row for row in rows}

    missing_ids: list[str] = []
    parse_failed_ids: list[str] = []
    payload: list[dict[str, Any]] = []

    for row_id, row in row_map.items():
        result = results.get(row_id)
        if row_id not in results:
            missing_ids.append(row_id)
            continue
        if result is None:
            parse_failed_ids.append(row_id)
            continue
        payload.append(_result_to_db_payload(row=row, result=result, write_batch_tag=write_batch_tag))

    db_failed_ids: list[str] = []
    ok = 0

    if not no_write:
        for item in payload:
            try:
                with engine.begin() as conn:
                    conn.execute(UPDATE_QUERY, [item])
                ok += 1
            except Exception as exc:  # noqa: BLE001
                db_failed_ids.append(str(item.get("id")))
                logger.error("db row update failed id=%s err=%s", item.get("id"), exc)
    else:
        ok = len(payload)

    return {
        "fetched": len(rows),
        "ok": ok,
        "missing": len(missing_ids),
        "parse_failed": len(parse_failed_ids),
        "db_failed": len(db_failed_ids),
        "missing_ids": missing_ids,
        "parse_failed_ids": parse_failed_ids,
        "db_failed_ids": db_failed_ids,
    }


if __name__ == "__main__":
    raise SystemExit(run())
