import argparse
import json
import logging
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI
from sqlalchemy import create_engine, text

from app.config import settings
from app.logging import configure_logging


logger = logging.getLogger("jobl.normalize.llm_label")


SYSTEM_PROMPT = """
You normalize raw job data for a jobs platform.

Rules:
- Work from raw inputs exactly as provided.
- Do NOT include salaries, dates, addresses, company names, or location fragments in normalized title.
- Preserve legal title markers like (m/w/d), (m/f/d), (w/m/d), when present.
- title_normalized: cleaned role title only, no location/company/salary/date noise.
- description_html: clean HTML description.
  Remove style/script/noise. Keep semantic content.
  Allowed tags only: <p>, <ul>, <ol>, <li>, <i>, <b>, <em>, <strong>, <u>, <h2>, <h3>, <h4>, <br>, <a>.
  Remove job-title repetition at the beginning of description_html.
  If the first heading/paragraph is just the same as title_normalized (or a close variant), drop it.
  Example to remove: <p><strong>Security Business Partner</strong></p> when title_normalized is "Security Business Partner".
  Do not rewrite meaning and do not invent content.
- If uncertain, preserve information instead of hallucinating.

Return strict JSON with keys:
{
  "title_normalized": string,
  "description_html": string
}
""".strip()

RESPONSE_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "job_normalization",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "title_normalized": {"type": "string"},
                "description_html": {"type": "string"},
            },
            "required": ["title_normalized", "description_html"],
            "additionalProperties": False,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label normalization_samples with OpenAI LLM output")
    parser.add_argument("--batch-tag", type=str, default=None, help="Set/overwrite batch_tag for processed rows")
    parser.add_argument("--batch-id", type=str, default=None, help="Resume polling and apply an existing OpenAI batch id")
    parser.add_argument("--limit", type=int, default=200, help="Maximum rows to process in this run")
    parser.add_argument("--batch-size", type=int, default=20, help="Rows fetched per DB query page in --no-batch mode")
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Disable OpenAI Batch API and run one-by-one requests",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print full prompt and full raw response (only with --no-batch)",
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    configure_logging(settings.log_level)

    if not settings.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is required for jobl-normalize-llm-label")
    if args.debug and not args.no_batch:
        raise SystemExit("--debug is supported only with --no-batch")
    if args.batch_id and args.no_batch:
        raise SystemExit("--batch-id cannot be used with --no-batch")

    client = OpenAI(api_key=settings.openai_api_key, timeout=settings.openai_timeout_seconds)
    engine = create_engine(settings.target_database_url, pool_pre_ping=True)
    logger.info(
        "llm label started mode=%s model=%s prompt_version=%s limit=%s batch_size=%s overwrite=%s batch_tag=%s batch_id=%s",
        "resume" if args.batch_id else ("direct" if args.no_batch else "batch"),
        settings.openai_model,
        settings.llm_prompt_version,
        args.limit,
        args.batch_size,
        False,
        args.batch_tag,
        args.batch_id,
    )

    try:
        if args.batch_id:
            processed, updated = _resume_existing_batch(
                engine=engine,
                client=client,
                batch_id=args.batch_id,
                write_batch_tag=args.batch_tag,
            )
        elif args.no_batch:
            processed, updated = _run_direct_mode(
                engine=engine,
                client=client,
                write_batch_tag=args.batch_tag,
                limit=args.limit,
                batch_size=args.batch_size,
                debug=args.debug,
            )
        else:
            processed, updated = _run_batch_mode(
                engine=engine,
                client=client,
                write_batch_tag=args.batch_tag,
                limit=args.limit,
            )
    finally:
        engine.dispose()

    logger.info("llm label completed processed=%s updated=%s", processed, updated)


def _run_direct_mode(
    *,
    engine,
    client: OpenAI,
    write_batch_tag: str | None,
    limit: int,
    batch_size: int,
    debug: bool,
) -> tuple[int, int]:
    processed = 0
    updated = 0

    while processed < limit:
        remaining = limit - processed
        fetch_size = min(batch_size, remaining)
        rows = _fetch_rows(
            engine=engine,
            limit=fetch_size,
        )
        if not rows:
            break

        payload: list[dict[str, Any]] = []
        for row in rows:
            result = _label_row_direct(client=client, row=row, debug=debug)
            payload.append(_result_to_db_payload(row=row, result=result, write_batch_tag=write_batch_tag))

        _update_rows(engine=engine, payload=payload)
        processed += len(rows)
        updated += len(payload)
        last_id = int(rows[-1]["id"])

        logger.info(
            "llm label progress mode=direct fetched=%s updated=%s total_processed=%s/%s last_id=%s",
            len(rows),
            len(payload),
            processed,
            limit,
            last_id,
        )
    return processed, updated


def _run_batch_mode(
    *,
    engine,
    client: OpenAI,
    write_batch_tag: str | None,
    limit: int,
) -> tuple[int, int]:
    rows = _fetch_rows(
        engine=engine,
        limit=limit,
    )
    if not rows:
        logger.info("no rows selected for batch labeling")
        return 0, 0

    row_map = {str(row["id"]): row for row in rows}
    requests = [_build_batch_request(row) for row in rows]

    with tempfile.TemporaryDirectory(prefix="jobl_llm_batch_") as tmpdir:
        in_path = Path(tmpdir) / "requests.jsonl"
        _write_jsonl(path=in_path, records=requests)
        with in_path.open("rb") as f:
            in_file = client.files.create(file=f, purpose="batch")

        batch = client.batches.create(
            input_file_id=in_file.id,
            endpoint="/v1/chat/completions",
            completion_window=settings.openai_batch_completion_window,
            metadata={
                "service": "jobl-normalize",
                "prompt_version": settings.llm_prompt_version,
            },
        )
        logger.info("batch submitted batch_id=%s input_file_id=%s rows=%s", batch.id, in_file.id, len(rows))

        terminal = {"completed", "failed", "expired", "cancelled"}
        while batch.status not in terminal:
            time.sleep(max(1, settings.openai_batch_poll_seconds))
            batch = client.batches.retrieve(batch.id)
            counts = getattr(batch, "request_counts", None)
            completed = counts.get("completed") if isinstance(counts, dict) else getattr(counts, "completed", None)
            failed = counts.get("failed") if isinstance(counts, dict) else getattr(counts, "failed", None)
            logger.info(
                "batch polling batch_id=%s status=%s completed=%s failed=%s",
                batch.id,
                batch.status,
                completed,
                failed,
            )

        if batch.status != "completed" or not batch.output_file_id:
            logger.warning("batch did not complete status=%s batch_id=%s; falling back to direct mode", batch.status, batch.id)
            payload: list[dict[str, Any]] = []
            for row in rows:
                try:
                    direct_result = _label_row_direct(client=client, row=row)
                    payload.append(_result_to_db_payload(row=row, result=direct_result, write_batch_tag=write_batch_tag))
                except Exception:  # noqa: BLE001
                    logger.exception("direct fallback failed id=%s", row.get("id"))
            _update_rows(engine=engine, payload=payload)
            logger.info(
                "llm label progress mode=batch-fallback fetched=%s updated=%s failed=%s",
                len(rows),
                len(payload),
                max(0, len(rows) - len(payload)),
            )
            return len(rows), len(payload)

        output_resp = client.files.content(batch.output_file_id)
        output_text = _file_content_to_text(output_resp)
        results = _parse_batch_output_lines(output_text)

    payload: list[dict[str, Any]] = []
    failed_ids: set[str] = set()
    for row_id, result in results.items():
        row = row_map.get(row_id)
        if not row:
            continue
        if result is None:
            failed_ids.add(row_id)
            continue
        payload.append(_result_to_db_payload(row=row, result=result, write_batch_tag=write_batch_tag))

    # Any rows not present in output are also treated as failed and retried directly.
    for row_id in row_map:
        if row_id not in results:
            failed_ids.add(row_id)

    if failed_ids:
        logger.warning("batch returned failures=%s; retrying directly", len(failed_ids))
        for row_id in sorted(failed_ids, key=lambda v: int(v)):
            row = row_map[row_id]
            try:
                direct_result = _label_row_direct(client=client, row=row)
                payload.append(_result_to_db_payload(row=row, result=direct_result, write_batch_tag=write_batch_tag))
            except Exception:  # noqa: BLE001
                logger.exception("direct retry failed id=%s", row_id)

    _update_rows(engine=engine, payload=payload)
    logger.info(
        "llm label progress mode=batch fetched=%s updated=%s failed_after_retry=%s",
        len(rows),
        len(payload),
        max(0, len(rows) - len(payload)),
    )
    return len(rows), len(payload)


def _resume_existing_batch(
    *,
    engine,
    client: OpenAI,
    batch_id: str,
    write_batch_tag: str | None,
) -> tuple[int, int]:
    batch = client.batches.retrieve(batch_id)
    terminal = {"completed", "failed", "expired", "cancelled"}

    while batch.status not in terminal:
        time.sleep(max(1, settings.openai_batch_poll_seconds))
        batch = client.batches.retrieve(batch.id)
        counts = getattr(batch, "request_counts", None)
        completed = counts.get("completed") if isinstance(counts, dict) else getattr(counts, "completed", None)
        failed = counts.get("failed") if isinstance(counts, dict) else getattr(counts, "failed", None)
        logger.info(
            "batch polling batch_id=%s status=%s completed=%s failed=%s",
            batch.id,
            batch.status,
            completed,
            failed,
        )

    if batch.status != "completed" or not batch.output_file_id:
        logger.warning("resumed batch not completed status=%s batch_id=%s", batch.status, batch.id)
        return 0, 0

    output_resp = client.files.content(batch.output_file_id)
    output_text = _file_content_to_text(output_resp)
    results = _parse_batch_output_lines(output_text)

    result_ids = sorted(int(row_id) for row_id in results)
    rows = _fetch_rows_by_ids(engine=engine, ids=result_ids)
    row_map = {str(row["id"]): row for row in rows}

    payload: list[dict[str, Any]] = []
    for row_id, result in results.items():
        row = row_map.get(row_id)
        if not row or result is None:
            continue
        payload.append(_result_to_db_payload(row=row, result=result, write_batch_tag=write_batch_tag))

    _update_rows(engine=engine, payload=payload)
    logger.info(
        "llm label progress mode=resume fetched=%s updated=%s failed=%s",
        len(rows),
        len(payload),
        max(0, len(rows) - len(payload)),
    )
    return len(rows), len(payload)


def _fetch_rows(
    engine,
    limit: int,
) -> list[dict[str, Any]]:
    where = ["1=1"]
    params: dict[str, Any] = {"limit_rows": limit}
    where.append(
        "("
        "ns.expected_title_normalized IS NULL OR "
        "ns.expected_description_html IS NULL"
        ")"
    )

    query = text(
        f"""
        SELECT
            ns.id,
            ns.title_raw,
            ns.description_raw,
            ns.country_code,
            ns.language_code,
            ns.country_name,
            ns.city_title,
            ns.region_title,
            ns.review_notes
        FROM normalization_samples ns
        WHERE {' AND '.join(where)}
        ORDER BY ns.id
        LIMIT :limit_rows
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, params).mappings()
        return [dict(row) for row in rows]


def _fetch_rows_by_ids(engine, ids: list[int]) -> list[dict[str, Any]]:
    if not ids:
        return []

    query = text(
        """
        SELECT
            ns.id,
            ns.title_raw,
            ns.description_raw,
            ns.country_code,
            ns.language_code,
            ns.country_name,
            ns.city_title,
            ns.region_title,
            ns.review_notes
        FROM normalization_samples ns
        WHERE ns.id = ANY(:ids)
        ORDER BY ns.id
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"ids": ids}).mappings()
        return [dict(row) for row in rows]


def _label_row_direct(client: OpenAI, row: dict[str, Any], debug: bool = False) -> dict[str, str]:
    user_payload = {
        "prompt_version": settings.llm_prompt_version,
        "title_raw": row.get("title_raw") or "",
        "description_raw": row.get("description_raw") or "",
        "location_context": {
            "language_code": row.get("language_code"),
            "country_code": row.get("country_code"),
            "country_name": row.get("country_name"),
            "region_title": row.get("region_title"),
            "city_title": row.get("city_title"),
        },
    }

    content = ""
    if debug:
        logger.info(
            "LLM DEBUG REQUEST id=%s\nSYSTEM:\n%s\nUSER:\n%s",
            row.get("id"),
            SYSTEM_PROMPT,
            json.dumps(user_payload, ensure_ascii=False, indent=2),
        )
    for attempt in range(1, settings.openai_max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
                response_format=RESPONSE_JSON_SCHEMA,
            )
            content = (resp.choices[0].message.content or "").strip()
            if debug:
                logger.info("LLM DEBUG RESPONSE id=%s\n%s", row.get("id"), content)
            parsed = _parse_json(content)
            result = _validated_result_or_none(parsed)
            if result is None:
                raise ValueError("invalid structured output: missing or empty required fields")
            return result
        except Exception as exc:  # noqa: BLE001
            if attempt >= settings.openai_max_retries:
                logger.exception("llm labeling failed id=%s content=%s", row.get("id"), content[:300])
                raise
            sleep_seconds = min(2**attempt, 10)
            logger.warning("llm labeling retry id=%s attempt=%s err=%s", row.get("id"), attempt, exc)
            time.sleep(sleep_seconds)
    raise RuntimeError("unreachable")


def _build_batch_request(row: dict[str, Any]) -> dict[str, Any]:
    user_payload = {
        "prompt_version": settings.llm_prompt_version,
        "title_raw": row.get("title_raw") or "",
        "description_raw": row.get("description_raw") or "",
        "location_context": {
            "language_code": row.get("language_code"),
            "country_code": row.get("country_code"),
            "country_name": row.get("country_name"),
            "region_title": row.get("region_title"),
            "city_title": row.get("city_title"),
        },
    }
    return {
        "custom_id": str(row["id"]),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": settings.openai_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "response_format": RESPONSE_JSON_SCHEMA,
        },
    }


def _parse_json(content: str) -> dict[str, Any]:
    if not content:
        return {}
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).strip()
    data = json.loads(cleaned)
    if isinstance(data, dict):
        return data
    return {}


def _update_rows(engine, payload: list[dict[str, Any]]) -> None:
    if not payload:
        return
    query = text(
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
    with engine.begin() as conn:
        conn.execute(query, payload)


def _result_to_db_payload(
    *,
    row: dict[str, Any],
    result: dict[str, str],
    write_batch_tag: str | None,
) -> dict[str, Any]:
    return {
        "id": row["id"],
        "expected_title_normalized": result["title_normalized"],
        "expected_description_html": result["description_html"],
        "batch_tag": write_batch_tag,
        "review_notes": _merge_review_notes(
            existing=row.get("review_notes"),
            model=settings.openai_model,
            prompt_version=settings.llm_prompt_version,
        ),
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _file_content_to_text(resp: Any) -> str:
    text_attr = getattr(resp, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    if callable(text_attr):
        maybe = text_attr()
        if isinstance(maybe, str):
            return maybe

    read_attr = getattr(resp, "read", None)
    if callable(read_attr):
        data = read_attr()
        if isinstance(data, bytes):
            return data.decode("utf-8")
        if isinstance(data, str):
            return data

    content_attr = getattr(resp, "content", None)
    if isinstance(content_attr, bytes):
        return content_attr.decode("utf-8")
    if isinstance(content_attr, str):
        return content_attr

    return str(resp)


def _parse_batch_output_lines(text_data: str) -> dict[str, dict[str, str] | None]:
    parsed: dict[str, dict[str, str] | None] = {}
    for raw_line in text_data.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        item = json.loads(line)
        custom_id = str(item.get("custom_id") or "")
        if not custom_id:
            continue
        if item.get("error"):
            logger.warning("batch line failed id=%s error=%s", custom_id, item.get("error"))
            parsed[custom_id] = None
            continue
        body = ((item.get("response") or {}).get("body") or {})
        content = ""
        try:
            content = body["choices"][0]["message"]["content"] or ""
        except Exception:  # noqa: BLE001
            logger.warning("batch line malformed id=%s body=%s", custom_id, body)
            parsed[custom_id] = None
            continue
        json_obj = _parse_json(content)
        parsed[custom_id] = _validated_result_or_none(json_obj)
    return parsed


def _validated_result_or_none(data: dict[str, Any]) -> dict[str, str] | None:
    title = str(data.get("title_normalized") or "").strip()
    html = str(data.get("description_html") or "").strip()
    if not title or not html:
        return None
    return {
        "title_normalized": title,
        "description_html": html,
    }


def _merge_review_notes(existing: Any, model: str, prompt_version: str) -> str:
    marker = (
        f"llm_label model={model} prompt={prompt_version} "
        f"at={datetime.now(timezone.utc).isoformat(timespec='seconds')}"
    )
    base = str(existing or "").strip()
    if not base:
        return marker
    if marker in base:
        return base
    return f"{base}\n{marker}"


if __name__ == "__main__":
    run()
