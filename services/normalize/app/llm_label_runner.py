import argparse
import html
import json
import logging
import re
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
- Return ONLY the cleaned role title in title_normalized.
- Keep it concise, typically 1-4 words.
- Do NOT include salaries, dates, addresses, company names, brand names, city/region/country fragments, store/location IDs, or hiring noise.
- Do NOT include schedule/employment qualifiers in title_normalized (e.g. part-time, full-time, temporary, contract, seasonal, internship, remote, hybrid, on-site).
- Do NOT include trailing descriptors separated by "-", "|", ",", ":" or "/" unless they are part of the role itself.
- If company_name is provided and appears in the title, remove it from title_normalized.
- You may use both title_raw and description_raw as context when deciding title_normalized.
- Preserve legal title markers like (m/w/d), (m/f/d), (w/m/d), when present.
- title_normalized must be role-focused and readable to candidates.
- If uncertain, choose the most likely role noun phrase and drop non-role text.
- If uncertain, preserve information instead of hallucinating.

Examples:
Input:
- title_raw: "Kiehl's since 1851 - Tigard, OR - Part-time Skincare Sales"
- company_name: "Kiehl's since 1851"
Output:
- title_normalized: "Keyholder"

Input:
- title_raw: "Senior Accountant (m/f/d) - Berlin - Full-time"
- company_name: "Acme GmbH"
Output:
- title_normalized: "Senior Accountant (m/f/d)"

Input:
- title_raw: "Nurse Practitioner | Phoenix AZ | $120k-$140k"
Output:
- title_normalized: "Nurse Practitioner"

Input:
- title_raw: "Software Engineer - Remote"
Output:
- title_normalized: "Software Engineer"

Return strict JSON with keys:
{
  "title_normalized": string
}
""".strip()

RESPONSE_JSON_SCHEMA = {
    "type": "json_schema",
    "name": "job_normalization",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "title_normalized": {"type": "string"},
        },
        "required": ["title_normalized"],
        "additionalProperties": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process normalization_samples titles with OpenAI Flex processing")
    parser.add_argument("--batch-tag", type=str, default=None, help="Set/overwrite batch_tag for processed rows")
    parser.add_argument("--limit", type=int, default=200, help="Maximum rows to process in this run")
    parser.add_argument("--batch-size", type=int, default=20, help="Rows fetched per DB query page")
    parser.add_argument(
        "--random",
        action="store_true",
        help="Select pending rows in random order instead of id order",
    )
    parser.add_argument(
        "--service-tier",
        choices=["flex", "auto", "default", "priority"],
        default="flex",
        help="OpenAI service tier used for title processing requests (default: flex)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print full prompt and full raw response",
    )
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    configure_logging(settings.log_level)

    if not settings.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is required for jobl-normalize-process-titles")

    client = OpenAI(api_key=settings.openai_api_key, timeout=settings.openai_timeout_seconds)
    engine = create_engine(settings.target_database_url, pool_pre_ping=True)
    logger.info(
        "llm label started mode=flex_direct model=%s prompt_version=%s service_tier=%s limit=%s batch_size=%s overwrite=%s batch_tag=%s",
        settings.openai_model,
        settings.llm_prompt_version,
        args.service_tier,
        args.limit,
        args.batch_size,
        False,
        args.batch_tag,
    )
    if args.random:
        logger.info("row selection order=random")

    try:
        try:
            processed, updated = _run_direct_mode(
                engine=engine,
                client=client,
                write_batch_tag=args.batch_tag,
                limit=args.limit,
                batch_size=args.batch_size,
                debug=args.debug,
                random_order=args.random,
                service_tier=args.service_tier,
            )
        except KeyboardInterrupt:
            logger.warning("interrupted by user (Ctrl+C), exiting gracefully")
            return 130
    finally:
        engine.dispose()

    logger.info("llm label completed processed=%s updated=%s", processed, updated)
    return 0


def _run_direct_mode(
    *,
    engine,
    client: OpenAI,
    write_batch_tag: str | None,
    limit: int,
    batch_size: int,
    debug: bool,
    random_order: bool,
    service_tier: str,
) -> tuple[int, int]:
    processed = 0
    updated = 0

    while processed < limit:
        remaining = limit - processed
        fetch_size = min(batch_size, remaining)
        rows = _fetch_rows(
            engine=engine,
            limit=fetch_size,
            random_order=random_order,
        )
        if not rows:
            break

        payload: list[dict[str, Any]] = []
        for row in rows:
            result = _label_row_direct(client=client, row=row, debug=debug, service_tier=service_tier)
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
    random_order: bool,
) -> tuple[int, int]:
    rows = _fetch_rows(
        engine=engine,
        limit=limit,
        random_order=random_order,
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
            endpoint="/v1/responses",
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
    failed_ids: list[str] = []
    for row_id, result in results.items():
        row = row_map.get(row_id)
        if not row or result is None:
            failed_ids.append(row_id)
            continue
        payload.append(_result_to_db_payload(row=row, result=result, write_batch_tag=write_batch_tag))

    _update_rows(engine=engine, payload=payload)
    if failed_ids:
        logger.warning(
            "resume failed row ids count=%s ids=%s",
            len(failed_ids),
            ",".join(sorted(failed_ids, key=lambda v: int(v))[:200]),
        )
    logger.info(
        "llm label progress mode=resume fetched=%s updated=%s failed=%s",
        len(rows),
        len(payload),
        len(failed_ids),
    )
    return len(rows), len(payload)


def _fetch_rows(
    engine,
    limit: int,
    random_order: bool = False,
) -> list[dict[str, Any]]:
    where = ["1=1"]
    params: dict[str, Any] = {"limit_rows": limit}
    where.append(
        "ns.expected_title_normalized IS NULL"
    )

    query = text(
        f"""
        SELECT
            ns.id,
            ns.title,
            ns.description,
            ns.company_name,
            ns.country_code,
            ns.language_code,
            ns.country_name,
            ns.city_title,
            ns.region_title,
            ns.review_notes
        FROM normalization_samples ns
        WHERE {' AND '.join(where)}
        ORDER BY {"RANDOM()" if random_order else "ns.id"}
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
            ns.title,
            ns.description,
            ns.company_name,
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


def _label_row_direct(
    client: OpenAI,
    row: dict[str, Any],
    debug: bool = False,
    service_tier: str = "flex",
) -> dict[str, str]:
    user_payload = {
        "prompt_version": settings.llm_prompt_version,
        "title_raw": row.get("title") or "",
        "description_raw": _sanitize_description_for_llm(row.get("description")),
        "company_name": row.get("company_name") or "",
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
            resp = client.responses.create(
                model=settings.openai_model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
                text={"format": RESPONSE_JSON_SCHEMA},
                service_tier=service_tier,
            )
            content = _extract_text_from_responses_object(resp).strip()
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
        "title_raw": row.get("title") or "",
        "description_raw": _sanitize_description_for_llm(row.get("description")),
        "company_name": row.get("company_name") or "",
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
        "url": "/v1/responses",
        "body": {
            "model": settings.openai_model,
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "text": {"format": RESPONSE_JSON_SCHEMA},
        },
    }


def _parse_json(content: str) -> dict[str, Any]:
    if not content:
        return {}

    candidates: list[str] = [content]
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).strip()
    candidates.append(cleaned)

    # Try to recover the first JSON object from content that has trailing text/noise.
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(cleaned):
        if ch != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(cleaned[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj

    for candidate in candidates:
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
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
            batch_tag = COALESCE(:batch_tag, batch_tag),
            review_notes = :review_notes,
            updated_at = NOW()
        WHERE id = :id
        """
    )
    try:
        with engine.begin() as conn:
            conn.execute(query, payload)
        return
    except Exception:  # noqa: BLE001
        logger.exception("bulk update failed; retrying row-by-row rows=%s", len(payload))

    success = 0
    failed = 0
    for row in payload:
        try:
            with engine.begin() as conn:
                conn.execute(query, [row])
            success += 1
        except Exception:  # noqa: BLE001
            failed += 1
            logger.exception("row update failed id=%s", row.get("id"))

    logger.warning("row-by-row update completed success=%s failed=%s", success, failed)


def _result_to_db_payload(
    *,
    row: dict[str, Any],
    result: dict[str, str],
    write_batch_tag: str | None,
) -> dict[str, Any]:
    return {
        "id": row["id"],
        "expected_title_normalized": _sanitize_text_for_postgres(result["title_normalized"]),
        "batch_tag": write_batch_tag,
        "review_notes": _sanitize_text_for_postgres(
            _merge_review_notes(
                existing=row.get("review_notes"),
                model=settings.openai_model,
                prompt_version=settings.llm_prompt_version,
            )
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
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("batch output line is not valid json; skipping line=%s", line[:200])
            continue
        custom_id = str(item.get("custom_id") or "")
        if not custom_id:
            continue
        if item.get("error"):
            logger.warning("batch line failed id=%s error=%s", custom_id, item.get("error"))
            parsed[custom_id] = None
            continue
        body = ((item.get("response") or {}).get("body") or {})
        content = _extract_text_from_response_body(body)
        if not content:
            logger.warning("batch line malformed id=%s body=%s", custom_id, body)
            parsed[custom_id] = None
            continue
        try:
            json_obj = _parse_json(content)
        except Exception:  # noqa: BLE001
            logger.warning("batch content parse failed id=%s", custom_id)
            parsed[custom_id] = None
            continue
        parsed[custom_id] = _validated_result_or_none(json_obj)
    return parsed


def _extract_text_from_responses_object(resp: Any) -> str:
    # Preferred fast path in SDK.
    text_value = getattr(resp, "output_text", None)
    if isinstance(text_value, str) and text_value.strip():
        return text_value

    # Fallback: parse object dump and reuse body extractor.
    dump = None
    model_dump = getattr(resp, "model_dump", None)
    if callable(model_dump):
        try:
            dump = model_dump()
        except Exception:  # noqa: BLE001
            dump = None
    if isinstance(dump, dict):
        return _extract_text_from_response_body(dump)
    return ""


def _extract_text_from_response_body(body: Any) -> str:
    if not isinstance(body, dict):
        return ""

    # Responses API format.
    output_text = body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output = body.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content_items = item.get("content")
            if not isinstance(content_items, list):
                continue
            for c in content_items:
                if not isinstance(c, dict):
                    continue
                txt = c.get("text")
                if isinstance(txt, str) and txt:
                    parts.append(txt)
        if parts:
            return "\n".join(parts).strip()
    return ""


def _validated_result_or_none(data: dict[str, Any]) -> dict[str, str] | None:
    title = str(data.get("title_normalized") or "").strip()
    if not title:
        return None
    return {
        "title_normalized": title,
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


def _sanitize_text_for_postgres(value: Any) -> str:
    # PostgreSQL text/varchar cannot contain NUL (0x00) bytes.
    text_value = str(value or "")
    if "\x00" in text_value:
        return text_value.replace("\x00", "")
    return text_value


def _sanitize_input_for_llm(value: Any) -> str:
    text_value = str(value or "")
    if "\x00" in text_value:
        text_value = text_value.replace("\x00", "")
    return text_value


def _sanitize_description_for_llm(value: Any) -> str:
    text_value = _sanitize_input_for_llm(value)
    text_value = html.unescape(text_value)
    text_value = re.sub(
        r"<\s*script\b[^>]*>.*?<\s*/\s*script\s*>",
        " ",
        text_value,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text_value = re.sub(
        r"<\s*style\b[^>]*>.*?<\s*/\s*style\s*>",
        " ",
        text_value,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text_value = re.sub(r"<!--.*?-->", " ", text_value, flags=re.DOTALL)
    text_value = re.sub(r"<[^>]+>", " ", text_value)
    text_value = re.sub(r"\s+", " ", text_value).strip()
    return text_value


if __name__ == "__main__":
    run()
