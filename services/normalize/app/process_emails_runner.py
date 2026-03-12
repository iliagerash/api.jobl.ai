import argparse
import html
import logging
import re
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import create_engine, text

from app.config import settings
from app.logging import configure_logging


logger = logging.getLogger("jobl.normalize.process_emails")

EMAIL_STUB = "***email_hidden***"
MAILTO_RE = re.compile(
    r"(?i)(mailto\s*:\s*)([A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,})(\?[^\"'\s>]*)?"
)
EMAIL_RE = re.compile(r"(?i)(?<![\w.%+\-])([A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,})(?![\w.\-])")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and mask emails in normalization_samples descriptions")
    parser.add_argument("--limit", type=int, default=0, help="Maximum rows to process in this run, 0 means all")
    parser.add_argument("--batch-size", type=int, default=200, help="Rows fetched per DB page")
    parser.add_argument("--batch-tag", type=str, default=None, help="Process only one normalization_samples batch_tag")
    parser.add_argument("--random", action="store_true", help="Select rows in random order")
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    configure_logging(settings.log_level)

    limit = max(0, int(args.limit))
    batch_size = max(1, int(args.batch_size))

    engine = create_engine(settings.target_database_url, pool_pre_ping=True)
    logger.info(
        "email processing started limit=%s batch_size=%s batch_tag=%s random=%s",
        "all" if limit == 0 else limit,
        batch_size,
        args.batch_tag,
        args.random,
    )

    processed = 0
    updated = 0
    extracted_total = 0
    try:
        try:
            while True:
                if limit > 0 and processed >= limit:
                    break
                remaining = max(0, limit - processed) if limit > 0 else batch_size
                rows = _fetch_rows(
                    engine=engine,
                    limit=min(batch_size, remaining),
                    batch_tag=args.batch_tag,
                    random_order=args.random,
                )
                if not rows:
                    break

                sample_updates: list[dict[str, Any]] = []

                for row in rows:
                    description = str(row.get("description") or "")
                    masked_description, emails = _extract_and_mask_emails(description)

                    sample_updates.append(
                        {
                            "id": int(row["id"]),
                            "email": _sanitize_text(", ".join(emails)) if emails else "",
                            "description": _sanitize_text(masked_description),
                            "review_notes": _sanitize_text(
                                _merge_review_notes(
                                    existing=row.get("review_notes"),
                                    extracted_count=len(emails),
                                )
                            ),
                        }
                    )
                    extracted_total += len(emails)

                _update_samples(engine=engine, payload=sample_updates)

                processed += len(rows)
                updated += len(sample_updates)
                logger.info(
                    "email processing progress fetched=%s processed=%s/%s updated=%s extracted_emails=%s",
                    len(rows),
                    processed,
                    "all" if limit == 0 else limit,
                    updated,
                    extracted_total,
                )
        except KeyboardInterrupt:
            logger.warning("interrupted by user (Ctrl+C), exiting gracefully")
            return 130
    finally:
        engine.dispose()

    logger.info(
        "email processing completed processed=%s updated=%s extracted_emails=%s",
        processed,
        updated,
        extracted_total,
    )
    return 0


def _fetch_rows(engine, *, limit: int, batch_tag: str | None, random_order: bool) -> list[dict[str, Any]]:
    where = [
        "ns.email IS NULL",
        "COALESCE(BTRIM(ns.description), '') <> ''",
        "(POSITION('@' IN ns.description) > 0 OR ns.description ILIKE '%mailto:%')",
        "ns.description NOT LIKE :stub_token",
    ]
    params: dict[str, Any] = {
        "limit_rows": limit,
        "stub_token": f"%{EMAIL_STUB}%",
    }
    if batch_tag:
        where.append("ns.batch_tag = :batch_tag")
        params["batch_tag"] = batch_tag

    query = text(
        f"""
        SELECT
            ns.id,
            ns.description,
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


def _update_samples(engine, *, payload: list[dict[str, Any]]) -> None:
    if not payload:
        return
    query = text(
        """
        UPDATE normalization_samples
        SET email = :email,
            description = :description,
            review_notes = :review_notes,
            updated_at = NOW()
        WHERE id = :id
        """
    )
    with engine.begin() as conn:
        conn.execute(query, payload)


def _extract_and_mask_emails(value: str) -> tuple[str, list[str]]:
    decoded = html.unescape(value or "")
    found: list[str] = []
    seen: set[str] = set()

    def track(raw_email: str) -> str:
        normalized = _normalize_email(raw_email)
        if normalized and normalized not in seen:
            seen.add(normalized)
            found.append(normalized)
        return EMAIL_STUB

    def replace_mailto(match: re.Match[str]) -> str:
        _prefix, email_value, suffix = match.group(1), match.group(2), match.group(3) or ""
        track(email_value)
        return f"mailto:{EMAIL_STUB}{suffix}"

    masked = MAILTO_RE.sub(replace_mailto, decoded)

    def replace_email(match: re.Match[str]) -> str:
        email_value = match.group(1)
        return track(email_value)

    masked = EMAIL_RE.sub(replace_email, masked)
    return masked, found


def _normalize_email(value: str) -> str:
    email = str(value or "").strip().strip("<>()[]{}\"'.,;:")
    email = email.lower()
    if "?" in email:
        email = email.split("?", 1)[0].strip()
    if not email or len(email) > 320:
        return ""
    if not re.fullmatch(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", email):
        return ""
    return email


def _merge_review_notes(*, existing: Any, extracted_count: int) -> str:
    marker = (
        f"email_mask extracted={extracted_count} "
        f"at={datetime.now(timezone.utc).isoformat(timespec='seconds')}"
    )
    base = str(existing or "").strip()
    if not base:
        return marker
    return f"{base}\n{marker}"


def _sanitize_text(value: Any) -> str:
    text_value = str(value or "")
    if "\x00" in text_value:
        text_value = text_value.replace("\x00", "")
    return text_value


if __name__ == "__main__":
    raise SystemExit(run())
