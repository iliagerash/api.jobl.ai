import argparse
import json
import logging
import sys
from urllib import error, request

from sqlalchemy import create_engine, text

from app.config import settings


logger = logging.getLogger("jobl.test.inference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test inference API using titles from jobs table")
    parser.add_argument("--limit", type=int, default=10, help="Number of titles to test")
    parser.add_argument("--random", action="store_true", help="Select random titles instead of latest by id")
    lang_group = parser.add_mutually_exclusive_group()
    lang_group.add_argument("--lang", default=None, help="Filter by jobs.language_code (for example: en, de)")
    lang_group.add_argument("--no-lang", action="store_true", help="Do not send language_code to inference API")
    parser.add_argument("--country", default=None, help="Filter by jobs.country_code (for example: DE, FR)")
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    try:
        limit = max(1, int(args.limit))
        rows = _fetch_titles(
            limit=limit,
            randomize=bool(args.random),
            language_code=args.lang,
            country_code=args.country,
        )
        if not rows:
            logger.warning("no titles found limit=%s", limit)
            return 0

        logger.info("testing inference rows=%s base_url=%s", len(rows), settings.inference_api_base_url)
        for row in rows:
            title = row["title"]
            language_code = None if args.no_lang else row.get("language_code")
            normalized = _normalize_title(title, language_code, omit_language=bool(args.no_lang))
            safe_original = str(title or "").replace("\n", " ").strip()
            safe_normalized = str(normalized or "").replace("\n", " ").strip()
            sys.stdout.write(f"{safe_original} ||| {safe_normalized}\n")
        return 0
    except KeyboardInterrupt:
        logger.warning("interrupted by user (Ctrl+C), exiting gracefully")
        return 130


def _fetch_titles(
    limit: int,
    randomize: bool,
    language_code: str | None,
    country_code: str | None,
) -> list[dict[str, str | None]]:
    engine = create_engine(settings.target_database_url, pool_pre_ping=True)
    order_clause = "RANDOM()" if randomize else "j.id DESC"
    where_filters: list[str] = []
    params: dict[str, object] = {"limit": limit}
    if language_code:
        where_filters.append("j.language_code = :language_code")
        params["language_code"] = str(language_code).strip().lower()
    if country_code:
        where_filters.append("UPPER(j.country_code) = :country_code")
        params["country_code"] = str(country_code).strip().upper()
    where_sql = ""
    if where_filters:
        where_sql = " AND " + " AND ".join(where_filters)
    sql = text(
        """
        SELECT j.title, j.language_code
        FROM jobs j
        WHERE COALESCE(BTRIM(j.title), '') <> ''
        """
        + where_sql
        + """
        ORDER BY """
        + order_clause
        + """
        LIMIT :limit
        """
    )
    with engine.connect() as conn:
        result = conn.execute(sql, params)
        rows: list[dict[str, str | None]] = []
        for row in result:
            if row[0] is None:
                continue
            rows.append(
                {
                    "title": str(row[0]),
                    "language_code": str(row[1]).strip().lower() if row[1] is not None else None,
                }
            )
        return rows


def _normalize_title(title_raw: str, language_code: str | None, omit_language: bool) -> str:
    base = settings.inference_api_base_url.rstrip("/")
    url = f"{base}/normalize"
    payload_obj: dict[str, str | None] = {"title_raw": str(title_raw or "")}
    if not omit_language:
        payload_obj["language_code"] = str(language_code or "").strip().lower() or None
    payload = json.dumps(payload_obj).encode("utf-8")
    req = request.Request(
        url=url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=settings.inference_timeout_seconds) as resp:
            body = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        logger.warning("inference http error status=%s title_raw=%r body=%r", exc.code, title_raw, body)
        return ""
    except Exception:
        logger.exception("inference request failed title_raw=%r", title_raw)
        return ""

    try:
        data = json.loads(body)
    except Exception:
        logger.warning("inference response parse failed title_raw=%r body=%r", title_raw, body)
        return ""
    return str(data.get("title_normalized") or "")


if __name__ == "__main__":
    raise SystemExit(run())
