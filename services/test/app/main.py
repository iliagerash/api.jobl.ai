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
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    limit = max(1, int(args.limit))
    rows = _fetch_titles(limit=limit, randomize=bool(args.random))
    if not rows:
        logger.warning("no titles found limit=%s", limit)
        return 0

    logger.info("testing inference rows=%s base_url=%s", len(rows), settings.inference_api_base_url)
    for title in rows:
        normalized = _normalize_title(title)
        safe_original = str(title or "").replace("\n", " ").strip()
        safe_normalized = str(normalized or "").replace("\n", " ").strip()
        sys.stdout.write(f"{safe_original} ||| {safe_normalized}\n")
    return 0


def _fetch_titles(limit: int, randomize: bool) -> list[str]:
    engine = create_engine(settings.target_database_url, pool_pre_ping=True)
    order_clause = "RANDOM()" if randomize else "j.id DESC"
    sql = text(
        """
        SELECT j.title
        FROM jobs j
        WHERE COALESCE(BTRIM(j.title), '') <> ''
        ORDER BY """
        + order_clause
        + """
        LIMIT :limit
        """
    )
    with engine.connect() as conn:
        result = conn.execute(sql, {"limit": limit})
        return [str(row[0]) for row in result if row[0] is not None]


def _normalize_title(title_raw: str) -> str:
    base = settings.inference_api_base_url.rstrip("/")
    url = f"{base}/normalize"
    payload = json.dumps({"title_raw": str(title_raw or "")}).encode("utf-8")
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
