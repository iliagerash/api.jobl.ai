import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib import error, request

from sqlalchemy import create_engine, text

from app.config import settings


logger = logging.getLogger("jobl.test.inference.stress")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress test inference API with concurrent requests")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent requests")
    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="Repeat interval in seconds; 0 runs once",
    )
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    concurrency = max(1, int(args.concurrency))
    repeat_seconds = max(0, int(args.repeat))
    titles = _fetch_titles(concurrency=concurrency)
    if not titles:
        logger.warning("no titles found for stress test language_code=en limit=%s", concurrency)
        return 0

    logger.info(
        "stress test started concurrency=%s repeat_seconds=%s base_url=%s rows=%s",
        concurrency,
        repeat_seconds,
        settings.inference_api_base_url,
        len(titles),
    )

    round_no = 0
    try:
        while True:
            round_no += 1
            started = time.perf_counter()
            _run_round(titles=titles, concurrency=concurrency)
            latency_ms = (time.perf_counter() - started) * 1000
            logger.info("stress round completed round=%s latency_ms=%.2f", round_no, latency_ms)
            if repeat_seconds <= 0:
                break
            time.sleep(repeat_seconds)
        return 0
    except KeyboardInterrupt:
        logger.warning("interrupted by user (Ctrl+C), exiting gracefully")
        return 130


def _fetch_titles(concurrency: int) -> list[str]:
    engine = create_engine(settings.target_database_url, pool_pre_ping=True)
    sql = text(
        """
        SELECT j.title
        FROM jobs j
        WHERE COALESCE(BTRIM(j.title), '') <> ''
          AND j.language_code = 'en'
        ORDER BY RANDOM()
        LIMIT :limit
        """
    )
    with engine.connect() as conn:
        result = conn.execute(sql, {"limit": concurrency})
        return [str(row[0]) for row in result if row[0] is not None]


def _run_round(titles: list[str], concurrency: int) -> None:
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(_normalize_title, title) for title in titles]
        for future in as_completed(futures):
            original_title, normalized_title = future.result()
            safe_original = str(original_title or "").replace("\n", " ").strip()
            safe_normalized = str(normalized_title or "").replace("\n", " ").strip()
            sys.stdout.write(f"{safe_original} ||| {safe_normalized}\n")


def _normalize_title(title_raw: str) -> tuple[str, str]:
    base = settings.inference_api_base_url.rstrip("/")
    url = f"{base}/normalize"
    payload = json.dumps({"title_raw": str(title_raw or ""), "language_code": "en"}).encode("utf-8")
    req = request.Request(
        url=url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=settings.inference_timeout_seconds) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)
        return title_raw, str(data.get("title_normalized") or "")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        logger.warning("inference http error status=%s title_raw=%r body=%r", exc.code, title_raw, body)
        return title_raw, ""
    except Exception:
        logger.exception("inference request failed title_raw=%r", title_raw)
        return title_raw, ""


if __name__ == "__main__":
    raise SystemExit(run())
