"""
test_process_endpoint.py
────────────────────────
Fetches random EN/FR jobs from the DB and calls POST /v1/process for each,
printing a per-row debug summary and final statistics.

Usage:
    python scripts/test_process_endpoint.py --limit 100
    python scripts/test_process_endpoint.py --limit 50 --url http://localhost:8000
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import httpx
from sqlalchemy import text
from app.db.session import SessionLocal


def fetch_jobs(limit: int) -> list[dict]:
    db = SessionLocal()
    try:
        rows = db.execute(
            text("""
                SELECT id, title, description
                FROM jobs
                WHERE language_code IN ('en', 'fr')
                  AND title IS NOT NULL
                  AND description IS NOT NULL
                ORDER BY RANDOM()
                LIMIT :limit
            """),
            {"limit": limit},
        ).fetchall()
    finally:
        db.close()
    return [{"id": r[0], "title": r[1], "description": r[2]} for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Test /v1/process against random DB jobs")
    parser.add_argument("--limit", type=int, required=True, help="Number of random jobs to test")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL (default: http://localhost:8000)")
    args = parser.parse_args()

    print(f"Fetching {args.limit} random EN/FR jobs from DB ...")
    jobs = fetch_jobs(args.limit)
    print(f"  {len(jobs)} rows fetched\n")

    endpoint = f"{args.url.rstrip('/')}/v1/process"
    stats = {"ok": 0, "error": 0, "with_email": 0, "with_expiry": 0, "with_category": 0}
    total_ms = 0.0

    with httpx.Client(timeout=30) as client:
        for i, job in enumerate(jobs, 1):
            payload = {"title": job["title"], "description": job["description"]}
            t0 = time.perf_counter()
            try:
                resp = client.post(endpoint, json=payload)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                total_ms += elapsed_ms

                if resp.status_code == 200:
                    stats["ok"] += 1
                    data = resp.json()
                    has_email = bool(data.get("application_email"))
                    has_expiry = bool(data.get("expiry_date"))
                    has_category = data.get("category") is not None
                    if has_email:
                        stats["with_email"] += 1
                    if has_expiry:
                        stats["with_expiry"] += 1
                    if has_category:
                        stats["with_category"] += 1

                    print(
                        f"[{i:>{len(str(len(jobs)))}}/{len(jobs)}] id={job['id']} "
                        f"{elapsed_ms:6.0f}ms | "
                        f"{job['title'][:50]!r} -> {data['title_normalized'][:50]!r} | "
                        f"email={'✓' if has_email else '✗'} "
                        f"expiry={'✓' if has_expiry else '✗'} "
                        f"category={data['category']['title'][:30] if has_category else '✗'}"
                    )
                else:
                    stats["error"] += 1
                    print(f"[{i}/{len(jobs)}] id={job['id']} ERROR {resp.status_code}: {resp.text[:120]}")

            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                total_ms += elapsed_ms
                stats["error"] += 1
                print(f"[{i}/{len(jobs)}] id={job['id']} EXCEPTION: {exc}")

    n = len(jobs)
    print(f"""
─────────────────────────────────────
Results ({n} jobs)
─────────────────────────────────────
  OK:           {stats['ok']} / {n}
  Errors:       {stats['error']}
  With email:   {stats['with_email']} ({stats['with_email']/n*100:.1f}%)
  With expiry:  {stats['with_expiry']} ({stats['with_expiry']/n*100:.1f}%)
  With category:{stats['with_category']} ({stats['with_category']/n*100:.1f}%)
  Avg latency:  {total_ms/n:.0f}ms
─────────────────────────────────────""")


if __name__ == "__main__":
    main()
