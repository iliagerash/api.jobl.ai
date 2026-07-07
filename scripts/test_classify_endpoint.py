"""
test_classify_endpoint.py
─────────────────────────
Fetches random resumes from the DB and calls POST /v1/classify for each,
printing position, a description snippet, and the resulting doc_type.

Usage:
    python scripts/test_classify_endpoint.py --limit 50
    python scripts/test_classify_endpoint.py --limit 100 --country au
    python scripts/test_classify_endpoint.py --limit 200 --country us,ca --url http://localhost:8001
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import httpx
from sqlalchemy import text
from app.db.session import SessionLocal

# Colour codes — fall back gracefully if the terminal doesn't support them
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_CYAN   = "\033[36m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_DIM    = "\033[2m"

_DOC_TYPE_COLOUR = {
    "resume":       _GREEN,
    "cover_letter": _CYAN,
    "stub":         _YELLOW,
    "other":        _RED,
}

DESC_SNIPPET_LEN = 200


def fetch_resumes(limit: int, countries: list[str] | None) -> list[dict]:
    db = SessionLocal()
    try:
        country_filter = ""
        params: dict = {"limit": limit}
        if countries:
            country_filter = "AND country_code = ANY(:countries)"
            params["countries"] = countries
        rows = db.execute(
            text(f"""
                SELECT id, title, description, country_code
                FROM resumes
                WHERE title IS NOT NULL
                  AND description IS NOT NULL
                  {country_filter}
                ORDER BY RANDOM()
                LIMIT :limit
            """),
            params,
        ).fetchall()
    finally:
        db.close()
    return [
        {"id": r[0], "title": r[1], "description": r[2], "country_code": r[3]}
        for r in rows
    ]


def _snippet(text: str | None) -> str:
    if not text:
        return ""
    clean = " ".join(text.split())
    return clean[:DESC_SNIPPET_LEN] + ("…" if len(clean) > DESC_SNIPPET_LEN else "")


def _colour(doc_type: str) -> str:
    return _DOC_TYPE_COLOUR.get(doc_type, "") + _BOLD + doc_type.upper() + _RESET


def main() -> None:
    parser = argparse.ArgumentParser(description="Test POST /v1/classify against random DB resumes")
    parser.add_argument("--limit", type=int, required=True, help="Number of random resumes to test")
    parser.add_argument("--country", default=None, help="Comma-separated country codes, e.g. us,au")
    parser.add_argument("--url", default="http://localhost:8001", help="API base URL (default: http://localhost:8001)")
    args = parser.parse_args()

    countries = [c.strip().upper() for c in args.country.split(",")] if args.country else None
    print(f"Fetching {args.limit} random resumes from DB ...")
    resumes = fetch_resumes(args.limit, countries=countries)

    if not resumes:
        print("No resumes found — check your --country code (e.g. --country=us)")
        sys.exit(1)

    print(f"  {len(resumes)} rows fetched\n")

    endpoint = f"{args.url.rstrip('/')}/v1/classify"
    counts: dict[str, int] = {"resume": 0, "cover_letter": 0, "stub": 0, "other": 0, "error": 0}
    total_ms = 0.0
    width = len(str(len(resumes)))

    with httpx.Client(timeout=10) as client:
        for i, row in enumerate(resumes, 1):
            payload = {"title": row["title"], "description": row["description"]}
            t0 = time.perf_counter()
            try:
                resp = client.post(endpoint, json=payload)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                total_ms += elapsed_ms

                if resp.status_code == 200:
                    doc_type = resp.json()["doc_type"]
                    counts[doc_type] = counts.get(doc_type, 0) + 1

                    print(
                        f"[{i:{width}}/{len(resumes)}] {elapsed_ms:4.0f}ms  "
                        f"{_colour(doc_type):<30}  "
                        f"{_BOLD}{row['title'][:60]}{_RESET}"
                    )
                    print(
                        f"{'':>{width + 14}}"
                        f"{_DIM}{_snippet(row['description'])}{_RESET}"
                    )
                    print()
                else:
                    counts["error"] += 1
                    print(f"[{i:{width}}/{len(resumes)}] ERROR HTTP {resp.status_code}: {resp.text[:80]}")

            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                total_ms += elapsed_ms
                counts["error"] += 1
                print(f"[{i:{width}}/{len(resumes)}] EXCEPTION: {exc}")

    n = len(resumes)
    avg_ms = total_ms / n if n else 0

    print("─" * 52)
    print(f"Results ({n} resumes)  avg latency: {avg_ms:.1f}ms")
    print("─" * 52)
    for doc_type in ("resume", "cover_letter", "stub", "other"):
        c = counts[doc_type]
        bar = "█" * int(c / n * 30) if n else ""
        print(f"  {doc_type:<14} {c:>5}  ({c/n*100:5.1f}%)  {bar}")
    if counts["error"]:
        print(f"  {'errors':<14} {counts['error']:>5}")
    print("─" * 52)


if __name__ == "__main__":
    main()
