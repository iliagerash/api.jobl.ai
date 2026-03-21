"""
test_process_endpoint.py
────────────────────────
Fetches random EN/FR jobs from the DB and calls POST /v1/process for each,
printing a per-row debug summary and final statistics.

Usage:
    python scripts/test_process_endpoint.py --limit 100
    python scripts/test_process_endpoint.py --limit 50 --url http://localhost:8000
    python scripts/test_process_endpoint.py --limit 50 --out report.html
"""

import argparse
import html as html_lib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import httpx
from sqlalchemy import text
from app.db.session import SessionLocal


def fetch_jobs(limit: int, countries: list[str] | None = None) -> list[dict]:
    db = SessionLocal()
    try:
        country_filter = ""
        params: dict = {"limit": limit}
        if countries:
            country_filter = "AND country_code = ANY(:countries)"
            params["countries"] = countries
        rows = db.execute(
            text(f"""
                SELECT id, title, description, category
                FROM jobs
                WHERE language_code IN ('en', 'fr')
                  AND title IS NOT NULL
                  AND description IS NOT NULL
                  {country_filter}
                ORDER BY RANDOM()
                LIMIT :limit
            """),
            params,
        ).fetchall()
    finally:
        db.close()
    return [{"id": r[0], "title": r[1], "description": r[2], "category": r[3]} for r in rows]


def _e(text: str) -> str:
    """HTML-escape a string."""
    return html_lib.escape(str(text or ""))


def generate_html(rows: list[dict], stats: dict, total_ms: float) -> str:
    n = len(rows)
    avg_ms = total_ms / n if n else 0

    job_rows_html = ""
    for r in rows:
        job = r["job"]
        data = r.get("data")
        error = r.get("error")

        if error:
            job_rows_html += f"""
            <tr class="error">
              <td colspan="2">Job {job['id']} — ERROR: {_e(error)}</td>
            </tr>"""
            continue

        cat = data.get("category")
        if cat:
            conf = cat.get("confidence")
            cat_out = f"{cat['title']} ({conf:.0%})" if conf is not None else cat["title"]
        else:
            cat_out = ""

        job_rows_html += f"""
        <tr class="job-header">
          <th>Source (id={job['id']})</th>
          <th>Processed</th>
        </tr>
        <tr>
          <td class="label">Title</td>
          <td class="label">Title (normalized)</td>
        </tr>
        <tr>
          <td>{_e(job['title'])}</td>
          <td>{_e(data['title_normalized'])}</td>
        </tr>
        <tr>
          <td class="label">Description</td>
          <td class="label">Description (cleaned)</td>
        </tr>
        <tr>
          <td class="desc">{job['description']}</td>
          <td class="desc">{data['description_clean']}</td>
        </tr>
        <tr>
          <td class="label">Category (source)</td>
          <td class="label">Category (predicted)</td>
        </tr>
        <tr>
          <td>{_e(job.get('category') or '—')}</td>
          <td>{_e(cat_out)}</td>
        </tr>
        <tr>
          <td class="label">Email</td>
          <td class="label">Email (extracted)</td>
        </tr>
        <tr>
          <td>—</td>
          <td>{_e(data.get('application_email') or '')}</td>
        </tr>
        <tr>
          <td class="label">Expiry date</td>
          <td class="label">Expiry date (extracted)</td>
        </tr>
        <tr class="job-end">
          <td>—</td>
          <td>{_e(data.get('expiry_date') or '')}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Process Endpoint Test Report</title>
<style>
  body {{ font-family: system-ui, sans-serif; font-size: 14px; margin: 24px; color: #222; }}
  h1 {{ font-size: 20px; }}
  .summary {{ background: #f5f5f5; padding: 12px 16px; border-radius: 6px; margin-bottom: 24px; display: inline-block; }}
  .summary span {{ margin-right: 24px; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 32px; }}
  th, td {{ padding: 6px 10px; text-align: left; vertical-align: top; border: 1px solid #ddd; }}
  tr.job-header th {{ background: #2c3e50; color: #fff; font-size: 13px; width: 50%; }}
  tr.job-end td {{ border-bottom: 3px solid #2c3e50; }}
  td.label {{ font-weight: 600; font-size: 12px; color: #555; background: #fafafa; }}
  td.desc {{ font-size: 12px; white-space: pre-wrap; word-break: break-word; }}
  tr.error td {{ background: #fff0f0; color: #c00; }}
</style>
</head>
<body>
<h1>Process Endpoint Test Report</h1>
<div class="summary">
  <span><b>Jobs:</b> {n}</span>
  <span><b>OK:</b> {stats['ok']}</span>
  <span><b>Errors:</b> {stats['error']}</span>
  <span><b>With email:</b> {stats['with_email']} ({stats['with_email']/n*100:.1f}%)</span>
  <span><b>With expiry:</b> {stats['with_expiry']} ({stats['with_expiry']/n*100:.1f}%)</span>
  <span><b>With category:</b> {stats['with_category']} ({stats['with_category']/n*100:.1f}%)</span>
  <span><b>Avg latency:</b> {avg_ms:.0f}ms</span>
</div>
<table>{job_rows_html}
</table>
</body>
</html>"""


def _fmt_category(cat: dict) -> str:
    conf = cat.get("confidence")
    label = cat["title"][:30]
    return f"{label} ({conf:.0%})" if conf is not None else label


def main() -> None:
    parser = argparse.ArgumentParser(description="Test /v1/process against random DB jobs")
    parser.add_argument("--limit", type=int, required=True, help="Number of random jobs to test")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL (default: http://localhost:8000)")
    parser.add_argument("--out", default=None, help="Write HTML report to this file")
    parser.add_argument("--country", default=None, help="Comma-separated country codes to filter (e.g. us,ca)")
    args = parser.parse_args()

    countries = [c.strip().upper() for c in args.country.split(",")] if args.country else None
    print(f"Fetching {args.limit} random EN/FR jobs from DB ...")
    jobs = fetch_jobs(args.limit, countries=countries)
    print(f"  {len(jobs)} rows fetched\n")

    endpoint = f"{args.url.rstrip('/')}/v1/process"
    stats = {"ok": 0, "error": 0, "with_email": 0, "with_expiry": 0, "with_category": 0}
    total_ms = 0.0
    html_rows: list[dict] = []

    with httpx.Client(timeout=30) as client:
        for i, job in enumerate(jobs, 1):
            payload = {"title": job["title"], "description": job["description"], "original_category": job.get("category")}
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

                    html_rows.append({"job": job, "data": data})
                    print(
                        f"[{i:>{len(str(len(jobs)))}}/{len(jobs)}] id={job['id']} "
                        f"{elapsed_ms:6.0f}ms | "
                        f"{job['title'][:80]!r} -> {data['title_normalized'][:80]!r} | "
                        f"email={'✓' if has_email else '✗'} "
                        f"expiry={'✓' if has_expiry else '✗'} "
                        f"category={_fmt_category(data['category']) if has_category else '✗'}"
                    )
                else:
                    stats["error"] += 1
                    error_msg = f"HTTP {resp.status_code}: {resp.text[:120]}"
                    html_rows.append({"job": job, "error": error_msg})
                    print(f"[{i}/{len(jobs)}] id={job['id']} ERROR {error_msg}")

            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                total_ms += elapsed_ms
                stats["error"] += 1
                html_rows.append({"job": job, "error": str(exc)})
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

    if args.out:
        html = generate_html(html_rows, stats, total_ms)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"HTML report written to {args.out}")


if __name__ == "__main__":
    main()
