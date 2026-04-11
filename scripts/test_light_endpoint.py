"""
test_light_endpoint.py
──────────────────────
Fetches random jobs from the DB and calls POST /v1/process with full=0
(light mode: description cleaning, email and expiry extraction only —
no title normalisation, no categorisation).

Usage:
    python scripts/test_light_endpoint.py --limit 100
    python scripts/test_light_endpoint.py --limit 50 --url http://localhost:8001
    python scripts/test_light_endpoint.py --limit 50 --country us,ca --out report_light.html
"""

import argparse
import html as html_lib
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import httpx
from sqlalchemy import text
from app.db.session import SessionLocal

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_DEBUG_CTX = 150  # characters of context to show around an email


def _debug_email_context(description: str, email: str, label: str) -> None:
    idx = description.find(email)
    if idx == -1:
        print(f"    [{label}] '{email}' not found verbatim in description")
        return
    before = description[max(0, idx - _DEBUG_CTX):idx].replace("\n", " ")
    after = description[idx + len(email):idx + len(email) + _DEBUG_CTX].replace("\n", " ")
    print(f"    [{label}] ...{before!r}  >>>>{email}<<<<  {after!r}...")


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

        job_rows_html += f"""
        <tr class="job-header">
          <th>Source (id={job['id']})</th>
          <th>Processed</th>
        </tr>
        <tr>
          <td class="label">Title</td>
          <td class="label">Title (unchanged)</td>
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
<title>Light Endpoint Test Report</title>
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
<h1>Light Endpoint Test Report</h1>
<div class="summary">
  <span><b>Jobs:</b> {n}</span>
  <span><b>OK:</b> {stats['ok']}</span>
  <span><b>Errors:</b> {stats['error']}</span>
  <span><b>With email:</b> {stats['with_email']} ({stats['with_email']/n*100:.1f}%)</span>
  <span><b>With expiry:</b> {stats['with_expiry']} ({stats['with_expiry']/n*100:.1f}%)</span>
  <span><b>Avg latency:</b> {avg_ms:.0f}ms</span>
</div>
<table>{job_rows_html}
</table>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Test /v1/process (full=0) against random DB jobs")
    parser.add_argument("--limit", type=int, required=True, help="Number of random jobs to test")
    parser.add_argument("--url", default="http://localhost:8001", help="API base URL (default: http://localhost:8001)")
    parser.add_argument("--out", default=None, help="Write HTML report to this file")
    parser.add_argument("--country", default=None, help="Comma-separated country codes to filter (e.g. us,ca)")
    parser.add_argument("--debug", action="store_true", help="Log email context (150 chars before/after) for regex matches and extracted emails")
    args = parser.parse_args()

    countries = [c.strip().upper() for c in args.country.split(",")] if args.country else None
    print(f"Fetching {args.limit} random jobs from DB ...")
    jobs = fetch_jobs(args.limit, countries=countries)
    print(f"  {len(jobs)} rows fetched\n")

    endpoint = f"{args.url.rstrip('/')}/v1/process"
    stats = {"ok": 0, "error": 0, "with_email": 0, "with_expiry": 0}
    total_ms = 0.0
    html_rows: list[dict] = []

    with httpx.Client(timeout=30) as client:
        for i, job in enumerate(jobs, 1):
            payload = {
                "title": job["title"],
                "description": job["description"],
                "original_category": job.get("category"),
                "full": 0,
            }
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
                    if has_email:
                        stats["with_email"] += 1
                    if has_expiry:
                        stats["with_expiry"] += 1

                    html_rows.append({"job": job, "data": data})
                    print(
                        f"[{i:>{len(str(len(jobs)))}}/{len(jobs)}] id={job['id']} "
                        f"{elapsed_ms:6.0f}ms | "
                        f"{job['title'][:80]!r} | "
                        f"email={'✓' if has_email else '✗'} "
                        f"expiry={'✓' if has_expiry else '✗'}"
                    )

                    if args.debug:
                        raw_desc = job["description"] or ""
                        regex_matches = _EMAIL_RE.findall(raw_desc)
                        if regex_matches:
                            for match in regex_matches:
                                _debug_email_context(raw_desc, match, "regex")
                        else:
                            print(f"    [regex] no email found in raw description")
                        extracted = data.get("application_email")
                        if extracted:
                            _debug_email_context(raw_desc, extracted, "extracted")
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
    if n == 0:
        print("No jobs fetched — check your --country code (use 2-letter ISO codes, e.g. --country=id,in)")
        return
    print(f"""
─────────────────────────────────────
Results ({n} jobs)
─────────────────────────────────────
  OK:           {stats['ok']} / {n}
  Errors:       {stats['error']}
  With email:   {stats['with_email']} ({stats['with_email']/n*100:.1f}%)
  With expiry:  {stats['with_expiry']} ({stats['with_expiry']/n*100:.1f}%)
  Avg latency:  {total_ms/n:.0f}ms
─────────────────────────────────────""")

    if args.out:
        html = generate_html(html_rows, stats, total_ms)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"HTML report written to {args.out}")


if __name__ == "__main__":
    main()
