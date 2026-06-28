"""
parse_ga.py
────────────
Parse GA4 export CSVs and join with jobl job metadata via destination_job_id.
Produces a unified dataset ready for profitability feature extraction.

The CSV files are one per domain, each containing aggregated sessions/revenue
per job page for the full export period.

Usage:
    python -u ml/scripts/parse_ga.py
    python -u ml/scripts/parse_ga.py --ga-dir data/ga_exports
    python -u ml/scripts/parse_ga.py --min-sessions 20

Outputs:
    ml/data/profitability/ga_joined.parquet
    ml/data/profitability/ga_stats.json
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

GA_DIR = "data/ga_exports"
OUTPUT_DIR = "ml/data/profitability"

JOB_PATH_RE = re.compile(r"^/jobs/view/(\d+)$")

# Map CSV filename (destination domain) to country code
DOMAIN_COUNTRY = {
    "au.workus.org": "AU",
    "australia.jobradars.com": "AU",
    "ca.workus.org": "CA",
    "ca.snapjobsearch.com": "CA",
    "canada.jobradars.com": "CA",
    "nz.workus.org": "NZ",
    "newzealand.jobradars.com": "NZ",
    "sg.workus.org": "SG",
    "singapore.jobradars.com": "SG",
    "uk.workus.org": "GB",
    "unitedkingdom.jobradars.com": "GB",
    "za.workus.org": "ZA",
    "jobspire.co.za": "ZA",
    "workus.org": "US",
    "jobradars.com": "US",
    "snapjobsearch.com": "US",
}


def parse_csv(csv_path: str) -> list[dict]:
    """Parse a single GA export CSV, extract job_id from landing page path."""
    destination = os.path.basename(csv_path).replace(".csv", "")
    country = DOMAIN_COUNTRY.get(destination)
    if not country:
        print(f"  WARNING: unknown domain {destination}, skipping")
        return []

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row.get("Landing page", "")
            m = JOB_PATH_RE.match(path)
            if not m:
                continue

            sessions = int(row.get("Sessions", 0))
            revenue = float(row.get("Total revenue", 0))

            rows.append({
                "destination_job_id": int(m.group(1)),
                "destination": destination,
                "country_code": country,
                "total_sessions": sessions,
                "active_users": int(row.get("Active users", 0)),
                "avg_engagement_time": float(row.get("Average engagement time per session", 0)),
                "total_revenue": revenue,
            })

    return rows


def main():
    parser = argparse.ArgumentParser(description="Parse GA exports and join with jobl jobs")
    parser.add_argument("--ga-dir", default=GA_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--min-sessions", type=int, default=0)
    parser.add_argument("--db-url", default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Parse all GA CSVs
    csv_files = sorted(glob.glob(os.path.join(args.ga_dir, "*.csv")))
    if not csv_files:
        sys.exit(f"No CSV files found in {args.ga_dir}/")

    print(f"Parsing {len(csv_files)} GA export files ...")
    all_rows = []
    for csv_path in csv_files:
        rows = parse_csv(csv_path)
        domain = os.path.basename(csv_path).replace(".csv", "")
        print(f"  {domain}: {len(rows):,} job pages")
        all_rows.extend(rows)

    ga_df = pd.DataFrame(all_rows)
    print(f"\nTotal: {len(ga_df):,} job page rows")

    # 2. Compute RPS
    ga_df["rps"] = ga_df["total_revenue"] / ga_df["total_sessions"].clip(lower=1)

    # 3. Apply minimum session threshold
    before = len(ga_df)
    ga_df = ga_df[ga_df["total_sessions"] >= args.min_sessions]
    print(f"Session filter (>={args.min_sessions}): {before:,} → {len(ga_df):,}")

    # 4. Join with jobl jobs
    db_url = args.db_url
    if not db_url:
        from dotenv import load_dotenv
        load_dotenv()
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            sys.exit("No DATABASE_URL")

    from sqlalchemy import create_engine, text
    engine = create_engine(db_url, pool_pre_ping=True)

    print("\nLoading job metadata from jobl ...")
    query = text("""
        SELECT id, title, company_name, city_title, region_title, country_code,
               category_id, is_remote, contract, description,
               salary_min, salary_max, salary_period,
               published_at, destination, destination_job_id
        FROM jobs
        WHERE destination IS NOT NULL
          AND destination_job_id IS NOT NULL
    """)

    with engine.connect() as conn:
        jobs_df = pd.DataFrame(conn.execute(query).mappings())

    engine.dispose()
    print(f"  {len(jobs_df):,} jobs with destination info")

    # Join GA data with jobl jobs
    merged = ga_df.merge(
        jobs_df,
        on=["destination", "destination_job_id"],
        how="inner",
        suffixes=("_ga", "_jobl"),
    )

    # Use jobl country_code if available, fall back to GA-derived
    if "country_code_jobl" in merged.columns:
        merged["country_code"] = merged["country_code_jobl"].fillna(merged["country_code_ga"])
        merged.drop(columns=["country_code_ga", "country_code_jobl"], inplace=True)

    print(f"\nJoined: {len(merged):,} rows (GA ∩ jobl)")

    # 5. Stats
    stats = {
        "total_ga_rows": len(all_rows),
        "after_session_filter": len(ga_df),
        "after_join": len(merged),
        "min_sessions": args.min_sessions,
        "per_country": {},
        "per_destination": {},
    }

    for cc in sorted(merged["country_code"].unique()):
        cc_df = merged[merged["country_code"] == cc]
        stats["per_country"][cc] = {
            "jobs": int(len(cc_df)),
            "total_sessions": int(cc_df["total_sessions"].sum()),
            "total_revenue": float(cc_df["total_revenue"].sum()),
            "median_sessions": int(cc_df["total_sessions"].median()),
            "median_revenue": float(cc_df["total_revenue"].median()),
            "median_rps": float(cc_df["rps"].median()),
        }

    for dest in sorted(merged["destination"].unique()):
        dest_df = merged[merged["destination"] == dest]
        stats["per_destination"][dest] = {
            "jobs": int(len(dest_df)),
            "matched_pct": round(len(dest_df) / max(len(ga_df[ga_df["destination"] == dest]), 1) * 100, 1),
        }

    # Save
    output_path = os.path.join(args.output_dir, "ga_joined.parquet")
    merged.to_parquet(output_path, index=False)
    print(f"\nSaved: {output_path} ({len(merged):,} rows)")

    stats_path = os.path.join(args.output_dir, "ga_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {stats_path}")

    # Summary
    print(f"\nPer-country summary:")
    print(f"{'Country':<8} {'Jobs':>8} {'Sessions':>10} {'Revenue':>10} {'Med RPS':>8}")
    for cc, s in sorted(stats["per_country"].items()):
        print(f"{cc:<8} {s['jobs']:>8,} {s['total_sessions']:>10,} {s['total_revenue']:>10,.0f} {s['median_rps']:>8.3f}")

    print(f"\nPer-destination match rate:")
    for dest, s in sorted(stats["per_destination"].items()):
        print(f"  {dest}: {s['jobs']:,} matched ({s['matched_pct']}%)")


if __name__ == "__main__":
    main()
