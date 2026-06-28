"""
export_ga.py
─────────────
Export Google Analytics 4 data for job pages via the GA4 Data API.

Usage:
    python -u scripts/export_ga.py --list --key data/ga_account/Workus.json
    python -u scripts/export_ga.py --key data/ga_account/Workus.json
    python -u scripts/export_ga.py --key data/ga_account/Workus.json --properties au.workus.org,ca.workus.org

Outputs:
    data/ga_exports/{property_name}.csv
"""

import argparse
import csv
import os
import sys

from google.analytics.admin import AnalyticsAdminServiceClient
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    DateRange,
    Dimension,
    DimensionExpression,
    Filter,
    FilterExpression,
    Metric,
    RunReportRequest,
)
from google.oauth2 import service_account

OUTPUT_DIR = "data/ga_exports"

PROPERTIES = {
    # Workus account
    "au.workus.org": {"start_date": "2026-04-06"},
    "ca.workus.org": {"start_date": "2026-04-06"},
    "nz.workus.org": {"start_date": "2026-04-06"},
    "sg.workus.org": {"start_date": "2026-04-06"},
    "uk.workus.org": {"start_date": "2026-04-06"},
    "za.workus.org": {"start_date": "2026-04-06"},
    "workus.org": {"start_date": "2026-06-10"},
    # JobRadars account
    "jobradars.com": {"start_date": "2026-01-01"},
    "australia.jobradars.com": {"start_date": "2026-01-01"},
    "canada.jobradars.com": {"start_date": "2026-01-01"},
    "newzealand.jobradars.com": {"start_date": "2026-01-01"},
    "singapore.jobradars.com": {"start_date": "2026-01-01"},
    "unitedkingdom.jobradars.com": {"start_date": "2026-01-01"},
    "jobspire.co.za": {"start_date": "2026-01-01"},
    # SnapJobSearch account
    "snapjobsearch.com": {"start_date": "2026-01-01"},
    "ca.snapjobsearch.com": {"start_date": "2026-01-01"},
}

SCOPES = [
    "https://www.googleapis.com/auth/analytics.readonly",
]


def get_credentials(key_path: str):
    return service_account.Credentials.from_service_account_file(
        key_path, scopes=SCOPES
    )


def list_properties(credentials) -> list[dict]:
    admin_client = AnalyticsAdminServiceClient(credentials=credentials)
    accounts = admin_client.list_accounts()
    properties = []
    for account in accounts:
        account_id = account.name.split("/")[-1]
        for prop in admin_client.list_properties(
            request={"filter": f"parent:accounts/{account_id}"}
        ):
            properties.append({
                "property_id": prop.name.split("/")[-1],
                "display_name": prop.display_name,
            })
    return properties


def export_property(
    credentials,
    property_id: str,
    display_name: str,
    start_date: str,
    end_date: str,
    output_dir: str,
) -> int:
    client = BetaAnalyticsDataClient(credentials=credentials)

    request = RunReportRequest(
        property=f"properties/{property_id}",
        date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
        dimensions=[Dimension(name="pagePath")],
        metrics=[
            Metric(name="sessions"),
            Metric(name="activeUsers"),
            Metric(name="averageSessionDuration"),
            Metric(name="totalRevenue"),
        ],
        dimension_filter=FilterExpression(
            filter=Filter(
                field_name="pagePath",
                string_filter=Filter.StringFilter(
                    match_type=Filter.StringFilter.MatchType.BEGINS_WITH,
                    value="/jobs/view/",
                ),
            )
        ),
        limit=100000,
    )

    response = client.run_report(request=request)

    output_path = os.path.join(output_dir, f"{display_name}.csv")
    row_count = 0
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Landing page", "Sessions", "Active users",
            "Average engagement time per session", "Total revenue",
        ])
        for row in response.rows:
            writer.writerow([
                row.dimension_values[0].value,
                row.metric_values[0].value,
                row.metric_values[1].value,
                row.metric_values[2].value,
                row.metric_values[3].value,
            ])
            row_count += 1

    return row_count


def main():
    parser = argparse.ArgumentParser(description="Export GA4 data for job pages")
    parser.add_argument("--key", required=True, help="Path to service account JSON key")
    parser.add_argument("--list", action="store_true", help="List all accessible properties")
    parser.add_argument("--properties", default=None, help="Comma-separated property names to export (default: all configured)")
    parser.add_argument("--end-date", default="today", help="End date (default: today)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    credentials = get_credentials(args.key)

    if args.list:
        print("Accessible properties:")
        for p in list_properties(credentials):
            print(f"  {p['property_id']}  {p['display_name']}")
        return

    # Discover property IDs by display name (strip common suffixes like " - GA4")
    all_props = list_properties(credentials)
    prop_map = {}
    for p in all_props:
        prop_map[p["display_name"]] = p["property_id"]
        clean_name = p["display_name"].replace(" - GA4", "").strip()
        prop_map[clean_name] = p["property_id"]

    if args.properties:
        target_names = {n.strip() for n in args.properties.split(",")}
    else:
        target_names = set(PROPERTIES.keys()) & set(prop_map.keys())

    os.makedirs(args.output_dir, exist_ok=True)

    for name in sorted(target_names):
        if name not in prop_map:
            print(f"  {name}: property not found, skipping")
            continue

        config = PROPERTIES.get(name, {"start_date": "2026-04-06"})
        property_id = prop_map[name]

        print(f"\n{name} (property {property_id}) — {config['start_date']} to {args.end_date}")

        row_count = export_property(
            credentials=credentials,
            property_id=property_id,
            display_name=name,
            start_date=config["start_date"],
            end_date=args.end_date,
            output_dir=args.output_dir,
        )
        print(f"  {row_count:,} rows exported")

    print("\nDone.")


if __name__ == "__main__":
    main()
