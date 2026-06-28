"""
profitability_features.py
──────────────────────────
Extract features for the profitability ranker from the GA-joined dataset.
Adds city population tier, target encoding, and temporal features.

Usage:
    python -u ml/scripts/profitability_features.py
    python -u ml/scripts/profitability_features.py --data-dir ml/data/profitability

Outputs:
    ml/data/profitability/train.parquet
    ml/data/profitability/test.parquet
    ml/data/profitability/target_encodings.json
    ml/data/profitability/stats.json
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DATA_DIR = "ml/data/profitability"
CITIES_DIR = "data/cities"

# (large_threshold, medium_threshold) per country
POP_TIERS = {
    "AU": (500_000, 100_000),
    "CA": (500_000, 100_000),
    "GB": (500_000, 100_000),
    "NZ": (100_000, 30_000),
    "US": (1_000_000, 200_000),
    "ZA": (500_000, 100_000),
}


def load_city_populations(cities_dir: str) -> dict[tuple[str, str], int]:
    """Load city populations from CSV dumps. Returns {(city_title, region_title): population}."""
    city_pop = {}

    for csv_path in glob.glob(os.path.join(cities_dir, "*.csv")):
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                city = row.get("title", "").strip()
                region = row.get("region_title") or row.get("region") or ""
                region = region.strip()
                pop = row.get("population", "0")
                try:
                    pop = int(pop)
                except (ValueError, TypeError):
                    pop = 0
                if city:
                    city_pop[(city, region)] = pop

    return city_pop


def population_tier(pop: int, country_code: str) -> int:
    large, medium = POP_TIERS.get(country_code, (500_000, 100_000))
    if pop >= large:
        return 3
    if pop >= medium:
        return 2
    return 1


def target_encode_cv(df: pd.DataFrame, col: str, target: str, n_folds: int = 5, smoothing: int = 10) -> pd.Series:
    global_mean = df[target].mean()
    encoded = pd.Series(np.nan, index=df.index)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(df):
        train = df.iloc[train_idx]
        stats = train.groupby(col)[target].agg(["mean", "count"])
        smooth = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
        encoded.iloc[val_idx] = df.iloc[val_idx][col].map(smooth).fillna(global_mean)
    return encoded


def target_encode_mapping(df: pd.DataFrame, col: str, target: str, smoothing: int = 10) -> dict:
    global_mean = df[target].mean()
    stats = df.groupby(col)[target].agg(["mean", "count"])
    smooth = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
    mapping = smooth.to_dict()
    mapping["__global_mean__"] = global_mean
    return mapping


def count_words(text: str | None) -> int:
    if not text:
        return 0
    return len(re.findall(r"\w+", str(text)[:5000]))


def main():
    parser = argparse.ArgumentParser(description="Extract profitability features")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--cities-dir", default=CITIES_DIR)
    args = parser.parse_args()

    # Load GA-joined data
    joined_path = os.path.join(args.data_dir, "ga_joined.parquet")
    df = pd.read_parquet(joined_path)
    print(f"Loaded {len(df):,} rows from {joined_path}")

    # Load city populations
    print("Loading city populations ...")
    city_pop = load_city_populations(args.cities_dir)
    print(f"  {len(city_pop):,} cities loaded")

    # Add city_population_tier
    def get_tier(row):
        cc = row.get("country_code", "")
        if cc == "SG":
            return 3
        pop = city_pop.get((row.get("city_title", ""), row.get("region_title", "")), 0)
        return population_tier(pop, cc)

    df["city_population_tier"] = df.apply(get_tier, axis=1)
    tier_counts = df["city_population_tier"].value_counts().sort_index()
    print(f"  Population tiers: {dict(tier_counts)}")

    # Build features
    print("Building features ...")
    target_col = "total_revenue"

    df["description_word_count"] = df["description"].apply(count_words)
    df["salary_present"] = (df["salary_min"].notna() & (df["salary_min"] > 0)).astype(int)
    df["work_mode"] = df["is_remote"].map({True: "remote", False: "onsite"}).fillna("onsite").astype("category")
    df["contract_type"] = df["contract"].fillna("unknown").astype("category")
    df["salary_period_cat"] = df["salary_period"].fillna("unknown").astype("category")
    df["country"] = df["country_code"].astype("category")
    df["category_id"] = df["category_id"].fillna(0).astype(int)

    # Temporal features
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["day_of_week_posted"] = df["published_at"].dt.dayofweek
    df["month_posted"] = df["published_at"].dt.month

    # 80/20 time-based split per country
    train_parts = []
    test_parts = []
    for cc in df["country_code"].unique():
        cc_df = df[df["country_code"] == cc].sort_values("published_at")
        split_idx = int(len(cc_df) * 0.8)
        train_parts.append(cc_df.iloc[:split_idx])
        test_parts.append(cc_df.iloc[split_idx:])
    train_df = pd.concat(train_parts).copy()
    test_df = pd.concat(test_parts).copy()
    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")
    for cc in sorted(df["country_code"].unique()):
        cc_train = train_df[train_df["country_code"] == cc]
        cc_test = test_df[test_df["country_code"] == cc]
        print(f"    {cc}: train={len(cc_train):,}  test={len(cc_test):,}")

    # Target encoding
    print("Target encoding ...")
    encodings = {}

    for col in ["title", "company_name", "city_title", "region_title"]:
        train_df[f"{col}_encoded"] = target_encode_cv(train_df, col, target_col)
        mapping = target_encode_mapping(train_df, col, target_col)
        global_mean = mapping.pop("__global_mean__")
        test_df[f"{col}_encoded"] = test_df[col].map(mapping).fillna(global_mean)
        mapping["__global_mean__"] = global_mean
        encodings[col] = {str(k): float(v) for k, v in mapping.items()}

    # category_id target encoding
    train_df["category_encoded"] = target_encode_cv(train_df, "category_id", target_col)
    cat_mapping = target_encode_mapping(train_df, "category_id", target_col)
    cat_global = cat_mapping.pop("__global_mean__")
    test_df["category_encoded"] = test_df["category_id"].map(cat_mapping).fillna(cat_global)
    cat_mapping["__global_mean__"] = cat_global
    encodings["category_id"] = {str(k): float(v) for k, v in cat_mapping.items()}

    # Final columns
    feature_cols = [
        "title_encoded", "company_name_encoded", "city_title_encoded",
        "region_title_encoded", "category_encoded", "country",
        "salary_present", "salary_min", "salary_max", "salary_period_cat",
        "work_mode", "contract_type", "description_word_count",
        "city_population_tier", "day_of_week_posted", "month_posted",
    ]
    target_cols = ["total_revenue", "rps", "total_sessions"]
    meta_cols = ["country_code", "destination", "destination_job_id", "title",
                 "company_name", "city_title", "published_at"]

    train_out = train_df[feature_cols + target_cols + meta_cols].copy()
    test_out = test_df[feature_cols + target_cols + meta_cols].copy()

    # Force salary to numeric
    for out in [train_out, test_out]:
        out["salary_min"] = pd.to_numeric(out["salary_min"], errors="coerce").fillna(0).astype(float)
        out["salary_max"] = pd.to_numeric(out["salary_max"], errors="coerce").fillna(0).astype(float)

    # Save
    train_path = os.path.join(args.data_dir, "train.parquet")
    test_path = os.path.join(args.data_dir, "test.parquet")
    train_out.to_parquet(train_path, index=False)
    test_out.to_parquet(test_path, index=False)
    print(f"\nSaved: {train_path} ({len(train_out):,} rows)")
    print(f"Saved: {test_path} ({len(test_out):,} rows)")

    enc_path = os.path.join(args.data_dir, "target_encodings.json")
    with open(enc_path, "w") as f:
        json.dump(encodings, f)
    print(f"Saved: {enc_path}")

    # Stats
    stats = {
        "total_jobs": len(df),
        "train_size": len(train_out),
        "test_size": len(test_out),
        "per_country": {},
    }
    for cc in sorted(df["country_code"].unique()):
        subset = df[df["country_code"] == cc]
        stats["per_country"][cc] = {
            "count": int(len(subset)),
            "revenue_mean": float(subset["total_revenue"].mean()),
            "revenue_median": float(subset["total_revenue"].median()),
            "sessions_median": float(subset["total_sessions"].median()),
            "rps_median": float(subset["rps"].median()),
        }

    stats_path = os.path.join(args.data_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {stats_path}")


if __name__ == "__main__":
    main()
