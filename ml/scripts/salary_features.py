"""
salary_features.py
───────────────────
Extract features for the salary predictor from the jobl database.
Pulls jobs with stated salary, applies target encoding, and splits
80/20 by time for training/evaluation.

Usage:
    python -u ml/scripts/salary_features.py
    python -u ml/scripts/salary_features.py --output-dir ml/data/salary
    python -u ml/scripts/salary_features.py --min-samples 50

Outputs:
    ml/data/salary/train.parquet
    ml/data/salary/test.parquet
    ml/data/salary/target_encodings.json
    ml/data/salary/stats.json
"""

import argparse
import json
import os
import sys
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

OUTPUT_DIR = "ml/data/salary"

COUNTRY_CURRENCY = {
    "AU": "AUD",
    "CA": "CAD",
    "SG": "SGD",
    "GB": "GBP",
    "US": "USD",
    "ZA": "ZAR",
}

# Countries where salary is typically stated monthly
MONTHLY_COUNTRIES = {"SG", "ZA"}

PERIOD_MULTIPLIERS_TO_YEARLY = {
    "hour": 2080, "hourly": 2080,
    "day": 260, "daily": 260,
    "week": 52, "weekly": 52,
    "month": 12, "monthly": 12,
    "year": 1, "yearly": 1,
}

PERIOD_MULTIPLIERS_TO_MONTHLY = {
    "hour": 173.33, "hourly": 173.33,
    "day": 21.67, "daily": 21.67,
    "week": 4.33, "weekly": 4.33,
    "month": 1, "monthly": 1,
    "year": 1 / 12, "yearly": 1 / 12,
}


def normalize_salary(row: pd.Series) -> tuple[float, float]:
    """Normalize salary to the default period for the country."""
    period = str(row["salary_period"]).lower() if row.get("salary_period") else "yearly"
    country = row["country_code"]
    sal_min = float(row["salary_min"])
    sal_max = float(row["salary_max"])

    if country in MONTHLY_COUNTRIES:
        multiplier = PERIOD_MULTIPLIERS_TO_MONTHLY.get(period, 1)
    else:
        multiplier = PERIOD_MULTIPLIERS_TO_YEARLY.get(period, 1)

    return sal_min * multiplier, sal_max * multiplier


def target_encode_cv(df: pd.DataFrame, col: str, target: str, n_folds: int = 5, smoothing: int = 10) -> pd.Series:
    """Target-encode a column using k-fold CV to prevent leakage."""
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
    """Compute target encoding mapping from the full training set (for inference)."""
    global_mean = df[target].mean()
    stats = df.groupby(col)[target].agg(["mean", "count"])
    smooth = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
    mapping = smooth.to_dict()
    mapping["__global_mean__"] = global_mean
    return mapping


def count_words(text: str | None) -> int:
    if not text:
        return 0
    return len(re.findall(r"\w+", text[:5000]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract salary prediction features")
    parser.add_argument("--db-url", default=None)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--min-samples", type=int, default=50, help="Min jobs per country to include")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    db_url = args.db_url
    if not db_url:
        from dotenv import load_dotenv
        load_dotenv()
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            sys.exit("No DATABASE_URL")

    from sqlalchemy import create_engine, text
    engine = create_engine(db_url, pool_pre_ping=True)

    print("Loading jobs with stated salary ...")
    query = text("""
        SELECT title, country_code, city_title, category_id,
               is_remote, contract, description,
               salary_min, salary_max, salary_period,
               published_at
        FROM jobs
        WHERE salary_min IS NOT NULL
          AND salary_max IS NOT NULL
          AND salary_min > 0
          AND salary_max > 0
          AND salary_max >= salary_min
          AND country_code IN ('AU', 'CA', 'SG', 'GB', 'US', 'ZA')
          AND category_id IS NOT NULL
          AND published_at IS NOT NULL
        ORDER BY published_at
    """)

    with engine.connect() as conn:
        df = pd.DataFrame(conn.execute(query).mappings())

    engine.dispose()
    print(f"  {len(df):,} jobs loaded")

    # Filter countries with enough data
    country_counts = df["country_code"].value_counts()
    print(f"\nJobs per country:")
    for cc, cnt in country_counts.items():
        status = "✓" if cnt >= args.min_samples else "✗ excluded"
        print(f"  {cc}: {cnt:,} {status}")

    valid_countries = country_counts[country_counts >= args.min_samples].index
    df = df[df["country_code"].isin(valid_countries)].copy()
    print(f"\n{len(df):,} jobs after country filter")

    # Normalize salaries to default period
    print("Normalizing salaries ...")
    normalized = df.apply(normalize_salary, axis=1, result_type="expand")
    df["target_salary_min"] = normalized[0]
    df["target_salary_max"] = normalized[1]

    # Filter mislabeled salary periods (e.g. hourly rates labeled as yearly)
    # Minimum reasonable normalized salary per country
    SALARY_FLOOR = {
        "AU": 15000,  # yearly AUD
        "CA": 15000,  # yearly CAD
        "GB": 10000,  # yearly GBP
        "US": 15000,  # yearly USD
        "SG": 1000,   # monthly SGD
        "ZA": 5000,   # monthly ZAR
    }
    before = len(df)
    df = df[df.apply(lambda r: r["target_salary_min"] >= SALARY_FLOOR.get(r["country_code"], 0), axis=1)]
    print(f"  Salary floor filter: {before:,} → {len(df):,} ({before - len(df)} likely mislabeled periods removed)")

    # Filter outliers (per country)
    before = len(df)
    filtered = []
    for cc in df["country_code"].unique():
        mask = df["country_code"] == cc
        subset = df[mask]
        q01 = subset["target_salary_min"].quantile(0.01)
        q99 = subset["target_salary_max"].quantile(0.99)
        subset = subset[(subset["target_salary_min"] >= q01) & (subset["target_salary_max"] <= q99)]
        filtered.append(subset)
    df = pd.concat(filtered)
    print(f"  Outlier filter: {before:,} → {len(df):,} ({before - len(df)} removed)")

    # Build features
    print("Building features ...")
    target_col = "target_salary_min"  # use min for target encoding

    df["description_word_count"] = df["description"].apply(count_words)
    df["work_mode"] = df["is_remote"].map({True: "remote", False: "onsite"}).astype("category")
    df["contract_type"] = df["contract"].fillna("unknown").astype("category")
    df["country"] = df["country_code"].astype("category")
    df["category_id"] = df["category_id"].astype(int)

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

    # Target encoding (CV on train, mapping applied to test)
    print("Target encoding ...")
    encodings = {}

    for col in ["title", "city_title"]:
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

    # Select final columns
    feature_cols = [
        "title_encoded", "country", "city_title_encoded", "category_encoded",
        "work_mode", "contract_type", "description_word_count",
    ]
    target_cols = ["target_salary_min", "target_salary_max"]
    meta_cols = ["country_code", "title", "city_title", "published_at"]

    train_out = train_df[feature_cols + target_cols + meta_cols].copy()
    test_out = test_df[feature_cols + target_cols + meta_cols].copy()

    # Save
    train_path = os.path.join(args.output_dir, "train.parquet")
    test_path = os.path.join(args.output_dir, "test.parquet")
    train_out.to_parquet(train_path, index=False)
    test_out.to_parquet(test_path, index=False)
    print(f"\nSaved: {train_path} ({len(train_out):,} rows)")
    print(f"Saved: {test_path} ({len(test_out):,} rows)")

    enc_path = os.path.join(args.output_dir, "target_encodings.json")
    with open(enc_path, "w") as f:
        json.dump(encodings, f)
    print(f"Saved: {enc_path}")

    # Stats
    stats = {
        "total_jobs": len(df),
        "train_size": len(train_out),
        "test_size": len(test_out),
        "countries": {
            cc: int(cnt) for cc, cnt in df["country_code"].value_counts().items()
        },
        "salary_stats": {},
    }
    for cc in sorted(df["country_code"].unique()):
        subset = df[df["country_code"] == cc]
        period = "monthly" if cc in MONTHLY_COUNTRIES else "yearly"
        stats["salary_stats"][cc] = {
            "period": period,
            "min_median": float(subset["target_salary_min"].median()),
            "max_median": float(subset["target_salary_max"].median()),
            "count": int(len(subset)),
        }

    stats_path = os.path.join(args.output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {stats_path}")

    print("\nSalary stats per country:")
    for cc, s in stats["salary_stats"].items():
        print(f"  {cc} ({s['period']}): median {s['min_median']:,.0f} - {s['max_median']:,.0f} ({s['count']:,} jobs)")


if __name__ == "__main__":
    main()
