"""
eval_salary.py
───────────────
Evaluate trained salary models on the holdout test set.
Produces per-country and per-category breakdowns.

Usage:
    python -u ml/scripts/eval_salary.py
    python -u ml/scripts/eval_salary.py --data-dir ml/data/salary --models-dir models
"""

import argparse
import json
import os
import pickle
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DATA_DIR = "ml/data/salary"
MODELS_DIR = "models"

FEATURE_COLS = [
    "title_encoded", "city_title_encoded", "category_encoded",
    "work_mode", "contract_type", "description_word_count",
]

CAT_FEATURES = ["work_mode", "contract_type"]


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = np.mean(np.abs(y_pred - y_true))
    mape = np.mean(np.abs((y_pred - y_true) / y_true.clip(min=1))) * 100
    within_20 = np.mean(np.abs((y_pred - y_true) / y_true.clip(min=1)) <= 0.2) * 100
    within_30 = np.mean(np.abs((y_pred - y_true) / y_true.clip(min=1)) <= 0.3) * 100
    median_ae = np.median(np.abs(y_pred - y_true))
    return {
        "mae": float(mae),
        "median_ae": float(median_ae),
        "mape": float(mape),
        "within_20pct": float(within_20),
        "within_30pct": float(within_30),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate salary predictor models")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--models-dir", default=MODELS_DIR)
    args = parser.parse_args()

    test_df = pd.read_parquet(os.path.join(args.data_dir, "test.parquet"))
    print(f"Test set: {len(test_df):,} jobs")

    countries = sorted(test_df["country_code"].unique())
    all_results = {}

    for cc in countries:
        model_path = os.path.join(args.models_dir, f"salary_{cc}.pkl")
        if not os.path.exists(model_path):
            print(f"\n{cc}: model not found at {model_path}, skipping")
            continue

        with open(model_path, "rb") as f:
            artifact = pickle.load(f)

        cc_test = test_df[test_df["country_code"] == cc].copy()
        if len(cc_test) < 5:
            continue

        X_test = cc_test[FEATURE_COLS].copy()
        for col in CAT_FEATURES:
            X_test[col] = X_test[col].astype("category")

        print(f"\n{'='*60}")
        print(f"{cc}: {len(cc_test):,} test jobs")

        country_results = {}
        for target_key, label in [("target_salary_min", "min"), ("target_salary_max", "max")]:
            booster = lgb.Booster(model_str=artifact["boosters"][target_key])
            preds = booster.predict(X_test)
            y_true = cc_test[target_key].values
            metrics = evaluate(y_true, preds)
            country_results[label] = metrics
            print(f"  {label}: MAE={metrics['mae']:,.0f}  MedAE={metrics['median_ae']:,.0f}  MAPE={metrics['mape']:.1f}%  ±20%={metrics['within_20pct']:.1f}%  ±30%={metrics['within_30pct']:.1f}%")

            # Per-category breakdown
            cc_test[f"pred_{label}"] = preds

        # Category breakdown
        if "category_encoded" in cc_test.columns:
            print(f"\n  Per-category breakdown (salary_min):")
            booster_min = lgb.Booster(model_str=artifact["boosters"]["target_salary_min"])
            for cat_id in sorted(cc_test["category_encoded"].unique())[:15]:
                cat_mask = cc_test["category_encoded"] == cat_id
                if cat_mask.sum() < 5:
                    continue
                cat_metrics = evaluate(
                    cc_test.loc[cat_mask, "target_salary_min"].values,
                    cc_test.loc[cat_mask, "pred_min"].values,
                )
                print(f"    cat={cat_id:.0f}: n={cat_mask.sum():,}  MAE={cat_metrics['mae']:,.0f}  MAPE={cat_metrics['mape']:.1f}%")

        # Bias check: mean predicted vs mean actual
        print(f"\n  Bias check:")
        for label in ["min", "max"]:
            actual_mean = cc_test[f"target_salary_{label}"].mean()
            pred_mean = cc_test[f"pred_{label}"].mean()
            bias = (pred_mean - actual_mean) / actual_mean * 100
            print(f"    {label}: actual_mean={actual_mean:,.0f}  pred_mean={pred_mean:,.0f}  bias={bias:+.1f}%")

        all_results[cc] = country_results

    # Summary table
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'Country':<8} {'MAE min':>10} {'MAE max':>10} {'MAPE min':>9} {'MAPE max':>9} {'±20% min':>9} {'±20% max':>9}")
    for cc, r in sorted(all_results.items()):
        print(f"{cc:<8} {r['min']['mae']:>10,.0f} {r['max']['mae']:>10,.0f} {r['min']['mape']:>8.1f}% {r['max']['mape']:>8.1f}% {r['min']['within_20pct']:>8.1f}% {r['max']['within_20pct']:>8.1f}%")

    results_path = os.path.join(args.data_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
