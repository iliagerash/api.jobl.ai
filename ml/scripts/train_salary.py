"""
train_salary.py
────────────────
Train LightGBM salary predictor models — one per currency/country.
Each model predicts salary_min and salary_max simultaneously.

Usage:
    python -u ml/scripts/train_salary.py
    python -u ml/scripts/train_salary.py --data-dir ml/data/salary
    python -u ml/scripts/train_salary.py --tune   # Optuna hyperparameter search

Outputs:
    models/salary_{country_code}.pkl   — per-country model + encodings
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
OUTPUT_DIR = "models"

FEATURE_COLS = [
    "title_encoded", "city_title_encoded", "category_encoded",
    "work_mode", "contract_type", "description_word_count",
]

CAT_FEATURES = ["work_mode", "contract_type"]

DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "n_jobs": -1,
    "verbosity": -1,
}


def train_country(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    country_code: str,
    params: dict,
    n_rounds: int,
    early_stopping: int,
) -> dict:
    """Train a salary model for one country, return metrics + booster."""
    X_train = train_df[FEATURE_COLS].copy()
    X_test = test_df[FEATURE_COLS].copy()

    for col in CAT_FEATURES:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    results = {}
    boosters = {}

    for target in ["target_salary_min", "target_salary_max"]:
        y_train = train_df[target]
        y_test = test_df[target]

        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=CAT_FEATURES, free_raw_data=False)
        val_data = lgb.Dataset(X_test, label=y_test, reference=train_data, free_raw_data=False)

        booster = lgb.train(
            params,
            train_data,
            num_boost_round=n_rounds,
            valid_sets=[val_data],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping, verbose=False),
                lgb.log_evaluation(period=50),
            ],
        )

        preds = booster.predict(X_test)
        mae = np.mean(np.abs(preds - y_test))
        mape = np.mean(np.abs((preds - y_test) / y_test.clip(lower=1))) * 100
        within_20 = np.mean(np.abs((preds - y_test) / y_test.clip(lower=1)) <= 0.2) * 100

        label = "min" if "min" in target else "max"
        results[f"mae_{label}"] = float(mae)
        results[f"mape_{label}"] = float(mape)
        results[f"within_20pct_{label}"] = float(within_20)
        boosters[target] = booster

        print(f"    {label}: MAE={mae:,.0f}  MAPE={mape:.1f}%  within±20%={within_20:.1f}%  best_iter={booster.best_iteration}")

    # Feature importance
    importance = boosters["target_salary_min"].feature_importance(importance_type="gain")
    feat_names = boosters["target_salary_min"].feature_name()
    top_features = sorted(zip(feat_names, importance), key=lambda x: -x[1])
    results["top_features"] = [(name, float(imp)) for name, imp in top_features[:10]]

    return {"boosters": boosters, "metrics": results}


def tune_country(train_df: pd.DataFrame, test_df: pd.DataFrame, country_code: str, n_trials: int) -> dict:
    """Tune hyperparameters with Optuna for one country."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_train = train_df[FEATURE_COLS].copy()
    y_train = train_df["target_salary_min"]
    X_test = test_df[FEATURE_COLS].copy()
    y_test = test_df["target_salary_min"]

    for col in CAT_FEATURES:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=CAT_FEATURES, free_raw_data=False)
    val_data = lgb.Dataset(X_test, label=y_test, reference=train_data, free_raw_data=False)

    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "mae",
            "num_leaves": trial.suggest_int("num_leaves", 31, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
            "n_jobs": -1,
            "verbosity": -1,
        }

        booster = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
            ],
        )

        preds = booster.predict(X_test)
        return float(np.mean(np.abs(preds - y_test)))

    def print_progress(study, trial):
        if (trial.number + 1) % 5 == 0 or trial.number == 0:
            print(f"    trial {trial.number + 1}/{n_trials}  MAE={trial.value:,.0f}  best={study.best_value:,.0f}")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, callbacks=[print_progress])

    print(f"    Best MAE: {study.best_value:,.0f}")
    print(f"    Best params: {study.best_params}")

    best_params = {**DEFAULT_PARAMS, **study.best_params}
    return best_params


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM salary predictor per country")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--n-rounds", type=int, default=500)
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter search")
    parser.add_argument("--tune-trials", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_df = pd.read_parquet(os.path.join(args.data_dir, "train.parquet"))
    test_df = pd.read_parquet(os.path.join(args.data_dir, "test.parquet"))

    with open(os.path.join(args.data_dir, "target_encodings.json")) as f:
        encodings = json.load(f)

    print(f"Loaded train={len(train_df):,}  test={len(test_df):,}")

    countries = sorted(train_df["country_code"].unique())
    all_metrics = {}

    for cc in countries:
        cc_train = train_df[train_df["country_code"] == cc]
        cc_test = test_df[test_df["country_code"] == cc]

        if len(cc_test) < 10:
            print(f"\n{cc}: skipping (only {len(cc_test)} test samples)")
            continue

        print(f"\n{'='*60}")
        print(f"{cc}: train={len(cc_train):,}  test={len(cc_test):,}")

        if args.tune:
            print(f"  Tuning ({args.tune_trials} trials) ...")
            params = tune_country(cc_train, cc_test, cc, args.tune_trials)
        else:
            params = DEFAULT_PARAMS.copy()

        result = train_country(cc_train, cc_test, cc, params, args.n_rounds, args.early_stopping)

        # Save per-country model
        artifact = {
            "boosters": {k: v.model_to_string() for k, v in result["boosters"].items()},
            "params": params,
            "encodings": encodings,
            "country_code": cc,
        }
        output_path = os.path.join(args.output_dir, f"salary_{cc}.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(artifact, f)
        print(f"  Saved: {output_path}")

        all_metrics[cc] = result["metrics"]

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'Country':<8} {'MAE min':>10} {'MAE max':>10} {'MAPE min':>9} {'MAPE max':>9} {'±20% min':>9} {'±20% max':>9}")
    for cc, m in sorted(all_metrics.items()):
        print(f"{cc:<8} {m['mae_min']:>10,.0f} {m['mae_max']:>10,.0f} {m['mape_min']:>8.1f}% {m['mape_max']:>8.1f}% {m['within_20pct_min']:>8.1f}% {m['within_20pct_max']:>8.1f}%")

    metrics_path = os.path.join(args.output_dir, "salary_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
