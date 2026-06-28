"""
train_profitability.py
───────────────────────
Train LightGBM profitability ranker — predicts total_revenue and rps.
Single model trained on all countries (unlike salary which is per-country).

Usage:
    python -u ml/scripts/train_profitability.py
    python -u ml/scripts/train_profitability.py --tune --tune-trials 20
    python -u ml/scripts/train_profitability.py --tune --gpu

Outputs:
    models/profitability_revenue.pkl
    models/profitability_rps.pkl
    models/profitability_metrics.json
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

DATA_DIR = "ml/data/profitability"
OUTPUT_DIR = "models"

FEATURE_COLS = [
    "title_encoded", "company_name_encoded", "city_title_encoded",
    "region_title_encoded", "category_encoded", "country",
    "salary_present", "salary_min", "salary_max", "salary_period_cat",
    "work_mode", "contract_type", "description_word_count",
    "city_population_tier", "day_of_week_posted", "month_posted",
]

CAT_FEATURES = ["country", "salary_period_cat", "work_mode", "contract_type"]

DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "feature_pre_filter": False,
    "n_jobs": -1,
    "verbosity": -1,
}

TARGETS = {
    "total_revenue": "profitability_revenue",
    "rps": "profitability_rps",
}


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLS].copy()
    for col in CAT_FEATURES:
        X[col] = X[col].astype("category")
    return X


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = np.mean(np.abs(y_pred - y_true))
    nonzero = y_true > 0
    if nonzero.sum() > 0:
        mape_nonzero = np.mean(np.abs((y_pred[nonzero] - y_true[nonzero]) / y_true[nonzero])) * 100
    else:
        mape_nonzero = 0
    spearman = pd.Series(y_pred).corr(pd.Series(y_true), method="spearman")

    sorted_idx = np.argsort(-y_pred)
    top_100_pred = set(sorted_idx[:100])
    top_100_actual = set(np.argsort(-y_true)[:100])
    precision_100 = len(top_100_pred & top_100_actual) / 100 * 100

    top_50pct_idx = sorted_idx[:len(sorted_idx) // 2]
    revenue_capture = y_true[top_50pct_idx].sum() / max(y_true.sum(), 1e-10) * 100

    return {
        "mae": float(mae),
        "mape_nonzero": float(mape_nonzero),
        "spearman": float(spearman),
        "precision_at_100": float(precision_100),
        "revenue_capture_top50pct": float(revenue_capture),
    }


def tune_model(X_train, y_train, X_test, y_test, n_trials: int, gpu: bool) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=CAT_FEATURES, free_raw_data=False, params={"feature_pre_filter": False})
    val_data = lgb.Dataset(X_test, label=y_test, reference=train_data, free_raw_data=False)

    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "mae",
            "feature_pre_filter": False,
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
        if gpu:
            params["device"] = "gpu"

        booster = lgb.train(
            params, train_data, num_boost_round=500,
            valid_sets=[val_data], valid_names=["val"],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        preds = booster.predict(X_test)
        return float(np.mean(np.abs(preds - y_test)))

    def print_progress(study, trial):
        print(f"    trial {trial.number + 1}/{n_trials}  MAE={trial.value:.6f}  best={study.best_value:.6f}", flush=True)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, callbacks=[print_progress])

    print(f"    Best MAE: {study.best_value:.6f}")
    print(f"    Best params: {study.best_params}")
    return {**DEFAULT_PARAMS, **study.best_params}


def main():
    parser = argparse.ArgumentParser(description="Train profitability ranker")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--n-rounds", type=int, default=500)
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--tune-trials", type=int, default=50)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_df = pd.read_parquet(os.path.join(args.data_dir, "train.parquet"))
    test_df = pd.read_parquet(os.path.join(args.data_dir, "test.parquet"))

    with open(os.path.join(args.data_dir, "target_encodings.json")) as f:
        encodings = json.load(f)

    print(f"Loaded train={len(train_df):,}  test={len(test_df):,}")

    X_train = prepare_features(train_df)
    X_test = prepare_features(test_df)

    all_metrics = {}

    for target_col, model_name in TARGETS.items():
        print(f"\n{'='*60}")
        print(f"Training: {target_col}")

        y_train = train_df[target_col].values
        y_test = test_df[target_col].values

        if args.tune:
            print(f"  Tuning ({args.tune_trials} trials) ...")
            params = tune_model(X_train, y_train, X_test, y_test, args.tune_trials, args.gpu)
        else:
            params = DEFAULT_PARAMS.copy()

        if args.gpu:
            params["device"] = "gpu"

        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=CAT_FEATURES, free_raw_data=False)
        val_data = lgb.Dataset(X_test, label=y_test, reference=train_data, free_raw_data=False)

        booster = lgb.train(
            params, train_data, num_boost_round=args.n_rounds,
            valid_sets=[val_data], valid_names=["val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=args.early_stopping, verbose=False),
                lgb.log_evaluation(period=50),
            ],
        )

        preds = booster.predict(X_test)
        metrics = evaluate(y_test, preds)
        all_metrics[target_col] = metrics

        print(f"\n  Results:")
        print(f"    MAE:                  {metrics['mae']:.6f}")
        print(f"    MAPE (nonzero):       {metrics['mape_nonzero']:.1f}%")
        print(f"    Spearman correlation: {metrics['spearman']:.4f}")
        print(f"    Precision@100:        {metrics['precision_at_100']:.1f}%")
        print(f"    Revenue capture (top 50%): {metrics['revenue_capture_top50pct']:.1f}%")

        # Feature importance
        importance = booster.feature_importance(importance_type="gain")
        feat_names = booster.feature_name()
        top = sorted(zip(feat_names, importance), key=lambda x: -x[1])[:10]
        print(f"\n  Top features:")
        for name, imp in top:
            print(f"    {name}: {imp:.0f}")

        # Save model
        artifact = {
            "booster": booster.model_to_string(),
            "params": params,
            "encodings": encodings,
            "target": target_col,
        }
        output_path = os.path.join(args.output_dir, f"{model_name}.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(artifact, f)
        print(f"\n  Saved: {output_path}")

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    for target, m in all_metrics.items():
        print(f"  {target}: MAE={m['mae']:.6f}  Spearman={m['spearman']:.4f}  P@100={m['precision_at_100']:.1f}%  RevCapture={m['revenue_capture_top50pct']:.1f}%")

    metrics_path = os.path.join(args.output_dir, "profitability_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
