"""
train_profitability.py
───────────────────────
Train two-stage LightGBM profitability ranker:
  Stage 1: Classifier — predicts whether a job generates any revenue (binary)
  Stage 2: Regressor — predicts how much revenue, trained only on revenue > 0 jobs

Final prediction = P(revenue > 0) × predicted_revenue_if_positive

Usage:
    python -u ml/scripts/train_profitability.py
    python -u ml/scripts/train_profitability.py --tune --tune-trials 20
    python -u ml/scripts/train_profitability.py --tune --gpu

Outputs:
    models/profitability_classifier.pkl
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
    "region_title_encoded", "category_encoded", "country", "destination_cat",
    "salary_present", "salary_min", "salary_max", "salary_period_cat",
    "work_mode", "contract_type", "description_word_count",
    "city_population_tier", "day_of_week_posted", "month_posted",
]

CAT_FEATURES = ["country", "destination_cat", "salary_period_cat", "work_mode", "contract_type"]

DEFAULT_CLASSIFIER_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "feature_pre_filter": False,
    "is_unbalance": True,
    "n_jobs": -1,
    "verbosity": -1,
}

DEFAULT_REGRESSOR_PARAMS = {
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


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLS].copy()
    for col in CAT_FEATURES:
        X[col] = X[col].astype("category")
    return X


def evaluate_ranking(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = np.mean(np.abs(y_pred - y_true))
    nonzero = y_true > 0
    mape_nonzero = np.mean(np.abs((y_pred[nonzero] - y_true[nonzero]) / y_true[nonzero])) * 100 if nonzero.sum() > 0 else 0
    spearman = pd.Series(y_pred).corr(pd.Series(y_true), method="spearman")

    sorted_idx = np.argsort(-y_pred)
    top_100_pred = set(sorted_idx[:100])
    top_100_actual = set(np.argsort(-y_true)[:100])
    precision_100 = len(top_100_pred & top_100_actual) / min(100, len(y_true)) * 100

    top_50pct_idx = sorted_idx[:len(sorted_idx) // 2]
    revenue_capture = y_true[top_50pct_idx].sum() / max(y_true.sum(), 1e-10) * 100

    return {
        "mae": float(mae),
        "mape_nonzero": float(mape_nonzero),
        "spearman": float(spearman),
        "precision_at_100": float(precision_100),
        "revenue_capture_top50pct": float(revenue_capture),
    }


def evaluate_classifier(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score, precision_score, recall_score
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_pred_proba)),
        "precision": float(precision_score(y_true, y_pred_binary, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_binary, zero_division=0)),
        "positive_rate_actual": float(y_true.mean()),
        "positive_rate_predicted": float(y_pred_binary.mean()),
    }


def tune_model(X_train, y_train, X_test, y_test, n_trials: int, gpu: bool, objective: str, metric: str) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    is_classifier = objective == "binary"
    base_params = DEFAULT_CLASSIFIER_PARAMS if is_classifier else DEFAULT_REGRESSOR_PARAMS

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=CAT_FEATURES, free_raw_data=False, params={"feature_pre_filter": False})
    val_data = lgb.Dataset(X_test, label=y_test, reference=train_data, free_raw_data=False)

    def trial_objective(trial):
        params = {
            "objective": objective,
            "metric": metric,
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
        if is_classifier:
            params["is_unbalance"] = True
        if gpu:
            params["device"] = "gpu"

        booster = lgb.train(
            params, train_data, num_boost_round=500,
            valid_sets=[val_data], valid_names=["val"],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        preds = booster.predict(X_test)

        if is_classifier:
            from sklearn.metrics import roc_auc_score
            return -float(roc_auc_score(y_test, preds))
        else:
            return float(np.mean(np.abs(preds - y_test)))

    def print_progress(study, trial):
        val = -trial.value if is_classifier else trial.value
        best = -study.best_value if is_classifier else study.best_value
        label = "AUC" if is_classifier else "MAE"
        print(f"    trial {trial.number + 1}/{n_trials}  {label}={val:.6f}  best={best:.6f}", flush=True)

    direction = "minimize"
    study = optuna.create_study(direction=direction)
    study.optimize(trial_objective, n_trials=n_trials, callbacks=[print_progress])

    best_val = -study.best_value if is_classifier else study.best_value
    label = "AUC" if is_classifier else "MAE"
    print(f"    Best {label}: {best_val:.6f}")
    print(f"    Best params: {study.best_params}")
    return {**base_params, **study.best_params}


def main():
    parser = argparse.ArgumentParser(description="Train two-stage profitability ranker")
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

    # ── Stage 1: Classifier (revenue > 0 vs = 0) ──
    print(f"\n{'='*60}")
    print("Stage 1: Classifier (has revenue?)")

    y_train_cls = (train_df["total_revenue"] > 0).astype(int).values
    y_test_cls = (test_df["total_revenue"] > 0).astype(int).values
    print(f"  Train: {y_train_cls.sum():,} positive / {len(y_train_cls):,} total ({y_train_cls.mean()*100:.1f}%)")
    print(f"  Test:  {y_test_cls.sum():,} positive / {len(y_test_cls):,} total ({y_test_cls.mean()*100:.1f}%)")

    if args.tune:
        print(f"  Tuning ({args.tune_trials} trials) ...")
        cls_params = tune_model(X_train, y_train_cls, X_test, y_test_cls, args.tune_trials, args.gpu, "binary", "auc")
    else:
        cls_params = DEFAULT_CLASSIFIER_PARAMS.copy()

    if args.gpu:
        cls_params["device"] = "gpu"

    cls_train_data = lgb.Dataset(X_train, label=y_train_cls, categorical_feature=CAT_FEATURES, free_raw_data=False)
    cls_val_data = lgb.Dataset(X_test, label=y_test_cls, reference=cls_train_data, free_raw_data=False)

    cls_booster = lgb.train(
        cls_params, cls_train_data, num_boost_round=args.n_rounds,
        valid_sets=[cls_val_data], valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=args.early_stopping, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )

    cls_preds = cls_booster.predict(X_test)
    cls_metrics = evaluate_classifier(y_test_cls, cls_preds)
    all_metrics["classifier"] = cls_metrics

    print(f"\n  Classifier results:")
    print(f"    AUC:       {cls_metrics['auc']:.4f}")
    print(f"    Precision: {cls_metrics['precision']:.4f}")
    print(f"    Recall:    {cls_metrics['recall']:.4f}")

    cls_artifact = {
        "booster": cls_booster.model_to_string(),
        "params": cls_params,
        "encodings": encodings,
        "target": "has_revenue",
    }
    cls_path = os.path.join(args.output_dir, "profitability_classifier.pkl")
    with open(cls_path, "wb") as f:
        pickle.dump(cls_artifact, f)
    print(f"  Saved: {cls_path}")

    # ── Stage 2: Regressors (revenue and rps, trained on positive-only) ──
    train_pos = train_df[train_df["total_revenue"] > 0]
    test_pos = test_df[test_df["total_revenue"] > 0]
    X_train_pos = prepare_features(train_pos)
    X_test_pos = prepare_features(test_pos)
    print(f"\n  Positive samples: train={len(train_pos):,}  test={len(test_pos):,}")

    for target_col, model_name in [("total_revenue", "profitability_revenue"), ("rps", "profitability_rps")]:
        print(f"\n{'='*60}")
        print(f"Stage 2: Regressor ({target_col}, positive-only)")

        y_train_reg = train_pos[target_col].values
        y_test_reg = test_pos[target_col].values

        if args.tune:
            print(f"  Tuning ({args.tune_trials} trials) ...")
            reg_params = tune_model(X_train_pos, y_train_reg, X_test_pos, y_test_reg, args.tune_trials, args.gpu, "regression", "mae")
        else:
            reg_params = DEFAULT_REGRESSOR_PARAMS.copy()

        if args.gpu:
            reg_params["device"] = "gpu"

        reg_train_data = lgb.Dataset(X_train_pos, label=y_train_reg, categorical_feature=CAT_FEATURES, free_raw_data=False)
        reg_val_data = lgb.Dataset(X_test_pos, label=y_test_reg, reference=reg_train_data, free_raw_data=False)

        reg_booster = lgb.train(
            reg_params, reg_train_data, num_boost_round=args.n_rounds,
            valid_sets=[reg_val_data], valid_names=["val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=args.early_stopping, verbose=False),
                lgb.log_evaluation(period=50),
            ],
        )

        # Evaluate regressor on positive-only
        reg_preds = reg_booster.predict(X_test_pos)
        reg_metrics = evaluate_ranking(y_test_reg, reg_preds)
        print(f"\n  Regressor results (positive-only):")
        print(f"    MAE:      {reg_metrics['mae']:.6f}")
        print(f"    Spearman: {reg_metrics['spearman']:.4f}")

        # Feature importance
        importance = reg_booster.feature_importance(importance_type="gain")
        feat_names = reg_booster.feature_name()
        top = sorted(zip(feat_names, importance), key=lambda x: -x[1])[:10]
        print(f"\n  Top features:")
        for name, imp in top:
            print(f"    {name}: {imp:.0f}")

        reg_artifact = {
            "booster": reg_booster.model_to_string(),
            "params": reg_params,
            "encodings": encodings,
            "target": target_col,
        }
        reg_path = os.path.join(args.output_dir, f"{model_name}.pkl")
        with open(reg_path, "wb") as f:
            pickle.dump(reg_artifact, f)
        print(f"  Saved: {reg_path}")

        # ── Combined evaluation: P(revenue>0) × predicted_revenue ──
        if target_col == "total_revenue":
            print(f"\n{'='*60}")
            print("Combined evaluation (classifier × regressor):")

            combined_preds = cls_preds * reg_booster.predict(X_test)
            y_test_revenue = test_df["total_revenue"].values
            combined_metrics = evaluate_ranking(y_test_revenue, combined_preds)
            all_metrics["combined_revenue"] = combined_metrics

            print(f"    MAE:                  {combined_metrics['mae']:.6f}")
            print(f"    Spearman correlation: {combined_metrics['spearman']:.4f}")
            print(f"    Precision@100:        {combined_metrics['precision_at_100']:.1f}%")
            print(f"    Revenue capture (top 50%): {combined_metrics['revenue_capture_top50pct']:.1f}%")

        all_metrics[target_col] = reg_metrics

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Classifier: AUC={all_metrics['classifier']['auc']:.4f}")
    if "combined_revenue" in all_metrics:
        m = all_metrics["combined_revenue"]
        print(f"  Combined revenue: Spearman={m['spearman']:.4f}  P@100={m['precision_at_100']:.1f}%  RevCapture={m['revenue_capture_top50pct']:.1f}%")
    for target in ["total_revenue", "rps"]:
        if target in all_metrics:
            m = all_metrics[target]
            print(f"  {target} (pos-only): Spearman={m['spearman']:.4f}  P@100={m['precision_at_100']:.1f}%")

    metrics_path = os.path.join(args.output_dir, "profitability_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
