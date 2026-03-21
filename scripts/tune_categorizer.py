"""
tune_categorizer.py
────────────────────
Optuna hyperparameter search for the LightGBM job categorizer.

Vectorizes the training data once, then runs Optuna trials varying the
LightGBM params. Reports the best accuracy and the params dict ready to
paste into train_categorizer.py.

Usage:
    python scripts/tune_categorizer.py \\
        --data data/categorizer_training.csv \\
        --n-trials 50

Options:
    --data        Path to categorizer_training.csv (default: data/categorizer_training.csv)
    --n-trials    Number of Optuna trials (default: 50)
    --timeout     Stop after N seconds regardless of trial count (optional)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import optuna
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb

from scripts.generate_training_data import CATEGORIES


def build_text(row: pd.Series) -> str:
    title = str(row.get("title") or "")
    description = str(row.get("description_plaintext") or "")
    return f"{title} {title} {title} {description}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune LightGBM categorizer hyperparameters with Optuna")
    parser.add_argument("--data", default="data/categorizer_training.csv")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=None, help="Stop after N seconds")
    args = parser.parse_args()

    print(f"Loading training data from {args.data} ...")
    df = pd.read_csv(args.data).fillna("")
    print(f"  {len(df)} rows loaded")
    if df.empty:
        sys.exit("Error: training data is empty. Run generate_training_data.py first.")

    num_classes = len(CATEGORIES)
    X_text = df.apply(build_text, axis=1).tolist()
    y = (df["category_id"].astype(int) - 1).tolist()

    print("Vectorizing text (TF-IDF) ...", flush=True)
    n_rows = len(df)
    max_features = min(20_000, max(2_000, n_rows // 2))
    min_df = 1 if n_rows < 10_000 else 5
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=min_df, sublinear_tf=True)
    X_vec = tfidf.fit_transform(X_text)
    print(f"  Done — matrix shape: {X_vec.shape}", flush=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X_vec, y, test_size=0.1, random_state=42, stratify=y
    )
    # Keep raw data in the Dataset so it can be reused across trials
    train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, free_raw_data=False)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "multiclass",
            "num_class": num_classes,
            "metric": "multi_logloss",
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "n_jobs": -1,
            "force_col_wise": True,
            "verbosity": -1,
        }
        booster = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=9999),
            ],
        )
        y_pred = np.argmax(booster.predict(X_val), axis=1)
        return float(accuracy_score(y_val, y_pred))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    print(f"\nRunning {args.n_trials} trials ...\n", flush=True)
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, show_progress_bar=True)

    print(f"\nBest validation accuracy: {study.best_value:.4f}")
    print("\nBest params (paste into train_categorizer.py):")
    print("    params = {")
    print('        "objective": "multiclass",')
    print(f'        "num_class": {num_classes},')
    print('        "metric": "multi_logloss",')
    for k, v in study.best_params.items():
        if isinstance(v, float):
            print(f'        "{k}": {v},')
        else:
            print(f'        "{k}": {v},')
    print('        "n_jobs": -1,')
    print('        "force_col_wise": True,')
    print('        "verbosity": -1,')
    print("    }")


if __name__ == "__main__":
    main()
