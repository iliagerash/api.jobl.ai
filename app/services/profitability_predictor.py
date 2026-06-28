import json
import logging
import os
import pickle
import re

import lightgbm as lgb
import numpy as np

logger = logging.getLogger("jobl.api.profitability_predictor")

PRIORITY_TIERS = [
    (0.90, "high"),
    (0.60, "medium"),
    (0.25, "low"),
    (0.0, "minimal"),
]


class ProfitabilityPredictor:
    def __init__(self, models_dir: str) -> None:
        self._ready = False

        try:
            cls_path = os.path.join(models_dir, "profitability_classifier.pkl")
            rev_path = os.path.join(models_dir, "profitability_revenue.pkl")
            rps_path = os.path.join(models_dir, "profitability_rps.pkl")

            if not all(os.path.exists(p) for p in [cls_path, rev_path, rps_path]):
                logger.warning("profitability models not found in %s", models_dir)
                return

            with open(cls_path, "rb") as f:
                cls_artifact = pickle.load(f)
            self._classifier = lgb.Booster(model_str=cls_artifact["booster"])
            self._encodings = cls_artifact["encodings"]

            with open(rev_path, "rb") as f:
                rev_artifact = pickle.load(f)
            self._revenue_model = lgb.Booster(model_str=rev_artifact["booster"])

            with open(rps_path, "rb") as f:
                rps_artifact = pickle.load(f)
            self._rps_model = lgb.Booster(model_str=rps_artifact["booster"])

            cat_path = os.path.join(models_dir, "profitability_categories.json")
            if os.path.exists(cat_path):
                with open(cat_path) as f:
                    self._cat_values = json.load(f)
            else:
                self._cat_values = {}

            self._pandas_categorical = self._classifier.dump_model().get("pandas_categorical", [])

            self._ready = True
            logger.info("profitability predictor loaded from %s", models_dir)
        except Exception as exc:
            raise RuntimeError(f"failed to load profitability predictor from {models_dir}") from exc

    def is_ready(self) -> bool:
        return self._ready

    def predict(self, title: str, company_name: str | None, country_code: str,
                city_title: str | None, region_title: str | None,
                category_id: int | None, destination: str | None,
                is_remote: bool, contract: str | None,
                salary_min: float | None, salary_max: float | None,
                salary_period: str | None, description: str | None,
                city_population_tier: int = 1,
                published_at=None) -> dict | None:

        enc = self._encodings

        def te(col, value):
            mapping = enc.get(col, {})
            return mapping.get(str(value) if value is not None else "", mapping.get("__global_mean__", 0))

        word_count = len(re.findall(r"\w+", (description or "")[:5000]))

        from datetime import datetime
        if published_at:
            dow = published_at.weekday() if hasattr(published_at, "weekday") else 0
            month = published_at.month if hasattr(published_at, "month") else 1
        else:
            now = datetime.utcnow()
            dow = now.weekday()
            month = now.month

        features = {
            "title_encoded": te("title", title),
            "company_name_encoded": te("company_name", company_name),
            "city_title_encoded": te("city_title", city_title),
            "region_title_encoded": te("region_title", region_title),
            "category_encoded": te("category_id", category_id),
            "country": country_code or "US",
            "destination_cat": destination or "unknown",
            "salary_present": 1 if salary_min and salary_min > 0 else 0,
            "salary_min": float(salary_min or 0),
            "salary_max": float(salary_max or 0),
            "salary_period_cat": salary_period or "unknown",
            "work_mode": "remote" if is_remote else "onsite",
            "contract_type": contract or "unknown",
            "description_word_count": word_count,
            "city_population_tier": city_population_tier,
            "day_of_week_posted": dow,
            "month_posted": month,
        }

        import pandas as pd

        row = {
            "title_encoded": features["title_encoded"],
            "company_name_encoded": features["company_name_encoded"],
            "city_title_encoded": features["city_title_encoded"],
            "region_title_encoded": features["region_title_encoded"],
            "category_encoded": features["category_encoded"],
            "country": features["country"],
            "destination_cat": features["destination_cat"],
            "salary_present": features["salary_present"],
            "salary_min": features["salary_min"],
            "salary_max": features["salary_max"],
            "salary_period_cat": features["salary_period_cat"],
            "work_mode": features["work_mode"],
            "contract_type": features["contract_type"],
            "description_word_count": features["description_word_count"],
            "city_population_tier": features["city_population_tier"],
            "day_of_week_posted": features["day_of_week_posted"],
            "month_posted": features["month_posted"],
        }
        df = pd.DataFrame([row])

        cat_cols = ["country", "destination_cat", "salary_period_cat", "work_mode", "contract_type"]
        for i, col in enumerate(cat_cols):
            if i < len(self._pandas_categorical):
                cats = list(self._pandas_categorical[i])
                val = df[col].iloc[0]
                if val not in cats:
                    cats.append(val)
                df[col] = pd.Categorical(df[col], categories=cats)

        p_revenue = float(self._classifier.predict(df)[0])
        predicted_revenue = float(self._revenue_model.predict(df)[0])
        predicted_rps = float(self._rps_model.predict(df)[0])

        combined_revenue = p_revenue * max(0, predicted_revenue)

        # Compute priority tier based on combined revenue percentile
        # Using fixed thresholds (can be tuned)
        if combined_revenue > 0.1:
            tier = "high"
        elif combined_revenue > 0.01:
            tier = "medium"
        elif combined_revenue > 0.001:
            tier = "low"
        else:
            tier = "minimal"

        return {
            "predicted_revenue": round(combined_revenue, 6),
            "predicted_rps": round(max(0, predicted_rps), 6),
            "priority_tier": tier,
        }
