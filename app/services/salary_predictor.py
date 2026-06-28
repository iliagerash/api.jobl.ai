import logging
import os
import pickle
import re

import lightgbm as lgb
import numpy as np

logger = logging.getLogger("jobl.api.salary_predictor")

MONTHLY_COUNTRIES = {"SG", "ZA"}


class SalaryPredictor:
    def __init__(self, models_dir: str) -> None:
        self._ready = False
        self._models: dict[str, dict] = {}

        try:
            for filename in os.listdir(models_dir):
                if filename.startswith("salary_") and filename.endswith(".pkl"):
                    cc = filename.replace("salary_", "").replace(".pkl", "")
                    path = os.path.join(models_dir, filename)
                    with open(path, "rb") as f:
                        artifact = pickle.load(f)
                    self._models[cc] = {
                        "min": lgb.Booster(model_str=artifact["boosters"]["target_salary_min"]),
                        "max": lgb.Booster(model_str=artifact["boosters"]["target_salary_max"]),
                        "encodings": artifact["encodings"],
                    }
            if self._models:
                self._ready = True
                logger.info("salary predictor loaded: %s", sorted(self._models.keys()))
        except Exception as exc:
            raise RuntimeError(f"failed to load salary predictor from {models_dir}") from exc

    def is_ready(self) -> bool:
        return self._ready

    def predict(self, title: str, country_code: str, city_title: str | None,
                region_title: str | None, category_id: int | None,
                is_remote: bool, contract: str | None,
                description: str | None) -> dict | None:
        if country_code not in self._models:
            return None

        model = self._models[country_code]
        enc = model["encodings"]

        title_enc = enc.get("title", {})
        title_encoded = title_enc.get(title, title_enc.get("__global_mean__", 0))

        city_enc = enc.get("city_title", {})
        city_encoded = city_enc.get(city_title or "", city_enc.get("__global_mean__", 0))

        region_enc = enc.get("region_title", {})
        region_encoded = region_enc.get(region_title or "", region_enc.get("__global_mean__", 0))

        cat_enc = enc.get("category_id", {})
        cat_encoded = cat_enc.get(str(category_id or 0), cat_enc.get("__global_mean__", 0))

        word_count = len(re.findall(r"\w+", (description or "")[:5000]))

        features = {
            "title_encoded": title_encoded,
            "city_title_encoded": city_encoded,
            "region_title_encoded": region_encoded,
            "category_encoded": cat_encoded,
            "country": country_code,
            "work_mode": "remote" if is_remote else "onsite",
            "contract_type": contract or "unknown",
            "description_word_count": word_count,
        }

        import pandas as pd
        df = pd.DataFrame([features])
        df["country"] = df["country"].astype("category")
        df["work_mode"] = df["work_mode"].astype("category")
        df["contract_type"] = df["contract_type"].astype("category")

        salary_min = float(model["min"].predict(df)[0])
        salary_max = float(model["max"].predict(df)[0])

        salary_min = max(0, salary_min)
        salary_max = max(salary_min, salary_max)

        period = "monthly" if country_code in MONTHLY_COUNTRIES else "yearly"

        return {
            "salary_min": round(salary_min, 2),
            "salary_max": round(salary_max, 2),
            "salary_period": period,
        }
