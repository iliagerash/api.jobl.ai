import json
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
                        "cat_values": artifact.get("cat_values", {}),
                    }
            cat_path = os.path.join(models_dir, "salary_categories.json")
            if os.path.exists(cat_path):
                with open(cat_path) as f:
                    self._cat_values = json.load(f)
            else:
                self._cat_values = {}

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

        cat_cols = {"country": 3, "work_mode": 4, "contract_type": 5}
        row = [
            features["title_encoded"],
            features["city_title_encoded"],
            features["category_encoded"],
            features["country"],
            features["work_mode"],
            features["contract_type"],
            features["description_word_count"],
        ]
        # Encode categoricals as integer codes matching training order
        for col, idx in cat_cols.items():
            cats = self._cat_values.get(col, [])
            val = row[idx]
            row[idx] = cats.index(val) if val in cats else len(cats)

        data = np.array([row], dtype=np.float64)

        salary_min = float(model["min"].predict(data)[0])
        salary_max = float(model["max"].predict(data)[0])

        salary_min = max(0, salary_min)
        salary_max = max(salary_min, salary_max)

        period = "monthly" if country_code in MONTHLY_COUNTRIES else "yearly"

        return {
            "salary_min": round(salary_min, 2),
            "salary_max": round(salary_max, 2),
            "salary_period": period,
        }
