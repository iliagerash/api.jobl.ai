import logging
import pickle

import numpy as np

logger = logging.getLogger("jobl.api.categorizer")

# If the model's top confidence for these category IDs falls below the threshold,
# the prediction is downgraded to Other (id=22) to avoid confident mislabels.
_CONFIDENCE_THRESHOLDS: dict[int, float] = {
    17: 0.5,  # Energy & Natural Resources — low F1, needs higher bar
}


class JobCategorizer:
    def __init__(self, model_path: str) -> None:
        self._ready = False
        try:
            with open(model_path, "rb") as f:
                artifact = pickle.load(f)
            self._tfidf = artifact["tfidf"]
            self._booster = artifact["booster"]
            self._id_to_category = artifact["id_to_category"]
            self._ready = True
            logger.info("categorizer loaded from %s", model_path)
        except Exception as exc:
            raise RuntimeError(f"failed to load categorizer from {model_path}") from exc

    def is_ready(self) -> bool:
        return self._ready

    def predict(self, title: str, desc_plain: str) -> dict:
        text = f"{title} {title} {title} {desc_plain}"
        X = self._tfidf.transform([text])
        probs = self._booster.predict(X)           # shape (1, num_classes)
        pred_class = int(np.argmax(probs[0]))      # 0-based
        confidence = float(probs[0][pred_class])
        category_id = pred_class + 1
        min_conf = _CONFIDENCE_THRESHOLDS.get(category_id)
        if min_conf is not None and confidence < min_conf:
            other_id = max(self._id_to_category)   # Other is always the last category
            return {**self._id_to_category[other_id], "confidence": confidence}
        return {**self._id_to_category[category_id], "confidence": confidence}
