import logging
import pickle

import numpy as np

logger = logging.getLogger("jobl.api.categorizer")


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

    def predict(self, title: str, original_category: str | None, desc_plain: str) -> dict:
        text = f"{title} {original_category or ''} {desc_plain[:1000]}"
        X = self._tfidf.transform([text])
        probs = self._booster.predict(X)           # shape (1, num_classes)
        pred_class = int(np.argmax(probs[0]))      # 0-based
        return self._id_to_category[pred_class + 1]  # back to 1-based id
