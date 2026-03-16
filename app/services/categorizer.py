import logging
import pickle

logger = logging.getLogger("jobl.api.categorizer")


class JobCategorizer:
    def __init__(self, model_path: str) -> None:
        self._ready = False
        try:
            with open(model_path, "rb") as f:
                artifact = pickle.load(f)
            self._pipeline = artifact["pipeline"]            # sklearn Pipeline (TF-IDF + LightGBM)
            self._id_to_category = artifact["id_to_category"]  # {int: {"id": int, "title": str}}
            self._ready = True
            logger.info("categorizer loaded from %s", model_path)
        except Exception as exc:
            raise RuntimeError(f"failed to load categorizer from {model_path}") from exc

    def is_ready(self) -> bool:
        return self._ready

    def predict(self, title: str, original_category: str | None, desc_plain: str) -> dict:
        text = f"{title} {original_category or ''} {desc_plain[:1000]}"
        pred_id = int(self._pipeline.predict([text])[0])
        return self._id_to_category[pred_id]
