import logging
import re

from sqlalchemy import text

logger = logging.getLogger("jobl.api.skill_extractor")

SKILL_NER_MODEL = "Nucha/Nucha_ITSkillNER_BERT"


class SkillExtractor:
    """Hybrid skill extraction:
    - English: BERT-based NER (Nucha_ITSkillNER_BERT) for tech + soft skills
    - All languages: ESCO taxonomy matching from Postgres as baseline/fallback
    """

    def __init__(self, db_session_factory, skill_ner_model: str | None = SKILL_NER_MODEL) -> None:
        self._ready = False
        self._ner_pipeline = None
        self._esco_labels: dict[str, dict[str, str]] = {}
        self._esco_all: dict[str, str] = {}

        # Load ESCO labels from Postgres
        try:
            db = db_session_factory()
            try:
                rows = db.execute(text("""
                    SELECT sl.language_code, sl.label, s.preferred_label_en
                    FROM skill_labels sl
                    JOIN skills s ON s.uri = sl.skill_uri
                """)).fetchall()
            finally:
                db.close()

            for lang, label, preferred_en in rows:
                lower = label.lower()
                self._esco_labels.setdefault(lang, {})[lower] = preferred_en
                self._esco_all[lower] = preferred_en

            for lang in self._esco_labels:
                self._esco_labels[lang] = dict(
                    sorted(self._esco_labels[lang].items(), key=lambda x: -len(x[0]))
                )
            self._esco_all = dict(
                sorted(self._esco_all.items(), key=lambda x: -len(x[0]))
            )

            total_labels = sum(len(v) for v in self._esco_labels.values())
            logger.info("ESCO taxonomy loaded: %d languages, %d labels", len(self._esco_labels), total_labels)
        except Exception:
            logger.exception("failed to load ESCO taxonomy from database")

        # Load BERT NER for English skill extraction
        if skill_ner_model:
            try:
                from transformers import pipeline
                self._ner_pipeline = pipeline(
                    "ner",
                    model=skill_ner_model,
                    aggregation_strategy="simple",
                )
                logger.info("Skill NER loaded: %s", skill_ner_model)
            except Exception as exc:
                logger.warning("Skill NER failed to load: %s — English uses ESCO fallback only", exc)

        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def extract_skills(self, text_input: str, language: str | None = None) -> list[str]:
        """Extract skills from text. Returns deduplicated English skill labels."""
        if not text_input:
            return []

        found: dict[str, bool] = {}

        # For English: use BERT NER first (catches tech skills and informal mentions)
        if language in ("en", None) and self._ner_pipeline is not None:
            try:
                entities = self._ner_pipeline(text_input[:2000])
                for ent in entities:
                    label = ent.get("word", "").strip()
                    if label and len(label) >= 2:
                        # Clean up BERT tokenizer artifacts
                        label = re.sub(r"\s*##\s*", "", label).strip()
                        if label:
                            found[label] = True
            except Exception:
                logger.debug("Skill NER extraction failed, falling back to ESCO", exc_info=True)

        # ESCO dictionary matching (all languages, supplements NER for English)
        text_lower = text_input.lower()

        label_dicts = []
        if language and language in self._esco_labels:
            label_dicts.append(self._esco_labels[language])
        label_dicts.append(self._esco_all)

        for labels in label_dicts:
            for label_lower, preferred_en in labels.items():
                if preferred_en in found:
                    continue
                if len(label_lower) < 2:
                    continue
                if len(label_lower) <= 4:
                    if re.search(rf"\b{re.escape(label_lower)}\b", text_lower):
                        found[preferred_en] = True
                else:
                    if label_lower in text_lower:
                        found[preferred_en] = True

        return list(found.keys())
