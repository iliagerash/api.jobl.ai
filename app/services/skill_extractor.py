import logging
import re

from sqlalchemy import text

logger = logging.getLogger("jobl.api.skill_extractor")


class SkillExtractor:
    """Hybrid skill extraction:
    - English: SkillNER (NER-based, 60K EMSI skills) for richer extraction
    - All languages: ESCO taxonomy matching from Postgres as baseline/fallback
    """

    def __init__(self, db_session_factory) -> None:
        self._ready = False
        self._skillner = None
        self._nlp = None
        self._esco_labels: dict[str, dict[str, str]] = {}  # lang -> {lowercase_label: preferred_en}
        self._esco_all: dict[str, str] = {}  # lowercase_label -> preferred_en

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

            # Sort by label length descending (longer matches take priority)
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

        # Load SkillNER for English
        try:
            import spacy
            from skillNer.general_params import SKILL_DB
            from skillNer.skill_extractor_class import SkillExtractor as _SkillNER

            nlp = spacy.load("en_core_web_lg")
            self._skillner = _SkillNER(nlp, SKILL_DB)
            self._nlp = nlp
            logger.info("SkillNER loaded (en_core_web_lg + EMSI skill DB)")
        except ImportError:
            logger.warning("SkillNER or spaCy not available; English uses ESCO fallback only")
        except OSError:
            logger.warning("en_core_web_lg not installed; English uses ESCO fallback only. Install with: python -m spacy download en_core_web_lg")

        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def extract_skills(self, text_input: str, language: str | None = None) -> list[str]:
        """Extract skills from text. Returns deduplicated English skill labels."""
        if not text_input:
            return []

        found: dict[str, bool] = {}

        # For English: use SkillNER first (catches informal mentions)
        if language in ("en", None) and self._skillner is not None:
            try:
                annotations = self._skillner.annotate(text_input)
                for match in annotations.get("results", {}).get("full_matches", []):
                    label = match.get("doc_node_value", "")
                    if label:
                        found[label] = True
                for match in annotations.get("results", {}).get("ngram_scored", []):
                    if match.get("score", 0) >= 0.5:
                        label = match.get("doc_node_value", "")
                        if label:
                            found[label] = True
            except Exception:
                logger.debug("SkillNER extraction failed, falling back to ESCO", exc_info=True)

        # ESCO dictionary matching (all languages, supplements SkillNER for English)
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
