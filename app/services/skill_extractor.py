import logging
import re

from sqlalchemy import text

logger = logging.getLogger("jobl.api.skill_extractor")


class SkillExtractor:
    """Extracts skills from text by matching against the skill taxonomy in Postgres.

    Loads all skill labels into memory at init, grouped by language.
    Matching is case-insensitive with word-boundary checks for short labels.
    """

    def __init__(self, db_session_factory) -> None:
        self._ready = False
        self._labels: dict[str, dict[str, str]] = {}  # lang -> {lowercase_label: preferred_label_en}
        self._all_labels: dict[str, str] = {}  # lowercase_label -> preferred_label_en

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
                self._labels.setdefault(lang, {})[lower] = preferred_en
                self._all_labels[lower] = preferred_en

            # Sort by label length descending so longer matches take priority
            for lang in self._labels:
                self._labels[lang] = dict(
                    sorted(self._labels[lang].items(), key=lambda x: -len(x[0]))
                )
            self._all_labels = dict(
                sorted(self._all_labels.items(), key=lambda x: -len(x[0]))
            )

            total_labels = sum(len(v) for v in self._labels.values())
            self._ready = True
            logger.info("skill extractor loaded: %d languages, %d labels", len(self._labels), total_labels)
        except Exception:
            logger.exception("failed to load skill taxonomy from database")

    def is_ready(self) -> bool:
        return self._ready

    def extract_skills(self, text_input: str, language: str | None = None) -> list[str]:
        """Extract skills from text. Returns deduplicated English skill labels."""
        if not text_input:
            return []

        text_lower = text_input.lower()
        found: dict[str, bool] = {}

        # Try language-specific labels first, then fall back to all labels
        label_dicts = []
        if language and language in self._labels:
            label_dicts.append(self._labels[language])
        label_dicts.append(self._all_labels)

        for labels in label_dicts:
            for label_lower, preferred_en in labels.items():
                if preferred_en in found:
                    continue
                if len(label_lower) < 3:
                    continue
                if re.search(rf"\b{re.escape(label_lower)}\b", text_lower):
                    found[preferred_en] = True

        return list(found.keys())
