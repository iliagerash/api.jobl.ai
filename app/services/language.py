from __future__ import annotations

from dataclasses import dataclass

import langid


ALLOWED_LANGUAGE_CODES = {"en", "es", "de", "fr", "pt", "it", "gr", "uk", "nl", "da"}

# Default language assumptions per country, based on current source topology.
PRIMARY_LANGUAGE_BY_COUNTRY = {
    "US": "en",
    "AU": "en",
    "NZ": "en",
    "GB": "en",
    "ZA": "en",
    "IN": "en",
    "DE": "de",
    "AT": "de",
    "FR": "fr",
    "PT": "pt",
    "BR": "pt",
    "GR": "gr",
    "DK": "da",
    "NL": "nl",
    "UA": "uk",
    "IT": "it",
}

MIXED_LANGUAGE_COUNTRIES = {"CA", "CH", "SG"}
PRIMARY_LANGUAGE_BY_SOURCE_DB = {"americas": "es"}


@dataclass(frozen=True)
class LanguageDetectionResult:
    language_code: str | None
    detector_code: str | None
    score: float | None


def detect_language_code(
    *,
    title: str | None,
    description: str | None,
    country_code: str | None,
    source_db: str | None,
) -> LanguageDetectionResult:
    normalized_country = (country_code or "").strip().upper() or None
    normalized_source_db = (source_db or "").strip().lower() or None

    text_blob = _build_text_blob(title=title, description=description)
    detector_code = None
    score = None
    detected = None

    if text_blob:
        detector_code, score = langid.classify(text_blob)
        detected = _map_detector_language(detector_code)

    if detected in ALLOWED_LANGUAGE_CODES:
        return LanguageDetectionResult(language_code=detected, detector_code=detector_code, score=score)

    # For mixed-language countries, do not force a default if detector is outside allowlist.
    if normalized_country in MIXED_LANGUAGE_COUNTRIES:
        return LanguageDetectionResult(language_code=None, detector_code=detector_code, score=score)

    fallback = None
    if normalized_source_db and normalized_source_db in PRIMARY_LANGUAGE_BY_SOURCE_DB:
        fallback = PRIMARY_LANGUAGE_BY_SOURCE_DB[normalized_source_db]
    elif normalized_country and normalized_country in PRIMARY_LANGUAGE_BY_COUNTRY:
        fallback = PRIMARY_LANGUAGE_BY_COUNTRY[normalized_country]

    if fallback in ALLOWED_LANGUAGE_CODES:
        return LanguageDetectionResult(language_code=fallback, detector_code=detector_code, score=score)

    return LanguageDetectionResult(language_code=None, detector_code=detector_code, score=score)


def _build_text_blob(*, title: str | None, description: str | None) -> str:
    title_text = (title or "").strip()
    description_text = (description or "").strip()
    if not description_text:
        return title_text[:400]
    return f"{title_text}\n{description_text[:1500]}".strip()


def _map_detector_language(lang: str | None) -> str | None:
    if not lang:
        return None
    code = lang.strip().lower()
    if code == "el":
        return "gr"
    return code
