import langid


ALLOWED_LANGUAGE_CODES = {"en", "es", "de", "fr", "pt", "it", "gr", "uk", "nl", "da"}


def detect_language_code(title: str | None) -> str | None:
    text = str(title or "").strip()[:400]
    if not text:
        return None
    detector_code, _score = langid.classify(text)
    mapped = _map_detector_language(detector_code)
    if mapped in ALLOWED_LANGUAGE_CODES:
        return mapped
    return None


def _map_detector_language(lang: str | None) -> str | None:
    if not lang:
        return None
    code = lang.strip().lower()
    if code == "el":
        return "gr"
    return code
