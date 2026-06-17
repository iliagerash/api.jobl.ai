"""
pii_scrub.py
────────────
Phase 0, step 3: strip PII from resume descriptions before any ML training
data is constructed.  Hard blocker — manual review of 200 scrubbed samples is
required before proceeding.

Detection layers (applied in order, with span deduplication):
  1. Regex — emails, URLs, phone numbers, street addresses  (high precision)
  2. spaCy NER — PERSON entities via language-specific models (en, es, pt)
     with a validation filter to suppress common false positives
  3. Pattern-based — name-introduction phrases in top languages (ES/EN/PT/DE/FR)

All detected spans are merged (overlapping spans consolidated) and replaced
with typed placeholders: [NAME], [EMAIL], [PHONE], [URL], [ADDRESS].

Setup (one-time, on the server):
    pip install -e ".[ml]"
    python -m spacy download en_core_web_sm
    python -m spacy download es_core_news_sm
    python -m spacy download pt_core_news_sm
    python -m spacy download fr_core_news_sm

Usage:
    python -u ml/scripts/pii_scrub.py
    python -u ml/scripts/pii_scrub.py --sample 200
    python -u ml/scripts/pii_scrub.py --input ml/data/raw/resumes.jsonl --output-dir ml/data/interim

Outputs:
    ml/data/interim/resumes.jsonl          — PII-scrubbed resumes
    ml/data/interim/pii_scrub_sample.jsonl — before/after pairs (with --sample)
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

HEARTBEAT_EVERY = 10_000

PLACEHOLDER = {
    "EMAIL": "[EMAIL]",
    "URL": "[URL]",
    "PHONE": "[PHONE]",
    "ADDRESS": "[ADDRESS]",
    "NAME": "[NAME]",
}

# ── Regex patterns ──────────────────────────────────────────────────────────

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

URL_RE = re.compile(r"https?://\S+")

# Phone: international format (+XX ...) — always a phone number
INTL_PHONE_RE = re.compile(
    r"\+\d{1,3}[\s\-\.]?\(?\d{1,4}\)?[\s\-\.]\d[\d\s\-\.]{4,14}\d"
)

# Phone: preceded by a keyword (Tel, Mobil, Celular, Téléphone, etc.)
PHONE_CONTEXT_RE = re.compile(
    r"(?:Tel(?:[eé]fon[eo]?)?|Phone|Mobil(?:e)?|Handy|Fax|Cell"
    r"|T[eé]l[eé]?(?:phone)?|Celular|Τηλ(?:έφωνο)?|Телефон"
    r"|Contact|Phone\s*number)"
    r"\s*[:\.\-]?\s*"
    r"([\+\d\(\[][\d\s\-\./\(\)\[\]]{6,20}\d[\]\)]?)",
    re.IGNORECASE,
)

# Phone: Australian mobile (04XX) — always a phone number regardless of context
AU_MOBILE_RE = re.compile(r"\b04\d{2}[\s\-\.]?\d{3}[\s\-\.]?\d{3}\b")

# Phone: bare domestic with separators (0XXX XXX XXXX) — 8+ digits starting with 0
BARE_DOMESTIC_PHONE_RE = re.compile(r"\b0\d{1,3}[\s\-\.]\d{3,4}[\s\-\.]\d{3,5}\b")

# ── Address patterns ───────────────────────────────────────────────────────

# German street address: "Musterstraße 12" / "Am Markt 5a"
STREET_DE_RE = re.compile(
    r"\b[A-ZÄÖÜ][a-zäöüß]*(?:stra[sß]e|str\.|weg|gasse|platz|ring|allee|damm)"
    r"\s+\d+[a-z]?\b",
    re.IGNORECASE,
)

# English/AU street address: "123 Main Street" / "5a Baker Rd" / "42 Smith Ave"
STREET_EN_RE = re.compile(
    r"\b\d+[a-z]?\s+"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+"
    r"(?:Street|St|Road|Rd|Avenue|Ave|Drive|Dr|Boulevard|Blvd"
    r"|Lane|Ln|Place|Pl|Crescent|Cres|Circuit|Cct|Court|Ct"
    r"|Close|Cl|Loop|Way|Parade|Pde|Terrace|Tce|TCE)\b\.?",
    re.IGNORECASE,
)

# AU/general: suburb + state abbreviation + postcode (e.g. "Taringa QLD 4068")
AU_SUBURB_STATE_POST_RE = re.compile(
    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+"
    r"(?:NSW|VIC|QLD|SA|WA|TAS|NT|ACT)\s*,?\s*"
    r"\d{4}\b"
)

# Postal code + city (DE: 5 digits, ES: 5 digits, PT: 4-4 or 4 digits, FR: 5 digits)
POSTAL_CITY_RE = re.compile(
    r"\b\d{4,5}[\s\-]?\d{0,3}\s+[A-ZÄÖÜÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÃÕÑÇ][a-zäöüßáéíóúàèìòùâêîôûãõñç]{2,}\b"
)

# Address preceded by context keyword
ADDRESS_LABEL_RE = re.compile(
    r"(?:Address|Adresse|Anschrift|Direc(?:ción|tion)|Endereço|Location)"
    r"\s*[:\.\-]\s*"
    r"(.+?)(?:\n|$)",
    re.IGNORECASE,
)

# ── Name-introduction patterns (top languages) ──────────────────────────────

_NAME_CHUNK = r"[A-ZÄÖÜÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÃÕÑÇΑ-ΩҐЄІЇ][a-zäöüßáéíóúàèìòùâêîôûãõñçα-ωґєії']+"

NAME_INTRO_RE = re.compile(
    r"(?:"
    # Spanish
    r"[Mm]i nombre (?:es|completo es)"
    r"|[Mm]e llamo"
    # English
    r"|[Mm]y name is"
    # Portuguese
    r"|[Mm]eu nome [eé]"
    r"|[Mm]e chamo"
    # German
    r"|[Mm]ein Name ist"
    r"|[Ii]ch hei[sß]e"
    # French
    r"|[Jj]e m['']appelle"
    r")\s*"
    rf"({_NAME_CHUNK}(?:\s+{_NAME_CHUNK}){{1,4}})",
    re.UNICODE,
)

# "Vor- und Zuname:", "Nombre:", "Nome:", "Name:", "Full name:" etc.
NAME_LABEL_RE = re.compile(
    r"(?:Vor-?\s*(?:und|&)\s*(?:Zu|Nach)name|(?:Nombre|Nome)\s*(?:completo)?|Full\s*name|Name)"
    r"\s*[:\.\-]\s*"
    rf"({_NAME_CHUNK}(?:\s+{_NAME_CHUNK}){{1,4}})",
    re.IGNORECASE | re.UNICODE,
)

# Salutations: Herr/Frau, Sr./Sra., Mr./Mrs./Ms.
SALUTATION_RE = re.compile(
    r"(?:Herr|Frau|[Ss]r\.?a?|[Dd]on|[Dd]oña|Mr\.?|Mrs\.?|Ms\.?|Mme\.?)"
    r"\s+"
    rf"({_NAME_CHUNK}(?:\s+{_NAME_CHUNK}){{1,3}})",
    re.UNICODE,
)

# ── NER false-positive filter ──────────────────────────────────────────────

# Common words the multilingual/small NER models wrongly tag as PERSON.
# Built from actual false positives in the 200-sample gate review.
_NER_BLOCKLIST = frozenset(w.lower() for w in [
    # Job titles / roles
    "Sales", "Customer", "Cook", "Cleaner", "Cleaning", "Supervisor",
    "Process", "Worker", "Cashiering", "Sanitizing", "Catering",
    "Porter", "Nurse", "Carer", "Barring", "Agile", "Scrum",
    "Tableau", "Genesys", "Remedy", "Sutherland", "Analytics",
    "Respiratory", "Allied", "Cert", "Strong", "Complementary",
    "Adaptable", "Flexible", "Enthusiastic", "Motivated", "Resume",
    "Profile", "Curriculum", "Vitae", "Gaming", "Retail", "Data",
    "Analyst", "Designer", "Assistant", "Attendant", "Electrician",
    "Store", "Unit", "Manager", "Garment", "Senior", "Junior",
    "Individual", "Support", "Health", "Drug", "Covid",
    # Common sentence starters / adjectives NER misreads
    "Dear", "Hi", "Hello", "Thank", "Good", "Please",
    "Seeking", "Looking", "Experienced", "Dedicated", "Passionate",
    "Reliable", "Hardworking", "Quick", "Fast", "Professional",
    # Tools / brands wrongly tagged
    "Winyama", "Clough", "Moxy", "Excavator", "Loader",
    "Hobart", "Altman", "Solon", "Bhatbhatni",
    # Generic words
    "Fluids", "Certificates", "Resident", "Engineer",
])


def _is_plausible_name(text: str) -> bool:
    """Filter NER PERSON entities to reduce false positives.

    Accepts an entity only if it:
      - Has 2+ words (single common words are almost always false positives), OR
      - Is a single word not in the blocklist AND is title-cased (not all-lower, not all-upper)
      - Is not entirely numeric or a single character
    """
    stripped = text.strip()
    if len(stripped) <= 1:
        return False
    words = stripped.split()
    if len(words) == 1:
        word = words[0]
        if word.lower() in _NER_BLOCKLIST:
            return False
        if not (word[0].isupper() and not word.isupper()):
            return False
        return True
    # Multi-word: check if any word is blocklisted
    if any(w.lower() in _NER_BLOCKLIST for w in words):
        return False
    return True


# ── Span utilities ──────────────────────────────────────────────────────────

def _merge_spans(spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    if not spans:
        return []
    sorted_spans = sorted(spans, key=lambda s: (s[0], -(s[1] - s[0])))
    merged = [sorted_spans[0]]
    for start, end, pii_type in sorted_spans[1:]:
        prev_start, prev_end, prev_type = merged[-1]
        if start < prev_end:
            if end > prev_end:
                keep_type = prev_type if (prev_end - prev_start) >= (end - start) else pii_type
                merged[-1] = (prev_start, end, keep_type)
        else:
            merged.append((start, end, pii_type))
    return merged


def _replace_spans(text: str, spans: list[tuple[int, int, str]]) -> str:
    merged = _merge_spans(spans)
    for start, end, pii_type in reversed(merged):
        text = text[:start] + PLACEHOLDER[pii_type] + text[end:]
    return text


# ── Scrubber ────────────────────────────────────────────────────────────────

# Map resume language_code → spaCy model name
_LANG_TO_MODEL = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "pt": "pt_core_news_sm",
    "de": "de_core_news_sm",
    "fr": "fr_core_news_sm",
}


class PiiScrubber:
    def __init__(self) -> None:
        self._models: dict[str, object] = {}
        self._failed: set[str] = set()
        try:
            import spacy  # noqa: F401
            self._spacy = spacy
        except ImportError:
            print(
                "WARNING: spaCy not installed — falling back to regex-only. Install with:\n"
                "  pip install -e '.[ml]'"
            )
            self._spacy = None

    def _get_nlp(self, language_code: str | None):
        if self._spacy is None:
            return None
        code = (language_code or "").strip().lower()
        model_name = _LANG_TO_MODEL.get(code)
        if model_name is None:
            return None
        if model_name in self._failed:
            return None
        if model_name not in self._models:
            try:
                nlp = self._spacy.load(model_name)
                nlp.max_length = 2_000_000
                self._models[model_name] = nlp
                print(f"  Loaded spaCy model: {model_name}")
            except OSError:
                print(
                    f"  WARNING: spaCy model '{model_name}' not found — "
                    f"no NER for lang={code}. Install with: python -m spacy download {model_name}"
                )
                self._failed.add(model_name)
                return None
        return self._models[model_name]

    def scrub(self, text: str | None, language_code: str | None = None) -> tuple[str, dict[str, int]]:
        if not text:
            return text or "", {}

        spans: list[tuple[int, int, str]] = []
        stats: dict[str, int] = {}

        def _add(start: int, end: int, pii_type: str) -> None:
            spans.append((start, end, pii_type))
            stats[pii_type] = stats.get(pii_type, 0) + 1

        # Layer 1: regex — structured PII (high precision)
        for m in EMAIL_RE.finditer(text):
            _add(m.start(), m.end(), "EMAIL")

        for m in URL_RE.finditer(text):
            _add(m.start(), m.end(), "URL")

        for m in INTL_PHONE_RE.finditer(text):
            _add(m.start(), m.end(), "PHONE")

        for m in PHONE_CONTEXT_RE.finditer(text):
            _add(m.start(1), m.end(1), "PHONE")

        for m in AU_MOBILE_RE.finditer(text):
            _add(m.start(), m.end(), "PHONE")

        for m in BARE_DOMESTIC_PHONE_RE.finditer(text):
            _add(m.start(), m.end(), "PHONE")

        for m in STREET_DE_RE.finditer(text):
            _add(m.start(), m.end(), "ADDRESS")

        for m in STREET_EN_RE.finditer(text):
            _add(m.start(), m.end(), "ADDRESS")

        for m in AU_SUBURB_STATE_POST_RE.finditer(text):
            _add(m.start(), m.end(), "ADDRESS")

        for m in POSTAL_CITY_RE.finditer(text):
            _add(m.start(), m.end(), "ADDRESS")

        for m in ADDRESS_LABEL_RE.finditer(text):
            _add(m.start(1), m.end(1), "ADDRESS")

        # Layer 2: spaCy NER — PERSON entities (language-routed, filtered)
        nlp = self._get_nlp(language_code)
        if nlp is not None:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in ("PER", "PERSON") and _is_plausible_name(ent.text):
                    _add(ent.start_char, ent.end_char, "NAME")

        # Layer 3: pattern-based name detection (catches names NER may miss)
        for m in NAME_INTRO_RE.finditer(text):
            _add(m.start(1), m.end(1), "NAME")

        for m in NAME_LABEL_RE.finditer(text):
            _add(m.start(1), m.end(1), "NAME")

        for m in SALUTATION_RE.finditer(text):
            _add(m.start(1), m.end(1), "NAME")

        scrubbed = _replace_spans(text, spans)
        return scrubbed, stats


# ── I/O ─────────────────────────────────────────────────────────────────────

def iter_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _json_default(v):
    from datetime import date, datetime
    from decimal import Decimal
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    raise TypeError(type(v).__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="PII-scrub resume descriptions for ML training data safety")
    parser.add_argument("--input", default="ml/data/raw/resumes.jsonl")
    parser.add_argument("--output-dir", default="ml/data/interim")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Process only N resumes and write before/after pairs for the manual gate review (plan requires 200)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "resumes.jsonl")
    sample_path = os.path.join(args.output_dir, "pii_scrub_sample.jsonl")

    scrubber = PiiScrubber()

    total = 0
    total_pii: dict[str, int] = {}
    rows_with_pii = 0
    sample_pairs: list[dict] = []

    limit = args.sample if args.sample > 0 else None
    writing_sample = args.sample > 0

    print(f"Scrubbing resumes from {args.input} ...")
    with open(output_path, "w", encoding="utf-8") as out:
        for record in iter_jsonl(args.input):
            lang = record.get("language_code")
            original_desc = record.get("description") or ""
            original_title = record.get("title") or ""

            scrubbed_desc, desc_stats = scrubber.scrub(original_desc, lang)
            scrubbed_title, title_stats = scrubber.scrub(original_title, lang)

            combined_stats: dict[str, int] = {}
            for d in (desc_stats, title_stats):
                for k, v in d.items():
                    combined_stats[k] = combined_stats.get(k, 0) + v

            if combined_stats:
                rows_with_pii += 1
            for k, v in combined_stats.items():
                total_pii[k] = total_pii.get(k, 0) + v

            record["description"] = scrubbed_desc
            record["title"] = scrubbed_title
            out.write(json.dumps(record, default=_json_default, ensure_ascii=False))
            out.write("\n")

            if writing_sample:
                sample_pairs.append({
                    "id": record.get("id"),
                    "language_code": lang,
                    "title_before": original_title,
                    "title_after": scrubbed_title,
                    "desc_before": original_desc[:500],
                    "desc_after": scrubbed_desc[:500],
                    "pii_found": combined_stats,
                })

            total += 1
            if total % HEARTBEAT_EVERY == 0:
                print(f"  ... {total:,} resumes processed, {rows_with_pii:,} had PII")
            if limit and total >= limit:
                break

    print(f"\nDone. {total:,} resumes processed -> {output_path}")
    print(f"  Resumes with PII detected: {rows_with_pii:,} / {total:,} ({rows_with_pii / total:.1%})" if total else "")
    print(f"  PII detections by type:")
    for pii_type in ("NAME", "EMAIL", "PHONE", "URL", "ADDRESS"):
        count = total_pii.get(pii_type, 0)
        if count:
            print(f"    {pii_type:>8}: {count:,}")

    if writing_sample:
        with open(sample_path, "w", encoding="utf-8") as f:
            for pair in sample_pairs:
                f.write(json.dumps(pair, ensure_ascii=False))
                f.write("\n")
        print(f"\n  Sample of {len(sample_pairs)} before/after pairs written to {sample_path}")
        print("  ** Review this file to confirm no PII leakage before proceeding (hard blocker gate) **")


if __name__ == "__main__":
    main()
