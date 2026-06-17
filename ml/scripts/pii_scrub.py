"""
pii_scrub.py
────────────
Phase 0, step 3: strip PII from resume descriptions before any ML training
data is constructed.  Hard blocker — manual review of 200 scrubbed samples is
required before proceeding.

Detection layers (applied in order, with span deduplication):
  1. Regex — emails, URLs, phone numbers, street addresses  (high precision)
  2. spaCy NER — PERSON entities via the multilingual xx_ent_wiki_sm model
  3. Pattern-based — name-introduction phrases in top languages (ES/EN/PT/DE/FR)

All detected spans are merged (overlapping spans consolidated) and replaced
with typed placeholders: [NAME], [EMAIL], [PHONE], [URL], [ADDRESS].

Setup (one-time, on the server):
    pip install -e ".[ml]"
    python -m spacy download xx_ent_wiki_sm

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
    r"|T[eé]l[eé]?(?:phone)?|Celular|Τηλ(?:έφωνο)?|Телефон)"
    r"\s*[:\.\-]?\s*"
    r"([\+\d\(][\d\s\-\./\(\)]{6,20}\d)",
    re.IGNORECASE,
)

# German street address: "Musterstraße 12" / "Am Markt 5a"
STREET_DE_RE = re.compile(
    r"\b[A-ZÄÖÜ][a-zäöüß]*(?:stra[sß]e|str\.|weg|gasse|platz|ring|allee|damm)"
    r"\s+\d+[a-z]?\b",
    re.IGNORECASE,
)

# Postal code + city (DE: 5 digits, ES: 5 digits, PT: 4-4 or 4 digits, FR: 5 digits)
POSTAL_CITY_RE = re.compile(
    r"\b\d{4,5}[\s\-]?\d{0,3}\s+[A-ZÄÖÜÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÃÕÑÇ][a-zäöüßáéíóúàèìòùâêîôûãõñç]+\b"
)

# ── Name-introduction patterns (top languages) ──────────────────────────────

_NAME_CHUNK = r"[A-ZÄÖÜÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÃÕÑÇΑ-ΩҐЄІЇ][a-zäöüßáéíóúàèìòùâêîôûãõñçα-ωґєії']+"

NAME_INTRO_RE = re.compile(
    r"(?:"
    # Spanish
    r"[Mm]i nombre (?:es|completo es)"
    r"|[Mm]e llamo"
    r"|[Ss]oy\s"
    # English
    r"|[Mm]y name is"
    r"|I am\s"
    # Portuguese
    r"|[Mm]eu nome [eé]"
    r"|[Mm]e chamo"
    # German
    r"|[Mm]ein Name ist"
    r"|[Ii]ch hei[sß]e"
    r"|[Ii]ch bin\s"
    # French
    r"|[Jj]e m['']appelle"
    r"|[Jj]e suis\s"
    r")\s*"
    rf"({_NAME_CHUNK}(?:\s+{_NAME_CHUNK}){{1,4}})",
    re.UNICODE,
)

# "Vor- und Zuname:", "Nombre:", "Nome:", "Name:", "Nombre completo:" etc.
NAME_LABEL_RE = re.compile(
    r"(?:Vor-?\s*(?:und|&)\s*(?:Zu|Nach)name|(?:Nombre|Nome)\s*(?:completo)?|Full\s*name|Name)"
    r"\s*[:\.\-]\s*"
    rf"({_NAME_CHUNK}(?:\s+{_NAME_CHUNK}){{1,4}})",
    re.IGNORECASE | re.UNICODE,
)

# Salutations: Herr/Frau, Sr./Sra., Mr./Mrs./Ms.
SALUTATION_RE = re.compile(
    r"(?:Herr|Frau|[Ss]r\.?a?|[Dd]on|[Dd]oña|Mr\.?|Mrs\.?|Ms\.?|M\.?|Mme\.?)"
    r"\s+"
    rf"({_NAME_CHUNK}(?:\s+{_NAME_CHUNK}){{1,3}})",
    re.UNICODE,
)


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

class PiiScrubber:
    def __init__(self) -> None:
        self._nlp = None
        try:
            import spacy
            self._nlp = spacy.load("xx_ent_wiki_sm")
            # Raise max doc length for long resumes (default 1M chars is fine)
            self._nlp.max_length = 2_000_000
            print("Loaded spaCy model: xx_ent_wiki_sm (multilingual NER for PERSON entities)")
        except OSError:
            print(
                "WARNING: spaCy model 'xx_ent_wiki_sm' not found — falling back to "
                "regex-only (no NER-based name detection). Install with:\n"
                "  python -m spacy download xx_ent_wiki_sm"
            )
        except ImportError:
            print(
                "WARNING: spaCy not installed — falling back to regex-only. Install with:\n"
                "  pip install -e '.[ml]' && python -m spacy download xx_ent_wiki_sm"
            )

    def scrub(self, text: str | None) -> tuple[str, dict[str, int]]:
        if not text:
            return text or "", {}

        spans: list[tuple[int, int, str]] = []
        stats: dict[str, int] = {}

        def _add(start: int, end: int, pii_type: str) -> None:
            spans.append((start, end, pii_type))
            stats[pii_type] = stats.get(pii_type, 0) + 1

        # Layer 1: regex — structured PII
        for m in EMAIL_RE.finditer(text):
            _add(m.start(), m.end(), "EMAIL")

        for m in URL_RE.finditer(text):
            _add(m.start(), m.end(), "URL")

        for m in INTL_PHONE_RE.finditer(text):
            _add(m.start(), m.end(), "PHONE")

        for m in PHONE_CONTEXT_RE.finditer(text):
            _add(m.start(1), m.end(1), "PHONE")

        for m in STREET_DE_RE.finditer(text):
            _add(m.start(), m.end(), "ADDRESS")

        for m in POSTAL_CITY_RE.finditer(text):
            _add(m.start(), m.end(), "ADDRESS")

        # Layer 2: spaCy NER — PERSON entities
        if self._nlp is not None:
            doc = self._nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PER":
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
            original_desc = record.get("description") or ""
            original_title = record.get("title") or ""

            scrubbed_desc, desc_stats = scrubber.scrub(original_desc)
            scrubbed_title, title_stats = scrubber.scrub(original_title)

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
                    "language_code": record.get("language_code"),
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
