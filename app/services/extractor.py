import html
import json
import logging
import re

logger = logging.getLogger("jobl.api.extractor")

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _clean_description(desc: str | None) -> str:
    text = html.unescape(desc or "")
    text = re.sub(r"<\s*script\b[^>]*>.*?</\s*script\s*>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<\s*style\b[^>]*>.*?</\s*style\s*>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = _HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:3000]


SYSTEM_PROMPT = """You extract structured fields from job postings for a multilingual jobs platform covering 39 countries.

Input: a raw job posting prefixed with locale tags [lang=XX][country=XX].

Output: a JSON object with these fields:

- normalized_title (string): English-normalized canonical job title, concise (1-5 words). Remove company names, salaries, locations, employment type qualifiers. Preserve legal markers like (m/w/d).
- original_title (string): the original title exactly as provided.
- seniority (enum): one of: intern, junior, mid, senior, lead, executive. Infer from title, description, and experience requirements. Default to "mid" if unclear.
- occupation_category (string): one of the following categories:
  "Manufacturing & Engineering"
  "Automotive"
  "Food & Beverage Manufacturing"
  "IT & Telecommunications"
  "Construction & Infrastructure"
  "Consulting & Advisory"
  "Human Resources"
  "Transportation & Logistics"
  "Healthcare & Medical Services"
  "Aerospace & Defense"
  "Financial Services & Banking"
  "Real Estate & Architecture"
  "Marketing, Advertising & Media"
  "Hospitality & Restaurants"
  "Retail, Wholesale & Customer Service"
  "Education & Science"
  "Energy & Natural Resources"
  "Nonprofit & Government"
  "Arts, Entertainment & Recreation"
  "Legal Services"
  "Security & Surveillance"
  "Other"
  Choose the single best match. Use "Other" only if no category fits.
- employment_type (enum): one of: full_time, part_time, contract, freelance, internship.
- contract_type (enum): one of: permanent, temporary, fixed_term, freelance.
- work_mode (enum): one of: onsite, hybrid, remote. Default to "onsite" if not stated.
- location_city (string or null): city name extracted from the posting.
- location_country (string): ISO 3166-1 alpha-2 country code.
- language (string): ISO 639-1 language code of the posting text.
- salary_present (boolean): true if any salary information is mentioned.
- salary_min (number or null): minimum salary if present, as a number.
- salary_max (number or null): maximum salary if present, as a number.
- salary_currency (string or null): ISO 4217 currency code if salary is present.
- skills (array of strings): extracted skill keywords, normalized to English. Include technical skills, tools, certifications, and soft skills explicitly mentioned.
- experience_years_min (number or null): minimum years of experience if stated.
- is_expired_signal (boolean): true if there are heuristic clues the posting is stale (past dates, "closed", "filled", etc.).
- is_duplicate_signal (boolean): always false for single-posting extraction.

Rules:
- Work from the provided text only. Do not hallucinate information not present.
- If a field cannot be determined, use null (for nullable fields) or the stated default.
- Skills should be normalized to English even if the posting is in another language.
- Return ONLY valid JSON, no markdown fencing, no explanation."""


class JobExtractor:
    def __init__(self, model_path: str, n_threads: int = 4) -> None:
        self._ready = False
        try:
            from llama_cpp import Llama
            self._llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_threads=n_threads,
                verbose=False,
            )
            self._ready = True
            logger.info("extractor loaded from %s (n_threads=%d)", model_path, n_threads)
        except Exception as exc:
            raise RuntimeError(f"failed to load extractor from {model_path}") from exc

    def is_ready(self) -> bool:
        return self._ready

    def extract(
        self,
        title: str,
        description: str,
        language: str | None = None,
        country: str | None = None,
        record_type: str = "job",
    ) -> dict:
        """Extract structured fields from a job posting or resume. Returns a dict."""
        lang = language or "unknown"
        cc = country or "XX"
        desc_clean = _clean_description(description)
        user_msg = f"[lang={lang}][country={cc}][type={record_type}] {title} — {desc_clean}"

        response = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=500,
            temperature=0.0,
        )

        content = response["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.strip("`").replace("json\n", "", 1).strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("extractor returned invalid JSON: %s", content[:200])
            return {"_error": "invalid JSON", "_raw": content[:500]}
