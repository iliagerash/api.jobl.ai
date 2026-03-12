import logging
import re
from dataclasses import dataclass

from sqlalchemy import create_engine, text


logger = logging.getLogger("jobl.normalize")
TITLE_MODE_MARKER_PATTERN = r"(?:remote|fully\s+remote|hybrid|on[-\s]?site|onsite|wfh|work\s+from\s+home)"
TITLE_TRAILING_MODE_REGEXES = (
    re.compile(r"\s*[-|,/:]\s*remote\s*/\s*on[-\s]?site\b\.?\s*$", re.IGNORECASE),
    re.compile(r"\s*[-|,/:]\s*on[-\s]?site\s*/\s*remote\b\.?\s*$", re.IGNORECASE),
    re.compile(rf"\s*[-|,/:]\s*(?:\d{{1,3}}%\s*)?{TITLE_MODE_MARKER_PATTERN}\b\.?\s*$", re.IGNORECASE),
    re.compile(rf"\s*[-|,/:]\s*{TITLE_MODE_MARKER_PATTERN}\s*/\s*{TITLE_MODE_MARKER_PATTERN}\b\.?\s*$", re.IGNORECASE),
    re.compile(rf"\s*[-|,/:]\s*remote\s+work\b\.?\s*$", re.IGNORECASE),
    re.compile(r"\s*[-|,/:]\s*remote\s+position\b\.?\s*$", re.IGNORECASE),
)
TITLE_SPAM_PREFIX_REGEX = re.compile(
    r"^\s*(?:apply\s+now(?:\s+for)?|hiring\s+now|urgent(?:ly)?\s+hiring|immediate\s+start|we(?:'| a)?re\s+hiring)\s*[:!\-]*\s*",
    re.IGNORECASE,
)
TITLE_SPAM_INLINE_REGEX = re.compile(
    r"\b(?:hiring\s+now|urgent(?:ly)?\s+hiring|apply\s+now)\b",
    re.IGNORECASE,
)
TITLE_JOB_LABEL_PREFIX_REGEX = re.compile(
    r"^\s*[A-Za-z][A-Za-z0-9&+/ ]{0,20}\s+job\s*[-:]\s*",
    re.IGNORECASE,
)
TITLE_JOB_CODE_REGEX = re.compile(
    r"\b(?:job\s*(?:id|code)|req(?:uisition)?|reference|ref|position\s*id|vacancy)\s*[:#-]?\s*[A-Z0-9][A-Z0-9/_-]{1,}\b",
    re.IGNORECASE,
)
TITLE_SALARY_REGEX = re.compile(
    r"(?ix)\b(?:"
    r"(?:[$€£]\s?\d[\d.,]*(?:\s?[kK])?(?:\s*[-–]\s*[$€£]?\s?\d[\d.,]*(?:\s?[kK])?)?(?:\s*/\s*(?:h|hr|hour|day|week|month|year|yr))?)"
    r"|(?:\d[\d.,]*(?:\s?[kK])?\s*(?:usd|eur|gbp|aud|cad|chf|sek|nok|dkk|pln|ron|uah))"
    r"|(?:(?:up\s*to|upto)\s*[$€£]?\s?\d[\d.,]*(?:\s?[kK])?(?:\s*/\s*(?:h|hr|hour|hourly|day|week|month|year|yr))?)"
    r")\b"
)
TITLE_DATE_REGEX = re.compile(
    r"(?ix)\b(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{4})\b"
)
TITLE_COMPANY_SUFFIX_HINT_REGEX = re.compile(
    r"(?ix)\b(?:inc|llc|ltd|limited|corp|corporation|company|gmbh|ag|bv|oy|s\.?a\.?|srl|sro|pte|pty)\b"
)
TITLE_ADDRESS_HINT_REGEX = re.compile(
    r"(?ix)\b(?:street|st\.|avenue|ave\.|road|rd\.|boulevard|blvd\.|suite|building|bldg|floor|fl\.|zip|postal)\b|\d{4,6}\b"
)


@dataclass
class NormalizeResult:
    scanned: int
    updated: int
    batches: int


class NormalizeWorker:
    def __init__(self, target_database_url: str) -> None:
        self.target_database_url = target_database_url

    def run(self, batch_size: int, max_batches: int | None = None, from_id: int = 0) -> NormalizeResult:
        total_scanned = 0
        total_updated = 0
        batch_no = 0
        cursor = from_id

        engine = create_engine(self.target_database_url, pool_pre_ping=True)
        try:
            while True:
                if max_batches is not None and batch_no >= max_batches:
                    break

                rows = self._fetch_batch(engine=engine, batch_size=batch_size, from_id=cursor)
                if not rows:
                    break

                payload = []
                for row in rows:
                    title = row["title"] or ""
                    title_clean = self._normalize_title(
                        title,
                        city_title=row.get("city_title"),
                        region_title=row.get("region_title"),
                        country_code=row.get("country_code"),
                        country_name=row.get("country_name"),
                        country_alternate_names=row.get("country_alternate_names"),
                    )

                    payload.append(
                        {
                            "id": row["id"],
                            "title_clean": title_clean,
                        }
                    )

                self._update_batch(engine=engine, payload=payload)
                batch_no += 1
                scanned_count = len(rows)
                total_scanned += scanned_count
                total_updated += len(payload)
                cursor = int(rows[-1]["id"])

                logger.info(
                    "batch progress batch=%s scanned=%s updated=%s totals(scanned=%s, updated=%s) last_id=%s",
                    batch_no,
                    scanned_count,
                    len(payload),
                    total_scanned,
                    total_updated,
                    cursor,
                )
        finally:
            engine.dispose()

        return NormalizeResult(scanned=total_scanned, updated=total_updated, batches=batch_no)

    def _fetch_batch(self, engine, batch_size: int, from_id: int) -> list[dict[str, object | None]]:
        query = text(
            """
            SELECT
                j.id,
                j.title,
                j.city_title,
                j.region_title,
                j.country_code,
                c.name AS country_name,
                c.alternate_names AS country_alternate_names
            FROM jobs
            LEFT JOIN countries c ON c.code = j.country_code
            WHERE id > :from_id
              AND title_clean IS NULL
            ORDER BY id
            LIMIT :batch_size
            """
        )
        with engine.connect() as conn:
            rows = conn.execute(query, {"from_id": from_id, "batch_size": batch_size}).mappings()
            return [dict(row) for row in rows]

    def _update_batch(self, engine, payload: list[dict[str, object | None]]) -> None:
        if not payload:
            return
        query = text(
            """
            UPDATE jobs
            SET title_clean = :title_clean
            WHERE id = :id
            """
        )
        with engine.begin() as conn:
            conn.execute(query, payload)

    @staticmethod
    def _normalize_title(
        value: str,
        city_title: str | None = None,
        region_title: str | None = None,
        country_code: str | None = None,
        country_name: str | None = None,
        country_alternate_names: list[str] | tuple[str, ...] | None = None,
    ) -> str:
        title = value.strip()
        title = title.replace("–", "-").replace("—", "-")
        title = re.sub(r"\s+", " ", title)
        title = TITLE_SPAM_PREFIX_REGEX.sub("", title)
        title = TITLE_JOB_LABEL_PREFIX_REGEX.sub("", title)
        title = TITLE_SPAM_INLINE_REGEX.sub("", title)
        title = NormalizeWorker._clean_trailing_mode_parenthetical(title)

        for regex in TITLE_TRAILING_MODE_REGEXES:
            title = regex.sub("", title)

        title = NormalizeWorker._remove_non_title_suffixes(title)
        title = TITLE_JOB_CODE_REGEX.sub("", title)
        title = TITLE_SALARY_REGEX.sub("", title)
        title = re.sub(r"\s*/\s*(?:h|hr|hour|hourly|day|week|month|year|yr)\b", "", title, flags=re.IGNORECASE)
        title = re.sub(r"\s*[|,:-]\s*(?:hourly|monthly|annually)\b\.?\s*$", "", title, flags=re.IGNORECASE)
        title = TITLE_DATE_REGEX.sub("", title)
        title = re.sub(r"\s*[-|,/:]\s*remote\s*,\s*", " - ", title, flags=re.IGNORECASE)
        title = re.sub(rf"\s+\|\s+(?:\d{{1,3}}%\s*)?{TITLE_MODE_MARKER_PATTERN}\b\.?\s*$", "", title, flags=re.IGNORECASE)
        title = re.sub(r"[*!]{2,}", " ", title)
        title = re.sub(r"^[*!|,;:\- ]+|[*!|,;:\- ]+$", "", title)
        title = re.sub(r"\s{2,}", " ", title)
        title = re.sub(r"(?<=\w)-(?=[A-Za-z])", " - ", title)
        title = re.sub(r"\s+,", ",", title)
        title = re.sub(r",\s*(?:remote|hybrid|on[-\s]?site|onsite|wfh)\b\.?\s*$", "", title, flags=re.IGNORECASE)
        title = NormalizeWorker._remove_trailing_location(
            title,
            city_title,
            region_title,
            country_code,
            country_name,
            country_alternate_names,
        )
        title = re.sub(r"\s*[-|,/:]+\s*$", "", title)
        title = re.sub(r"\s+", " ", title).strip(" -|,")
        return title

    @staticmethod
    def _clean_trailing_mode_parenthetical(value: str) -> str:
        match = re.search(r"\(([^)]*)\)\s*$", value)
        if not match:
            return value

        body = match.group(1)
        if re.fullmatch(r"\s*[mf]\s*/\s*[wf]\s*/\s*d\s*", body, flags=re.IGNORECASE):
            return value

        cleaned = re.sub(rf"\b(?:\d{{1,3}}%\s*)?{TITLE_MODE_MARKER_PATTERN}\b", "", body, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bremote\s+position\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bhybrid\s+work\s+schedule\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(full[-\s]?time|part[-\s]?time)\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*[,/|-]\s*", ", ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")

        if not cleaned:
            return value[: match.start()].rstrip()
        return f"{value[: match.start()].rstrip()} ({cleaned})"

    @staticmethod
    def _remove_non_title_suffixes(value: str) -> str:
        segments = [seg.strip() for seg in re.split(r"\s*[|]\s*", value) if seg.strip()]
        if len(segments) > 1:
            last = segments[-1]
            if TITLE_COMPANY_SUFFIX_HINT_REGEX.search(last) or TITLE_ADDRESS_HINT_REGEX.search(last):
                value = " | ".join(segments[:-1])

        for sep in (" - ", " / "):
            parts = [part.strip() for part in value.split(sep) if part.strip()]
            if len(parts) <= 1:
                continue
            last = parts[-1]
            if TITLE_COMPANY_SUFFIX_HINT_REGEX.search(last) or TITLE_ADDRESS_HINT_REGEX.search(last):
                value = sep.join(parts[:-1]).strip()
        return value

    @staticmethod
    def _remove_trailing_location(
        value: str,
        city_title: str | None,
        region_title: str | None,
        country_code: str | None,
        country_name: str | None,
        country_alternate_names: list[str] | tuple[str, ...] | None,
    ) -> str:
        location_tokens = NormalizeWorker._location_tokens(
            city_title,
            region_title,
            country_code,
            country_name,
            country_alternate_names,
        )
        title = value

        for token in location_tokens:
            escaped = re.escape(token)
            title = re.sub(rf"\s*[-|,/]\s*{escaped}\s*$", "", title, flags=re.IGNORECASE)
            title = re.sub(rf"\(\s*{escaped}\s*\)\s*$", "", title, flags=re.IGNORECASE)
            title = re.sub(rf",\s*{escaped}\s*$", "", title, flags=re.IGNORECASE)
            title = re.sub(rf"\s+\b(?:in|at|near|for)\s+{escaped}\s*$", "", title, flags=re.IGNORECASE)

        for token in location_tokens:
            escaped = re.escape(token)
            title = re.sub(rf"\(([^)]*?),\s*{escaped}\s*\)\s*$", r"(\1)", title, flags=re.IGNORECASE)

        title = re.sub(r"\(\s*[A-Z]{2}\s*\)\s*$", "", title)
        return re.sub(r"\s+", " ", title).strip()

    @staticmethod
    def _location_tokens(
        city_title: str | None,
        region_title: str | None,
        country_code: str | None,
        country_name: str | None,
        country_alternate_names: list[str] | tuple[str, ...] | None = None,
    ) -> list[str]:
        tokens: list[str] = []
        for raw in (city_title, region_title, country_name):
            token = (raw or "").strip()
            if token and len(token) >= 2:
                tokens.append(token)
        for raw in (country_alternate_names or []):
            token = str(raw).strip()
            if token and len(token) >= 2:
                tokens.append(token)
        code = (country_code or "").strip().upper()
        if len(code) == 2:
            tokens.append(code)
        unique_tokens: list[str] = []
        seen: set[str] = set()
        for token in sorted(tokens, key=len, reverse=True):
            key = token.lower()
            if key not in seen:
                unique_tokens.append(token)
                seen.add(key)
        return unique_tokens

    @staticmethod
    def normalize_title_for_ml(value: str) -> str:
        title = NormalizeWorker._normalize_title(value)

        # Remove gender/legal suffixes for ML features only, never for user-facing API fields.
        gender_marker = r"(?:m\s*[/|\\-]\s*w\s*[/|\\-]\s*d|m\s*[/|\\-]\s*f\s*[/|\\-]\s*d|f\s*[/|\\-]\s*m\s*[/|\\-]\s*d|m\s*[/|\\-]\s*x|f\s*[/|\\-]\s*m\s*[/|\\-]\s*x|all\s+genders|gn)"
        title = re.sub(rf"\(\s*{gender_marker}\s*\)", "", title, flags=re.IGNORECASE)
        title = re.sub(rf"\[\s*{gender_marker}\s*\]", "", title, flags=re.IGNORECASE)
        title = re.sub(rf"\b{gender_marker}\b", "", title, flags=re.IGNORECASE)
        title = re.sub(r"\s+", " ", title)
        title = re.sub(r"\s*[-|,/:]+\s*$", "", title)
        return title.strip(" -|,:/")
