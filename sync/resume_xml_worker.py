from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from xml.etree import ElementTree as ET

from sqlalchemy import create_engine, text

from app.services.language import detect_language_code

logger = logging.getLogger("jobl.sync.resume_xml")


@dataclass
class ResumeXmlImportResult:
    files_scanned: int
    files_processed: int
    parsed: int
    skipped_existing: int
    skipped_invalid: int
    inserted: int


class ResumeXmlWorker:
    def __init__(
        self,
        *,
        target_database_url: str,
        feeds_dir: str,
        lookback_hours: int = 24,
    ) -> None:
        self.target_database_url = target_database_url
        self.feeds_dir = Path(feeds_dir)
        self.lookback_hours = lookback_hours

    def run_once(
        self,
        *,
        batch_size: int,
        all_files: bool = False,
        only_files: set[str] | None = None,
    ) -> ResumeXmlImportResult:
        if not self.feeds_dir.is_dir():
            raise FileNotFoundError(f"Resume XML feeds directory not found: {self.feeds_dir}")

        files = self._discover_files(all_files=all_files, only_files=only_files)
        logger.info(
            "resume xml import started feeds_dir=%s files_to_process=%s lookback_hours=%s all_files=%s",
            self.feeds_dir,
            len(files),
            self.lookback_hours,
            all_files,
        )

        result = ResumeXmlImportResult(
            files_scanned=len(list(self.feeds_dir.glob("*.xml"))),
            files_processed=0,
            parsed=0,
            skipped_existing=0,
            skipped_invalid=0,
            inserted=0,
        )

        if not files:
            logger.info("no XML feeds changed within lookback window")
            return result

        engine = create_engine(self.target_database_url, pool_pre_ping=True)
        try:
            for path in files:
                inserted, parsed, skipped_existing, skipped_invalid = self._import_file(
                    engine=engine,
                    path=path,
                    batch_size=batch_size,
                )
                result.files_processed += 1
                result.parsed += parsed
                result.skipped_existing += skipped_existing
                result.skipped_invalid += skipped_invalid
                result.inserted += inserted
                logger.info(
                    "resume xml file processed path=%s parsed=%s inserted=%s skipped_existing=%s skipped_invalid=%s",
                    path.name,
                    parsed,
                    inserted,
                    skipped_existing,
                    skipped_invalid,
                )
        finally:
            engine.dispose()

        return result

    def _discover_files(
        self,
        *,
        all_files: bool,
        only_files: set[str] | None,
    ) -> list[Path]:
        candidates = sorted(self.feeds_dir.glob("*.xml"))
        if only_files:
            wanted = {name.lower() for name in only_files}
            candidates = [path for path in candidates if path.name.lower() in wanted]

        if all_files:
            return candidates

        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)
        selected: list[Path] = []
        for path in candidates:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if mtime >= cutoff:
                selected.append(path)
        return selected

    def _import_file(self, *, engine, path: Path, batch_size: int) -> tuple[int, int, int, int]:
        website, resumes = self._parse_feed(path)
        if not website:
            logger.warning("resume xml file missing <website> path=%s", path.name)
            return 0, 0, 0, len(resumes)

        existing_ids = self._load_existing_external_ids(engine=engine, source_website=website)
        pending: list[dict[str, object | None]] = []
        parsed = 0
        skipped_existing = 0
        skipped_invalid = 0
        inserted = 0

        for raw in resumes:
            parsed += 1
            row = self._map_resume(source_website=website, raw=raw)
            if row is None:
                skipped_invalid += 1
                continue
            if row["external_id"] in existing_ids:
                skipped_existing += 1
                continue
            pending.append(row)
            if len(pending) >= batch_size:
                inserted += self._insert_batch(engine=engine, rows=pending)
                existing_ids.update(int(row["external_id"]) for row in pending)
                pending.clear()

        if pending:
            inserted += self._insert_batch(engine=engine, rows=pending)

        return inserted, parsed, skipped_existing, skipped_invalid

    def _parse_feed(self, path: Path) -> tuple[str | None, list[ET.Element]]:
        tree = ET.parse(path)
        root = tree.getroot()
        if root.tag != "resumes":
            logger.warning("resume xml root is not <resumes> path=%s root=%s", path.name, root.tag)
            return None, []

        website = _element_text(root.find("website"))
        resume_nodes = [node for node in root if node.tag == "resume"]
        return website, resume_nodes

    def _map_resume(self, *, source_website: str, raw: ET.Element) -> dict[str, object | None] | None:
        title = _element_text(raw.find("position"))
        if not title:
            return None

        description = _element_text(raw.find("description"))
        if not description:
            return None

        created_at = _parse_created_at(_element_text(raw.find("created_at")))
        external_id = _parse_external_id(_element_text(raw.find("id")))
        if external_id is None:
            return None

        country_code = _normalize_country_code(_element_text(raw.find("country_code")))
        salary_currency = _normalize_currency(_element_text(raw.find("currency")))

        return {
            "source_website": source_website[:128],
            "external_id": external_id,
            "title": title[:255],
            "description": description,
            "city_title": _truncate(_element_text(raw.find("city_title")), 255),
            "region_title": _truncate(_element_text(raw.find("region_title")), 255),
            "country_code": country_code,
            "salary": _parse_salary(_element_text(raw.find("salary"))),
            "salary_period": _truncate(_element_text(raw.find("salary_period")), 20),
            "salary_currency": salary_currency,
            "contract": _truncate(_element_text(raw.find("contract_code")), 24),
            "published_at": created_at,
            "is_remote": _parse_remote(_element_text(raw.find("remote"))),
            "language_code": detect_language_code(
                title=title,
                description=description,
                country_code=country_code,
                source_db=source_website,
            ).language_code,
        }

    def _load_existing_external_ids(self, *, engine, source_website: str) -> set[int]:
        query = text("SELECT external_id FROM resumes WHERE source_website = :source_website")
        with engine.connect() as conn:
            rows = conn.execute(query, {"source_website": source_website}).fetchall()
        return {int(row[0]) for row in rows}

    def _insert_batch(self, *, engine, rows: list[dict[str, object | None]]) -> int:
        if not rows:
            return 0

        query = text(
            """
            INSERT INTO resumes (
                source_website, external_id, title, description,
                city_title, region_title, country_code,
                salary, salary_period, salary_currency, contract,
                published_at, is_remote, language_code
            ) VALUES (
                :source_website, :external_id, :title, :description,
                :city_title, :region_title, :country_code,
                :salary, :salary_period, :salary_currency, :contract,
                :published_at, :is_remote, :language_code
            )
            ON CONFLICT ON CONSTRAINT uq_resumes_source_external DO NOTHING
            """
        )
        with engine.begin() as conn:
            result = conn.execute(query, rows)
        return int(result.rowcount or 0)


def _element_text(node: ET.Element | None) -> str | None:
    if node is None:
        return None
    text = (node.text or "").strip()
    return text or None


def _truncate(value: str | None, max_len: int) -> str | None:
    if value is None:
        return None
    trimmed = value.strip()
    if not trimmed:
        return None
    return trimmed[:max_len]


def _normalize_country_code(value: str | None) -> str | None:
    if not value:
        return None
    code = value.strip().upper()
    return code[:2] if code else None


def _normalize_currency(value: str | None) -> str | None:
    if not value:
        return None
    code = value.strip().upper()
    return code[:3] if code else None


def _parse_salary(value: str | None) -> Decimal | None:
    if not value:
        return None
    cleaned = value.strip().replace(",", "")
    if not cleaned:
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def _parse_remote(value: str | None) -> bool:
    if not value:
        return False
    normalized = value.strip().lower()
    return normalized in {"1", "true", "yes", "y"}


def _parse_created_at(value: str | None) -> datetime | None:
    if not value:
        return None
    cleaned = value.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S%z"):
        try:
            parsed = datetime.strptime(cleaned, fmt)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def _parse_external_id(value: str | None) -> int | None:
    if not value:
        return None
    cleaned = value.strip()
    if not cleaned.isdigit():
        return None
    return int(cleaned)
