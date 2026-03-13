import argparse
import html
import logging
import re
from pathlib import Path
from typing import Any

from app.io_utils import read_jsonl, write_jsonl
from app.logging import configure_logging


logger = logging.getLogger("jobl.training.build_jsonl")

# Keep this pre-strip logic identical to production inference pre-strip behavior.
PRE_STRIP_PATTERNS = [
    (
        "salary_rate_symbol",
        re.compile(
            r"[$€£]\s?\d[\d,k.]*\s?(/hr|/hour|/yr|/year|ph|pa)?",
            re.IGNORECASE,
        ),
    ),
    ("salary_rate_upto", re.compile(r"\bup\s?to\s?[$€£]\s?[\d,k.]+\b", re.IGNORECASE)),
    ("job_code_hash", re.compile(r"\s*#[A-Z]{2,6}\b", re.IGNORECASE)),
    ("numeric_job_code_parens", re.compile(r"\(\s*\d{4}[-–]\d{2,6}\s*\)", re.IGNORECASE)),
    (
        "employment_type",
        re.compile(
            r"full[\s-]?time|part[\s-]?time|casual|\btemporary\b|\btemp(?!\w)\b|fixed[\s-]?term|\bpermanent(?!\s+reliever)\b|perm(?!\w)|on[\s-]?call|sur appel",
            re.IGNORECASE,
        ),
    ),
    ("early_careers", re.compile(r"early careers?|new\s?grad(uate)?", re.IGNORECASE)),
    (
        "campaign_text",
        re.compile(
            r"apply\s+now|hiring\s+now|no\s+experience\s+needed|start\s+your\s+career\s+with\s+us\s+today[!?]*|possibility\s+for\s+conversion\s+to\s+perm",
            re.IGNORECASE,
        ),
    ),
    ("multiple_positions", re.compile(r"\(?\bmultiple\s+positions?\b\)?", re.IGNORECASE)),
    ("opportunities_suffix", re.compile(r"\bopportunities\b", re.IGNORECASE)),
]

COVERAGE_PATTERNS = {
    "salary": re.compile(
        r"[$€£]\s?\d[\d,k.]*\s?(/hr|/hour|/yr|/year|ph|pa)?|\bup\s?to\s?[$€£]\s?[\d,k.]+\b",
        re.IGNORECASE,
    ),
    "job_code_hash": re.compile(r"\s*#[A-Z]{2,6}\b", re.IGNORECASE),
    "employment_type": re.compile(
        r"full[\s-]?time|part[\s-]?time|casual|temp(orary)?|locum|fixed[\s-]?term|permanent|perm(?!\w)|on[\s-]?call|sur appel",
        re.IGNORECASE,
    ),
    "early_careers": re.compile(r"early careers?|new\s?grad(uate)?", re.IGNORECASE),
    "multilingual": re.compile(
        r"\b(emploi|poste|trabajo|puesto|lavoro|posizione|vaga|emprego|vacature|functie|stilling|fuldtid|deltid|híbrido|hibrido|hybride|sur appel)\b",
        re.IGNORECASE,
    ),
    "location_in_title": re.compile(r"\b(remote|hybrid)\b", re.IGNORECASE),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build instruction JSONL for SFT from split files")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Dataset split name used for default --in/--out paths",
    )
    parser.add_argument(
        "--in",
        dest="input_path",
        default=None,
        help="Input split JSONL path (default: data/splits/<split>.jsonl)",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        default=None,
        help="Output instruction JSONL path (default: data/sft/<split>.jsonl)",
    )
    parser.add_argument(
        "--coverage-check",
        dest="coverage_check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warn when train-split noise-pattern coverage is below threshold",
    )
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    configure_logging("INFO")
    input_path = Path(args.input_path) if args.input_path else Path(f"data/splits/{args.split}.jsonl")
    output_path = Path(args.output_path) if args.output_path else Path(f"data/sft/{args.split}.jsonl")

    try:
        rows = read_jsonl(input_path)
        converted = [_convert_row(row) for row in rows]
        write_jsonl(output_path, converted)
        if args.coverage_check and args.split == "train":
            _log_coverage(rows, min_rows=20)
    except KeyboardInterrupt:
        logger.warning("interrupted by user (Ctrl+C), exiting gracefully")
        return 130

    logger.info("instruction jsonl built split=%s rows=%s output=%s", args.split, len(converted), output_path)
    return 0


def _convert_row(row: dict[str, Any]) -> dict[str, Any]:
    title_raw = row.get("title")
    if title_raw is None:
        title_raw = row.get("title_raw")
    return {
        "id": row.get("id"),
        "language_code": row.get("language_code"),
        "input": "normalize job title: " + pre_strip(str(title_raw or "")),
        "target": str(row.get("expected_title_normalized") or ""),
    }


def pre_strip(title: str) -> str:
    cleaned = html.unescape(str(title or ""))
    for _name, pattern in PRE_STRIP_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
        cleaned = _cleanup_separators_and_spaces(cleaned)
    return cleaned


def _cleanup_separators_and_spaces(value: str) -> str:
    cleaned = str(value or "")
    while True:
        previous = cleaned
        cleaned = re.sub(r"^\s*[-–—|,]+\s*", "", cleaned)
        cleaned = re.sub(r"\s*[-–—|,]+\s*$", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned == previous:
            break
    return cleaned


def _log_coverage(rows: list[dict[str, Any]], min_rows: int) -> None:
    counts = {name: 0 for name in COVERAGE_PATTERNS}
    for row in rows:
        title_raw = row.get("title")
        if title_raw is None:
            title_raw = row.get("title_raw")
        title_text = html.unescape(str(title_raw or ""))
        for name, pattern in COVERAGE_PATTERNS.items():
            if pattern.search(title_text):
                counts[name] += 1

    for name, count in counts.items():
        if count < min_rows:
            logger.warning(
                "coverage check warning split=train pattern=%s matched_rows=%s threshold=%s",
                name,
                count,
                min_rows,
            )


if __name__ == "__main__":
    raise SystemExit(run())
