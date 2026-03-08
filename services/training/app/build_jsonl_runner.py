import argparse
import json
import logging
from pathlib import Path
from typing import Any

from app.io_utils import read_jsonl, write_jsonl
from app.logging import configure_logging


logger = logging.getLogger("jobl.training.build_jsonl")

SYSTEM_PROMPT = (
    "You normalize raw job data for a jobs platform. "
    "Return strict JSON with fields title_normalized and description_html. "
    "Do not include location/company/salary/date noise in title_normalized. "
    "Preserve legal markers like (m/w/d). "
    "For description_html keep only safe tags: "
    "<p>, <ul>, <ol>, <li>, <i>, <b>, <em>, <strong>, <u>, <h2>, <h3>, <h4>, <br>, <a>."
)


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
    parser.add_argument("--prompt-version", type=str, default="v1", help="Prompt version label")
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    configure_logging("INFO")
    input_path = Path(args.input_path) if args.input_path else Path(f"data/splits/{args.split}.jsonl")
    output_path = Path(args.output_path) if args.output_path else Path(f"data/sft/{args.split}.jsonl")

    try:
        rows = read_jsonl(input_path)
        converted = [_convert_row(row, prompt_version=args.prompt_version) for row in rows]
        write_jsonl(output_path, converted)
    except KeyboardInterrupt:
        logger.warning("interrupted by user (Ctrl+C), exiting gracefully")
        return 130

    logger.info("instruction jsonl built split=%s rows=%s output=%s", args.split, len(converted), output_path)
    return 0


def _convert_row(row: dict[str, Any], prompt_version: str) -> dict[str, Any]:
    user_payload = {
        "prompt_version": prompt_version,
        "title_raw": row.get("title_raw") or "",
        "description_raw": row.get("description_raw") or "",
        "location_context": {
            "language_code": row.get("language_code"),
            "country_code": row.get("country_code"),
            "country_name": row.get("country_name"),
            "region_title": row.get("region_title"),
            "city_title": row.get("city_title"),
        },
    }
    target_payload = {
        "title_normalized": row.get("expected_title_normalized") or "",
        "description_html": row.get("expected_description_html") or "",
    }
    return {
        "id": row.get("id"),
        "language_code": row.get("language_code"),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps(target_payload, ensure_ascii=False)},
        ],
    }


if __name__ == "__main__":
    raise SystemExit(run())
