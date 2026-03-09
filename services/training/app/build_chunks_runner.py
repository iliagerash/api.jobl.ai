import argparse
import json
import logging
from pathlib import Path
from typing import Any

from app.chunking import split_text_chunks
from app.io_utils import read_jsonl, write_jsonl
from app.logging import configure_logging


logger = logging.getLogger("jobl.training.build_chunks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build chunk-level instruction JSONL from SFT JSONL")
    parser.add_argument("--in", dest="input_path", required=True, help="Input SFT JSONL path")
    parser.add_argument("--out", dest="output_path", required=True, help="Output chunk-level JSONL path")
    parser.add_argument("--max-chars", type=int, default=3500, help="Max chars per description chunk")
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    configure_logging("INFO")

    try:
        rows = read_jsonl(Path(args.input_path))
        out_rows: list[dict[str, Any]] = []
        skipped = 0

        for row in rows:
            converted = _to_chunk_rows(row=row, max_chars=args.max_chars)
            if not converted:
                skipped += 1
                continue
            out_rows.extend(converted)

        write_jsonl(Path(args.output_path), out_rows)
    except KeyboardInterrupt:
        logger.warning("interrupted by user (Ctrl+C), exiting gracefully")
        return 130

    logger.info(
        "chunk dataset built source_rows=%s chunk_rows=%s skipped=%s output=%s",
        len(rows),
        len(out_rows),
        skipped,
        args.output_path,
    )
    return 0


def _to_chunk_rows(*, row: dict[str, Any], max_chars: int) -> list[dict[str, Any]]:
    messages = row.get("messages") or []
    if not isinstance(messages, list) or len(messages) < 3:
        return []

    system_msg = messages[0] if isinstance(messages[0], dict) else {"role": "system", "content": ""}
    user_text = messages[1].get("content") if isinstance(messages[1], dict) else ""
    assistant_text = messages[2].get("content") if isinstance(messages[2], dict) else ""

    try:
        user_payload = json.loads(str(user_text or ""))
        assistant_payload = json.loads(str(assistant_text or ""))
    except json.JSONDecodeError:
        return []

    desc_raw = str(user_payload.get("description_raw") or "")
    desc_html = str(assistant_payload.get("description_html") or "")
    title_norm = str(assistant_payload.get("title_normalized") or "")

    raw_chunks = split_text_chunks(desc_raw, max_chars=max_chars)
    html_chunks = split_text_chunks(desc_html, max_chars=max_chars)
    total = max(len(raw_chunks), len(html_chunks), 1)

    out: list[dict[str, Any]] = []
    for idx in range(total):
        chunk_user = dict(user_payload)
        chunk_user["description_raw"] = raw_chunks[idx] if idx < len(raw_chunks) else ""
        chunk_user["chunk_context"] = {
            "index": idx + 1,
            "total": total,
        }

        chunk_assistant = {
            "title_normalized": title_norm,
            "description_html": html_chunks[idx] if idx < len(html_chunks) else "",
        }

        out.append(
            {
                "id": f"{row.get('id')}_{idx + 1}",
                "parent_id": row.get("id"),
                "language_code": row.get("language_code"),
                "chunk_index": idx + 1,
                "chunk_total": total,
                "messages": [
                    {"role": "system", "content": system_msg.get("content") or ""},
                    {"role": "user", "content": json.dumps(chunk_user, ensure_ascii=False)},
                    {"role": "assistant", "content": json.dumps(chunk_assistant, ensure_ascii=False)},
                ],
            }
        )

    return out


if __name__ == "__main__":
    raise SystemExit(run())
