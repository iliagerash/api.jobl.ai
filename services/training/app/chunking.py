import re
from typing import Iterable


_SPLIT_RE = re.compile(r"(</p>|</li>|<br\s*/?>|\n\n+|\n)", flags=re.IGNORECASE)


def split_text_chunks(text: str, max_chars: int) -> list[str]:
    value = (text or "").strip()
    if not value:
        return []
    if max_chars <= 0:
        return [value]

    parts = _split_with_separators(value)
    chunks: list[str] = []
    buf = ""

    for part in parts:
        piece = part
        if not piece:
            continue
        if len(piece) > max_chars:
            # Hard-cut oversized piece.
            if buf:
                chunks.append(buf.strip())
                buf = ""
            for i in range(0, len(piece), max_chars):
                sub = piece[i : i + max_chars].strip()
                if sub:
                    chunks.append(sub)
            continue

        candidate = f"{buf}{piece}"
        if buf and len(candidate) > max_chars:
            chunks.append(buf.strip())
            buf = piece
        else:
            buf = candidate

    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def _split_with_separators(text: str) -> list[str]:
    raw = _SPLIT_RE.split(text)
    out: list[str] = []
    it: Iterable[str] = (r for r in raw if r is not None)
    for token in it:
        if token:
            out.append(token)
    return out
