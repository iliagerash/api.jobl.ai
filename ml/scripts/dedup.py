"""
dedup.py
────────
Phase 0, step 2 of the ML pipeline implementation plan: deduplicate the
exported jobs corpus.

Exact duplicates — SHA-256 hash of normalized (title + plain-text description).
The first-seen row per hash is kept; the rest are dropped.

Near-duplicates — MinHash + LSH (Jaccard similarity over 5-word shingles) run
on the deduped set. These are NOT dropped, only flagged (`is_near_duplicate`,
`near_duplicate_of`) so the Component 3 `is_duplicate_signal` field has a real
source to learn from; downstream training-data construction decides whether to
exclude or down-weight them. Detection queries the LSH index before inserting
the current row, so a row is never matched against itself and the index never
needs more than one signature per already-seen row in memory.

Runs in three streaming passes over JSONL — no attempt to hold the full ~1.3M
job corpus in memory at once. Passes 1 and 2 each re-parse HTML to normalize
text; this trades some CPU for materially lower peak memory, which is the
right tradeoff for a Phase 0 prep step with no tight time budget.

Usage:
    python ml/scripts/dedup.py
    python ml/scripts/dedup.py --input ml/data/raw/jobs.jsonl --output-dir ml/data/interim
    python ml/scripts/dedup.py --skip-near-dup       # exact dedup only, faster
    python ml/scripts/dedup.py --threshold 0.85      # stricter near-dup match

Outputs:
    ml/data/interim/jobs.jsonl        — deduped jobs; each row gains
                                         is_near_duplicate (bool) and
                                         near_duplicate_of (id | null)
    ml/data/interim/near_dup_sample.jsonl — up to 50 flagged pairs, for the
                                         gate's manual sanity-check step
"""

import argparse
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bs4 import BeautifulSoup

_WHITESPACE_RE = re.compile(r"\s+")

SHINGLE_SIZE = 5
NUM_PERM = 128
SAMPLE_SIZE = 50
HEARTBEAT_EVERY = 100_000
# Pass 3 (MinHash+LSH) is the slowest per-row, so heartbeat more often.
PASS3_HEARTBEAT_EVERY = 20_000


def normalize_text(title: str, description: str | None) -> str:
    """Strip HTML, lowercase, collapse whitespace — for hashing/shingling only.

    Not the same normalization used for training inputs; this is purely to
    make near-identical postings (different whitespace/casing, boilerplate
    HTML wrappers) hash/shingle the same.
    """
    desc_plain = BeautifulSoup(description or "", "lxml").get_text(separator=" ")
    combined = f"{title or ''} {desc_plain}"
    return _WHITESPACE_RE.sub(" ", combined).strip().lower()


def content_hash(normalized: str) -> str:
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def iter_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def pass1_find_exact_duplicates(input_path: str) -> tuple[dict[str, int], dict[str, int]]:
    """First streaming pass: hash -> id of first occurrence, hash -> total count."""
    first_seen: dict[str, int] = {}
    counts: dict[str, int] = {}
    scanned = 0
    for record in iter_jsonl(input_path):
        h = content_hash(normalize_text(record["title"], record.get("description")))
        counts[h] = counts.get(h, 0) + 1
        if h not in first_seen:
            first_seen[h] = record["id"]
        scanned += 1
        if scanned % HEARTBEAT_EVERY == 0:
            print(f"  ... {scanned:,} rows scanned")
    return first_seen, counts


def pass2_write_deduped(input_path: str, output_path: str, first_seen: dict[str, int]) -> int:
    """Second streaming pass: write only the kept (first-seen) row per hash."""
    written = 0
    seen = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for record in iter_jsonl(input_path):
            h = content_hash(normalize_text(record["title"], record.get("description")))
            seen += 1
            if first_seen.get(h) != record["id"]:
                continue  # exact duplicate of an earlier row — drop
            out.write(json.dumps(record, ensure_ascii=False))
            out.write("\n")
            written += 1
            if seen % HEARTBEAT_EVERY == 0:
                print(f"  ... {seen:,} rows processed, {written:,} written so far")
    return written


def _shingles(normalized: str, size: int = SHINGLE_SIZE) -> set[str]:
    words = normalized.split()
    if len(words) <= size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def pass3_flag_near_duplicates(
    deduped_path: str,
    sample_path: str,
    *,
    num_perm: int = NUM_PERM,
    threshold: float = 0.8,
) -> tuple[int, int]:
    """Third pass: MinHash+LSH near-dup flagging on the deduped corpus, in place."""
    from datasketch import MinHash, MinHashLSH

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    titles: dict[str, str] = {}  # id -> title, for the sample file only
    flagged = 0
    total = 0
    sample_pairs: list[dict] = []

    tmp_path = deduped_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as out:
        for record in iter_jsonl(deduped_path):
            total += 1
            record_id = str(record["id"])
            normalized = normalize_text(record["title"], record.get("description"))

            m = MinHash(num_perm=num_perm)
            for shingle in _shingles(normalized):
                m.update(shingle.encode("utf-8"))

            matches = lsh.query(m)
            if matches:
                near_dup_of = int(matches[0])
                record["is_near_duplicate"] = True
                record["near_duplicate_of"] = near_dup_of
                flagged += 1
                if len(sample_pairs) < SAMPLE_SIZE:
                    sample_pairs.append(
                        {
                            "id": record["id"],
                            "title": record["title"],
                            "near_duplicate_of": near_dup_of,
                            "near_duplicate_of_title": titles.get(str(near_dup_of)),
                        }
                    )
            else:
                record["is_near_duplicate"] = False
                record["near_duplicate_of"] = None

            lsh.insert(record_id, m)
            titles[record_id] = record["title"]

            out.write(json.dumps(record, ensure_ascii=False))
            out.write("\n")

            if total % PASS3_HEARTBEAT_EVERY == 0:
                flag_ratio = flagged / total
                print(f"  ... {total:,} rows processed, {flagged:,} flagged so far ({flag_ratio:.1%})")

    os.replace(tmp_path, deduped_path)

    with open(sample_path, "w", encoding="utf-8") as f:
        for pair in sample_pairs:
            f.write(json.dumps(pair, ensure_ascii=False))
            f.write("\n")

    return total, flagged


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate the exported jobs corpus (exact + near-duplicates)")
    parser.add_argument("--input", default="ml/data/raw/jobs.jsonl")
    parser.add_argument("--output-dir", default="ml/data/interim")
    parser.add_argument("--skip-near-dup", action="store_true", help="Exact dedup only, skip MinHash/LSH pass")
    parser.add_argument(
        "--threshold", type=float, default=0.8, help="Jaccard similarity threshold for near-dup (default: 0.8)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "jobs.jsonl")
    n_passes = "2" if args.skip_near_dup else "3"

    print(f"Pass 1/{n_passes}: scanning {args.input} for exact duplicates ...")
    first_seen, counts = pass1_find_exact_duplicates(args.input)
    total_rows = sum(counts.values())
    unique_rows = len(first_seen)
    exact_dupes = total_rows - unique_rows
    dupe_ratio = exact_dupes / total_rows if total_rows else 0.0
    print(
        f"  {total_rows:,} rows scanned, {unique_rows:,} unique content hashes, "
        f"{exact_dupes:,} exact duplicates ({dupe_ratio:.1%}) will be dropped"
    )

    print(f"Pass 2/{n_passes}: writing deduped output -> {output_path} ...")
    written = pass2_write_deduped(args.input, output_path, first_seen)
    assert written == unique_rows, f"expected {unique_rows} rows written, got {written}"
    print(f"  wrote {written:,} rows")

    if args.skip_near_dup:
        print("Skipping near-duplicate detection (--skip-near-dup).")
        return

    sample_path = os.path.join(args.output_dir, "near_dup_sample.jsonl")
    print(f"Pass 3/3: MinHash/LSH near-duplicate detection (threshold={args.threshold}) ...")
    total, flagged = pass3_flag_near_duplicates(output_path, sample_path, threshold=args.threshold)
    flag_ratio = flagged / total if total else 0.0
    print(
        f"  {flagged:,} / {total:,} rows flagged as near-duplicates ({flag_ratio:.1%}); "
        f"sample of up to {SAMPLE_SIZE} pairs written to {sample_path} for manual sanity-check"
    )


if __name__ == "__main__":
    main()
