"""
load_keywords.py
─────────────────
Load taxonomy keywords into the keywords table and generate biencoder
embeddings for each.

Reads taxonomy JSON (en/fr/gr titles with canonical_id and category_id),
inserts every language variant present on each entry, and encodes them
with the biencoder.

Usage:
    python -u scripts/load_keywords.py
    python -u scripts/load_keywords.py --input ml/data/taxonomy/taxonomy_reviewed.json
    python -u scripts/load_keywords.py --biencoder-path models/biencoder-onnx
    python -u scripts/load_keywords.py --clear   # delete existing keywords first
    python -u scripts/load_keywords.py --languages gr   # only (re-)embed and load one language
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

INPUT_PATH = "ml/data/taxonomy/taxonomy_reviewed.json"
LANGUAGE_FIELDS = ["en", "fr", "gr"]  # taxonomy keys to load; must match app/services/language.py codes


def main():
    parser = argparse.ArgumentParser(description="Load taxonomy into keywords table")
    parser.add_argument("--input", default=INPUT_PATH)
    parser.add_argument("--biencoder-path", default="models/biencoder-onnx")
    parser.add_argument("--clear", action="store_true", help="Delete existing keywords before loading")
    parser.add_argument("--db-url", default=None)
    parser.add_argument(
        "--languages", default=None,
        help=f"Comma-separated subset of language fields to (re-)embed and load, e.g. 'gr'. Default: all of {LANGUAGE_FIELDS}",
    )
    args = parser.parse_args()

    db_url = args.db_url or os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("No DATABASE_URL")

    languages = args.languages.split(",") if args.languages else LANGUAGE_FIELDS
    unknown = set(languages) - set(LANGUAGE_FIELDS)
    if unknown:
        sys.exit(f"Unknown language(s): {unknown}. Known: {LANGUAGE_FIELDS}")

    with open(args.input) as f:
        taxonomy = json.load(f)

    print(f"Loaded {len(taxonomy)} keywords from {args.input}")

    # Build keyword rows — one per requested language field present on each entry
    rows = []
    for item in taxonomy:
        for lang in languages:
            title = item.get(lang)
            if not title:
                continue
            rows.append({
                "canonical_id": item["canonical_id"],
                "title": title,
                "language_code": lang,
                "category_id": item["category_id"],
            })

    print(f"  {len(rows)} keyword rows ({', '.join(languages)})")

    # Generate embeddings
    print(f"Loading biencoder from {args.biencoder_path} ...")
    from app.services.biencoder import BiEncoder
    biencoder = BiEncoder(args.biencoder_path)

    print("Generating embeddings ...")
    texts = [f"[lang={r['language_code']}][country=XX][type=job] {r['title']}" for r in rows]
    embeddings = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings.extend(biencoder.encode_batch(batch))
        if (i + batch_size) % 500 < batch_size:
            print(f"  {len(embeddings)}/{len(texts)} encoded ...", flush=True)
    print(f"  {len(embeddings)} embeddings generated")

    # Insert into database
    engine = create_engine(db_url, pool_pre_ping=True)

    try:
        with engine.begin() as conn:
            if args.clear:
                conn.execute(text("DELETE FROM job_keywords"))
                conn.execute(text("DELETE FROM keywords"))
                print("  Cleared existing keywords")

            inserted = 0
            for row, emb in zip(rows, embeddings):
                vec_str = "[" + ",".join(f"{x:.6f}" for x in emb) + "]"
                conn.execute(text("""
                    INSERT INTO keywords (canonical_id, title, language_code, category_id, embedding)
                    VALUES (:canonical_id, :title, :language_code, :category_id, CAST(:embedding AS vector))
                    ON CONFLICT (title, language_code, category_id) DO UPDATE
                    SET canonical_id = EXCLUDED.canonical_id,
                        embedding = EXCLUDED.embedding
                """), {
                    "canonical_id": row["canonical_id"],
                    "title": row["title"],
                    "language_code": row["language_code"],
                    "category_id": row["category_id"],
                    "embedding": vec_str,
                })
                inserted += 1

                if inserted % 500 == 0:
                    print(f"  {inserted}/{len(rows)} inserted ...", flush=True)

        print(f"\nDone. {inserted} keywords inserted/updated.")

        # Verify
        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM keywords")).scalar()
            with_emb = conn.execute(text("SELECT COUNT(*) FROM keywords WHERE embedding IS NOT NULL")).scalar()
            print(f"  Keywords in DB: {count} ({with_emb} with embeddings)")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
