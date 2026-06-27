"""
test_keywords.py
─────────────────
Insert a few test keywords, generate embeddings with the biencoder,
and run vector similarity queries against existing jobs.

Usage:
    python -u scripts/test_keywords.py
    python -u scripts/test_keywords.py --biencoder-path models/biencoder
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEST_KEYWORDS = [
    # canonical_id, language_code, category_id, title
    (1, "en", 1, "Software Developer"),
    (1, "fr", 1, "Développeur Logiciel"),
    (2, "en", 2, "Architect"),       # construction
    (3, "en", 1, "Architect"),       # IT — same title, different category
    (4, "en", 1, "Data Scientist"),
]

# Category names for display (subset — adjust IDs to match your categories table)
CATEGORY_NAMES = {
    1: "IT",
    2: "Construction",
}


def build_keyword_text(title: str, language_code: str) -> str:
    return f"[lang={language_code}][country=XX][type=job] {title}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--biencoder-path", default="models/biencoder")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    from sqlalchemy import text
    from app.db.session import SessionLocal
    from app.services.biencoder import BiEncoder

    biencoder = BiEncoder(args.biencoder_path)
    db = SessionLocal()

    try:
        # 1. Insert test keywords
        print("Inserting test keywords ...")
        for canonical_id, lang, cat_id, title in TEST_KEYWORDS:
            db.execute(text("""
                INSERT INTO keywords (canonical_id, title, language_code, category_id)
                VALUES (:canonical_id, :title, :lang, :cat_id)
                ON CONFLICT (title, language_code, category_id) DO NOTHING
            """), {"canonical_id": canonical_id, "title": title, "lang": lang, "cat_id": cat_id})
        db.commit()
        print(f"  {len(TEST_KEYWORDS)} keywords inserted")

        # 2. Generate embeddings
        print("\nGenerating embeddings ...")
        rows = db.execute(text("SELECT id, title, language_code FROM keywords WHERE embedding IS NULL")).fetchall()
        if not rows:
            print("  All keywords already have embeddings")
        else:
            texts = [build_keyword_text(r.title, r.language_code) for r in rows]
            embeddings = biencoder.encode_batch(texts)
            for row, emb in zip(rows, embeddings):
                vec_str = "[" + ",".join(f"{x:.6f}" for x in emb) + "]"
                db.execute(
                    text("UPDATE keywords SET embedding = CAST(:emb AS vector) WHERE id = :id"),
                    {"emb": vec_str, "id": row.id},
                )
            db.commit()
            print(f"  {len(rows)} embeddings generated")

        # 3. Test similarity queries
        print("\n" + "=" * 80)
        keywords = db.execute(text("""
            SELECT id, canonical_id, title, language_code, category_id
            FROM keywords ORDER BY id
        """)).fetchall()

        for kw in keywords:
            cat_name = CATEGORY_NAMES.get(kw.category_id, f"cat={kw.category_id}")
            print(f"\nKeyword: \"{kw.title}\" [{kw.language_code}] ({cat_name})")
            print("-" * 60)

            # Find nearest jobs — filtered by language_code and category
            # Category filter uses the jobs.category text field for now
            results = db.execute(text("""
                SELECT j.id, j.title, j.language_code, j.category,
                       j.embedding <=> k.embedding AS distance
                FROM jobs j, keywords k
                WHERE k.id = :kw_id
                  AND j.embedding IS NOT NULL
                  AND j.language_code = k.language_code
                ORDER BY j.embedding <=> k.embedding
                LIMIT :top_k
            """), {"kw_id": kw.id, "top_k": args.top_k}).fetchall()

            if not results:
                print("  (no matching jobs found)")
            for r in results:
                print(f"  dist={r.distance:.4f}  [{r.language_code}] {r.title}  (cat: {r.category})")

        # 4. Test cross-language lookup via canonical_id
        print("\n" + "=" * 80)
        print("\nCross-language test: jobs matching canonical_id=1 (Software Developer / Développeur Logiciel)")
        print("-" * 60)
        results = db.execute(text("""
            WITH canonical_keywords AS (
                SELECT id, title, language_code, embedding FROM keywords WHERE canonical_id = 1
            )
            SELECT DISTINCT ON (j.id)
                j.id, j.title, j.language_code, j.category,
                j.embedding <=> k.embedding AS distance,
                k.title AS matched_keyword
            FROM jobs j
            CROSS JOIN canonical_keywords k
            WHERE j.embedding IS NOT NULL
              AND j.language_code = k.language_code
            ORDER BY j.id, j.embedding <=> k.embedding
            LIMIT :top_k
        """), {"top_k": args.top_k * 2}).fetchall()

        for r in results:
            print(f"  dist={r.distance:.4f}  [{r.language_code}] {r.title}  → matched: {r.matched_keyword}")

    finally:
        db.close()


if __name__ == "__main__":
    main()
