"""
backfill_embeddings.py
───────────────────────
Step 1 of the backfill: load pre-computed vectors from numpy files into
Jobl Postgres jobs.embedding and resumes.embedding (pgvector columns).

Usage:
    python -u scripts/backfill_embeddings.py
    python -u scripts/backfill_embeddings.py --table jobs
    python -u scripts/backfill_embeddings.py --table resumes
    python -u scripts/backfill_embeddings.py --vectors-dir ~/Jobl/ml-artifacts/data/vectors
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BATCH_SIZE = 1000
HEARTBEAT_EVERY = 10_000


def vec_to_pgvector(vec: np.ndarray) -> str:
    """Convert a numpy vector to pgvector string format '[0.0123, -0.0456, ...]'."""
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def backfill_table(table: str, vectors_path: str, ids_path: str) -> None:
    from sqlalchemy import text
    from app.db.session import SessionLocal

    print(f"Loading vectors from {vectors_path} ...")
    vectors = np.load(vectors_path)
    with open(ids_path) as f:
        ids = json.load(f)

    assert len(vectors) == len(ids), f"Vector count {len(vectors)} != ID count {len(ids)}"
    print(f"  {len(vectors):,} vectors loaded ({vectors.nbytes / 1e9:.2f} GB)")

    db = SessionLocal()
    try:
        updated = 0
        skipped = 0
        start_time = time.time()

        for i in range(0, len(ids), BATCH_SIZE):
            batch_ids = ids[i:i + BATCH_SIZE]
            batch_vecs = vectors[i:i + BATCH_SIZE]

            params = []
            for j, (record_id, vec) in enumerate(zip(batch_ids, batch_vecs)):
                params.append({"id": record_id, "embedding": vec_to_pgvector(vec)})

            if params:
                db.execute(
                    text(f"UPDATE {table} SET embedding = CAST(:embedding AS vector) WHERE id = :id"),
                    params,
                )
                db.commit()
                updated += len(params)

            if updated % HEARTBEAT_EVERY < BATCH_SIZE:
                elapsed = time.time() - start_time
                rate = updated / elapsed if elapsed > 0 else 0
                print(f"  ... {updated:,} / {len(ids):,} updated ({rate:.0f} rows/sec)")

        elapsed = time.time() - start_time
        print(f"\nDone. {updated:,} rows updated in {elapsed:.1f}s ({updated/elapsed:.0f} rows/sec)")
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="Backfill embeddings from pre-computed numpy vectors")
    parser.add_argument("--table", choices=["jobs", "resumes", "both"], default="both")
    parser.add_argument("--vectors-dir", default=os.path.expanduser("~/Jobl/ml-artifacts/data/vectors"))
    args = parser.parse_args()

    if args.table in ("jobs", "both"):
        job_vectors = os.path.join(args.vectors_dir, "job_vectors.npy")
        job_ids = os.path.join(args.vectors_dir, "job_ids.json")
        if os.path.exists(job_vectors):
            print("Backfilling jobs ...")
            backfill_table("jobs", job_vectors, job_ids)
        else:
            print(f"SKIP: {job_vectors} not found")

    if args.table in ("resumes", "both"):
        resume_vectors = os.path.join(args.vectors_dir, "resume_vectors.npy")
        resume_ids = os.path.join(args.vectors_dir, "resume_ids.json")
        if os.path.exists(resume_vectors):
            print("\nBackfilling resumes ...")
            backfill_table("resumes", resume_vectors, resume_ids)
        else:
            print(f"SKIP: {resume_vectors} not found")


if __name__ == "__main__":
    main()
