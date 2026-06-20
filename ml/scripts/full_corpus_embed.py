"""
full_corpus_embed.py
─────────────────────
Phase 5: encode all jobs and resumes with the trained bi-encoder, producing
dense vectors for ANN search in production.

Processes the train+val pool files (= full corpus after dedup/splits) in
batches on GPU. Outputs numpy .npy files + ID mapping for loading into
pgvector or Qdrant.

Usage:
    python -u ml/scripts/full_corpus_embed.py
    python -u ml/scripts/full_corpus_embed.py --mode jobs       # jobs only
    python -u ml/scripts/full_corpus_embed.py --mode resumes    # resumes only
    python -u ml/scripts/full_corpus_embed.py --batch-size 256
    python -u ml/scripts/full_corpus_embed.py --limit 1000      # test run

Outputs:
    ml/data/vectors/job_vectors.npy         — float32 matrix (N_jobs × embedding_dim)
    ml/data/vectors/job_ids.json            — ordered list of job IDs matching vector rows
    ml/data/vectors/resume_vectors.npy      — float32 matrix (N_resumes × embedding_dim)
    ml/data/vectors/resume_ids.json         — ordered list of resume IDs matching vector rows
"""

import argparse
import html
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

HEARTBEAT_EVERY = 10_000
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _build_text(record: dict, record_type: str) -> str:
    """Build prefixed text for encoding, matching training format."""
    lang = record.get("language_fasttext") or record.get("language_code") or "unknown"
    country = record.get("country_code") or "XX"
    title = record.get("title", "")
    desc = record.get("description") or ""
    # Strip HTML for jobs (resumes are already plain text from PII scrubbing)
    if "<" in desc:
        desc = html.unescape(desc)
        desc = _HTML_TAG_RE.sub(" ", desc)
        desc = re.sub(r"\s+", " ", desc).strip()
    desc = desc[:2000]
    return f"[lang={lang}][country={country}][type={record_type}] {title} — {desc}"


def iter_jsonl(path: str):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def encode_corpus(
    model,
    input_paths: list[str],
    record_type: str,
    vectors_path: str,
    ids_path: str,
    batch_size: int,
    limit: int | None,
) -> int:
    """Encode all records from input files, save vectors + ID mapping."""
    # Collect texts and IDs
    print(f"  Loading texts from {len(input_paths)} file(s) ...")
    texts = []
    ids = []
    for path in input_paths:
        if not os.path.exists(path):
            print(f"    SKIP: {path} not found")
            continue
        for record in iter_jsonl(path):
            texts.append(_build_text(record, record_type))
            ids.append(record.get("id"))
            if limit and len(texts) >= limit:
                break
        if limit and len(texts) >= limit:
            break
    print(f"  {len(texts):,} texts loaded")

    if not texts:
        return 0

    # Encode in batches
    print(f"  Encoding with batch_size={batch_size} ...")
    start_time = time.time()
    all_embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    elapsed = time.time() - start_time
    rate = len(texts) / elapsed
    print(f"  Encoded {len(texts):,} texts in {elapsed:.1f}s ({rate:.0f} texts/s)")

    # Save
    vectors = np.array(all_embeddings, dtype=np.float32)
    np.save(vectors_path, vectors)
    print(f"  Vectors saved: {vectors_path} ({vectors.shape}, {vectors.nbytes / 1e9:.2f} GB)")

    with open(ids_path, "w") as f:
        json.dump(ids, f)
    print(f"  IDs saved: {ids_path} ({len(ids):,} entries)")

    return len(texts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode full corpus with bi-encoder for ANN search")
    parser.add_argument("--model", default="ml/models/exported/biencoder")
    parser.add_argument("--mode", choices=["both", "jobs", "resumes"], default="both")
    parser.add_argument("--output-dir", default="ml/data/vectors")
    parser.add_argument("--batch-size", type=int, default=256, help="Encoding batch size (default: 256)")
    parser.add_argument("--limit", type=int, default=None, help="Limit records per corpus (for testing)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from sentence_transformers import SentenceTransformer

    print(f"Loading bi-encoder from {args.model} ...")
    model = SentenceTransformer(args.model, trust_remote_code=True)
    print(f"  Embedding dim: {model.get_sentence_embedding_dimension()}")

    total = 0

    if args.mode in ("both", "jobs"):
        print(f"\nEncoding jobs ...")
        job_paths = [
            "ml/data/splits/train_pool_jobs.jsonl",
            "ml/data/splits/val_pool_jobs.jsonl",
        ]
        n = encode_corpus(
            model, job_paths, "job",
            os.path.join(args.output_dir, "job_vectors.npy"),
            os.path.join(args.output_dir, "job_ids.json"),
            args.batch_size, args.limit,
        )
        total += n

    if args.mode in ("both", "resumes"):
        print(f"\nEncoding resumes ...")
        resume_paths = [
            "ml/data/splits/train_pool_resumes.jsonl",
            "ml/data/splits/val_pool_resumes.jsonl",
        ]
        n = encode_corpus(
            model, resume_paths, "resume",
            os.path.join(args.output_dir, "resume_vectors.npy"),
            os.path.join(args.output_dir, "resume_ids.json"),
            args.batch_size, args.limit,
        )
        total += n

    print(f"\nDone. {total:,} total vectors encoded.")


if __name__ == "__main__":
    main()
