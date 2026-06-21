"""Quick test for the bi-encoder service.

Run: python -u tests/test_biencoder.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.environ.get("BIENCODER_MODEL_PATH", "models/biencoder-onnx")


def main():
    from app.services.biencoder import BiEncoder

    print(f"Loading bi-encoder from {MODEL_PATH} ...")
    start = time.time()
    model = BiEncoder(MODEL_PATH)
    print(f"  Loaded in {time.time() - start:.1f}s, ready={model.is_ready()}")

    # Single encoding
    text = "[lang=en][country=US][type=job] Senior Software Engineer — 5+ years Python, AWS, Docker"
    start = time.time()
    emb = model.encode(text)
    ms = (time.time() - start) * 1000
    print(f"  Single encode: {len(emb)} dims, {ms:.1f}ms")

    # Verify normalization
    norm = sum(x * x for x in emb) ** 0.5
    print(f"  L2 norm: {norm:.4f} (should be ~1.0)")

    # Batch encoding
    texts = [
        "[lang=en][country=US][type=job] Senior Software Engineer — Python, AWS",
        "[lang=es][country=MX][type=job] Desarrollador Full Stack — React, Node.js",
        "[lang=en][country=US][type=resume] Python Developer — 7 years experience",
    ]
    start = time.time()
    embs = model.encode_batch(texts)
    ms = (time.time() - start) * 1000
    print(f"  Batch encode ({len(texts)} texts): {ms:.1f}ms ({ms/len(texts):.1f}ms/text)")

    # Similarity check
    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    job_emb = embs[0]
    resume_emb = embs[2]
    other_job = embs[1]
    good_sim = dot(job_emb, resume_emb)
    cross_sim = dot(job_emb, other_job)
    print(f"  Similarity job↔resume (good match): {good_sim:.4f}")
    print(f"  Similarity job↔job (cross-lang): {cross_sim:.4f}")

    # Assertions
    assert model.is_ready(), "Model not ready"
    assert len(emb) == 1024, f"Expected 1024 dims, got {len(emb)}"
    assert 0.99 < norm < 1.01, f"Expected unit norm, got {norm}"
    assert good_sim > 0, "Good match should have positive similarity"

    print("\n  ALL TESTS PASSED")


if __name__ == "__main__":
    main()
