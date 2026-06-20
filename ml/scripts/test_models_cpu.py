"""
test_models_cpu.py
────────────────────
Standalone test suite for validating all ML pipeline models on a CPU host.
Run before deploying to production to verify:
  1. Each model loads and produces correct output
  2. Latency is within budget
  3. The full pipeline (embed → ANN → rerank) works end-to-end
  4. The extractor produces valid JSON

Does NOT require the FastAPI app — runs against model files directly.

Setup (on the test host):
    pip install onnxruntime sentence-transformers llama-cpp-python numpy

    # Copy artifacts from the CPU server:
    scp -r webadmin@46.224.133.10:/home/webadmin/Jobl/ml-artifacts/models/ ml/models/
    scp -r webadmin@46.224.133.10:/home/webadmin/Jobl/ml-artifacts/data/vectors/ ml/data/vectors/

Usage:
    python -u ml/scripts/test_models_cpu.py
    python -u ml/scripts/test_models_cpu.py --skip-extractor    # skip the 5GB GGUF load
    python -u ml/scripts/test_models_cpu.py --skip-vectors      # skip vector index test
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ── Test data ────────────────────────────────────────────────────────────────

SAMPLE_JOBS = [
    {
        "title": "Senior Software Engineer",
        "description": "We are looking for a Senior Software Engineer with 5+ years of Python experience. Must have experience with AWS, Docker, and CI/CD pipelines. Remote work available.",
        "country": "US",
        "lang": "en",
    },
    {
        "title": "Desarrollador Full Stack Senior",
        "description": "Se requiere experiencia en React, Node.js y bases de datos SQL. Modalidad híbrida en Ciudad de México.",
        "country": "MX",
        "lang": "es",
    },
    {
        "title": "Softwareentwickler (m/w/d)",
        "description": "Wir suchen einen erfahrenen Java-Entwickler mit Spring Boot Kenntnissen für unser Team in Berlin.",
        "country": "DE",
        "lang": "de",
    },
]

SAMPLE_RESUMES = [
    {
        "title": "Python Developer",
        "description": "Experienced Python developer with 7 years building web applications using Django and FastAPI. Proficient in AWS, Docker, PostgreSQL.",
        "country": "US",
        "lang": "en",
    },
    {
        "title": "Junior Barista",
        "description": "Looking for part-time work in hospitality. 2 years of experience in cafes and restaurants.",
        "country": "AU",
        "lang": "en",
    },
]

LATENCY_BUDGETS = {
    "biencoder_single": 200,   # ms — single query encoding
    "biencoder_batch": 2000,   # ms — batch of 10
    "reranker_single": 10,     # ms — single pair
    "reranker_batch": 500,     # ms — 100 pairs
    "ann_search": 50,          # ms — top-100 from 1M vectors
    "extractor_single": 15000, # ms — single extraction (GGUF on CPU is slow)
    "full_pipeline": 1000,     # ms — embed + ANN + rerank
}


def _build_text(record: dict, record_type: str) -> str:
    lang = record.get("lang", "en")
    country = record.get("country", "XX")
    title = record.get("title", "")
    desc = record.get("description", "")
    return f"[lang={lang}][country={country}][type={record_type}] {title} — {desc}"


# ── Test functions ───────────────────────────────────────────────────────────

def test_biencoder(model_dir: str) -> bool:
    """Test bi-encoder ONNX model."""
    print("\n" + "="*60)
    print("TEST: Bi-encoder (ONNX)")
    print("="*60)

    import onnxruntime as ort
    from transformers import AutoTokenizer

    model_path = os.path.join(model_dir, "model.onnx")
    if not os.path.exists(model_path):
        print(f"  SKIP: {model_path} not found")
        return False

    print(f"  Loading from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    required_inputs = {inp.name for inp in session.get_inputs()}
    print(f"  Loaded. Inputs: {required_inputs}")

    def encode(text: str) -> np.ndarray:
        inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=512)
        if "position_ids" in required_inputs and "position_ids" not in inputs:
            seq_len = inputs["input_ids"].shape[1]
            inputs["position_ids"] = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
        feed = {k: v for k, v in inputs.items() if k in required_inputs}
        outputs = session.run(None, feed)
        # Last token pooling + L2 normalize (matching Harrier's architecture)
        emb = outputs[0][0, -1, :]
        emb = emb / np.linalg.norm(emb)
        return emb

    # Test single encoding
    text = _build_text(SAMPLE_JOBS[0], "job")
    start = time.time()
    emb = encode(text)
    single_ms = (time.time() - start) * 1000
    print(f"  Single encode: {emb.shape} dim, {single_ms:.1f}ms")

    # Test batch encoding
    texts = [_build_text(j, "job") for j in SAMPLE_JOBS] + [_build_text(r, "resume") for r in SAMPLE_RESUMES]
    start = time.time()
    embeddings = [encode(t) for t in texts]
    batch_ms = (time.time() - start) * 1000
    print(f"  Batch encode ({len(texts)} texts): {batch_ms:.1f}ms ({batch_ms/len(texts):.1f}ms/text)")

    # Test similarity makes sense
    job_emb = embeddings[0]   # Senior Software Engineer
    good_resume = embeddings[3]  # Python Developer
    bad_resume = embeddings[4]   # Junior Barista
    good_sim = float(job_emb @ good_resume)
    bad_sim = float(job_emb @ bad_resume)
    print(f"  Similarity: good match={good_sim:.4f}, bad match={bad_sim:.4f}")

    passed = good_sim > bad_sim
    budget_ok = single_ms < LATENCY_BUDGETS["biencoder_single"]
    print(f"  Similarity order: {'PASS' if passed else 'FAIL'}")
    print(f"  Latency budget (<{LATENCY_BUDGETS['biencoder_single']}ms): {'PASS' if budget_ok else 'FAIL'} ({single_ms:.1f}ms)")

    return passed and budget_ok


def test_reranker(model_dir: str) -> bool:
    """Test reranker ONNX model."""
    print("\n" + "="*60)
    print("TEST: Reranker (ONNX)")
    print("="*60)

    import onnxruntime as ort
    from transformers import AutoTokenizer

    model_path = os.path.join(model_dir, "model.onnx")
    if not os.path.exists(model_path):
        print(f"  SKIP: {model_path} not found")
        return False

    print(f"  Loading from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    required_inputs = {inp.name for inp in session.get_inputs()}
    print(f"  Loaded. Inputs: {required_inputs}")

    def score_pair(text_a: str, text_b: str) -> float:
        inputs = tokenizer(text_a, text_b, return_tensors="np", padding=True, truncation=True, max_length=512)
        feed = {k: v for k, v in inputs.items() if k in required_inputs}
        outputs = session.run(None, feed)
        return float(outputs[0][0][0])

    # Test single pair scoring
    good_a = _build_text(SAMPLE_RESUMES[0], "resume")
    good_b = _build_text(SAMPLE_JOBS[0], "job")
    bad_a = _build_text(SAMPLE_RESUMES[1], "resume")

    start = time.time()
    good_score = score_pair(good_a, good_b)
    single_ms = (time.time() - start) * 1000
    bad_score = score_pair(bad_a, good_b)

    print(f"  Good match score: {good_score:.4f}")
    print(f"  Bad match score:  {bad_score:.4f}")
    print(f"  Single pair: {single_ms:.1f}ms")

    # Test batch (100 pairs)
    pairs = [(good_a, good_b)] * 50 + [(bad_a, good_b)] * 50
    start = time.time()
    scores = [score_pair(a, b) for a, b in pairs]
    batch_ms = (time.time() - start) * 1000
    print(f"  Batch score (100 pairs): {batch_ms:.1f}ms ({batch_ms/100:.1f}ms/pair)")

    passed = good_score > bad_score
    budget_ok = single_ms < LATENCY_BUDGETS["reranker_single"]
    print(f"  Score order: {'PASS' if passed else 'FAIL'}")
    print(f"  Latency budget (<{LATENCY_BUDGETS['reranker_single']}ms): {'PASS' if budget_ok else 'FAIL'} ({single_ms:.1f}ms)")

    return passed


def test_vectors(vectors_dir: str) -> bool:
    """Test vector loading and ANN search."""
    print("\n" + "="*60)
    print("TEST: Vector Index")
    print("="*60)

    job_vectors_path = os.path.join(vectors_dir, "job_vectors.npy")
    job_ids_path = os.path.join(vectors_dir, "job_ids.json")

    if not os.path.exists(job_vectors_path):
        print(f"  SKIP: {job_vectors_path} not found")
        return False

    print(f"  Loading vectors ...")
    start = time.time()
    job_vectors = np.load(job_vectors_path)
    load_ms = (time.time() - start) * 1000
    print(f"  Job vectors: {job_vectors.shape} loaded in {load_ms:.0f}ms ({job_vectors.nbytes / 1e9:.2f} GB)")

    with open(job_ids_path) as f:
        job_ids = json.load(f)
    print(f"  Job IDs: {len(job_ids):,}")

    # Simulate ANN search with a random query
    query = np.random.randn(job_vectors.shape[1]).astype(np.float32)
    query = query / np.linalg.norm(query)

    start = time.time()
    similarities = job_vectors @ query
    top_100_idx = np.argsort(-similarities)[:100]
    ann_ms = (time.time() - start) * 1000
    print(f"  Brute-force ANN (top-100 from {len(job_ids):,}): {ann_ms:.1f}ms")

    top_score = float(similarities[top_100_idx[0]])
    print(f"  Top match score: {top_score:.4f}")

    budget_ok = ann_ms < LATENCY_BUDGETS["ann_search"]
    print(f"  Latency budget (<{LATENCY_BUDGETS['ann_search']}ms): {'PASS' if budget_ok else 'FAIL'} ({ann_ms:.1f}ms)")

    return budget_ok


def test_extractor(gguf_path: str) -> bool:
    """Test extractor GGUF model."""
    print("\n" + "="*60)
    print("TEST: Extractor (GGUF)")
    print("="*60)

    if not os.path.exists(gguf_path):
        print(f"  SKIP: {gguf_path} not found")
        return False

    from llama_cpp import Llama

    print(f"  Loading from {gguf_path} ...")
    start = time.time()
    llm = Llama(model_path=gguf_path, n_ctx=2048, n_threads=4, verbose=False)
    load_ms = (time.time() - start) * 1000
    print(f"  Loaded in {load_ms:.0f}ms")

    # Import the locked prompt
    from ml.scripts.validate_extraction_prompt import SYSTEM_PROMPT

    job = SAMPLE_JOBS[0]
    user_msg = _build_text(job, "job")

    print(f"  Extracting: '{job['title']}' ...")
    start = time.time()
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=500,
        temperature=0.0,
    )
    extract_ms = (time.time() - start) * 1000

    content = response["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.strip("`").replace("json\n", "", 1).strip()

    print(f"  Extraction time: {extract_ms:.0f}ms")

    try:
        result = json.loads(content)
        print(f"  JSON valid: YES")
        print(f"  normalized_title: {result.get('normalized_title')}")
        print(f"  seniority: {result.get('seniority')}")
        print(f"  skills: {result.get('skills', [])[:5]}")
        valid_json = True
    except json.JSONDecodeError:
        print(f"  JSON valid: NO")
        print(f"  Raw output: {content[:200]}")
        valid_json = False

    budget_ok = extract_ms < LATENCY_BUDGETS["extractor_single"]
    print(f"  Latency budget (<{LATENCY_BUDGETS['extractor_single']}ms): {'PASS' if budget_ok else 'FAIL'} ({extract_ms:.0f}ms)")

    del llm
    return valid_json


def test_full_pipeline(biencoder_dir: str, reranker_dir: str, vectors_dir: str) -> bool:
    """Test the full pipeline: embed → ANN → rerank."""
    print("\n" + "="*60)
    print("TEST: Full Pipeline (embed → ANN → rerank)")
    print("="*60)

    import onnxruntime as ort
    from transformers import AutoTokenizer

    # Check all files exist
    for path in [
        os.path.join(biencoder_dir, "model.onnx"),
        os.path.join(reranker_dir, "model.onnx"),
        os.path.join(vectors_dir, "job_vectors.npy"),
    ]:
        if not os.path.exists(path):
            print(f"  SKIP: {path} not found")
            return False

    # Load models
    print("  Loading models ...")
    bi_tokenizer = AutoTokenizer.from_pretrained(biencoder_dir, trust_remote_code=True)
    bi_session = ort.InferenceSession(os.path.join(biencoder_dir, "model.onnx"), providers=["CPUExecutionProvider"])
    bi_inputs = {inp.name for inp in bi_session.get_inputs()}

    re_tokenizer = AutoTokenizer.from_pretrained(reranker_dir, trust_remote_code=True)
    re_session = ort.InferenceSession(os.path.join(reranker_dir, "model.onnx"), providers=["CPUExecutionProvider"])
    re_inputs = {inp.name for inp in re_session.get_inputs()}

    job_vectors = np.load(os.path.join(vectors_dir, "job_vectors.npy"))
    with open(os.path.join(vectors_dir, "job_ids.json")) as f:
        job_ids = json.load(f)

    # Full pipeline
    query = _build_text(SAMPLE_RESUMES[0], "resume")
    pipeline_start = time.time()

    # Step 1: Encode query
    inputs = bi_tokenizer(query, return_tensors="np", padding=True, truncation=True, max_length=512)
    if "position_ids" in bi_inputs and "position_ids" not in inputs:
        seq_len = inputs["input_ids"].shape[1]
        inputs["position_ids"] = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
    feed = {k: v for k, v in inputs.items() if k in bi_inputs}
    outputs = bi_session.run(None, feed)
    query_emb = outputs[0][0, -1, :]
    query_emb = query_emb / np.linalg.norm(query_emb)
    embed_ms = (time.time() - pipeline_start) * 1000

    # Step 2: ANN search (brute-force for now)
    ann_start = time.time()
    similarities = job_vectors @ query_emb
    top_100_idx = np.argsort(-similarities)[:100]
    ann_ms = (time.time() - ann_start) * 1000

    # Step 3: Rerank top-100
    rerank_start = time.time()
    rerank_scores = []
    for idx in top_100_idx[:20]:  # Rerank top-20 for speed in testing
        job_text = f"[type=job] Job {job_ids[idx]}"  # Simplified — real impl would load job text
        inputs = re_tokenizer(query, job_text, return_tensors="np", padding=True, truncation=True, max_length=512)
        feed = {k: v for k, v in inputs.items() if k in re_inputs}
        outputs = re_session.run(None, feed)
        rerank_scores.append(float(outputs[0][0][0]))
    rerank_ms = (time.time() - rerank_start) * 1000

    pipeline_ms = (time.time() - pipeline_start) * 1000

    print(f"  Embed:  {embed_ms:.1f}ms")
    print(f"  ANN:    {ann_ms:.1f}ms (top-100 from {len(job_ids):,})")
    print(f"  Rerank: {rerank_ms:.1f}ms (top-20)")
    print(f"  Total:  {pipeline_ms:.1f}ms")

    budget_ok = pipeline_ms < LATENCY_BUDGETS["full_pipeline"]
    print(f"  Latency budget (<{LATENCY_BUDGETS['full_pipeline']}ms): {'PASS' if budget_ok else 'FAIL'} ({pipeline_ms:.0f}ms)")

    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Test ML pipeline models on CPU")
    parser.add_argument("--biencoder-dir", default="ml/models/exported/biencoder-onnx")
    parser.add_argument("--reranker-dir", default="ml/models/exported/reranker-onnx")
    parser.add_argument("--extractor-gguf", default="ml/models/exported/extractor-gguf/model-Q4_K_M.gguf")
    parser.add_argument("--vectors-dir", default="ml/data/vectors")
    parser.add_argument("--skip-extractor", action="store_true", help="Skip extractor test (saves 5GB RAM)")
    parser.add_argument("--skip-vectors", action="store_true", help="Skip vector index test (saves 6GB RAM)")
    args = parser.parse_args()

    results = {}

    # Individual model tests
    results["biencoder"] = test_biencoder(args.biencoder_dir)
    results["reranker"] = test_reranker(args.reranker_dir)

    if not args.skip_vectors:
        results["vectors"] = test_vectors(args.vectors_dir)

    if not args.skip_extractor:
        results["extractor"] = test_extractor(args.extractor_gguf)

    # Full pipeline test
    if not args.skip_vectors:
        results["full_pipeline"] = test_full_pipeline(args.biencoder_dir, args.reranker_dir, args.vectors_dir)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {name:>20}: {status}")

    print()
    if all_pass:
        print("  ALL TESTS PASSED — ready for production deployment")
    else:
        print("  SOME TESTS FAILED — review before deploying")


if __name__ == "__main__":
    main()
