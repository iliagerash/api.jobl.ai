"""
generate_taxonomy.py
─────────────────────
Generate a taxonomy of ~1-2K normalized job title keywords from the jobl
database using k-means clustering + Qwen2.5-72B naming.

Steps:
1. Export distinct (title, language_code, category_id) with one vector each
2. K-means cluster into ~2000 groups
3. Sample representative titles per cluster
4. Prompt Qwen via vLLM to produce canonical name + French equivalent
5. Output JSON ready for load_keywords.py

Usage (on GPU host with vLLM running):
    # Start vLLM first:
    # vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ --quantization awq --max-model-len 4096

    python -u ml/scripts/generate_taxonomy.py
    python -u ml/scripts/generate_taxonomy.py --clusters 2000
    python -u ml/scripts/generate_taxonomy.py --step export     # export only
    python -u ml/scripts/generate_taxonomy.py --step cluster    # cluster only
    python -u ml/scripts/generate_taxonomy.py --step name       # name only (requires prior steps)

Outputs:
    ml/data/taxonomy/titles_vectors.npz     — distinct titles + vectors
    ml/data/taxonomy/clusters.json          — cluster assignments + samples
    ml/data/taxonomy/keywords.json          — final taxonomy
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

OUTPUT_DIR = "ml/data/taxonomy"
HEARTBEAT_EVERY = 10_000
SAMPLES_PER_CLUSTER = 20

SYSTEM_PROMPT = """You are a job title taxonomy expert. Given a list of sample job titles from a cluster, produce ONE canonical normalized job title that best represents the group.

Rules:
- The canonical title should be a clean, standard job title (e.g. "Software Developer", not "Sr. Software Dev - Backend (Remote)")
- Remove seniority prefixes (Junior, Senior, Lead, etc.) — the canonical title should be level-agnostic
- Remove company names, locations, team names, and parenthetical details
- Use standard industry terminology
- Keep it concise: 1-4 words typically

Respond with ONLY a JSON object, no markdown or explanation:
{"en": "English Canonical Title", "fr": "French Equivalent Title"}"""


def step_export(db_url: str, output_dir: str) -> None:
    """Export distinct titles with vectors from jobl DB."""
    from sqlalchemy import create_engine, text

    print("Step 1: Exporting distinct titles with vectors ...")
    engine = create_engine(db_url, pool_pre_ping=True)

    query = text("""
        SELECT DISTINCT ON (title, language_code, category_id)
            title, language_code, category_id, embedding
        FROM jobs
        WHERE embedding IS NOT NULL
          AND category_id IS NOT NULL
          AND title IS NOT NULL
          AND language_code IN ('en', 'fr')
        ORDER BY title, language_code, category_id, id DESC
    """)

    titles = []
    vectors = []
    start = time.time()

    with engine.connect() as conn:
        result = conn.execution_options(stream_results=True).execute(query)
        for i, row in enumerate(result):
            titles.append({
                "title": row.title,
                "language_code": row.language_code,
                "category_id": int(row.category_id),
            })
            vectors.append(np.frombuffer(row.embedding, dtype=np.float32) if isinstance(row.embedding, (bytes, memoryview)) else np.array(row.embedding, dtype=np.float32))

            if (i + 1) % HEARTBEAT_EVERY == 0:
                elapsed = time.time() - start
                print(f"  {i + 1:,} titles exported ({elapsed:.1f}s)")

    engine.dispose()

    vectors_array = np.stack(vectors)
    print(f"  {len(titles):,} distinct titles, vectors shape: {vectors_array.shape}")

    np.savez_compressed(
        os.path.join(output_dir, "titles_vectors.npz"),
        vectors=vectors_array,
    )
    with open(os.path.join(output_dir, "titles.json"), "w") as f:
        json.dump(titles, f)

    elapsed = time.time() - start
    print(f"  Export done in {elapsed:.1f}s")


def step_cluster(output_dir: str, n_clusters: int) -> None:
    """K-means cluster the exported vectors."""
    print(f"Step 2: Clustering into {n_clusters} groups ...")
    start = time.time()

    data = np.load(os.path.join(output_dir, "titles_vectors.npz"))
    vectors = data["vectors"]
    with open(os.path.join(output_dir, "titles.json")) as f:
        titles = json.load(f)

    print(f"  Loaded {len(titles):,} titles, vectors shape: {vectors.shape}")

    try:
        import faiss
        print("  Using faiss for clustering ...")
        d = vectors.shape[1]
        kmeans = faiss.Kmeans(d, n_clusters, niter=20, verbose=True, gpu=True)
        kmeans.train(vectors.astype(np.float32))
        _, assignments = kmeans.index.search(vectors.astype(np.float32), 1)
        labels = assignments.flatten()
    except ImportError:
        print("  faiss not available, using scikit-learn ...")
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=n_clusters, batch_size=10000, n_init=3, random_state=42)
        labels = km.fit_predict(vectors)

    clusters = {}
    for idx, label in enumerate(labels):
        label = int(label)
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)

    cluster_samples = {}
    for label, indices in clusters.items():
        sample_indices = indices[:SAMPLES_PER_CLUSTER] if len(indices) <= SAMPLES_PER_CLUSTER else list(np.random.choice(indices, SAMPLES_PER_CLUSTER, replace=False))
        cluster_samples[label] = {
            "size": len(indices),
            "sample_titles": [titles[i]["title"] for i in sample_indices],
            "sample_languages": [titles[i]["language_code"] for i in sample_indices],
            "sample_categories": [titles[i]["category_id"] for i in sample_indices],
            "all_indices": indices,
        }

    with open(os.path.join(output_dir, "clusters.json"), "w") as f:
        json.dump(cluster_samples, f, indent=2, default=int)

    elapsed = time.time() - start
    print(f"  Clustering done in {elapsed:.1f}s — {len(clusters)} clusters")

    sizes = [c["size"] for c in cluster_samples.values()]
    print(f"  Cluster sizes: min={min(sizes)}, max={max(sizes)}, median={sorted(sizes)[len(sizes)//2]}, mean={sum(sizes)/len(sizes):.0f}")


def step_name(output_dir: str, api_base: str, model: str, workers: int) -> None:
    """Use Qwen to name each cluster."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from openai import OpenAI

    print("Step 3: Naming clusters with Qwen ...")
    start = time.time()

    with open(os.path.join(output_dir, "clusters.json")) as f:
        clusters = json.load(f)

    with open(os.path.join(output_dir, "titles.json")) as f:
        titles = json.load(f)

    output_path = os.path.join(output_dir, "keywords.json")
    existing = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        print(f"  Resuming — {len(existing)} clusters already named")

    client = OpenAI(base_url=api_base, api_key="not-needed")
    keywords = dict(existing)
    errors = 0

    def name_cluster(cluster_id: str, cluster_data: dict) -> tuple[str, dict | None]:
        sample_titles = cluster_data["sample_titles"]
        user_msg = "Sample job titles from this cluster:\n" + "\n".join(f"- {t}" for t in sample_titles)

        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=200,
                    temperature=0.0,
                )
                content = resp.choices[0].message.content or ""
                content = content.strip()
                if content.startswith("```"):
                    content = content.strip("`").replace("json\n", "", 1).strip()
                result = json.loads(content)
                if "en" in result and "fr" in result:
                    return cluster_id, result
            except Exception as e:
                if attempt == 2:
                    return cluster_id, {"_error": str(e), "sample": sample_titles[:3]}
                time.sleep(1)
        return cluster_id, None

    to_process = {cid: cdata for cid, cdata in clusters.items() if cid not in existing}
    print(f"  {len(to_process)} clusters to name ...")

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(name_cluster, cid, cdata): cid for cid, cdata in to_process.items()}
        for future in as_completed(futures):
            cluster_id, result = future.result()
            if result:
                keywords[cluster_id] = result
                if "_error" in result:
                    errors += 1
            completed += 1

            if completed % 50 == 0:
                elapsed = time.time() - start
                rate = completed / elapsed
                remaining = (len(to_process) - completed) / rate if rate > 0 else 0
                print(f"  {completed}/{len(to_process)} named ({rate:.1f}/s, ~{remaining/60:.0f}m remaining, {errors} errors)")
                with open(output_path, "w") as f:
                    json.dump(keywords, f, indent=2, ensure_ascii=False)

    with open(output_path, "w") as f:
        json.dump(keywords, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start
    print(f"  Naming done in {elapsed:.1f}s — {len(keywords)} keywords, {errors} errors")


def build_final_output(output_dir: str) -> None:
    """Build the final keywords list with category_id assignments."""
    print("Building final taxonomy output ...")

    with open(os.path.join(output_dir, "clusters.json")) as f:
        clusters = json.load(f)

    with open(os.path.join(output_dir, "titles.json")) as f:
        titles = json.load(f)

    with open(os.path.join(output_dir, "keywords.json")) as f:
        keywords = json.load(f)

    taxonomy = []
    canonical_id = 1

    for cluster_id, kw in sorted(keywords.items(), key=lambda x: int(x[0])):
        if "_error" in kw:
            continue

        cluster = clusters.get(str(cluster_id), {})
        categories = cluster.get("sample_categories", [])
        from collections import Counter
        most_common_cat = Counter(categories).most_common(1)[0][0] if categories else 0

        taxonomy.append({
            "canonical_id": canonical_id,
            "en": kw["en"],
            "fr": kw["fr"],
            "category_id": most_common_cat,
            "cluster_size": cluster.get("size", 0),
        })
        canonical_id += 1

    output_path = os.path.join(output_dir, "taxonomy_final.json")
    with open(output_path, "w") as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)

    print(f"  {len(taxonomy)} canonical keywords written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate job title taxonomy via clustering + Qwen")
    parser.add_argument("--db-url", default=None, help="Database URL (default: from .env)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--clusters", type=int, default=2000, help="Number of k-means clusters")
    parser.add_argument("--api-base", default="http://localhost:8000/v1", help="vLLM server URL")
    parser.add_argument("--model", default="Qwen/Qwen2.5-72B-Instruct-AWQ")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent workers for Qwen")
    parser.add_argument("--step", choices=["all", "export", "cluster", "name", "finalize"], default="all")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    db_url = args.db_url
    if not db_url:
        from dotenv import load_dotenv
        load_dotenv()
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            sys.exit("No DATABASE_URL found in environment or --db-url")

    if args.step in ("all", "export"):
        step_export(db_url, args.output_dir)

    if args.step in ("all", "cluster"):
        step_cluster(args.output_dir, args.clusters)

    if args.step in ("all", "name"):
        step_name(args.output_dir, args.api_base, args.model, args.workers)

    if args.step in ("all", "finalize"):
        build_final_output(args.output_dir)


if __name__ == "__main__":
    main()
