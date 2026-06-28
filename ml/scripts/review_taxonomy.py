"""
review_taxonomy.py
───────────────────
Review and fix taxonomy keywords using OpenAI API (gpt-5-nano).
Submits keywords in batches of 50, asks the model to:
- Flag and fix generic titles (e.g. "Specialist" → "HR Specialist")
- Fix inaccurate French translations
- Ensure titles are real, standard job titles

Uses the cluster sample titles for context when fixing generic titles.

Usage:
    python -u ml/scripts/review_taxonomy.py
    python -u ml/scripts/review_taxonomy.py --dry-run
    python -u ml/scripts/review_taxonomy.py --batch-size 50
    python -u ml/scripts/review_taxonomy.py --model gpt-5-nano

Outputs:
    ml/data/taxonomy/taxonomy_reviewed.json
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

TAXONOMY_DIR = "ml/data/taxonomy"

SYSTEM_PROMPT = """You are a job title taxonomy expert reviewing a list of canonical job titles with their French translations.

For each entry, check:
1. Is the English title specific enough to be useful? Single-word titles like "Specialist", "Manager", "Officer", "Director", "Coordinator" are too generic — they need a domain qualifier based on the sample job titles provided.
2. Is the French translation accurate and natural?
3. Is this a real, standard job title?

You will receive a JSON array of entries, each with:
- "id": canonical ID
- "en": English title
- "fr": French title
- "category_id": job category
- "samples": sample job titles from the cluster (use these to determine the right specialization for generic titles)

Return a JSON array with ALL entries. For each entry return:
- "id": same canonical ID
- "en": corrected English title (or original if fine)
- "fr": corrected French title (or original if fine)
- "changed": true if you modified either title, false if both were fine

Rules:
- Do NOT add seniority levels (Junior, Senior, Lead)
- Keep titles concise: 1-4 words typically
- For generic titles, pick the most common specialization from the samples
- Do NOT over-specify (e.g. "Software Developer" is fine, don't make it "Full-Stack Software Developer")
- Respond with a JSON object containing a "results" key with an array of all entries. Example: {"results": [{"id": 1, "en": "...", "fr": "...", "changed": false}, ...]}"""


def load_data(taxonomy_dir: str):
    with open(os.path.join(taxonomy_dir, "taxonomy_deduped.json")) as f:
        taxonomy = json.load(f)
    clusters_path = os.path.join(taxonomy_dir, "clusters.json")
    clusters = {}
    if os.path.exists(clusters_path):
        with open(clusters_path) as f:
            clusters = json.load(f)
    return taxonomy, clusters


def build_batch(taxonomy: list, clusters: dict, start: int, batch_size: int) -> list:
    batch = []
    for item in taxonomy[start:start + batch_size]:
        cluster_id = str(item["canonical_id"] - 1)
        cluster = clusters.get(cluster_id, {})
        samples = cluster.get("sample_titles", [])[:5]

        batch.append({
            "id": item["canonical_id"],
            "en": item["en"],
            "fr": item["fr"],
            "category_id": item["category_id"],
            "samples": samples,
        })
    return batch


def review_batch(client, model: str, batch: list) -> list | None:
    user_msg = json.dumps(batch, ensure_ascii=False, indent=2)

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or ""
            if not content:
                print(f"\n    EMPTY RESPONSE (attempt {attempt + 1}/3)", flush=True)
                if attempt < 2:
                    time.sleep(2)
                    continue
                return None
            result = json.loads(content)
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                for v in result.values():
                    if isinstance(v, list):
                        return v
            print(f"\n    UNEXPECTED FORMAT: {str(result)[:300]}", flush=True)
            return None
        except Exception as e:
            print(f"\n    ERROR (attempt {attempt + 1}/3): {e}", flush=True)
            if hasattr(resp, 'choices') and resp.choices:
                print(f"    RAW: {(resp.choices[0].message.content or '')[:300]}", flush=True)
            if attempt == 2:
                return None
            time.sleep(2)
    return None


def main():
    parser = argparse.ArgumentParser(description="Review taxonomy with OpenAI")
    parser.add_argument("--taxonomy-dir", default=TAXONOMY_DIR)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    taxonomy, clusters = load_data(args.taxonomy_dir)
    print(f"Loaded {len(taxonomy)} keywords")

    if args.dry_run:
        batch = build_batch(taxonomy, clusters, 0, 3)
        print("Sample batch:")
        print(json.dumps(batch, indent=2, ensure_ascii=False))
        return

    from openai import OpenAI
    client = OpenAI()

    total_changed = 0
    total_batches = (len(taxonomy) + args.batch_size - 1) // args.batch_size
    results_map = {}

    for i in range(0, len(taxonomy), args.batch_size):
        batch_num = i // args.batch_size + 1
        batch = build_batch(taxonomy, clusters, i, args.batch_size)

        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} items) ...", end="", flush=True)
        result = review_batch(client, args.model, batch)

        if result:
            changed = sum(1 for r in result if r.get("changed"))
            total_changed += changed
            for r in result:
                results_map[r["id"]] = r
            print(f" {changed} changed", flush=True)
        else:
            print(" FAILED — retrying ...", end="", flush=True)
            time.sleep(3)
            result = review_batch(client, args.model, batch)
            if result:
                changed = sum(1 for r in result if r.get("changed"))
                total_changed += changed
                for r in result:
                    results_map[r["id"]] = r
                print(f" {changed} changed (retry)", flush=True)
            else:
                print(" FAILED again — keeping originals", flush=True)

    # Apply changes
    for item in taxonomy:
        reviewed = results_map.get(item["canonical_id"])
        if reviewed and reviewed.get("changed"):
            item["en"] = reviewed["en"]
            item["fr"] = reviewed["fr"]

    # Dedup again after fixes (same en + category_id might now match)
    seen = {}
    for item in taxonomy:
        key = (item["en"], item["category_id"])
        if key not in seen or item["cluster_size"] > seen[key]["cluster_size"]:
            seen[key] = item
    taxonomy = sorted(seen.values(), key=lambda x: -x["cluster_size"])
    for i, item in enumerate(taxonomy, 1):
        item["canonical_id"] = i

    output_path = os.path.join(args.taxonomy_dir, "taxonomy_reviewed.json")
    with open(output_path, "w") as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {total_changed} titles changed, {len(taxonomy)} keywords after dedup.")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
