"""
refine_taxonomy.py
───────────────────
Post-process the generated taxonomy:
1. Re-prompt Qwen for generic/duplicate titles with more specific instructions
2. Merge remaining true duplicates (same en + category_id)
3. Output refined taxonomy

Usage (on GPU host with vLLM running):
    python -u ml/scripts/refine_taxonomy.py
    python -u ml/scripts/refine_taxonomy.py --dry-run          # preview only
    python -u ml/scripts/refine_taxonomy.py --model ml/models/base/qwen-72b-awq
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

TAXONOMY_DIR = "ml/data/taxonomy"

REFINE_PROMPT = """You are a job title taxonomy expert. The previous attempt to name this cluster produced a title that is too generic: "{previous_title}".

Given the sample job titles below, produce a MORE SPECIFIC canonical title that captures what makes this cluster distinct from other "{previous_title}" clusters. Include the domain or specialization (e.g., "Restaurant Manager", "IT Project Manager", "HR Specialist").

Sample job titles:
{samples}

Respond with ONLY a JSON object:
{{"en": "More Specific English Title", "fr": "French Equivalent"}}"""


def load_data(taxonomy_dir: str):
    with open(os.path.join(taxonomy_dir, "taxonomy_final.json")) as f:
        taxonomy = json.load(f)
    with open(os.path.join(taxonomy_dir, "clusters.json")) as f:
        clusters = json.load(f)
    with open(os.path.join(taxonomy_dir, "keywords.json")) as f:
        keywords = json.load(f)
    return taxonomy, clusters, keywords


def find_items_to_refine(taxonomy: list, min_duplicates: int = 2, max_word_count: int = 1) -> list:
    """Find taxonomy items that need refinement: duplicates and single-word generics."""
    en_counts = Counter(d["en"] for d in taxonomy)

    to_refine = []
    for item in taxonomy:
        is_duplicate = en_counts[item["en"]] >= min_duplicates
        is_generic = len(item["en"].split()) <= max_word_count
        if is_duplicate or is_generic:
            to_refine.append(item)

    return to_refine


def refine_one(client, model: str, item: dict, cluster_data: dict) -> dict | None:
    samples = cluster_data.get("sample_titles", [])[:15]
    samples_text = "\n".join(f"- {t}" for t in samples)

    prompt = REFINE_PROMPT.format(
        previous_title=item["en"],
        samples=samples_text,
    )

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.0,
            )
            content = resp.choices[0].message.content or ""
            content = content.strip()
            if content.startswith("```"):
                content = content.strip("`").replace("json\n", "", 1).strip()
            result = json.loads(content)
            if "en" in result and "fr" in result:
                return result
        except Exception:
            if attempt == 2:
                return None
            time.sleep(1)
    return None


def merge_duplicates(taxonomy: list) -> list:
    """Merge items with identical (en, category_id) — keep the one with largest cluster."""
    seen = {}
    for item in taxonomy:
        key = (item["en"], item["category_id"])
        if key not in seen or item["cluster_size"] > seen[key]["cluster_size"]:
            seen[key] = item

    merged = sorted(seen.values(), key=lambda x: x["canonical_id"])
    # Re-assign canonical_ids
    for i, item in enumerate(merged, 1):
        item["canonical_id"] = i

    return merged


def main():
    parser = argparse.ArgumentParser(description="Refine taxonomy: fix generic/duplicate titles")
    parser.add_argument("--taxonomy-dir", default=TAXONOMY_DIR)
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-72B-Instruct-AWQ")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Preview what would be refined")
    args = parser.parse_args()

    taxonomy, clusters, keywords = load_data(args.taxonomy_dir)

    to_refine = find_items_to_refine(taxonomy)
    print(f"Total keywords: {len(taxonomy)}")
    print(f"Need refinement: {len(to_refine)} (duplicates + single-word generics)")

    if args.dry_run:
        en_counts = Counter(d["en"] for d in taxonomy)
        print("\nDuplicates:")
        for title, count in sorted(en_counts.items(), key=lambda x: -x[1])[:30]:
            if count >= 2:
                print(f"  {count}x  {title}")
        print(f"\nSingle-word titles: {sum(1 for d in taxonomy if len(d['en'].split()) <= 1)}")
        return

    from openai import OpenAI
    client = OpenAI(base_url=args.api_base, api_key="not-needed")

    print(f"\nRe-prompting {len(to_refine)} clusters ...")
    refined_count = 0
    errors = 0

    def process_item(item):
        cluster_id = str(item["canonical_id"] - 1)
        cluster_data = clusters.get(cluster_id, {})
        return item["canonical_id"], refine_one(client, args.model, item, cluster_data)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_item, item): item for item in to_refine}
        for future in as_completed(futures):
            canonical_id, result = future.result()
            if result:
                for item in taxonomy:
                    if item["canonical_id"] == canonical_id:
                        old = item["en"]
                        item["en"] = result["en"]
                        item["fr"] = result["fr"]
                        refined_count += 1
                        break
            else:
                errors += 1

            if (refined_count + errors) % 50 == 0:
                print(f"  {refined_count + errors}/{len(to_refine)} processed ({errors} errors)", flush=True)

    print(f"\nRefined {refined_count} titles ({errors} errors)")

    # Merge remaining duplicates
    before = len(taxonomy)
    taxonomy = merge_duplicates(taxonomy)
    print(f"Merged duplicates: {before} → {len(taxonomy)} keywords")

    # Save
    output_path = os.path.join(args.taxonomy_dir, "taxonomy_refined.json")
    with open(output_path, "w") as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)
    print(f"Saved: {output_path}")

    # Summary
    en_counts = Counter(d["en"] for d in taxonomy)
    remaining_dupes = sum(1 for v in en_counts.values() if v > 1)
    single_word = sum(1 for d in taxonomy if len(d["en"].split()) <= 1)
    print(f"\nAfter refinement:")
    print(f"  Total keywords: {len(taxonomy)}")
    print(f"  Remaining duplicates: {remaining_dupes}")
    print(f"  Single-word titles: {single_word}")


if __name__ == "__main__":
    main()
