"""
dedup_taxonomy.py
──────────────────
Merge duplicate keywords in the taxonomy. Duplicates are defined as
entries with the same (en, category_id). Keeps the one with the
largest cluster_size.

Usage:
    python -u ml/scripts/dedup_taxonomy.py
    python -u ml/scripts/dedup_taxonomy.py --input ml/data/taxonomy/taxonomy_final.json
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    parser = argparse.ArgumentParser(description="Deduplicate taxonomy keywords")
    parser.add_argument("--input", default="ml/data/taxonomy/taxonomy_final.json")
    parser.add_argument("--output", default=None, help="Output path (default: taxonomy_deduped.json in same dir)")
    args = parser.parse_args()

    with open(args.input) as f:
        taxonomy = json.load(f)

    print(f"Input: {len(taxonomy)} keywords")

    # Analyze before
    en_counts = Counter(d["en"] for d in taxonomy)
    en_cat_counts = Counter((d["en"], d["category_id"]) for d in taxonomy)
    dupes_en = sum(1 for v in en_counts.values() if v > 1)
    dupes_en_cat = sum(1 for v in en_cat_counts.values() if v > 1)

    print(f"Duplicate English titles: {dupes_en}")
    print(f"Duplicate (en + category_id): {dupes_en_cat}")

    # Merge: same (en, category_id) → keep largest cluster
    seen = {}
    for item in taxonomy:
        key = (item["en"], item["category_id"])
        if key not in seen or item["cluster_size"] > seen[key]["cluster_size"]:
            seen[key] = item

    deduped = sorted(seen.values(), key=lambda x: -x["cluster_size"])

    # Re-assign canonical_ids
    for i, item in enumerate(deduped, 1):
        item["canonical_id"] = i

    output_path = args.output or os.path.join(os.path.dirname(args.input), "taxonomy_deduped.json")
    with open(output_path, "w") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)

    # Summary
    new_en_counts = Counter(d["en"] for d in deduped)
    remaining_dupes = sum(1 for v in new_en_counts.values() if v > 1)
    single_word = sum(1 for d in deduped if len(d["en"].split()) <= 1)

    print(f"\nOutput: {len(deduped)} keywords")
    print(f"Remaining duplicate English titles: {remaining_dupes} (different categories — expected)")
    print(f"Single-word titles: {single_word}")
    print(f"Saved: {output_path}")

    # Show remaining duplicates (same en, different category)
    if remaining_dupes:
        print(f"\nSame title, different categories:")
        for title, count in sorted(new_en_counts.items(), key=lambda x: -x[1])[:15]:
            if count > 1:
                cats = [d["category_id"] for d in deduped if d["en"] == title]
                print(f"  {count}x  {title}  (categories: {cats})")


if __name__ == "__main__":
    main()
