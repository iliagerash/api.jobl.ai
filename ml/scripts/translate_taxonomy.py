"""
translate_taxonomy.py
──────────────────────
Add a translation of the reviewed taxonomy into a new language, using
OpenAI (gpt-5-nano by default — same model review_taxonomy.py uses).

Submits keywords in batches, asks the model for a natural, standard
job-title translation of each "en" entry, and writes the result back
as a new field (e.g. "gr") on every entry in taxonomy_reviewed.json.

Usage:
    export OPENAI_API_KEY=...
    python -u ml/scripts/translate_taxonomy.py --lang gr --lang-name Greek
    python -u ml/scripts/translate_taxonomy.py --lang gr --lang-name Greek --dry-run
    python -u ml/scripts/translate_taxonomy.py --lang gr --lang-name Greek --batch-size 50

Input:
    ml/data/taxonomy/taxonomy_reviewed.json  (must already have "en"/"fr"/etc.)

Output:
    Same file, in place — each entry gains the new language field.
    A timestamped .bak copy of the input is written first.
"""

import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

TAXONOMY_DIR = "ml/data/taxonomy"

SYSTEM_PROMPT_TEMPLATE = """You are a job title taxonomy expert translating canonical job titles into {lang_name}.

You will receive a JSON array of entries, each with:
- "id": canonical ID
- "en": English title (source of truth)
- "category_id": job category

For each entry, produce a natural, standard {lang_name} job title that a native {lang_name}-speaking
recruiter or job seeker would actually use and search for.

Rules:
- Translate the job title concept, not word-for-word — use the term a native speaker would actually use
- Keep it concise: 1-4 words typically, matching the English title's specificity
- Do NOT add seniority levels (Junior, Senior, Lead, etc.)
- Do NOT transliterate — use the real {lang_name} job-title vocabulary
- If a widely-used loanword/anglicism is the standard term in {lang_name} for this role, use it

Return a JSON object with a "results" key containing an array of ALL entries, each with:
- "id": same canonical ID
- "{lang_code}": the {lang_name} translation
Example: {{"results": [{{"id": 1, "{lang_code}": "..."}}, ...]}}"""


def load_data(taxonomy_dir: str) -> list:
    with open(os.path.join(taxonomy_dir, "taxonomy_reviewed.json"), encoding="utf-8") as f:
        return json.load(f)


def build_batch(taxonomy: list, start: int, batch_size: int) -> list:
    batch = []
    for item in taxonomy[start:start + batch_size]:
        batch.append({
            "id": item["canonical_id"],
            "en": item["en"],
            "category_id": item["category_id"],
        })
    return batch


def translate_batch(client, model: str, system_prompt: str, batch: list) -> list | None:
    user_msg = json.dumps(batch, ensure_ascii=False, indent=2)

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or ""
            if not content:
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
            if attempt == 2:
                return None
            time.sleep(2)
    return None


def main():
    parser = argparse.ArgumentParser(description="Translate reviewed taxonomy into a new language")
    parser.add_argument("--taxonomy-dir", default=TAXONOMY_DIR)
    parser.add_argument("--lang", required=True, help="Target language code, e.g. 'gr' (matches app/services/language.py codes)")
    parser.add_argument("--lang-name", required=True, help="Target language name for the prompt, e.g. 'Greek'")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--dry-run", action="store_true", help="Preview the first batch's prompt, make no API calls")
    args = parser.parse_args()

    taxonomy = load_data(args.taxonomy_dir)
    print(f"Loaded {len(taxonomy)} keywords")

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(lang_name=args.lang_name, lang_code=args.lang)

    if args.dry_run:
        batch = build_batch(taxonomy, 0, 3)
        print("System prompt:\n" + system_prompt)
        print("\nSample batch:")
        print(json.dumps(batch, indent=2, ensure_ascii=False))
        return

    from openai import OpenAI
    client = OpenAI()

    total_translated = 0
    total_errors = 0
    total_batches = (len(taxonomy) + args.batch_size - 1) // args.batch_size
    results_map = {}

    for i in range(0, len(taxonomy), args.batch_size):
        batch_num = i // args.batch_size + 1
        batch = build_batch(taxonomy, i, args.batch_size)

        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} items) ...", end="", flush=True)
        result = translate_batch(client, args.model, system_prompt, batch)

        if result:
            for r in result:
                if args.lang in r:
                    results_map[r["id"]] = r[args.lang]
            print(f" {len(result)} translated", flush=True)
            total_translated += len(result)
        else:
            print(" FAILED — skipping batch", flush=True)
            total_errors += len(batch)

    missing = [item["canonical_id"] for item in taxonomy if item["canonical_id"] not in results_map]
    if missing:
        print(f"\n{len(missing)} entries missing a translation (API errors): {missing[:20]}{'...' if len(missing) > 20 else ''}")

    # Apply translations
    for item in taxonomy:
        translation = results_map.get(item["canonical_id"])
        if translation:
            item[args.lang] = translation

    # Backup before overwriting
    input_path = os.path.join(args.taxonomy_dir, "taxonomy_reviewed.json")
    backup_path = input_path + f".bak.{int(time.time())}"
    shutil.copyfile(input_path, backup_path)
    print(f"\nBackup saved: {backup_path}")

    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)

    print(f"Done. {total_translated} translated ({total_errors} skipped on error).")
    print(f"Saved: {input_path}")


if __name__ == "__main__":
    main()
