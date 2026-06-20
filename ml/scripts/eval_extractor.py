"""
eval_extractor.py
───────────────────
Phase 4: evaluate the fine-tuned extractor on the validation set.

Runs inference on val examples, parses JSON output, and computes field-level
accuracy against the teacher's extractions (treated as ground truth).

Metrics: per-field accuracy, JSON validity rate, overall accuracy.

Usage:
    python -u ml/scripts/eval_extractor.py
    python -u ml/scripts/eval_extractor.py --model ml/models/exported/extractor-merged
    python -u ml/scripts/eval_extractor.py --limit 100
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

HEARTBEAT_EVERY = 50

EVAL_FIELDS = [
    "normalized_title",
    "seniority",
    "occupation_category",
    "employment_type",
    "contract_type",
    "work_mode",
    "location_country",
    "language",
    "salary_present",
]


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate extractor on validation set")
    parser.add_argument("--model", default="ml/models/exported/extractor-merged")
    parser.add_argument("--val-data", default="ml/data/splits/extractor_val.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for generation")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("  Model loaded")

    print(f"Loading validation data from {args.val_data} ...")
    val_records = list(iter_jsonl(args.val_data))
    if args.limit:
        val_records = val_records[:args.limit]
    print(f"  {len(val_records):,} validation examples")

    total = 0
    valid_json = 0
    field_correct = {f: 0 for f in EVAL_FIELDS}
    field_total = {f: 0 for f in EVAL_FIELDS}

    for record in val_records:
        messages = record.get("messages", [])
        if len(messages) < 3:
            continue

        # Ground truth from teacher
        try:
            ground_truth = json.loads(messages[2]["content"])
        except (json.JSONDecodeError, KeyError):
            continue

        # Build prompt (system + user, no assistant)
        prompt_messages = messages[:2]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Generate
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated tokens
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True).strip()

        # Parse JSON
        if response.startswith("```"):
            response = response.strip("`").replace("json\n", "", 1).strip()

        total += 1
        try:
            predicted = json.loads(response)
            valid_json += 1
        except json.JSONDecodeError:
            if total % HEARTBEAT_EVERY == 0:
                print(f"  ... {total:,} processed, {valid_json:,} valid JSON")
            continue

        # Compare fields
        for field in EVAL_FIELDS:
            gt_val = ground_truth.get(field)
            pred_val = predicted.get(field)
            field_total[field] += 1
            if gt_val == pred_val:
                field_correct[field] += 1

        if total % HEARTBEAT_EVERY == 0:
            print(f"  ... {total:,} processed, {valid_json:,} valid JSON")

    # Results
    print(f"\n{'='*60}")
    print(f"Results ({total:,} examples)")
    print(f"{'='*60}")
    print(f"  JSON validity: {valid_json}/{total} ({valid_json/total:.1%})" if total else "")

    print(f"\n  Field-level accuracy:")
    field_accuracies = []
    for field in EVAL_FIELDS:
        if field_total[field] > 0:
            acc = field_correct[field] / field_total[field]
            field_accuracies.append(acc)
            print(f"    {field:>25}: {acc:.4f} ({field_correct[field]}/{field_total[field]})")

    if field_accuracies:
        overall = sum(field_accuracies) / len(field_accuracies)
        print(f"\n  Overall field accuracy: {overall:.4f}")

        if overall >= 0.90:
            print(f"\n  GATE PASSED: overall accuracy = {overall:.4f} >= 0.90")
        else:
            print(f"\n  GATE FAILED: overall accuracy = {overall:.4f} < 0.90")

    print("\nDone.")


if __name__ == "__main__":
    main()
