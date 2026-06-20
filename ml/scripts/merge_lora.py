"""
merge_lora.py
──────────────
Phase 4: merge LoRA adapter weights into the base Qwen2.5-7B-Instruct model,
producing a standalone model ready for GGUF conversion.

Input:  ml/models/base/extractor/ (base model)
        ml/models/checkpoints/extractor-lora/ (LoRA adapter)
Output: ml/models/exported/extractor-merged/ (full merged model)

Usage:
    python -u ml/scripts/merge_lora.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base-model", default="ml/models/base/extractor")
    parser.add_argument("--lora-path", default="ml/models/checkpoints/extractor-lora")
    parser.add_argument("--output-dir", default="ml/models/exported/extractor-merged")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    os.makedirs(args.output_dir, exist_ok=True)

    # Load base model
    print(f"Loading base model from {args.base_model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Load and merge LoRA
    print(f"Loading LoRA adapter from {args.lora_path} ...")
    model = PeftModel.from_pretrained(model, args.lora_path)
    print("Merging LoRA weights ...")
    model = model.merge_and_unload()

    # Save merged model
    print(f"Saving merged model to {args.output_dir} ...")
    model.save_pretrained(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\nDone. Merged model saved to {args.output_dir}")
    print("Next: convert to GGUF with convert_gguf.sh")


if __name__ == "__main__":
    main()
