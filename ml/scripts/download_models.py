"""
download_models.py
───────────────────
Phase 0, step 7: download all base model weights from HuggingFace into
ml/models/base/ and verify each loads correctly.

Downloads ~50GB total — run on the CPU server before the A100 rental starts so
Day 1 isn't spent downloading over the GPU instance's network.

Models:
  1. microsoft/harrier-oss-v1-0.6b     (~1.2GB)  — bi-encoder base
  2. FacebookAI/xlm-roberta-large       (~2.2GB)  — reranker base
  3. Qwen/Qwen2.5-7B-Instruct          (~15GB)   — extractor base (LoRA fine-tune)
  4. Qwen/Qwen2.5-72B-Instruct-AWQ     (~41GB)   — teacher model (Day 1-2 only)

Setup:
    pip install -e ".[ml]"
    # If any model requires gated access:
    huggingface-cli login

Usage:
    python -u ml/scripts/download_models.py
    python -u ml/scripts/download_models.py --model harrier         # single model
    python -u ml/scripts/download_models.py --model reranker
    python -u ml/scripts/download_models.py --model extractor
    python -u ml/scripts/download_models.py --model teacher
    python -u ml/scripts/download_models.py --skip-teacher          # skip the 41GB teacher
    python -u ml/scripts/download_models.py --verify-only           # check existing downloads
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

MODELS = {
    "harrier": {
        "repo_id": "microsoft/harrier-oss-v1-0.6b",
        "description": "Bi-encoder base (0.6B, decoder-only, MIT license)",
        "size_approx": "~1.2 GB",
        "verify_lib": "sentence_transformers",
    },
    "reranker": {
        "repo_id": "FacebookAI/xlm-roberta-large",
        "description": "Reranker base (560M, encoder-only, MIT license)",
        "size_approx": "~2.2 GB",
        "verify_lib": "transformers",
    },
    "extractor": {
        "repo_id": "Qwen/Qwen2.5-7B-Instruct",
        "description": "Extractor base for LoRA fine-tuning (7B, Apache 2.0)",
        "size_approx": "~15 GB",
        "verify_lib": "transformers",
    },
    "teacher": {
        "repo_id": "Qwen/Qwen2.5-72B-Instruct-AWQ",
        "description": "Teacher model for Day 1-2 labeling (72B AWQ, Apache 2.0)",
        "size_approx": "~41 GB",
        "verify_lib": "transformers",
    },
}

BASE_DIR = "ml/models/base"


def download_model(name: str, info: dict) -> bool:
    """Download a model from HuggingFace Hub. Returns True on success."""
    from huggingface_hub import snapshot_download

    repo_id = info["repo_id"]
    local_dir = os.path.join(BASE_DIR, name)

    print(f"\n{'='*60}")
    print(f"Downloading: {repo_id}")
    print(f"  Description: {info['description']}")
    print(f"  Size: {info['size_approx']}")
    print(f"  Target: {local_dir}")
    print(f"{'='*60}")

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        print(f"  Download complete: {local_dir}")
        return True
    except Exception as e:
        print(f"  DOWNLOAD FAILED: {e}")
        return False


def verify_model(name: str, info: dict) -> bool:
    """Verify a downloaded model loads correctly."""
    local_dir = os.path.join(BASE_DIR, name)

    if not os.path.exists(local_dir):
        print(f"  VERIFY SKIP: {local_dir} does not exist")
        return False

    print(f"  Verifying {name} ({info['repo_id']}) ...")

    if info["verify_lib"] == "sentence_transformers":
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(local_dir, trust_remote_code=True)
            emb = model.encode(["test sentence"])
            dim = emb.shape[-1]
            print(f"  VERIFY OK: SentenceTransformer loaded, embedding dim={dim}")
            del model
            return True
        except Exception as e:
            print(f"  VERIFY FAILED (SentenceTransformer): {e}")
            # Fall back to transformers
            try:
                from transformers import AutoModel, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
                model = AutoModel.from_pretrained(local_dir, trust_remote_code=True)
                print(f"  VERIFY OK (fallback): AutoModel loaded, vocab_size={tokenizer.vocab_size}")
                del model, tokenizer
                return True
            except Exception as e2:
                print(f"  VERIFY FAILED (AutoModel fallback): {e2}")
                return False

    elif info["verify_lib"] == "transformers":
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
            vocab_size = tokenizer.vocab_size
            print(f"  VERIFY OK: tokenizer loaded, vocab_size={vocab_size}")

            # For the 72B teacher model, don't load the full model into RAM —
            # just verify the tokenizer and config exist
            if name == "teacher":
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(local_dir, trust_remote_code=True)
                print(f"  VERIFY OK: config loaded, model_type={config.model_type}, "
                      f"hidden_size={config.hidden_size}")
            else:
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(local_dir, trust_remote_code=True)
                print(f"  VERIFY OK: config loaded, model_type={config.model_type}, "
                      f"hidden_size={config.hidden_size}")

            del tokenizer
            return True
        except Exception as e:
            print(f"  VERIFY FAILED: {e}")
            return False

    return False


def check_disk_space() -> None:
    """Print available disk space."""
    try:
        stat = os.statvfs(BASE_DIR if os.path.exists(BASE_DIR) else ".")
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        print(f"Available disk space: {free_gb:.1f} GB")
        if free_gb < 60:
            print(f"WARNING: less than 60 GB free — the full download needs ~50 GB")
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Download base model weights from HuggingFace")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default=None,
        help="Download a single model (default: all)",
    )
    parser.add_argument("--skip-teacher", action="store_true", help="Skip the 41GB teacher model download")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing downloads, don't download")
    parser.add_argument("--token", default=None, help="HuggingFace token (default: uses cached login)")
    args = parser.parse_args()

    os.makedirs(BASE_DIR, exist_ok=True)

    if args.token:
        os.environ["HF_TOKEN"] = args.token

    # Determine which models to process
    if args.model:
        model_names = [args.model]
    elif args.skip_teacher:
        model_names = [k for k in MODELS if k != "teacher"]
    else:
        model_names = list(MODELS.keys())

    check_disk_space()

    results: dict[str, str] = {}

    for name in model_names:
        info = MODELS[name]

        if not args.verify_only:
            success = download_model(name, info)
            if not success:
                results[name] = "DOWNLOAD FAILED"
                continue

        verified = verify_model(name, info)
        results[name] = "OK" if verified else "VERIFY FAILED"

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    for name in model_names:
        info = MODELS[name]
        status = results.get(name, "SKIPPED")
        print(f"  {name:>10}: {status:>15}  ({info['repo_id']})")

    all_ok = all(v == "OK" for v in results.values())
    if all_ok:
        print(f"\nGATE PASSED: all {len(results)} models downloaded and verified")
    else:
        failed = [k for k, v in results.items() if v != "OK"]
        print(f"\nGATE FAILED: {failed} — fix before proceeding to the GPU window")


if __name__ == "__main__":
    main()
