"""
export_onnx.py
────────────────
Phase 6: export bi-encoder and reranker to ONNX format for fast CPU inference
via onnxruntime in production.

Usage:
    python -u ml/scripts/export_onnx.py --model biencoder
    python -u ml/scripts/export_onnx.py --model reranker
    python -u ml/scripts/export_onnx.py --model both
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def export_biencoder(input_dir: str, output_dir: str) -> None:
    """Export bi-encoder to ONNX via optimum."""
    print(f"Exporting bi-encoder from {input_dir} ...")
    os.makedirs(output_dir, exist_ok=True)

    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(input_dir, trust_remote_code=True)
        model = ORTModelForFeatureExtraction.from_pretrained(input_dir, export=True, trust_remote_code=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"  ONNX model saved to {output_dir}")
    except ImportError:
        print("  optimum not installed, falling back to torch.onnx.export ...")
        _export_biencoder_torch(input_dir, output_dir)


def _export_biencoder_torch(input_dir: str, output_dir: str) -> None:
    """Fallback ONNX export using torch.onnx."""
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(input_dir, trust_remote_code=True)
    transformer = model[0].auto_model
    tokenizer = model.tokenizer

    dummy = tokenizer("test sentence", return_tensors="pt", padding=True, truncation=True)

    torch.onnx.export(
        transformer,
        (dummy["input_ids"], dummy["attention_mask"]),
        os.path.join(output_dir, "model.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
        },
        opset_version=17,
    )
    tokenizer.save_pretrained(output_dir)
    print(f"  ONNX model saved to {output_dir}")


def export_reranker(input_dir: str, output_dir: str) -> None:
    """Export reranker CrossEncoder to ONNX."""
    print(f"Exporting reranker from {input_dir} ...")
    os.makedirs(output_dir, exist_ok=True)

    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(input_dir, trust_remote_code=True)
        model = ORTModelForSequenceClassification.from_pretrained(input_dir, export=True, trust_remote_code=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"  ONNX model saved to {output_dir}")
    except ImportError:
        print("  optimum not installed, falling back to torch.onnx.export ...")
        _export_reranker_torch(input_dir, output_dir)


def _export_reranker_torch(input_dir: str, output_dir: str) -> None:
    """Fallback ONNX export using torch.onnx."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(input_dir)
    model = AutoModelForSequenceClassification.from_pretrained(input_dir)
    model.eval()

    dummy = tokenizer("query text", "passage text", return_tensors="pt", padding=True, truncation=True)

    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        os.path.join(output_dir, "model.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )
    tokenizer.save_pretrained(output_dir)
    print(f"  ONNX model saved to {output_dir}")


def verify_biencoder(onnx_dir: str) -> None:
    """Quick verification that the ONNX bi-encoder produces embeddings."""
    print(f"  Verifying bi-encoder ONNX ...")
    import onnxruntime as ort
    from transformers import AutoTokenizer
    import numpy as np

    tokenizer = AutoTokenizer.from_pretrained(onnx_dir, trust_remote_code=True)
    session = ort.InferenceSession(os.path.join(onnx_dir, "model.onnx"))

    # Get required input names from the ONNX model
    required_inputs = {inp.name for inp in session.get_inputs()}
    print(f"  ONNX inputs: {required_inputs}")

    inputs = tokenizer("Senior Software Engineer with Python experience", return_tensors="np", padding=True, truncation=True)

    # Add position_ids if required
    if "position_ids" in required_inputs and "position_ids" not in inputs:
        seq_len = inputs["input_ids"].shape[1]
        inputs["position_ids"] = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

    feed = {k: v for k, v in inputs.items() if k in required_inputs}
    outputs = session.run(None, feed)
    print(f"  Output shape: {outputs[0].shape}")
    print(f"  VERIFY OK")


def verify_reranker(onnx_dir: str) -> None:
    """Quick verification that the ONNX reranker produces scores."""
    print(f"  Verifying reranker ONNX ...")
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(onnx_dir, trust_remote_code=True)
    session = ort.InferenceSession(os.path.join(onnx_dir, "model.onnx"))

    inputs = tokenizer("resume text", "job text", return_tensors="np", padding=True, truncation=True)
    outputs = session.run(None, {k: v for k, v in inputs.items() if k in ["input_ids", "attention_mask"]})
    print(f"  Score: {outputs[0][0]}")
    print(f"  VERIFY OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export models to ONNX for production CPU inference")
    parser.add_argument("--model", choices=["biencoder", "reranker", "both"], default="both")
    parser.add_argument("--biencoder-input", default="ml/models/exported/biencoder")
    parser.add_argument("--reranker-input", default="ml/models/exported/reranker-final")
    parser.add_argument("--output-dir", default="ml/models/exported")
    parser.add_argument("--verify", action="store_true", default=True, help="Verify ONNX output")
    args = parser.parse_args()

    if args.model in ("biencoder", "both"):
        onnx_dir = os.path.join(args.output_dir, "biencoder-onnx")
        export_biencoder(args.biencoder_input, onnx_dir)
        if args.verify:
            verify_biencoder(onnx_dir)

    if args.model in ("reranker", "both"):
        onnx_dir = os.path.join(args.output_dir, "reranker-onnx")
        export_reranker(args.reranker_input, onnx_dir)
        if args.verify:
            verify_reranker(onnx_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
