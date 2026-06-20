#!/usr/bin/env bash
set -euo pipefail

# convert_gguf.sh
# ─────────────────
# Phase 6: convert the merged extractor model to GGUF Q4_K_M format
# for production CPU inference via llama.cpp / llama-cpp-python.
#
# Prerequisites:
#   pip install llama-cpp-python
#   # OR clone llama.cpp and build the converter:
#   git clone https://github.com/ggerganov/llama.cpp /tmp/llama.cpp
#   pip install -r /tmp/llama.cpp/requirements.txt
#
# Usage:
#   ./ml/scripts/convert_gguf.sh
#   ./ml/scripts/convert_gguf.sh --quantize Q5_K_M    # higher quality, larger file

INPUT_DIR="${1:-ml/models/exported/extractor-merged}"
OUTPUT_DIR="${2:-ml/models/exported/extractor-gguf}"
QUANT_TYPE="${3:-Q4_K_M}"

LLAMA_CPP_DIR="/tmp/llama.cpp"

echo "============================================================"
echo "Converting extractor to GGUF"
echo "  Input:  $INPUT_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Quant:  $QUANT_TYPE"
echo "============================================================"

mkdir -p "$OUTPUT_DIR"

# Step 1: Check if llama.cpp converter is available
if [ ! -f "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" ]; then
    echo ""
    echo "Cloning llama.cpp for GGUF converter ..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMA_CPP_DIR"
    pip install -r "$LLAMA_CPP_DIR/requirements.txt"
fi

# Step 2: Convert HF model to GGUF (float16)
FP16_PATH="$OUTPUT_DIR/model-f16.gguf"
echo ""
echo "Step 1/2: Converting to GGUF (float16) ..."
python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
    "$INPUT_DIR" \
    --outfile "$FP16_PATH" \
    --outtype f16

echo "  Float16 GGUF: $FP16_PATH ($(du -sh "$FP16_PATH" | cut -f1))"

# Step 3: Quantize to Q4_K_M
QUANT_PATH="$OUTPUT_DIR/model-${QUANT_TYPE}.gguf"
echo ""
echo "Step 2/2: Quantizing to $QUANT_TYPE ..."

# Build llama-quantize if not present
if [ ! -f "$LLAMA_CPP_DIR/build/bin/llama-quantize" ]; then
    echo "  Building llama.cpp quantizer ..."
    cd "$LLAMA_CPP_DIR"
    cmake -B build
    cmake --build build --target llama-quantize -j$(nproc)
    cd -
fi

"$LLAMA_CPP_DIR/build/bin/llama-quantize" "$FP16_PATH" "$QUANT_PATH" "$QUANT_TYPE"

echo ""
echo "  Quantized GGUF: $QUANT_PATH ($(du -sh "$QUANT_PATH" | cut -f1))"

# Clean up float16 intermediate
rm -f "$FP16_PATH"
echo "  Cleaned up float16 intermediate"

echo ""
echo "============================================================"
echo "Done. GGUF model: $QUANT_PATH"
echo "  Test with: llama-cli -m $QUANT_PATH -p 'Hello'"
echo "============================================================"
