#!/usr/bin/env bash
set -euo pipefail

# pull_models.sh
# ────────────────
# Pull base model weights from the CPU server to the A100 instance via scp.
# Use this instead of downloading from HuggingFace when the GPU host has
# slow internet but fast connectivity to the CPU server.
#
# Prerequisites: models must already be downloaded on the CPU server
# (via python -u ml/scripts/download_models.py)
#
# Usage (run ON the A100 instance):
#   ./ml/scripts/pull_models.sh
#   ./ml/scripts/pull_models.sh --dry-run
#   ./ml/scripts/pull_models.sh harrier reranker    # specific models only
#
# Configuration:
#   PULL_REMOTE_HOST   — CPU server SSH host
#   PULL_REMOTE_BASE   — remote project root

REMOTE_HOST="${PULL_REMOTE_HOST:-}"
REMOTE_BASE="${PULL_REMOTE_BASE:-}"
LOCAL_BASE="$(cd "$(dirname "$0")/../.." && pwd)"
DRY_RUN=""

MODELS="harrier reranker extractor teacher"

# Parse args
SELECTED=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN="yes" ;;
        harrier|reranker|extractor|teacher) SELECTED="$SELECTED $arg" ;;
        *) echo "Unknown argument: $arg"; echo "Usage: $0 [--dry-run] [harrier] [reranker] [extractor] [teacher]"; exit 1 ;;
    esac
done

if [ -n "$SELECTED" ]; then
    MODELS="$SELECTED"
fi

if [ -z "$REMOTE_HOST" ] || [ -z "$REMOTE_BASE" ]; then
    echo "ERROR: PULL_REMOTE_HOST and PULL_REMOTE_BASE must be set."
    echo ""
    echo "Example:"
    echo "  export PULL_REMOTE_HOST=webadmin@cpu-server.com"
    echo "  export PULL_REMOTE_BASE=/home/webadmin/Jobl/api.jobl.ai"
    exit 1
fi

echo "============================================================"
echo "Pulling base models from CPU server"
echo "  From:  $REMOTE_HOST:$REMOTE_BASE/ml/models/base/"
echo "  To:    $LOCAL_BASE/ml/models/base/"
if [ -n "$DRY_RUN" ]; then
    echo "  Mode:  DRY RUN"
fi
echo "============================================================"

for model in $MODELS; do
    src="$REMOTE_HOST:$REMOTE_BASE/ml/models/base/$model/"
    dest="$LOCAL_BASE/ml/models/base/$model/"

    echo ""
    echo "--- $model ---"

    # Check if remote directory exists
    if ! ssh "$REMOTE_HOST" "test -d '$REMOTE_BASE/ml/models/base/$model'" 2>/dev/null; then
        echo "  SKIP: remote directory does not exist"
        echo "  Run on CPU server first: python -u ml/scripts/download_models.py --model $model"
        continue
    fi

    # Get remote size
    remote_size=$(ssh "$REMOTE_HOST" "du -sh '$REMOTE_BASE/ml/models/base/$model' 2>/dev/null | cut -f1" || echo "unknown")
    echo "  PULL: $src ($remote_size) -> $dest"

    if [ -n "$DRY_RUN" ]; then
        echo "  (dry-run, skipping)"
        continue
    fi

    mkdir -p "$dest"
    scp -r "$src." "$dest"
    echo "  DONE: $model"
done

echo ""
echo "============================================================"
echo "Pull complete. Verify with:"
echo "  python -u ml/scripts/download_models.py --verify-only"
echo "============================================================"
