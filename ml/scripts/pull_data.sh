#!/usr/bin/env bash
set -euo pipefail

# pull_data.sh
# ─────────────
# Pull training data from the CPU server to the A100 instance at the start
# of the GPU window. Run once on Day 1 before training begins.
#
# Does NOT pull base model weights — those are downloaded directly from
# HuggingFace on the A100 (faster than scp between servers).
#
# Usage (run ON the A100 instance):
#   ./ml/scripts/pull_data.sh
#   ./ml/scripts/pull_data.sh --dry-run
#
# Configuration:
#   PULL_REMOTE_HOST   — CPU server SSH host (e.g. webadmin@cpu-server.com)
#   PULL_REMOTE_BASE   — remote project root (e.g. /home/webadmin/Jobl/api.jobl.ai)

REMOTE_HOST="${PULL_REMOTE_HOST:-}"
REMOTE_BASE="${PULL_REMOTE_BASE:-}"
LOCAL_BASE="$(cd "$(dirname "$0")/../.." && pwd)"
DRY_RUN=""

if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN="yes"
fi

if [ -z "$REMOTE_HOST" ] || [ -z "$REMOTE_BASE" ]; then
    echo "ERROR: PULL_REMOTE_HOST and PULL_REMOTE_BASE must be set."
    echo ""
    echo "Example:"
    echo "  export PULL_REMOTE_HOST=webadmin@cpu-server.com"
    echo "  export PULL_REMOTE_BASE=/home/webadmin/Jobl/api.jobl.ai"
    exit 1
fi

pull_dir() {
    local remote_subdir="$1"
    local local_subdir="$2"
    local src="$REMOTE_HOST:$REMOTE_BASE/$remote_subdir/"
    local dest="$LOCAL_BASE/$local_subdir/"

    echo "  PULL: $src -> $dest"

    if [ -n "$DRY_RUN" ]; then
        echo "  (dry-run, skipping)"
        return
    fi

    mkdir -p "$dest"
    scp -r "$src." "$dest"
    echo "  DONE"
}

echo "============================================================"
echo "Pulling training data to A100"
echo "  From:  $REMOTE_HOST:$REMOTE_BASE"
echo "  To:    $LOCAL_BASE"
if [ -n "$DRY_RUN" ]; then
    echo "  Mode:  DRY RUN"
fi
echo "============================================================"

echo ""
echo "--- Data: interim (deduped jobs + PII-scrubbed resumes with lang + resume_type) ---"
pull_dir "ml/data/interim" "ml/data/interim"

echo ""
echo "--- Data: splits (train/val pools) ---"
pull_dir "ml/data/splits" "ml/data/splits"

echo ""
echo "--- Scripts ---"
pull_dir "ml/scripts" "ml/scripts"

echo ""
echo "--- Configs ---"
mkdir -p "$LOCAL_BASE/ml"
if [ -z "$DRY_RUN" ]; then
    scp "$REMOTE_HOST:$REMOTE_BASE/ml/requirements-training.txt" "$LOCAL_BASE/ml/requirements-training.txt"
    echo "  DONE: requirements-training.txt"
else
    echo "  PULL: requirements-training.txt (dry-run, skipping)"
fi

echo ""
echo "============================================================"
echo "Pull complete. Next steps:"
echo "  1. pip install -r ml/requirements-training.txt"
echo "  2. python -u ml/scripts/download_models.py    # download models from HuggingFace"
echo "============================================================"
