#!/usr/bin/env bash
set -euo pipefail

# push_artifacts.sh
# ──────────────────
# Phase 0, step 9: push ML pipeline artifacts to remote storage via scp.
# Designed to run after each training phase on the A100 so artifacts are
# backed up incrementally — instance death mid-window should lose at most
# one day of work.
#
# Usage:
#   ./ml/scripts/push_artifacts.sh <phase> [--dry-run]
#
# Examples:
#   ./ml/scripts/push_artifacts.sh teacher_labels
#   ./ml/scripts/push_artifacts.sh biencoder
#   ./ml/scripts/push_artifacts.sh reranker
#   ./ml/scripts/push_artifacts.sh extractor
#   ./ml/scripts/push_artifacts.sh corpus_outputs
#   ./ml/scripts/push_artifacts.sh final
#   ./ml/scripts/push_artifacts.sh all
#   ./ml/scripts/push_artifacts.sh teacher_labels --dry-run
#
# Configuration:
#   Set these environment variables (or edit the defaults below):
#     PUSH_REMOTE_HOST    — SSH host (e.g. webadmin@your-server.com)
#     PUSH_REMOTE_BASE    — remote base directory (e.g. /home/webadmin/Jobl/ml-artifacts)

# ── Configuration ──────────────────────────────────────────────────────────

REMOTE_HOST="${PUSH_REMOTE_HOST:-}"
REMOTE_BASE="${PUSH_REMOTE_BASE:-}"
LOCAL_BASE="$(cd "$(dirname "$0")/../.." && pwd)"
DRY_RUN=""

# ── Argument parsing ───────────────────────────────────────────────────────

usage() {
    echo "Usage: $0 <phase> [--dry-run]"
    echo ""
    echo "Phases:"
    echo "  teacher_labels   Push ml/data/teacher_labels/"
    echo "  biencoder        Push ml/models/exported/biencoder/"
    echo "  reranker         Push ml/models/exported/reranker/"
    echo "  extractor        Push ml/models/exported/extractor/"
    echo "  corpus_outputs   Push ml/data/splits/ and ml/models/exported/"
    echo "  final            Push everything (models, data, configs)"
    echo "  all              Alias for final"
    echo ""
    echo "Environment variables:"
    echo "  PUSH_REMOTE_HOST   SSH host (required)"
    echo "  PUSH_REMOTE_BASE   Remote base directory (required)"
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

PHASE="$1"
shift

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN="--dry-run" ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
    shift
done

if [ -z "$REMOTE_HOST" ] || [ -z "$REMOTE_BASE" ]; then
    echo "ERROR: PUSH_REMOTE_HOST and PUSH_REMOTE_BASE must be set."
    echo ""
    echo "Example:"
    echo "  export PUSH_REMOTE_HOST=webadmin@storage.example.com"
    echo "  export PUSH_REMOTE_BASE=/home/webadmin/Jobl/ml-artifacts"
    exit 1
fi

# ── Helper ─────────────────────────────────────────────────────────────────

push_dir() {
    local src="$1"
    local dest_subdir="$2"

    if [ ! -d "$src" ]; then
        echo "  SKIP: $src does not exist"
        return
    fi

    local file_count
    file_count=$(find "$src" -type f | wc -l | tr -d ' ')
    local size
    size=$(du -sh "$src" 2>/dev/null | cut -f1)

    echo "  PUSH: $src ($file_count files, $size) -> $REMOTE_HOST:$REMOTE_BASE/$dest_subdir/"

    if [ -n "$DRY_RUN" ]; then
        echo "  (dry-run, skipping)"
        return
    fi

    # Create remote directory
    ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_BASE/$dest_subdir'"

    # Use scp -r for the transfer
    scp -r "$src/." "$REMOTE_HOST:$REMOTE_BASE/$dest_subdir/"

    echo "  DONE: $dest_subdir"
}

push_file() {
    local src="$1"
    local dest_subdir="$2"

    if [ ! -f "$src" ]; then
        echo "  SKIP: $src does not exist"
        return
    fi

    local size
    size=$(du -sh "$src" 2>/dev/null | cut -f1)
    local filename
    filename=$(basename "$src")

    echo "  PUSH: $src ($size) -> $REMOTE_HOST:$REMOTE_BASE/$dest_subdir/$filename"

    if [ -n "$DRY_RUN" ]; then
        echo "  (dry-run, skipping)"
        return
    fi

    ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_BASE/$dest_subdir'"
    scp "$src" "$REMOTE_HOST:$REMOTE_BASE/$dest_subdir/$filename"

    echo "  DONE: $dest_subdir/$filename"
}

# ── Phase dispatch ─────────────────────────────────────────────────────────

echo "============================================================"
echo "Pushing artifacts: phase=$PHASE"
echo "  Local:  $LOCAL_BASE"
echo "  Remote: $REMOTE_HOST:$REMOTE_BASE"
if [ -n "$DRY_RUN" ]; then
    echo "  Mode:   DRY RUN"
fi
echo "============================================================"

case "$PHASE" in
    teacher_labels)
        push_dir "$LOCAL_BASE/ml/data/teacher_labels" "data/teacher_labels"
        ;;
    biencoder)
        push_dir "$LOCAL_BASE/ml/models/exported/biencoder" "models/biencoder"
        ;;
    reranker)
        push_dir "$LOCAL_BASE/ml/models/exported/reranker" "models/reranker"
        ;;
    extractor)
        push_dir "$LOCAL_BASE/ml/models/exported/extractor-merged" "models/extractor-merged"
        push_dir "$LOCAL_BASE/ml/models/checkpoints/extractor-lora" "models/extractor-lora"
        ;;
    corpus_outputs)
        push_dir "$LOCAL_BASE/ml/data/splits" "data/splits"
        push_dir "$LOCAL_BASE/ml/models/exported" "models/exported"
        ;;
    final|all)
        echo ""
        echo "--- Models ---"
        push_dir "$LOCAL_BASE/ml/models/exported" "models/exported"
        echo ""
        echo "--- Data: teacher labels ---"
        push_dir "$LOCAL_BASE/ml/data/teacher_labels" "data/teacher_labels"
        echo ""
        echo "--- Data: splits ---"
        push_dir "$LOCAL_BASE/ml/data/splits" "data/splits"
        echo ""
        echo "--- Data: interim ---"
        push_dir "$LOCAL_BASE/ml/data/interim" "data/interim"
        echo ""
        echo "--- Configs & eval ---"
        push_file "$LOCAL_BASE/ml/requirements-training.txt" "configs"
        push_dir "$LOCAL_BASE/ml/scripts" "scripts"
        ;;
    *)
        echo "ERROR: unknown phase '$PHASE'"
        usage
        ;;
esac

echo ""
echo "============================================================"
echo "Push complete: phase=$PHASE"
echo "============================================================"
