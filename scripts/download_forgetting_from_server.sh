#!/usr/bin/env bash
set -euo pipefail

REMOTE_USER="${REMOTE_USER:-agromyko}"
REMOTE_HOST="${REMOTE_HOST:-login.farmshare.stanford.edu}"
REMOTE_RESULTS_DIR="${REMOTE_RESULTS_DIR:-/home/users/agromyko/websleuths-229/results/retention}"
LOCAL_DIR="${1:-$HOME/Documents/forgetting_results}"
MODEL_FILTER="${MODEL_FILTER:-}"

mkdir -p "$LOCAL_DIR"

REMOTE="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_RESULTS_DIR}/"

if [[ -n "$MODEL_FILTER" ]]; then
  PREFIX="retention_${MODEL_FILTER}"
  rsync -avz --prune-empty-dirs \
    --include="*/" \
    --include="${PREFIX}.json" \
    --include="${PREFIX}_checkpoint.json" \
    --include="${PREFIX}_state/***" \
    --exclude="*" \
    "$REMOTE" "$LOCAL_DIR/"
else
  rsync -avz --prune-empty-dirs \
    --include="*/" \
    --include="retention_*.json" \
    --include="retention_*_checkpoint.json" \
    --include="retention_*_state/***" \
    --exclude="*" \
    "$REMOTE" "$LOCAL_DIR/"
fi

echo "Downloaded forgetting artifacts to $LOCAL_DIR"
