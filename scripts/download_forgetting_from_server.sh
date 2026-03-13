#!/usr/bin/env bash
set -euo pipefail

REMOTE_USER="${REMOTE_USER:-agromyko}"
REMOTE_HOST="${REMOTE_HOST:-login.farmshare.stanford.edu}"
REMOTE_RESULTS_DIR="${REMOTE_RESULTS_DIR:-/home/users/agromyko/websleuths-229/results/retention}"
LOCAL_DIR="${1:-$HOME/Documents/forgetting_results}"
MODEL_FILTER="${MODEL_FILTER:-}"
INCLUDE_CHECKPOINTS="${INCLUDE_CHECKPOINTS:-0}"

mkdir -p "$LOCAL_DIR"

REMOTE="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_RESULTS_DIR}/"

rsync_args=(
  -avz
  --prune-empty-dirs
  --include="*/"
)

if [[ -n "$MODEL_FILTER" ]]; then
  PREFIX="retention_${MODEL_FILTER}"
  if [[ "$INCLUDE_CHECKPOINTS" == "1" ]]; then
    rsync_args+=(
      "--include=${PREFIX}_checkpoint.json"
      "--include=${PREFIX}_state/***"
      "--include=${PREFIX}.json"
    )
  else
    rsync_args+=(
      "--exclude=${PREFIX}_checkpoint.json"
      "--exclude=${PREFIX}_state/***"
      "--include=${PREFIX}.json"
    )
  fi
else
  if [[ "$INCLUDE_CHECKPOINTS" == "1" ]]; then
    rsync_args+=(
      "--include=retention_*_checkpoint.json"
      "--include=retention_*_state/***"
      "--include=retention_*.json"
    )
  else
    rsync_args+=(
      "--exclude=retention_*_checkpoint.json"
      "--exclude=retention_*_state/***"
      "--include=retention_*.json"
    )
  fi
fi

rsync_args+=(
  --exclude="*"
  "$REMOTE"
  "$LOCAL_DIR/"
)

rsync "${rsync_args[@]}"

echo "Downloaded forgetting artifacts to $LOCAL_DIR"
if [[ "$INCLUDE_CHECKPOINTS" == "1" ]]; then
  echo "Included checkpoint metadata and state directories"
else
  echo "Skipped checkpoint metadata and state directories (set INCLUDE_CHECKPOINTS=1 to include them)"
fi
