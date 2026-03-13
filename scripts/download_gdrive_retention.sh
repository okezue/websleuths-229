#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="${RETENTION_DIR:-results/retention}"
OUT_DIR="${1:-retention}"

if [[ ! -d "$SRC_DIR" ]]; then
  echo "missing source directory: $SRC_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
cp -a "$SRC_DIR"/. "$OUT_DIR"/
