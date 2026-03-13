#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="${ABLATIONS_DIR:-results/ablations}"
OUT_DIR="${1:-ablations}"

if [[ ! -d "$SRC_DIR" ]]; then
  echo "missing source directory: $SRC_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
cp -a "$SRC_DIR"/. "$OUT_DIR"/
