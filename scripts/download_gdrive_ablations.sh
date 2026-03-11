#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-ablations}"
mkdir -p "$OUT_DIR"
rclone copy gdrive:ablations "$OUT_DIR" --update
