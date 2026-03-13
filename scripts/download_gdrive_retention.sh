#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-retention}"
mkdir -p "$OUT_DIR"
rclone copy gdrive:retention "$OUT_DIR" --update
