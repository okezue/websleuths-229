#!/usr/bin/env python3
from __future__ import annotations

import argparse

from wm.config import load_config
from wm.experiments.baseline_matrix import BaselineMatrixRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched continual-learning baseline matrix")
    parser.add_argument("--config", default="configs/quick.yaml")
    parser.add_argument("--manifest")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifest = args.manifest or cfg.stream.manifest
    if not manifest:
        raise SystemExit("--manifest or stream.manifest is required")
    report = BaselineMatrixRunner(cfg, args.methods).run(manifest, args.output)
    for method, values in report["summary"].items():
        print(method, values)


if __name__ == "__main__":
    main()
