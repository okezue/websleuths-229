#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from wm.experiments.scaling import ScalingGrid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="runs/scaling")
    parser.add_argument("--models", nargs="+", default=["__tiny__"])
    parser.add_argument("--ranks", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--steps", nargs="+", type=int, default=[25, 50, 100])
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    grid = ScalingGrid(args.config, args.output)
    configs = grid.materialize({"model.name": args.models, "cell.max_rank": args.ranks, "training.max_steps": args.steps})
    print(json.dumps([str(path) for path in configs], indent=2))
    if args.run:
        grid.run_local(configs, args.manifest)


if __name__ == "__main__":
    main()
