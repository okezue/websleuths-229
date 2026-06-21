#!/usr/bin/env python3
from __future__ import annotations

import argparse

from wm.synthetic.micro_web import MicroWebGenerator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/micro_web")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = MicroWebGenerator(args.output, seed=args.seed).generate(args.episodes)
    print(f"generated {len(manifest['episodes'])} episodes at {args.output}")


if __name__ == "__main__":
    main()
