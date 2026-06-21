#!/usr/bin/env python3
from __future__ import annotations

import argparse

from wm.config import load_config
from wm.pipe.stream import StreamRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--manifest")
    args = parser.parse_args()
    runner = StreamRunner(load_config(args.config))
    try:
        report = runner.run(args.manifest)
        print(report["continual"])
    finally:
        runner.close()


if __name__ == "__main__":
    main()
