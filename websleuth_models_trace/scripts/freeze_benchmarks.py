#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from wm.core.io import write_jsonl
from wm.eval.adapters import load_examples
from wm.eval.registry import BenchmarkCatalog


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+")
    parser.add_argument("--registry", default="configs/benchmarks.yaml")
    parser.add_argument("--output", default="artifacts/frozen_benchmarks")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    catalog = BenchmarkCatalog.load(args.registry)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        spec = catalog.get(name)
        examples = load_examples(spec, args.samples, args.seed)
        write_jsonl(output / f"{name}.jsonl", [example.model_dump(mode="json") for example in examples])
        print(name, len(examples))


if __name__ == "__main__":
    main()
