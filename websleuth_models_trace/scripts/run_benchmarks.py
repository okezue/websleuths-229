#!/usr/bin/env python3
from __future__ import annotations

import argparse

from wm.config import load_config
from wm.eval.registry import BenchmarkCatalog
from wm.eval.runner import BenchmarkRunner
from wm.model.wrapper import TraceModel
from wm.pipe.loop import TracePipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    model = TraceModel.from_pretrained(cfg.model)
    pipeline = TracePipeline(cfg, model)
    try:
        runner = BenchmarkRunner(model, pipeline.router, BenchmarkCatalog.load(cfg.benchmarks.registry), max_samples=cfg.benchmarks.max_samples, seed=cfg.seed)
        for name, result in runner.evaluate_many(args.names).items():
            print(name, result.metrics, "n=", result.n)
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
