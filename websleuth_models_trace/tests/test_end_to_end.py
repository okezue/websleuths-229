from __future__ import annotations

from pathlib import Path

from wm.config import AppConfig
from wm.pipe.stream import StreamRunner
from wm.synthetic.micro_web import MicroWebGenerator


def test_one_episode_stream_allocates_and_promotes_a_cell(tmp_path: Path) -> None:
    web_root = tmp_path / "micro_web"
    manifest = MicroWebGenerator(web_root, seed=3).generate(1, domains=["finance"])
    # Keep the integration test independent of optional system media binaries. Dedicated
    # parser tests cover those paths; the end-to-end path uses two independent text sources.
    manifest["episodes"][0]["urls"] = [
        url for url in manifest["episodes"][0]["urls"] if url.endswith((".html", ".pdf"))
    ]
    from wm.core.io import write_json

    write_json(web_root / "manifest.json", manifest)
    run_root = tmp_path / "run"
    cfg = AppConfig.model_validate(
        {
            "seed": 3,
            "storage": {"root": str(run_root)},
            "fetch": {
                "allow_file_urls": True,
                "allow_private_network": False,
                "same_origin_only": False,
                "respect_robots": True,
                "max_pages": 10,
            },
            "parse": {"enable_ocr": False, "enable_transcription": False},
            "extraction": {"backend": "rules", "require_independent_sources": 1},
            "model": {
                "name": "__tiny__",
                "dtype": "float32",
                "device": "cpu",
                "max_input_tokens": 512,
                "max_new_tokens": 8,
                "target_module_patterns": [".mlp"],
                "target_last_fraction": 0.5,
            },
            "cell": {
                "min_rank": 2,
                "max_rank": 4,
                "max_active_cells": 1,
                "route_threshold": 0.1,
            },
            "training": {
                "learning_rate": 0.01,
                "max_steps": 2,
                "eval_every": 1,
                "early_stop_patience": 2,
                "distill_weight": 0.0,
            },
            "promotion": {
                "min_test_gain": 0.0,
                "min_test_accuracy": 0.0,
                "max_old_probe_logit_delta": 1e-6,
                "max_route_false_positive": 1.0,
                "min_proof_coverage": 0.5,
            },
            "benchmarks": {
                "registry": str(Path(__file__).parents[1] / "configs" / "benchmarks.yaml"),
                "max_samples": 1,
                "suites": {"domain": [], "general": [], "long_context": []},
            },
            "stream": {
                "manifest": str(web_root / "manifest.json"),
                "save_every_step": True,
                "fail_fast": True,
            },
        }
    ).resolved()
    runner = StreamRunner(cfg)
    try:
        report = runner.run()
        assert len(report["steps"]) == 1
        assimilation = report["steps"][0]["assimilation"]
        assert assimilation["trained_steps"] == 2
        assert assimilation["accepted"] is True
        assert assimilation["invariants"]["parameter_invariant"] is True
        assert len(runner.model.cells) == 1
        assert (run_root / "reports" / "stream_report.json").exists()
    finally:
        runner.close()
