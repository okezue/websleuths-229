from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class StorageConfig(BaseModel):
    root: str = "./runs/default"
    database: str = "evidence.sqlite"
    blobs: str = "blobs"
    cells: str = "cells"
    reports: str = "reports"

    def resolve(self) -> "StorageConfig":
        root = Path(os.path.expandvars(os.path.expanduser(self.root))).resolve()
        return self.model_copy(update={
            "root": str(root),
            "database": str(root / self.database),
            "blobs": str(root / self.blobs),
            "cells": str(root / self.cells),
            "reports": str(root / self.reports),
        })


class FetchConfig(BaseModel):
    user_agent: str = "WebsleuthModels/0.2 research crawler"
    timeout_seconds: float = 30.0
    max_bytes: int = 50_000_000
    max_pages: int = 100
    max_depth: int = 2
    same_origin_only: bool = True
    respect_robots: bool = True
    allow_private_network: bool = False
    allow_file_urls: bool = False
    browser_fallback: bool = False
    max_redirects: int = 5


class ParseConfig(BaseModel):
    max_text_chars_per_span: int = 4000
    image_max_side: int = 1280
    video_sample_fps: float = 0.2
    video_max_frames: int = 24
    enable_ocr: bool = True
    enable_transcription: bool = False


class ExtractionConfig(BaseModel):
    backend: Literal["rules", "local_llm", "hybrid"] = "rules"
    local_model: str | None = None
    min_claim_chars: int = 18
    max_claim_chars: int = 500
    min_support_overlap: float = 0.45
    require_independent_sources: int = 2


class ModelConfig(BaseModel):
    name: str = "Qwen/Qwen2.5-1.5B"
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    device: str = "auto"
    trust_remote_code: bool = False
    max_input_tokens: int = 2048
    max_new_tokens: int = 96
    target_module_patterns: list[str] = Field(default_factory=lambda: [".mlp", ".feed_forward", ".ffn"])
    target_last_fraction: float = 0.25


class CellConfig(BaseModel):
    min_rank: int = 4
    max_rank: int = 64
    rank_energy: float = 0.90
    max_active_cells: int = 2
    route_threshold: float = 0.22
    route_entity_bonus: float = 0.35
    dropout: float = 0.0
    scale: float = 1.0
    initializer_path: str | None = None


class TrainingConfig(BaseModel):
    enabled: bool = True
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_steps: int = 120
    batch_size: int = 1
    grad_clip: float = 1.0
    distill_weight: float = 0.25
    supervised_weight: float = 1.0
    early_stop_patience: int = 15
    eval_every: int = 10
    answer_template: str = "Question: {question}\nAnswer:"


class PromotionConfig(BaseModel):
    min_test_gain: float = 0.08
    min_test_accuracy: float = 0.25
    min_nll_gain: float = 0.0
    max_old_probe_logit_delta: float = 1e-5
    max_route_false_positive: float = 0.01
    max_old_route_churn: float = 0.0
    min_proof_coverage: float = 0.95
    require_parameter_invariance: bool = True


class BenchmarkConfig(BaseModel):
    registry: str = "configs/benchmarks.yaml"
    max_samples: int = 100
    general_every: int = 1
    context_every: int = 2
    generation_temperature: float = 0.0
    suites: dict[str, list[str]] = Field(default_factory=dict)


class StreamConfig(BaseModel):
    manifest: str | None = None
    domains: list[str] = Field(default_factory=lambda: ["finance", "legal", "chemistry", "medicine"])
    episodes_per_domain: int = 3
    evaluate_all_prior_domains: bool = True
    save_every_step: bool = True
    fail_fast: bool = False


class AimConfig(BaseModel):
    enabled: bool = False
    repo: str | None = None
    experiment: str = "trace-continual"
    run_name: str | None = None
    tags: list[str] = Field(default_factory=list)


class AppConfig(BaseModel):
    seed: int = 42
    storage: StorageConfig = Field(default_factory=StorageConfig)
    fetch: FetchConfig = Field(default_factory=FetchConfig)
    parse: ParseConfig = Field(default_factory=ParseConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    cell: CellConfig = Field(default_factory=CellConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    promotion: PromotionConfig = Field(default_factory=PromotionConfig)
    benchmarks: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    stream: StreamConfig = Field(default_factory=StreamConfig)
    aim: AimConfig = Field(default_factory=AimConfig)

    def resolved(self) -> "AppConfig":
        return self.model_copy(update={"storage": self.storage.resolve()})


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return AppConfig.model_validate(_expand(raw)).resolved()
