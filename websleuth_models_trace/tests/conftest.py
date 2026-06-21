from __future__ import annotations

from pathlib import Path

import pytest

from wm.config import AppConfig, CellConfig, ExtractionConfig, FetchConfig, ModelConfig, ParseConfig, StorageConfig
from wm.model.wrapper import TraceModel


@pytest.fixture
def cfg(tmp_path: Path) -> AppConfig:
    return AppConfig(
        seed=11,
        storage=StorageConfig(root=str(tmp_path / "run")),
        fetch=FetchConfig(max_pages=20, max_depth=1, allow_file_urls=True, respect_robots=True),
        parse=ParseConfig(enable_ocr=False, enable_transcription=False, image_max_side=512),
        extraction=ExtractionConfig(require_independent_sources=1, min_support_overlap=0.2),
        model=ModelConfig(name="__tiny__", dtype="float32", device="cpu", max_input_tokens=512, max_new_tokens=16, target_module_patterns=[".mlp"], target_last_fraction=0.5),
        cell=CellConfig(min_rank=2, max_rank=8, route_threshold=0.15),
    ).resolved()


@pytest.fixture
def tiny_model(cfg: AppConfig) -> TraceModel:
    model = TraceModel.from_pretrained(cfg.model)
    yield model
    model.close()
