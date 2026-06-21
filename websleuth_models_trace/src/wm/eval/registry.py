from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class BenchmarkSpec(BaseModel):
    name: str
    group: str
    loader: str
    dataset: str | None = None
    subset: str | None = None
    split: str | None = None
    path: str | None = None
    metric: str = "accuracy"
    reference: str = ""
    options: dict[str, Any] = Field(default_factory=dict)


class BenchmarkCatalog:
    def __init__(self, specs: dict[str, BenchmarkSpec]):
        self.specs = specs

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkCatalog":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        specs = {
            name: BenchmarkSpec(name=name, **value)
            for name, value in raw.get("benchmarks", {}).items()
        }
        return cls(specs)

    def get(self, name: str) -> BenchmarkSpec:
        if name not in self.specs:
            raise KeyError(f"unknown benchmark: {name}")
        return self.specs[name]

    def by_group(self, group: str) -> list[BenchmarkSpec]:
        return [spec for spec in self.specs.values() if spec.group == group]

    def rows(self) -> list[dict[str, Any]]:
        return [spec.model_dump() for spec in sorted(self.specs.values(), key=lambda s: (s.group, s.name))]
