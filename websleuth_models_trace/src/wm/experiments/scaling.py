from __future__ import annotations

import itertools
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from wm.core.io import ensure_dir, read_json, write_json


@dataclass
class ScalingPoint:
    model_parameters: float
    evidence_items: float
    web_requests: float
    added_parameters: float
    error: float
    metadata: dict[str, Any]


def fit_power_law(points: Iterable[ScalingPoint], floor: float | None = None) -> dict[str, float]:
    points = list(points)
    if len(points) < 4:
        raise ValueError("at least four scaling points are required")
    errors = np.asarray([point.error for point in points], dtype=float)
    floor_value = float(floor if floor is not None else max(0.0, errors.min() * 0.9))
    target = np.log(np.maximum(errors - floor_value, 1e-12))
    features = np.column_stack(
        [
            np.ones(len(points)),
            -np.log(np.maximum([p.evidence_items for p in points], 1.0)),
            -np.log(np.maximum([p.added_parameters for p in points], 1.0)),
            -np.log(np.maximum([p.web_requests for p in points], 1.0)),
            -np.log(np.maximum([p.model_parameters for p in points], 1.0)),
        ]
    )
    coefficients, *_ = np.linalg.lstsq(features, target, rcond=None)
    predicted = features @ coefficients
    ss_res = float(((target - predicted) ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    return {
        "floor": floor_value,
        "log_a": float(coefficients[0]),
        "alpha_evidence": float(coefficients[1]),
        "beta_added_parameters": float(coefficients[2]),
        "gamma_web_requests": float(coefficients[3]),
        "delta_model_parameters": float(coefficients[4]),
        "r2_log_space": 1.0 - ss_res / max(ss_tot, 1e-12),
    }


class ScalingGrid:
    """Builds and runs a reproducible local experiment grid. AWS submission is separate."""

    def __init__(self, base_config: str | Path, output: str | Path):
        self.base_config = Path(base_config)
        self.output = ensure_dir(output)
        self.base = yaml.safe_load(self.base_config.read_text(encoding="utf-8"))

    @staticmethod
    def _set(data: dict[str, Any], dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        current = data
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value

    def materialize(self, axes: dict[str, list[Any]]) -> list[Path]:
        configs: list[Path] = []
        names = list(axes)
        for index, values in enumerate(itertools.product(*(axes[name] for name in names))):
            data = json.loads(json.dumps(self.base))
            tags = []
            for name, value in zip(names, values, strict=True):
                self._set(data, name, value)
                tags.append(f"{name.replace('.', '-')}_{value}")
            run_dir = self.output / f"run_{index:04d}"
            data.setdefault("storage", {})["root"] = str(run_dir)
            config_path = self.output / ("__".join(tags) + ".yaml")
            config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
            configs.append(config_path)
        write_json(self.output / "grid.json", {"axes": axes, "configs": [str(path) for path in configs]})
        return configs

    def run_local(self, configs: Iterable[Path], manifest: str | Path, fail_fast: bool = False) -> list[dict[str, Any]]:
        results = []
        for config in configs:
            proc = subprocess.run(
                ["websleuth", "stream", "--config", str(config), "--manifest", str(manifest)],
                text=True,
                capture_output=True,
            )
            result = {"config": str(config), "returncode": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]}
            results.append(result)
            if fail_fast and proc.returncode:
                break
        write_json(self.output / "local_runs.json", results)
        return results
