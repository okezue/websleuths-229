from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from wm.core.io import ensure_dir


def plot_retention(report: dict[str, Any], output: str | Path) -> str:
    matrix = report.get("retention_matrix", {})
    path = Path(output)
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(10, 6))
    for domain, values in matrix.items():
        ax.plot(range(len(values)), values, marker="o", label=domain)
    ax.set_xlabel("Learning step")
    ax.set_ylabel("Held-out accuracy")
    ax.set_title("Iterative retention")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def plot_general_regression(report: dict[str, Any], output: str | Path) -> str:
    rows = []
    for step in report.get("steps", []):
        for benchmark, score in step.get("general_scores", {}).items():
            rows.append({"step": step["step"], "benchmark": benchmark, "score": score})
    path = Path(output)
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(10, 6))
    frame = pd.DataFrame(rows)
    if not frame.empty:
        for benchmark, group in frame.groupby("benchmark"):
            ax.plot(group["step"], group["score"], marker="o", label=benchmark)
        ax.legend(ncol=2)
    ax.set_xlabel("Learning step")
    ax.set_ylabel("Score")
    ax.set_title("General-knowledge regression and improvement")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)
