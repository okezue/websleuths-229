from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9.+%_-]+", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, answer: str) -> float:
    return float(normalize_text(prediction) == normalize_text(answer))


def contains_match(prediction: str, answer: str) -> float:
    pred = normalize_text(prediction)
    gold = normalize_text(answer)
    return float(bool(gold) and gold in pred)


def token_f1(prediction: str, answer: str) -> float:
    pred = normalize_text(prediction).split()
    gold = normalize_text(answer).split()
    common = Counter(pred) & Counter(gold)
    overlap = sum(common.values())
    if not pred or not gold or overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def extract_number(text: str) -> float | None:
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", text.replace("$", "").replace("%", ""))
    if not match:
        return None
    try:
        return float(match.group().replace(",", ""))
    except ValueError:
        return None


def numeric_match(prediction: str, answer: str, relative_tolerance: float = 0.01, absolute_tolerance: float = 1e-6) -> float:
    pred = extract_number(prediction)
    gold = extract_number(answer)
    if pred is None or gold is None:
        return contains_match(prediction, answer)
    return float(math.isclose(pred, gold, rel_tol=relative_tolerance, abs_tol=absolute_tolerance))


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / max(len(values), 1)


def bootstrap_ci(values: list[float], confidence: float = 0.95, samples: int = 2000, seed: int = 42) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    import numpy as np

    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = np.asarray([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(samples)])
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))
