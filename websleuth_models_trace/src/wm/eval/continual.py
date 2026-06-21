from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class ContinualMetrics:
    mean_current_accuracy: float = 0.0
    mean_retained_gain: float = 0.0
    mean_forgetting: float = 0.0
    worst_forgetting: float = 0.0
    backward_transfer: float = 0.0
    forward_transfer: float = 0.0
    retention_auc: float = 0.0
    domains_above_baseline: int = 0
    n_domains: int = 0


class IterativeEvaluator:
    """Tracks paired domain scores before and after each sequential update."""

    def __init__(self):
        self.baselines: dict[str, float] = {}
        self.prelearn: dict[str, float] = {}
        self.first_postlearn: dict[str, float] = {}
        self.history: list[dict[str, float]] = []
        self.learned_order: list[str] = []

    def set_baseline(self, scores: dict[str, float]) -> None:
        self.baselines = dict(scores)

    def record_prelearn(self, domain: str, score: float) -> None:
        self.prelearn.setdefault(domain, score)

    def record_post_step(self, learned_domain: str, scores: dict[str, float]) -> None:
        if learned_domain not in self.learned_order:
            self.learned_order.append(learned_domain)
            if learned_domain in scores:
                self.first_postlearn[learned_domain] = scores[learned_domain]
        self.history.append(dict(scores))

    def matrix(self, domains: Iterable[str] | None = None) -> dict[str, list[float]]:
        domains = list(domains or sorted({domain for row in self.history for domain in row}))
        return {domain: [row.get(domain, float("nan")) for row in self.history] for domain in domains}

    def compute(self) -> ContinualMetrics:
        if not self.history:
            return ContinualMetrics()
        final = self.history[-1]
        learned = [domain for domain in self.learned_order if domain in final]
        forgetting: list[float] = []
        gains: list[float] = []
        aucs: list[float] = []
        bwt: list[float] = []
        fwt: list[float] = []
        for domain in learned:
            values = [row[domain] for row in self.history if domain in row]
            if not values:
                continue
            forgetting.append(max(values) - values[-1])
            baseline = self.baselines.get(domain, self.prelearn.get(domain, values[0]))
            gains.append(values[-1] - baseline)
            aucs.append(sum(values) / len(values))
            if domain in self.first_postlearn:
                bwt.append(values[-1] - self.first_postlearn[domain])
            if domain in self.prelearn:
                fwt.append(self.prelearn[domain] - baseline)
        return ContinualMetrics(
            mean_current_accuracy=sum(final.get(domain, 0.0) for domain in learned) / max(len(learned), 1),
            mean_retained_gain=sum(gains) / max(len(gains), 1),
            mean_forgetting=sum(forgetting) / max(len(forgetting), 1),
            worst_forgetting=max(forgetting, default=0.0),
            backward_transfer=sum(bwt) / max(len(bwt), 1),
            forward_transfer=sum(fwt) / max(len(fwt), 1),
            retention_auc=sum(aucs) / max(len(aucs), 1),
            domains_above_baseline=sum(gain > 0 for gain in gains),
            n_domains=len(learned),
        )
