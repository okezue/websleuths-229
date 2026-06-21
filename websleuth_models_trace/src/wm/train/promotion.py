from __future__ import annotations

from dataclasses import dataclass

from wm.config import PromotionConfig


@dataclass
class PromotionDecision:
    accepted: bool
    reason: str
    checks: dict[str, bool]


class PromotionGate:
    def __init__(self, cfg: PromotionConfig):
        self.cfg = cfg

    def decide(
        self,
        *,
        test_gain: float,
        test_accuracy: float,
        proof_coverage: float,
        route_false_positive: float,
        max_old_logit_delta: float,
        parameter_invariant: bool,
        old_route_churn: float = 0.0,
    ) -> PromotionDecision:
        checks = {
            "test_gain": test_gain >= self.cfg.min_test_gain,
            "test_accuracy": test_accuracy >= self.cfg.min_test_accuracy,
            "proof_coverage": proof_coverage >= self.cfg.min_proof_coverage,
            "route_false_positive": route_false_positive <= self.cfg.max_route_false_positive,
            "old_route_churn": old_route_churn <= self.cfg.max_old_route_churn,
            "old_logit_delta": max_old_logit_delta <= self.cfg.max_old_probe_logit_delta,
            "parameter_invariant": parameter_invariant or not self.cfg.require_parameter_invariance,
        }
        failed = [name for name, passed in checks.items() if not passed]
        return PromotionDecision(not failed, "accepted" if not failed else "failed: " + ", ".join(failed), checks)
