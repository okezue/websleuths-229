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
        nll_gain: float = 0.0,
        old_route_churn: float = 0.0,
    ) -> PromotionDecision:
        acc_required = self.cfg.min_test_gain > 0 or self.cfg.min_test_accuracy > 0
        nll_required = self.cfg.min_nll_gain > 0
        acc_pass = test_gain >= self.cfg.min_test_gain and test_accuracy >= self.cfg.min_test_accuracy
        nll_pass = nll_gain >= self.cfg.min_nll_gain
        capability = (not acc_required or acc_pass) and (not nll_required or nll_pass) and (acc_required or nll_required)
        checks = {
            "capability": capability,
            "test_gain": test_gain >= self.cfg.min_test_gain,
            "test_accuracy": test_accuracy >= self.cfg.min_test_accuracy,
            "nll_gain": nll_gain >= self.cfg.min_nll_gain,
            "proof_coverage": proof_coverage >= self.cfg.min_proof_coverage,
            "route_false_positive": route_false_positive <= self.cfg.max_route_false_positive,
            "old_route_churn": old_route_churn <= self.cfg.max_old_route_churn,
            "old_logit_delta": max_old_logit_delta <= self.cfg.max_old_probe_logit_delta,
            "parameter_invariant": parameter_invariant or not self.cfg.require_parameter_invariance,
        }
        gate_keys = ["capability", "proof_coverage", "route_false_positive", "old_route_churn", "old_logit_delta", "parameter_invariant"]
        failed = [name for name in gate_keys if not checks[name]]
        return PromotionDecision(not failed, "accepted" if not failed else "failed: " + ", ".join(failed), checks)
