from wm.config import PromotionConfig
from wm.train.promotion import PromotionGate


def test_promotion_gate():
    gate = PromotionGate(PromotionConfig(min_test_gain=0.1, min_test_accuracy=0.5))
    accepted = gate.decide(test_gain=0.2, test_accuracy=0.6, proof_coverage=1.0, route_false_positive=0.0, max_old_logit_delta=0.0, parameter_invariant=True)
    rejected = gate.decide(test_gain=0.0, test_accuracy=0.6, proof_coverage=1.0, route_false_positive=0.0, max_old_logit_delta=0.0, parameter_invariant=True)
    assert accepted.accepted
    assert not rejected.accepted
