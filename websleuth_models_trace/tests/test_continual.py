import pytest
from wm.eval.continual import IterativeEvaluator


def test_continual_metrics():
    tracker = IterativeEvaluator()
    tracker.set_baseline({"finance": 0.2, "legal": 0.3})
    tracker.record_prelearn("finance", 0.2)
    tracker.record_post_step("finance", {"finance": 0.7})
    tracker.record_prelearn("legal", 0.3)
    tracker.record_post_step("legal", {"finance": 0.65, "legal": 0.8})
    metrics = tracker.compute()
    assert metrics.mean_forgetting == pytest.approx(0.025)
    assert metrics.domains_above_baseline == 2
    assert metrics.backward_transfer < 0
