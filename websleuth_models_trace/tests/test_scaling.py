from wm.experiments.scaling import ScalingPoint, fit_power_law


def test_scaling_fit_returns_positive_exponents():
    points = []
    for n in [10, 20, 40, 80, 160]:
        error = 0.1 + 2.0 * n ** -0.5
        points.append(ScalingPoint(model_parameters=1e8, evidence_items=n, web_requests=n, added_parameters=n * 1000, error=error, metadata={}))
    fit = fit_power_law(points, floor=0.1)
    assert fit["alpha_evidence"] > 0
    assert fit["r2_log_space"] > 0.9
