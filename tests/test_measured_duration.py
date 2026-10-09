import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "benchmarks/run_measured_duration.py"
    spec = importlib.util.spec_from_file_location("run_measured_duration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_heldout_inverse_linear_recovery():
    module = _module()
    rows = []
    for pulse in module.PULSES:
        for index, duration in enumerate((5., 6., 7., 8., 9., 10.)):
            rows.append(dict(pulse=pulse, duration_ns=duration,
                             measured_error=.001 / duration + .00001 * duration,
                             reported_one_sigma=.000001,
                             split="train" if index % 2 == 0 else "test"))
    fits = module.evaluate(rows)
    assert len(fits) == len(module.PULSES)
    for fit in fits:
        assert fit["test_mae"] == pytest.approx(0., abs=1e-12)
        assert fit["inverse_coefficient_error_ns"] == pytest.approx(.001)
        assert fit["linear_coefficient_error_per_ns"] == pytest.approx(.00001)
