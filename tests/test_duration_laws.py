import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "benchmarks/compare_duration_laws.py"
    spec = importlib.util.spec_from_file_location("compare_duration_laws", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("power", [1, 2])
def test_leave_one_out_recovers_exact_positive_law(power):
    module = _module()
    duration = np.array([5., 6., 7., 8., 9., 10.])
    error = .001 / duration ** power + .00001 * duration
    predicted = module.leave_one_out(duration, error, np.full(6, 1e-6), (-power, 1))
    assert predicted == pytest.approx(error, abs=1e-12)


def test_leave_one_out_rejects_bad_uncertainty():
    module = _module()
    with pytest.raises(ValueError):
        module.leave_one_out([1, 2, 3, 4], [1, 1, 1, 1], [1, 1, 0, 1], (-1, 1))
