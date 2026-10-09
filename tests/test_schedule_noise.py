import math

import pytest

from qcgp.schedule_noise import NoiseScenario, example_circuit, schedule, state_fidelity, timeline


POLICIES = ("ASAP", "ALAP", "ASAP+DD", "ALAP+DD")


def test_matched_makespan_and_noiseless_equivalence():
    circuit = example_circuit()
    noiseless = NoiseScenario("none", math.inf, math.inf, 0., 0., 0., 0.)
    for policy in POLICIES:
        scheduled = schedule(circuit, policy)
        _, makespan = timeline(scheduled)
        assert makespan == 168
        assert state_fidelity(circuit, scheduled, noiseless) == pytest.approx(1., abs=1e-12)


def test_dd_tradeoff_depends_on_noise_memory():
    circuit = example_circuit()
    markov = NoiseScenario("markov", 200_000., 250_000., 0., .00005, .00005, .0008)
    static_z = NoiseScenario("static", 200_000., 250_000., .02, .00005, .00005, .0008)
    asap = schedule(circuit, "ASAP")
    dd = schedule(circuit, "ASAP+DD")
    assert state_fidelity(circuit, dd, markov) < state_fidelity(circuit, asap, markov)
    assert state_fidelity(circuit, dd, static_z) > state_fidelity(circuit, asap, static_z)


def test_invalid_policy():
    with pytest.raises(ValueError):
        schedule(example_circuit(), "unspecified")
