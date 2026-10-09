import json
import math
from pathlib import Path

import numpy as np
import pytest

from qcgp.direct_compare import (
    event_fidelity, nominal_snapshot, qiskit_events, snapshot_anchors,
    variable_schedule,
)
from qcgp.schedule_noise import NoiseScenario, example_circuit, schedule


SNAPSHOT = json.loads((Path(__file__).resolve().parents[1] /
                       "evidence/observable_study/props_torino.json").read_text())


def test_equal_deadline_and_noiseless_logical_equivalence():
    circuit = example_circuit()
    base, _ = nominal_snapshot(circuit, SNAPSHOT)
    zero_anchors = {key: (duration, 0.)
                    for key, (duration, _) in snapshot_anchors(SNAPSHOT).items()}
    noiseless = NoiseScenario("none", math.inf, math.inf, 0., 0., 0., 0.)
    policies = [variable_schedule(circuit, base, 168., placement)
                for placement in ("ASAP", "ALAP")]
    for policy in ("ASAP", "ALAP", "ASAP+DD", "ALAP+DD"):
        events, deadline = qiskit_events(schedule(circuit, policy))
        assert deadline == 168.
        policies.append(events)
    for events in policies:
        assert event_fidelity(circuit, events, 168., zero_anchors,
                              true_power="flat", alpha=.8,
                              noise=noiseless) == pytest.approx(1., abs=1e-12)


def test_invalid_duration_schedule_is_rejected():
    circuit = example_circuit()
    base, _ = nominal_snapshot(circuit, SNAPSHOT)
    with pytest.raises(ValueError, match="deadline"):
        variable_schedule(circuit, base * 2, 168.)
    with pytest.raises(ValueError, match="positive finite"):
        variable_schedule(circuit, np.zeros(len(base)), 168.)


def test_direct_comparison_rank_reverses_under_static_detuning():
    circuit = example_circuit()
    base, _ = nominal_snapshot(circuit, SNAPSHOT)
    anchors = snapshot_anchors(SNAPSHOT)
    asap = variable_schedule(circuit, base, 168., "ASAP")
    dd, _ = qiskit_events(schedule(circuit, "ASAP+DD"))
    markov = NoiseScenario("markov", 200_000., 250_000., 0., 0., 0., 0.)
    static = NoiseScenario("static", 200_000., 250_000., .02, 0., 0., 0.)
    kwargs = dict(true_power=1., alpha=.8, anisotropy=(.1, .1, .8),
                  quadrature_order=5)
    assert event_fidelity(circuit, asap, 168., anchors, noise=markov, **kwargs) > \
           event_fidelity(circuit, dd, 168., anchors, noise=markov, **kwargs)
    assert event_fidelity(circuit, dd, 168., anchors, noise=static, **kwargs) > \
           event_fidelity(circuit, asap, 168., anchors, noise=static, **kwargs)
