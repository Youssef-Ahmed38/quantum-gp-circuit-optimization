"""Convergence audit for the static-detuning direct comparison."""
import json
from pathlib import Path

from qcgp.direct_compare import (
    event_fidelity, gp_policy, nominal_snapshot, qiskit_events,
    snapshot_anchors, variable_schedule,
)
from qcgp.schedule_noise import NoiseScenario, example_circuit, schedule


ROOT = Path(__file__).resolve().parents[1]


def main():
    snapshot = json.loads((ROOT / "evidence/observable_study/props_torino.json").read_text())
    circuit = example_circuit()
    base, error = nominal_snapshot(circuit, snapshot)
    anchors = snapshot_anchors(snapshot)
    gp = gp_policy(circuit, base, error, 168., 2., .8)
    policies = {
        "GP p=2 ALAP": variable_schedule(circuit, gp["durations"], 168., "ALAP"),
        "ASAP+DD": qiskit_events(schedule(circuit, "ASAP+DD"))[0],
    }
    noise = NoiseScenario("static", 200_000., 250_000., .02, 0., 0., 0.)
    rows = []
    for name, events in policies.items():
        for order in (5, 7, 9):
            fidelity = event_fidelity(circuit, events, 168., anchors,
                                      true_power=1., alpha=.8, noise=noise,
                                      anisotropy=(.1, .1, .8),
                                      quadrature_order=order)
            rows.append(dict(policy=name, quadrature_order=order,
                             fidelity=fidelity))
    payload = dict(all_free_local=True, hardware_jobs=0, rows=rows)
    path = ROOT / "evidence/direct_matched/quadrature_check.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
