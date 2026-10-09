"""Direct, equal-deadline comparison with Qiskit scheduling policies.

All variable pulse durations and duration-response curves are hypothetical.
No physical quantum hardware or paid simulator is used.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import qiskit

from qcgp.direct_compare import (
    event_fidelity, gp_policy, nominal_snapshot, qiskit_events,
    snapshot_anchors, variable_schedule,
)
from qcgp.schedule_noise import NoiseScenario, example_circuit, schedule


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "evidence/observable_study/props_torino.json"
OUT = ROOT / "evidence/direct_matched"
DEADLINE_NS = 168.
ALPHA = .8


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads(SNAPSHOT.read_text())
    circuit = example_circuit()
    base, nominal_error = nominal_snapshot(circuit, snapshot)
    anchors = snapshot_anchors(snapshot)
    policies = {}
    optimization = {}
    for power in (1., 2.):
        optimized = gp_policy(circuit, base, nominal_error, DEADLINE_NS, power, ALPHA)
        optimization[f"GP p={int(power)}"] = optimized
        for placement in ("ASAP", "ALAP"):
            policies[f"GP p={int(power)} {placement}"] = variable_schedule(
                circuit, optimized["durations"], DEADLINE_NS, placement)
    for name in ("ASAP", "ALAP", "ASAP+DD", "ALAP+DD"):
        events, deadline = qiskit_events(schedule(circuit, name))
        if not np.isclose(deadline, DEADLINE_NS):
            raise RuntimeError("Qiskit baseline has a different deadline")
        policies[name] = events
    scenarios = [
        ("aligned_p1_markovian", 1., 0., (1/3, 1/3, 1/3)),
        ("p2_zbiased_markovian", 2., 0., (.1, .1, .8)),
        ("flat_zbiased_markovian", "flat", 0., (.1, .1, .8)),
        ("p1_zbiased_static", 1., .02, (.1, .1, .8)),
    ]
    rows = []
    for scenario_name, true_power, static_sigma, bias in scenarios:
        noise = NoiseScenario(scenario_name, 200_000., 250_000., static_sigma, 0., 0., 0.)
        for name, events in policies.items():
            fidelity = event_fidelity(circuit, events, DEADLINE_NS, anchors,
                                      true_power=true_power, alpha=ALPHA,
                                      noise=noise, anisotropy=bias)
            rows.append(dict(scenario=scenario_name, policy=name, fidelity=fidelity,
                             deadline_ns=DEADLINE_NS,
                             logical_gate_count=len(circuit.data),
                             extra_x=sum(e.name == "x" for e in events),
                             total_idle_ns=float(sum(e.duration_ns for e in events if e.name == "delay"))))
    payload = dict(all_free_local=True, hardware_jobs=0, qiskit_version=qiskit.__version__,
                   snapshot_backend=snapshot["backend_name"],
                   snapshot_updated=snapshot["last_update_date"],
                   assumed_drive_fraction=ALPHA, deadline_ns=DEADLINE_NS,
                   assumption="Nominal cached gate errors anchor hypothetical duration curves; variable-duration gates are not characterized pulses",
                   evaluator="Dense density matrix with per-qubit Pauli gate noise and idle T1/Tphi; seven-node Gauss-Hermite for shot-static Z detuning",
                   note="GP uses extra continuous-duration freedom; Qiskit passes schedule nominal fixed-duration gates. All policies share circuit and deadline, not identical hardware action sets.",
                   nominal_gate_durations_ns=base.tolist(),
                   nominal_gate_errors=nominal_error.tolist(),
                   optimization=optimization, results=rows)
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.7), constrained_layout=True)
    names = list(policies)
    for ax, (scenario_name, _, _, _) in zip(axes.flat, scenarios):
        values = [next(row["fidelity"] for row in rows
                       if row["scenario"] == scenario_name and row["policy"] == name)
                  for name in names]
        ax.plot(np.arange(len(names)), [1. - value for value in values],
                marker="o", linestyle="none", markersize=5)
        ax.set_xticks(np.arange(len(names)), names, rotation=45, ha="right", fontsize=7)
        ax.set_title(scenario_name.replace("_", " "))
        ax.set_ylabel("Simulated infidelity (1 − fidelity)")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=.25)
    fig.savefig(ROOT / "figures/direct_matched.pdf")
    fig.savefig(ROOT / "figures/direct_matched.png", dpi=180)
    plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
