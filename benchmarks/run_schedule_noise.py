"""Matched free-simulator study of Qiskit ASAP/ALAP and XY-free X-X DD."""
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import platform

import numpy as np
import qiskit

from qcgp.schedule_noise import (
    NoiseScenario, example_circuit, schedule, state_fidelity, timeline,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence" / "schedule_noise"


def calibration_rows():
    """Cross-sectional reported calibration pairs; never fit a duration curve."""
    source = json.loads((ROOT / "evidence/observable_study/props_torino.json").read_text())
    rows = []
    for gate in source["gates"]:
        params = {item["name"]: item for item in gate["parameters"]}
        if gate["gate"] not in ("sx", "x", "cz") or not {"gate_error", "gate_length"} <= params.keys():
            continue
        length = params["gate_length"]
        if length["unit"] not in ("ns", "us"):
            continue
        duration_ns = float(length["value"]) * (1000 if length["unit"] == "us" else 1)
        error = float(params["gate_error"]["value"])
        if not np.isfinite(duration_ns + error) or duration_ns <= 0 or not 0 <= error <= 1:
            continue
        rows.append(dict(backend=source["backend_name"], snapshot_updated=source["last_update_date"],
                         gate=gate["gate"], qubits="-".join(map(str, gate["qubits"])),
                         duration_ns=duration_ns, reported_gate_error=error,
                         error_timestamp=params["gate_error"].get("date", ""),
                         duration_timestamp=length.get("date", "")))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    circuit = example_circuit()
    scenarios = [
        NoiseScenario("markovian_isotropic", 200_000., 250_000., 0., .0002, .0002, .0002),
        NoiseScenario("markovian_anisotropic_z", 200_000., 250_000., 0., .00005, .00005, .0008),
        NoiseScenario("quasistatic_z_plus_anisotropic", 200_000., 250_000., .02, .00005, .00005, .0008),
    ]
    policies = ("ASAP", "ALAP", "ASAP+DD", "ALAP+DD")
    results = []
    schedules = {}
    for policy in policies:
        scheduled = schedule(circuit, policy)
        events, duration = timeline(scheduled)
        schedules[policy] = scheduled
        for noise in scenarios:
            results.append(dict(policy=policy, scenario=noise.name,
                                fidelity=state_fidelity(circuit, scheduled, noise),
                                makespan_ns=duration,
                                x_count=scheduled.count_ops().get("x", 0),
                                idle_ns=sum(e[3] for e in events if e[0] == "delay")))
    rows = calibration_rows()
    with (OUT / "reported_calibration_pairs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    payload = dict(created_utc=datetime.now(timezone.utc).isoformat(), all_free_local=True,
                   hardware_jobs=0, qiskit_version=qiskit.__version__, python=platform.python_version(),
                   method="Qiskit scheduling passes, independent density-matrix evaluator; seven-node Gauss-Hermite per qubit for static Gaussian detuning",
                   circuit="three-qubit sx-cz-sx-cz-sx native Clifford example",
                   scenarios=[vars(s) for s in scenarios], results=results,
                   calibration_pair_count=len(rows),
                   calibration_caveat="Cross-sectional cached calibration entries; not a same-gate duration sweep or fitted duration-response curve.")
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), constrained_layout=True)
        for gate in ("sx", "x", "cz"):
            points = [r for r in rows if r["gate"] == gate]
            axes[0].scatter([r["duration_ns"] for r in points],
                            [r["reported_gate_error"] for r in points], s=12, alpha=.5, label=gate)
        axes[0].set(xlabel="Reported gate duration (ns)", ylabel="Reported calibration gate error",
                    title="Cross-sectional calibration entries")
        axes[0].set_yscale("log")
        axes[0].legend()
        x = np.arange(len(policies))
        for noise in scenarios:
            values = [next(r["fidelity"] for r in results if r["policy"] == p and r["scenario"] == noise.name) for p in policies]
            axes[1].plot(x, values, marker="o", label=noise.name.replace("_", " "))
        axes[1].set_xticks(x, policies)
        axes[1].tick_params(axis="x", rotation=25)
        axes[1].set(ylabel="Simulated ideal-state fidelity", title="Matched local scheduling comparison")
        axes[1].legend(fontsize=7)
        fig.savefig(ROOT / "figures/schedule_noise.pdf")
        fig.savefig(ROOT / "figures/schedule_noise.png", dpi=180)
        plt.close(fig)
    except ImportError:
        pass
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
