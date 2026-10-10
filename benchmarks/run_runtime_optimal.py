"""C3 study: QPU time to reach a PEC-mitigated precision target.

Workloads and calibration anchoring follow the manuscript's observable study
(saved ibm_torino snapshot; inverse-duration share ``alpha`` assumed). Each
policy is scored by the declared model's QPU time, and the PEC variance law is
validated by independent Pauli-frame sampling on a subset of instances.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from qcgp.mitigation import (best_uniform, error_optimal, optimize_runtime, pec_pauli_frame,
                             qpu_time)
from qcgp.observables import snapshot_model
from run_observables import workload

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "evidence/observable_study/props_torino.json"
OUT = ROOT / "evidence" / "runtime_optimal"
REP_DELAYS = {"no_repetition_delay": 0.0, "fast_reset_5us": 5.0, "ibm_default_250us": 250.0}


def main():
    snapshot = json.loads(SNAPSHOT.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    records, validations = [], []
    start = time.perf_counter()
    for family in ("graph", "scrambled"):
        for n in (8, 16, 32, 64):
            for seed in (11, 29):
                circuit, mapping = workload(snapshot, n, seed, family)
                for alpha in (0.5, 0.8):
                    model = snapshot_model(circuit, mapping, snapshot, alpha)
                    for label, rep in REP_DELAYS.items():
                        opt = optimize_runtime(model, rep_delay=rep)
                        candidates = {"nominal": model.base, "fastest": model.lower.copy(),
                                      "error_optimal": error_optimal(model),
                                      "best_uniform": best_uniform(model, rep_delay=rep),
                                      "runtime_optimal": opt["durations"]}
                        times = {k: qpu_time(model, v, rep_delay=rep)[0] for k, v in candidates.items()}
                        baseline = min(v for k, v in times.items() if k != "runtime_optimal")
                        records.append(dict(
                            family=family, qubits=n, seed=seed, alpha=alpha, rep_delay=label,
                            gates=len(model.base), qpu_time_us=times,
                            pec_overhead={k: float(np.exp(2 * np.sum(model.a / v + model.gamma * v)))
                                          for k, v in candidates.items()},
                            makespan_us={k: qpu_time(model, v, rep_delay=rep)[2] for k, v in candidates.items()},
                            reduction_vs_best_other=1 - times["runtime_optimal"] / baseline,
                            reduction_vs_nominal=1 - times["runtime_optimal"] / times["nominal"],
                            stationarity=opt["stationarity"], solve_s=opt["elapsed"]))
                        if times["runtime_optimal"] > baseline * (1 + 1e-6):
                            raise AssertionError("Convex optimum beaten by a feasible heuristic")
                    if family == "graph" and n in (8, 16) and seed == 11:
                        for name, tau in (("nominal", model.base), ("runtime_optimal_t0",
                                                                    optimize_runtime(model)["durations"])):
                            sim = pec_pauli_frame(circuit, model, tau, shots=200000, seed=7)
                            validations.append(dict(
                                qubits=n, alpha=alpha, schedule=name, gamma=sim["gamma"],
                                max_abs_bias_in_se=float(np.max(np.abs(sim["mean"] - 1) /
                                                                np.sqrt(sim["variance"] / 200000))),
                                variance_ratio_min=float(np.min(sim["variance"] / sim["predicted_variance"])),
                                variance_ratio_max=float(np.max(sim["variance"] / sim["predicted_variance"])),
                                unmitigated_mean_min=float(np.min(sim["unmitigated"]))))
                print(f"{family} n={n} seed={seed} {time.perf_counter() - start:6.1f}s", flush=True)
    summary = {}
    for label in REP_DELAYS:
        subset = [r for r in records if r["rep_delay"] == label]
        summary[label] = dict(
            cases=len(subset),
            median_reduction_vs_best_other=float(np.median([r["reduction_vs_best_other"] for r in subset])),
            min_reduction_vs_best_other=float(np.min([r["reduction_vs_best_other"] for r in subset])),
            max_reduction_vs_best_other=float(np.max([r["reduction_vs_best_other"] for r in subset])),
            median_reduction_vs_nominal=float(np.median([r["reduction_vs_nominal"] for r in subset])),
            winner_counts={k: sum(min(r["qpu_time_us"], key=r["qpu_time_us"].get) == k for r in subset)
                           for k in ("nominal", "fastest", "error_optimal", "best_uniform", "runtime_optimal")},
            stationarity_all_valid=all(r["stationarity"]["valid"] for r in subset),
            max_offcritical_flow=float(max(r["stationarity"]["max_offcritical_flow"] for r in subset)))
    summary["pec_validation"] = validations
    summary["elapsed_s"] = time.perf_counter() - start
    summary["hardware_jobs"] = 0
    (OUT / "results.json").write_text(json.dumps(records, indent=1, default=float))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
