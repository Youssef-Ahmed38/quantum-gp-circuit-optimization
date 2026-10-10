"""C2+C5 ablation: state-aware (non-unital) T1 weights inside the DD-GP.

Truth physics uses exact amplitude damping, so a qubit idling in |0> does not
decay and a qubit idling in |1> does. The state-blind GP folds T1 into a unital
rate fitted on |+>; the state-aware GP weights T1 exposure by the ideal excited
population of each window (Proposition C5.2), with 1/2 under an even CPMG block.
Both models are fitted only from simulated spectroscopy.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from qcgp.ddgp import (IdleModel, alap, asap, correlated_protection, fit_idle_model, fixed_pattern, model_objective,
                       solve_dd_allocation, to_schedule)
from qcgp.noisesim import GateLaw, QubitPhysics, gate, relaxation_spectroscopy, simulate, spectroscopy
from run_dd_gp_sim import build_problem, heterogeneous

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence" / "dd_state_aware"


def circuits():
    ghz = [gate("h", 0)] + [gate("cx", i, i + 1) for i in range(4)] + [gate("block", 5), gate("cx", 4, 5)]
    # Qubits 0-2 are flipped to |1> and idle during an ancilla wait; 3 stays in |0>.
    excited = [gate("x", 0), gate("x", 1), gate("x", 2), gate("h", 4), gate("cx", 3, 4), gate("block", 4),
               gate("cx", 2, 3), gate("cx", 1, 2), gate("cx", 0, 1), gate("x", 0), gate("x", 1)]
    rep = [gate("h", 0), gate("cx", 0, 2), gate("cx", 2, 4)]
    for _ in range(2):
        rep += [gate("cx", 0, 1), gate("cx", 2, 1), gate("cx", 2, 3), gate("cx", 4, 3),
                gate("block", 1), gate("block", 3)]
    return {"ghz5_with_ancilla_wait": (6, ghz), "excited_idle_chain": (5, excited), "rep3_two_cycles": (5, rep)}


def scenarios():
    base = QubitPhysics(t1=80.0, white_dephasing=1 / 600, quasi_static=0.0, spectral_s0=0.0,
                        pulse_error=3e-4, exact_damping=True)
    return {"t1_dominated": base,
            "t1_plus_one_over_f": replace(base, spectral_s0=4e-4, spectral_beta=1.0),
            "t1_plus_quasi_static": replace(base, quasi_static=0.04, spectral_s0=1e-4)}


def fit(physics, q):
    rows = spectroscopy(physics, [0.5, 1, 2, 4, 8, 16], [0, 2, 4, 8, 16], trajectories=1500, seed=100 + q)
    blind, _ = fit_idle_model(rows)
    rate = float(np.mean([r["rate"] for r in relaxation_spectroscopy(physics, [8., 32.], trajectories=4)]))
    aware = replace(blind, markov=max(blind.markov - rate / 2, 0.), relaxation=rate)
    return blind, aware


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records, start = [], time.perf_counter()
    for scenario, physics in scenarios().items():
        for name, (n, gates) in circuits().items():
            truth = heterogeneous(physics, n, seed=11)
            fits = [fit(p, q) for q, p in enumerate(truth)]
            blind = build_problem(n, gates, [f[0] for f in fits])
            aware = build_problem(n, gates, [f[1] for f in fits])
            _, _, deadline = asap(blind)
            law = GateLaw(blind.inverse, blind.linear)
            d, s, M = alap(blind, deadline)
            plans = {"alap": (d, s, M, np.zeros(len(blind.windows()))),
                     "alap_xx_qiskit_like": (d, s, M, fixed_pattern(blind, d, s, M, 2))}
            for label, problem in (("gp_dd_state_blind", blind), ("gp_dd_state_aware", aware)):
                r = solve_dd_allocation(problem, deadline, optimize_durations=False)
                plans[label] = (r["durations"], r["starts"], r["makespan"], r["pulses"])
            # Cross-window (correlated) model on the state-aware GP placement.
            d2, s2, M2, _ = plans["gp_dd_state_aware"]
            plans["gp_dd_correlated"] = (d2, s2, M2, correlated_protection(aware, d2, s2, M2))
            for label, (d, s, M, counts) in plans.items():
                fidelity = simulate(to_schedule(blind, d, s, M, counts), truth, law, trajectories=200, seed=2026)
                records.append(dict(scenario=scenario, circuit=name, policy=label, pulses=int(np.sum(counts)),
                                    fidelity=float(np.mean(fidelity)),
                                    per_trajectory=fidelity.tolist(),
                                    aware_model_hazard=model_objective(aware, d, s, M, counts)))
            print(f"{scenario:22s} {name:24s} {time.perf_counter() - start:6.1f}s", flush=True)
    summary = {}
    for key in sorted({(r["scenario"], r["circuit"]) for r in records}):
        rows = {r["policy"]: r for r in records if (r["scenario"], r["circuit"]) == key}
        aware = np.array(rows["gp_dd_state_aware"]["per_trajectory"])
        entry = {p: dict(fidelity=r["fidelity"], pulses=r["pulses"]) for p, r in rows.items()}
        corr = np.array(rows["gp_dd_correlated"]["per_trajectory"])
        for other in ("alap", "alap_xx_qiskit_like", "gp_dd_state_aware"):
            diff = corr - np.array(rows[other]["per_trajectory"])
            entry[f"correlated_minus_{other}"] = dict(mean=float(diff.mean()),
                                                      paired_se=float(diff.std(ddof=1) / np.sqrt(len(diff))))
        for other in ("alap", "alap_xx_qiskit_like", "gp_dd_state_blind"):
            diff = aware - np.array(rows[other]["per_trajectory"])
            entry[f"aware_minus_{other}"] = dict(mean=float(diff.mean()),
                                                 paired_se=float(diff.std(ddof=1) / np.sqrt(len(diff))))
        summary["|".join(key)] = entry
    summary["elapsed_s"] = time.perf_counter() - start
    summary["hardware_jobs"] = 0
    (OUT / "results.json").write_text(json.dumps(records, indent=1))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
