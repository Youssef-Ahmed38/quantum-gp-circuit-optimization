"""C2 study: joint timing + dynamical-decoupling GP versus standard schedules.

Pipeline per noise scenario (mirrors the planned hardware protocol H1/H2):
1. simulated CPMG spectroscopy on each qubit (never on the evaluation circuits);
2. fit the declared idle law per qubit;
3. optimise schedules with the fitted model at a matched deadline;
4. evaluate every policy in the independent time-correlated simulator with
   common random numbers.

All runs are local; no hardware is used.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from qcgp.ddgp import (DDProblem, alap, asap, correlated_protection, fit_idle_model, fixed_pattern, model_objective,
                       solve_dd_allocation, to_schedule)
from qcgp.noisesim import Gate, GateLaw, QubitPhysics, gate, simulate, spectroscopy
from qcgp.robust import RobustSpec, solve_certified_robust

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence" / "dd_gp"
PROTOCOL = "v1"
MIN_EFFECT = 1e-6  # fidelity differences below this are ties, whatever the paired s.e.
ONE_Q, TWO_Q, BLOCK = 0.036, 0.084, 1.2
ERROR_1Q, ERROR_2Q, ALPHA = 3e-4, 3e-3, 0.5


def dagger(g):
    return Gate(g.name + "dg", g.qubits, g.matrix.conj().T, g.timed, g.block)


def circuits(seed=7):
    rng = np.random.default_rng(seed)
    ghz = [gate("h", 0)] + [gate("cx", i, i + 1) for i in range(5)]
    rep = [gate("h", 0), gate("cx", 0, 2), gate("cx", 2, 4)]
    for _ in range(2):
        rep += [gate("cx", 0, 1), gate("cx", 2, 1), gate("cx", 2, 3), gate("cx", 4, 3),
                gate("block", 1), gate("block", 3)]
    layers = []
    for layer in range(4):
        for q in range(5):
            layers.append(gate(["h", "s", "sx"][rng.integers(3)], q))
        for q in range(layer % 2, 4, 2):
            layers.append(gate("cz", q, q + 1))
        if layer == 1:
            layers.append(gate("block", 4))
    mirror = layers + [dagger(g) for g in reversed(layers)]
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import QFTGate
    logical = QuantumCircuit(4)
    logical.h(range(4))
    logical.append(QFTGate(4), range(4))
    native = transpile(logical, basis_gates=["rz", "sx", "x", "cz"], optimization_level=1, seed_transpiler=3)
    qft = []
    for ins in native.data:
        qs = tuple(native.find_bit(q).index for q in ins.qubits)
        name = ins.operation.name
        qft.append(gate("rz", *qs, theta=float(ins.operation.params[0])) if name == "rz" else gate(name, *qs))
    return {"ghz6_chain": (6, ghz), "rep3_two_cycles": (5, rep), "mirror5_with_block": (5, mirror),
            "qft4_native": (4, qft)}


def scenarios():
    heron = QubitPhysics(t1=250.0, white_dephasing=1 / 500, quasi_static=0.0, spectral_s0=0.0,
                         pulse_error=3e-4)
    return {
        "white_markov": replace(heron, white_dephasing=1 / 120),
        "one_over_f": replace(heron, spectral_s0=1e-3, spectral_beta=1.0),
        "quasi_static": replace(heron, quasi_static=0.06, spectral_s0=2e-4, spectral_beta=1.0),
        "one_over_f2": replace(heron, spectral_s0=2e-6, spectral_beta=2.0, f_min=1e-3),
        "one_over_f_bad_pulses": replace(heron, spectral_s0=1e-3, spectral_beta=1.0, pulse_error=4e-3),
    }


def heterogeneous(physics, n, seed):
    rng = np.random.default_rng(seed)
    factors = rng.uniform(0.6, 1.6, size=n)
    return [replace(physics, spectral_s0=physics.spectral_s0 * f, quasi_static=physics.quasi_static * np.sqrt(f),
                    white_dephasing=physics.white_dephasing * f) for f in factors]


def build_problem(n, gates, idle, protocol="v1"):
    """v1: gate law a/t + g t on top of simulated decoherence (double-counts decoherence
    during gates). v2: the simulator is the only source of decoherence; the truth gate
    law is the drive term a/t, and the model's g_j is the fitted idle Markov rate of the
    gate's qubits (decoherence while driven is modelled like decoherence while idle)."""
    timed = [g for g in gates if g.timed]
    base = np.array([BLOCK if g.block else (ONE_Q if len(g.qubits) == 1 else TWO_Q) for g in timed])
    error = np.array([0. if g.block else (ERROR_1Q if len(g.qubits) == 1 else ERROR_2Q) for g in timed])
    inverse, linear = ALPHA * error * base, (1 - ALPHA) * error / base
    if protocol == "v2":
        linear = np.array([0. if g.block else sum(idle[q].markov + 0.5 * idle[q].relaxation for q in g.qubits)
                           for g in timed])
    block = np.array([g.block for g in timed])
    lower = np.where(block, base, 0.5 * base)
    upper = np.where(block, base, 3.0 * base)
    return DDProblem(n, tuple(gates), base, lower, upper, inverse, linear, tuple(idle))


def policies(problem, deadline):
    out = {}
    d, s, M = asap(problem)
    out["asap"] = (d, s, M, np.zeros(len(problem.windows())), "XX")
    d, s, M = alap(problem, deadline)
    out["alap"] = (d, s, M, np.zeros(len(problem.windows())), "XX")
    out["alap_xx_qiskit_like"] = (d, s, M, fixed_pattern(problem, d, s, M, 2), "XX")
    out["alap_xy4_qiskit_like"] = (d, s, M, fixed_pattern(problem, d, s, M, 4), "XY4")
    # Original manuscript method: gate-only law, deadline, then ASAP placement.
    timed = problem.timed
    edges = []
    last = {}
    for k, i in enumerate(timed):
        for q in problem.gates[i].qubits:
            if q in last:
                edges.append((last[q], k))
            last[q] = k
    spec = RobustSpec.independent(problem.inverse, problem.linear, np.zeros((len(timed), 2, 2)))
    old = solve_certified_robust(problem.lower, problem.upper, tuple(sorted(set(edges))), spec, 0.,
                                 deadline=deadline)
    d, s, M = asap(problem, old["durations"])
    out["manuscript_gp_gate_only"] = (d, s, deadline, np.zeros(len(problem.windows())), "XX")
    for label, kwargs in (("gp_idle_aware_no_dd", dict(optimize_durations=True, allow_dd=False)),
                          ("gp_dd_fixed_durations", dict(optimize_durations=False, allow_dd=True)),
                          ("gp_dd_joint", dict(optimize_durations=True, allow_dd=True))):
        r = solve_dd_allocation(problem, deadline, **kwargs)
        out[label] = (r["durations"], r["starts"], r["makespan"], r["pulses"], "XX",
                      dict(relaxation=r["relaxation_value"], integer=r["integer_value"], gap=r["relative_gap"]))
    for source, label in (("gp_dd_fixed_durations", "gp_dd_correlated_fixed_durations"),
                          ("gp_dd_joint", "gp_dd_correlated_joint")):
        d, s, M = out[source][:3]
        out[label] = (d, s, M, correlated_protection(problem, d, s, M), "XX")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=int, default=300)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--protocol", choices=["v1", "v2"], default="v1")
    args = parser.parse_args()
    global OUT, PROTOCOL
    PROTOCOL = args.protocol
    if PROTOCOL == "v2":
        OUT = ROOT / "evidence" / "dd_gp_v2"
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(deadline="ASAP makespan at nominal durations", trajectories=args.trajectories,
                    nominal_us=dict(one_qubit=ONE_Q, two_qubit=TWO_Q, block=BLOCK),
                    gate_errors=dict(one_qubit=ERROR_1Q, two_qubit=ERROR_2Q, alpha=ALPHA),
                    spectroscopy=dict(windows=[0.5, 1, 2, 4, 8, 16], pulses=[0, 2, 4, 8, 16], trajectories=2000),
                    note="Model fitted only from single-qubit spectroscopy; evaluation uses the independent simulator.",
                    hardware_jobs=0)
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    records, fits = [], {}
    start = time.perf_counter()
    circuit_set = circuits()
    scenario_set = scenarios()
    if args.quick:
        circuit_set = {k: circuit_set[k] for k in ["ghz6_chain"]}
        scenario_set = {k: scenario_set[k] for k in ["one_over_f"]}
    for scenario, physics in scenario_set.items():
        for name, (n, gates) in circuit_set.items():
            truth = heterogeneous(physics, n, seed=11)
            idle = []
            for q, p in enumerate(truth):
                rows = spectroscopy(p, [0.5, 1, 2, 4, 8, 16], [0, 2, 4, 8, 16], trajectories=2000, seed=100 + q)
                model, cost = fit_idle_model(rows)
                idle.append(model)
                fits[f"{scenario}|{name}|q{q}"] = dict(fit=asdict(model), log_residual_cost=cost)
            problem = build_problem(n, gates, idle, PROTOCOL)
            _, _, deadline = asap(problem)
            law = GateLaw(problem.inverse, problem.linear if PROTOCOL == "v1" else np.zeros_like(problem.linear))
            for policy, item in policies(problem, deadline).items():
                d, s, M, counts, sequence = item[:5]
                schedule = to_schedule(problem, d, s, M, counts, sequence)
                fidelity = simulate(schedule, truth, law, trajectories=args.trajectories, seed=2026)
                records.append(dict(
                    scenario=scenario, circuit=name, policy=policy, makespan=float(M), deadline=float(deadline),
                    pulses=int(np.sum(counts)), model_hazard=model_objective(problem, d, s, M, counts),
                    fidelity=float(np.mean(fidelity)), fidelity_se=float(np.std(fidelity, ddof=1) / np.sqrt(len(fidelity))),
                    per_trajectory=fidelity.tolist(), certificate=item[5] if len(item) > 5 else None))
            print(f"{scenario:22s} {name:20s} done {time.perf_counter() - start:7.1f}s", flush=True)
    summary = summarise(records)
    summary["elapsed_s"] = time.perf_counter() - start
    (OUT / "fits.json").write_text(json.dumps(fits, indent=1))
    (OUT / "results.json").write_text(json.dumps(records, indent=1))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["by_case"], indent=1)[:6000])
    print(json.dumps(summary["overall"], indent=2))


def summarise(records):
    cases = sorted({(r["scenario"], r["circuit"]) for r in records})
    by_case, wins, losses, rank_corr = {}, 0, 0, []
    joint_vs_best = []
    for scenario, circuit in cases:
        rows = {r["policy"]: r for r in records if r["scenario"] == scenario and r["circuit"] == circuit}
        best_std = max(["asap", "alap", "alap_xx_qiskit_like", "alap_xy4_qiskit_like"],
                       key=lambda p: rows[p]["fidelity"])
        entry = {}
        for policy, r in rows.items():
            diff = np.array(r["per_trajectory"]) - np.array(rows[best_std]["per_trajectory"])
            entry[policy] = dict(fidelity=r["fidelity"], se=r["fidelity_se"], pulses=r["pulses"],
                                 diff_vs_best_standard=float(diff.mean()),
                                 paired_se=float(diff.std(ddof=1) / np.sqrt(len(diff))),
                                 model_hazard=r["model_hazard"])
        entry["best_standard_policy"] = best_std
        by_case[f"{scenario}|{circuit}"] = entry
        joint = entry["gp_dd_joint"]
        joint_vs_best.append(joint["diff_vs_best_standard"])
        margin = max(2 * joint["paired_se"], MIN_EFFECT)
        if joint["diff_vs_best_standard"] > margin:
            wins += 1
        elif joint["diff_vs_best_standard"] < -margin:
            losses += 1
        hazards = [rows[p]["model_hazard"] for p in rows]
        infid = [1 - rows[p]["fidelity"] for p in rows]
        rank_corr.append(float(spearmanr(hazards, infid).statistic))
    overall = dict(cases=len(cases), gp_dd_joint_significant_wins=wins, gp_dd_joint_significant_losses=losses,
                   median_diff_vs_best_standard=float(np.median(joint_vs_best)),
                   mean_diff_vs_best_standard=float(np.mean(joint_vs_best)),
                   median_model_rank_correlation=float(np.median(rank_corr)))
    for policy in ("gp_dd_fixed_durations", "gp_idle_aware_no_dd", "manuscript_gp_gate_only",
                   "gp_dd_correlated_fixed_durations", "gp_dd_correlated_joint"):
        if any(policy not in by_case[k] for k in by_case):
            continue
        diffs = [by_case[k][policy]["diff_vs_best_standard"] for k in by_case]
        ses = [by_case[k][policy]["paired_se"] for k in by_case]
        overall[f"{policy}_significant_wins"] = sum(d > max(2 * e, MIN_EFFECT) for d, e in zip(diffs, ses))
        overall[f"{policy}_significant_losses"] = sum(d < -max(2 * e, MIN_EFFECT) for d, e in zip(diffs, ses))
        overall[f"{policy}_median_diff_vs_best_standard"] = float(np.median(diffs))
    return dict(by_case=by_case, overall=overall)


if __name__ == "__main__":
    main()
