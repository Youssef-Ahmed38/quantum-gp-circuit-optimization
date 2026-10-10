"""C5 study: anisotropic, state-aware hazards predict basis-dependent rankings.

Part 1 (physics). A single-qubit Lindblad + classical-noise simulator plays
the four native-X policies of the manuscript's Heron experiment. An
anisotropic, state-aware hazard model is calibrated ONLY from generic
characterisation runs (idle CPMG on |+>, T1 on |1>, spin-lock on |+>, driven
|0>) and then predicts each policy's survival in both bases. An isotropic
single-scalar model is fitted to the same calibration data for comparison.
The measured Heron ranking (Table X) is listed as a retrospective check only.

Part 2 (compilation). On Clifford workloads with Z-biased gate noise, the
anisotropic GP is compared with an isotropic GP of equal total intensity: the
isotropic plan is scored against the anisotropic truth.

All runs are local.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from qcgp.anisotropic import (SingleQubitPhysics, build_model, isotropic_split, optimize_anisotropic,
                              run_x_sequence, simulate_drive)
from qcgp.ddgp import fit_idle_model
from qcgp.observables import snapshot_model
from run_observables import workload

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence" / "anisotropic"
GATES, SLOW, FAST = 48, 1.0, 0.036
POLICIES = ["slow", "distributed", "tail", "dd_tail"]
HERON_TABLE_X = {"X": {"slow": 0.9010, "distributed": 0.8826, "tail": 0.3052, "dd_tail": 0.8586},
                 "Z": {"slow": 0.8296, "distributed": 0.8013, "tail": 0.9849, "dd_tail": 0.8753}}


def hazard(survival):
    return -math.log(max(2 * survival - 1, 1e-9))


def pulses(times, centres):
    on = np.zeros_like(times, dtype=bool)
    for c in centres:
        on |= (times >= c - FAST / 2) & (times < c + FAST / 2)
    return np.where(on, np.pi / FAST, 0.)


def calibrate(physics, trajectories):
    rows = []
    for w in (4., 16., 32., 48.):
        for n in (0, 2, 4, 8):
            centres = [w * (2 * i + 1) / (2 * n) for i in range(n)]
            s = simulate_drive(lambda t, c=centres: pulses(t, c), w, "+", physics,
                               trajectories=trajectories, seed=11)
            rows.append(dict(window=w, pulses=n, hazard=hazard(s)))
    idle_x, _ = fit_idle_model(rows)
    t1_rate = np.mean([hazard(simulate_drive(lambda t: np.zeros_like(t), w, "1", physics,
                                             trajectories=trajectories, seed=12)) / w for w in (16., 48.)])
    lock = np.mean([hazard(simulate_drive(lambda t: np.full_like(t, np.pi / SLOW), w, "+", physics,
                                          trajectories=trajectories, seed=13)) / w for w in (8., 24., 48.)])
    driven_z = np.mean([hazard(simulate_drive(lambda t: np.full_like(t, np.pi / SLOW), w, "0", physics,
                                              trajectories=trajectories, seed=14)) / w for w in (8., 24., 48.)])
    return dict(idle_x=idle_x, t1_rate=float(t1_rate), lock_rate=float(lock), driven_z_rate=float(driven_z),
                calibration_rows=rows)


def idle_hazard(model, w, n):
    return model.amplitude * w ** (1 + model.beta) * (n + 1) ** (-model.beta) + model.markov * w + model.pulse_error * n


def predict(cal):
    total, burst = GATES * SLOW, GATES * FAST
    tail = total - burst
    m = cal["idle_x"]
    x = {"slow": cal["lock_rate"] * total,
         "distributed": idle_hazard(m, total, GATES),
         "tail": idle_hazard(m, tail, 0),
         "dd_tail": idle_hazard(m, tail, 2)}
    # State-aware T1: hazard accrues only while the ideal state is |1>.
    z = {"slow": cal["driven_z_rate"] * total,
         "distributed": cal["t1_rate"] * (total - burst) / 2,
         "tail": cal["t1_rate"] * burst / 2,
         "dd_tail": cal["t1_rate"] * (burst / 2 + tail / 2)}
    survival = {b: {p: 0.5 * (1 + math.exp(-h[p])) for p in POLICIES} for b, h in (("X", x), ("Z", z))}
    # Isotropic single scalar per policy: average of the two basis hazards.
    iso = {p: 0.5 * (x[p] + z[p]) for p in POLICIES}
    isotropic = {b: {p: 0.5 * (1 + math.exp(-iso[p])) for p in POLICIES} for b in ("X", "Z")}
    return survival, isotropic


def order(values):
    return sorted(values, key=values.get, reverse=True)


def part_one(trajectories=300):
    physics = SingleQubitPhysics()
    truth = {b: {p: run_x_sequence(p, b, GATES, physics, slow=SLOW, fast=FAST,
                                   trajectories=trajectories, seed=99) for p in POLICIES} for b in ("X", "Z")}
    cal = calibrate(physics, trajectories)
    anisotropic, isotropic = predict(cal)
    result = dict(physics=physics.__dict__, simulated_truth=truth,
                  calibration={k: (v.__dict__ if hasattr(v, "__dict__") else v) for k, v in cal.items()},
                  anisotropic_prediction=anisotropic, isotropic_prediction=isotropic,
                  heron_table_x_retrospective=HERON_TABLE_X)
    for label, pred in (("anisotropic", anisotropic), ("isotropic", isotropic)):
        result[f"{label}_best_policy_correct"] = {b: order(pred[b])[0] == order(truth[b])[0] for b in ("X", "Z")}
        result[f"{label}_full_order_correct"] = {b: order(pred[b]) == order(truth[b]) for b in ("X", "Z")}
        result[f"{label}_best_matches_heron"] = {b: order(pred[b])[0] == order(HERON_TABLE_X[b])[0] for b in ("X", "Z")}
    result["truth_order"] = {b: order(truth[b]) for b in ("X", "Z")}
    result["heron_order"] = {b: order(HERON_TABLE_X[b]) for b in ("X", "Z")}
    return result


def part_two(bias=10.0):
    snapshot = json.loads((ROOT / "evidence/observable_study/props_torino.json").read_text())
    rows = []
    for n in (8, 16, 32):
        for seed in (11, 29):
            circuit, mapping = workload(snapshot, n, seed, "graph")
            iso_model = snapshot_model(circuit, mapping, snapshot, 0.5)
            totals_a = iso_model.a
            totals_g = iso_model.gamma
            # Per-Pauli weights: Z-type errors `bias` times more likely than X/Y on each qubit.
            aniso_a, aniso_g, iso_a, iso_g = [], [], [], []
            timed = [i for i, ins in enumerate(circuit.data) if ins.operation.name != "rz"]
            for j, index in enumerate(timed):
                k = len(circuit.data[index].qubits)
                weights = []
                for code in range(1, 4 ** k):
                    digits = [(code // 4 ** i) % 4 for i in range(k)]
                    weights.append(np.prod([1.0 if d == 0 else (bias if d == 3 else 1.0) for d in digits]))
                weights = np.array(weights) / np.sum(weights)
                aniso_a.append(totals_a[j] * weights)
                aniso_g.append(totals_g[j] * weights)
                ia, ig = isotropic_split(totals_a[j], totals_g[j], k)
                iso_a.append(ia)
                iso_g.append(ig)
            truth = build_model(circuit, iso_model.base, iso_model.lower, iso_model.upper, aniso_a, aniso_g)
            assumed = build_model(circuit, iso_model.base, iso_model.lower, iso_model.upper, iso_a, iso_g)
            low = float(np.min(truth.expectations(truth.lower)))
            high = float(np.min(truth.expectations(np.clip(np.sqrt(iso_model.a / iso_model.gamma),
                                                               truth.lower, truth.upper))))
            target = math.sqrt(low * high)
            plan_true = optimize_anisotropic(truth, target)
            plan_iso = optimize_anisotropic(assumed, target)
            record = dict(qubits=n, seed=seed, bias=bias, target=target,
                          anisotropic_feasible=plan_true["feasible"], isotropic_feasible=plan_iso["feasible"])
            if plan_true["feasible"]:
                record["anisotropic_makespan"] = plan_true["makespan"]
                record["anisotropic_min_true_expectation"] = float(np.min(truth.expectations(plan_true["durations"])))
            if plan_iso["feasible"]:
                actual = truth.expectations(plan_iso["durations"])
                record["isotropic_makespan"] = plan_iso["makespan"]
                record["isotropic_plan_true_violations"] = int(np.sum(actual < target * (1 - 1e-9)))
                record["isotropic_plan_min_true_expectation"] = float(np.min(actual))
            rows.append(record)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    one = part_one()
    two = part_two()
    summary = dict(part_one={k: v for k, v in one.items() if k.endswith(("correct", "heron", "order"))},
                   part_two=dict(
                       cases=len(two),
                       isotropic_plans_violating_true_targets=sum(r.get("isotropic_plan_true_violations", 0) > 0 for r in two),
                       median_anisotropic_vs_isotropic_makespan=float(np.median(
                           [r["anisotropic_makespan"] / r["isotropic_makespan"] for r in two
                            if "anisotropic_makespan" in r and "isotropic_makespan" in r]))),
                   elapsed_s=time.perf_counter() - start, hardware_jobs=0)
    (OUT / "spin_locking.json").write_text(json.dumps(one, indent=1, default=float))
    (OUT / "compilation.json").write_text(json.dumps(two, indent=1, default=float))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(one["simulated_truth"], indent=1))
    print(json.dumps(one["anisotropic_prediction"], indent=1))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
