"""Feasible large-circuit timing study under the declared Pauli model.

Targets are set within each instance's own attainable range. This tests solver
scaling, not calibration validity or physical-device performance.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from qcgp.core import critical_path
from qcgp.observables import (
    expectations,
    optimize_observables,
    optimal_uniform,
    snapshot_model,
)
from run_observables import workload


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "evidence/observable_study/props_torino.json"
OUT = ROOT / "evidence/feasible_scale"


def run() -> list[dict]:
    snapshot = json.loads(SNAPSHOT.read_text())
    protocol = {
        "purpose": "Exercise feasible GP solves at 32, 64, and 128 qubits",
        "sizes": [32, 64, 128],
        "seeds": [11, 29],
        "families": ["graph", "scrambled"],
        "alpha": [0.5, 0.8],
        "target_rule": "geometric mean of worst-generator expectations at the lower-duration vector and at the coordinatewise minimum-hazard vector",
        "target_source": "endogenous to the assumed model; target varies by instance",
        "snapshot_sha256": hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest(),
        "noise": "post-gate isotropic Pauli Markov channels; no idle decay or crosstalk",
        "hardware_jobs": 0,
        "paid_services": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    records = []
    for family in protocol["families"]:
        for n in protocol["sizes"]:
            for seed in protocol["seeds"]:
                circuit, mapping = workload(snapshot, n, seed, family)
                for alpha in protocol["alpha"]:
                    model = snapshot_model(circuit, mapping, snapshot, alpha)
                    tmin = np.clip(np.sqrt(model.a / model.gamma), model.lower, model.upper)
                    low = float(np.min(expectations(model, model.lower)))
                    high = float(np.min(expectations(model, tmin)))
                    if not high > low + 1e-9:
                        raise AssertionError("Target rule did not create a nontrivial interval")
                    target = float(np.sqrt(low * high))
                    start = time.perf_counter()
                    gp = optimize_observables(model, target, compute_dual=False)
                    gp_seconds = time.perf_counter() - start
                    direct = optimize_observables(model, target, method="convex", compute_dual=False)
                    if not (gp["feasible"] and direct["feasible"]):
                        raise AssertionError("A model-attainable target failed to solve")
                    if abs(gp["makespan"] - direct["makespan"]) > 2e-5 * gp["makespan"]:
                        raise AssertionError("Equivalent formulations disagree")
                    uniform = optimal_uniform(model, target)
                    if uniform is None:
                        raise AssertionError("Uniform baseline unexpectedly infeasible")
                    uniform_span = critical_path(uniform, model.edges).makespan
                    if gp["makespan"] > uniform_span + 1e-6:
                        raise AssertionError("GP is worse than a feasible uniform policy")
                    records.append({
                        "family": family, "qubits": n, "seed": seed, "alpha": alpha,
                        "timed_gates": len(model.base), "target": target,
                        "lower_duration_expectation": low,
                        "coordinatewise_min_hazard_expectation": high,
                        "gp_makespan_us": gp["makespan"],
                        "uniform_makespan_us": uniform_span,
                        "achieved_min_expectation": gp["minimum_expectation"],
                        "formulation_relative_difference": abs(gp["makespan"] - direct["makespan"]) / gp["makespan"],
                        "gp_elapsed_s": gp_seconds,
                    })
                    print(f"{family} {n}q seed={seed} alpha={alpha}: GP {gp_seconds:.3f}s", flush=True)
    (OUT / "results.json").write_text(json.dumps(records, indent=2) + "\n")
    (OUT / "environment.json").write_text(json.dumps({
        "python": platform.python_version(), "platform": platform.platform(),
        "hardware_jobs": 0, "paid_services": False,
    }, indent=2) + "\n")
    return records


if __name__ == "__main__":
    run()
