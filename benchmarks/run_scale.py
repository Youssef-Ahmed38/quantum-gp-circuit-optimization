"""C8 study: certified solves on very large circuit DAGs.

Primary: sparse vectorised interior point (Clarabel primal-dual certificate).
Independent check: the path-flow dual value after 500 Frank-Wolfe steps is a
valid lower bound, so (IP primal - FW dual) / IP primal bounds suboptimality.

Synthetic random circuits (no quantum simulation is needed for timing). The
level-vectorised solver is cross-checked against the reference implementation
and the interior-point GP on instances small enough for both.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from qcgp.pathflow import optimize_pathflow
from qcgp.scale import fast_pathflow, random_circuit_dag, sparse_interior_point
from qcgp.timing import TimingModel, optimize_timing, reference_metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence" / "scale"


def coefficients(arity, seed):
    rng = np.random.default_rng(seed)
    base = np.where(arity == 1, 0.036, 0.084)
    error = np.where(arity == 1, 3e-4, 3e-3) * rng.uniform(0.5, 2., len(arity))
    a = 0.5 * error * base * rng.uniform(1., 4., len(arity))   # optimum 1-2x nominal
    c = 0.5 * error / base
    return base, 0.5 * base, 3. * base, a, c


def balanced_weight(base, a, c, edges):
    """Makespan weight equal to total nominal gate error per unit nominal makespan."""
    from qcgp.scale import LevelOracle
    span, _ = LevelOracle(len(base), edges).longest_path(base)
    return float(np.sum(a / base + c * base) / span)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    # Cross-check on moderate instances.
    for n, depth in ((20, 10), (50, 20)):
        arity, edges = random_circuit_dag(n, depth, seed=n)
        base, lower, upper, a, c = coefficients(arity, n)
        b = balanced_weight(base, a, c, edges)
        model = TimingModel(base, lower, upper, a, c, np.ones(len(base)), b, tuple((int(i), int(j)) for i, j in edges))
        ref = np.array([1., 1., 1.])
        reference = optimize_pathflow(model, (1, 0, 0), references=ref, relative_tolerance=1e-7)
        gp = optimize_timing(model, (1, 0, 0), references=ref)
        fast = fast_pathflow(lower, upper, a, c, b, edges, tolerance=1e-7)
        rows.append(dict(kind="cross_check", qubits=n, gates=len(base), edges=len(edges),
                         fast_primal=fast["primal"], reference_primal=reference.timing.objective,
                         gp_objective=gp.objective,
                         max_relative_difference=float(max(abs(fast["primal"] - gp.objective),
                                                           abs(reference.timing.objective - gp.objective)) / gp.objective),
                         fast_s=fast["elapsed_s"], reference_s=reference.timing.elapsed, gp_s=gp.elapsed))
        print(rows[-1], flush=True)
    for n, depth in ((100, 50), (500, 60), (1000, 80), (2000, 100)):
        arity, edges = random_circuit_dag(n, depth, seed=n)
        base, lower, upper, a, c = coefficients(arity, n)
        b = balanced_weight(base, a, c, edges)
        ip = sparse_interior_point(lower, upper, a, c, b, edges)
        fast = fast_pathflow(lower, upper, a, c, b, edges, tolerance=1e-9, max_iterations=500)
        rows.append(dict(kind="scale", qubits=n, depth=depth, gates=len(base), edges=len(edges),
                         ip_status=ip["status"], ip_primal=ip["primal"], ip_build_s=ip["build_s"],
                         ip_solve_s=ip["solve_s"], elapsed_s=ip["elapsed_s"],
                         fw_dual_lower_bound=fast["dual"], fw_primal_500=fast["primal"], fw_s=fast["elapsed_s"],
                         relative_gap=(ip["primal"] - fast["dual"]) / ip["primal"],
                         makespan_us=ip["makespan"], makespan_weight=b))
        print(rows[-1], flush=True)
    scale = [r for r in rows if r["kind"] == "scale"]
    gates = np.array([r["gates"] for r in scale], dtype=float)
    seconds = np.array([r["elapsed_s"] for r in scale])
    exponent = float(np.polyfit(np.log(gates), np.log(seconds), 1)[0])
    summary = dict(rows=rows, empirical_time_exponent_in_gates=exponent, hardware_jobs=0)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
