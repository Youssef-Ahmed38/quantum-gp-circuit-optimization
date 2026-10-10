"""C6 + C7 numerical checks.

C6: on random small DAGs and random finite libraries inside the duration box,
verify  makespan(rounded) <= q D  and  error(rounded) <= q * OPT_library(D).
C7: on benchmark circuits, trace the latency-fidelity frontier, verify log-log
convexity and that the finite-difference slope equals minus the dual price.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from qcgp.approximation import bicriteria, library_optimum
from qcgp.core import benchmark, compile_candidate, critical_path, topology
from qcgp.frontier import pareto_frontier
from qcgp.timing import synthetic_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence" / "theory_checks"


def random_dag(m, rng):
    edges = [(i, j) for j in range(m) for i in range(j) if rng.random() < 0.35]
    return tuple(edges)


def c6(trials=300, seed=5):
    rng = np.random.default_rng(seed)
    rows = []
    for trial in range(trials):
        m = int(rng.integers(3, 7))
        edges = random_dag(m, rng)
        lower = rng.uniform(0.5, 1.5, m)
        upper = lower * rng.uniform(2., 6., m)
        inverse = rng.uniform(0.1, 2., m)
        linear = rng.uniform(0.01, 0.5, m)
        options = int(rng.integers(2, 5))
        library = [np.sort(rng.uniform(lower[j], upper[j], options)) for j in range(m)]
        library = [np.concatenate([[lower[j]], lib]) for j, lib in enumerate(library)]  # deadline feasible
        span_low = critical_path(lower, edges).makespan
        deadline = span_low * rng.uniform(1.05, 2.5)
        q = float(rng.choice([1.02, 1.1, 1.3]))
        opt, _ = library_optimum(library, inverse, linear, edges, deadline)
        result = bicriteria(lower, upper, inverse, linear, edges, deadline, q)
        rows.append(dict(trial=trial, m=m, q=q, deadline=deadline, library_optimum=opt,
                         rounded_error=result["error"], rounded_makespan=result["makespan"],
                         error_ratio=result["error"] / opt, makespan_ratio=result["makespan"] / deadline,
                         error_bound_holds=result["error"] <= q * opt * (1 + 1e-7),
                         makespan_bound_holds=result["makespan"] <= q * deadline * (1 + 1e-7)))
    return rows


def c7():
    rows = {}
    for name, n in (("qft", 4), ("grover", 4), ("vqe_ansatz", 4), ("qaoa", 4)):
        line = topology(n, "line")
        compiled, _ = compile_candidate(benchmark(name, n, 0), line, 0)
        model = synthetic_model(compiled, line)
        a, c = model.amplitude_noise, model.gate_noise_rate
        fastest = critical_path(model.lower, model.edges).makespan
        free = critical_path(np.clip(np.sqrt(a / c), model.lower, model.upper), model.edges).makespan
        deadlines = np.geomspace(fastest * 1.001, free * 0.999, 30)
        result = pareto_frontier(model.lower, model.upper, a, c, model.edges, deadlines)
        slope = np.array(result["finite_difference_slope"])
        price = np.array(result["dual_price"])
        interior = slice(2, -2)
        rows[f"{name}_{n}"] = dict(
            deadlines_us=deadlines.tolist(), errors=[p["error"] for p in result["points"]],
            dual_price=price.tolist(), finite_difference_slope=slope.tolist(),
            min_second_difference=result["min_second_difference"],
            max_slope_plus_price_abs=float(np.max(np.abs(slope[interior] + price[interior]))),
            error_reduction_full_range=1 - result["points"][-1]["error"] / result["points"][0]["error"])
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    a = c6()
    b = c7()
    summary = dict(
        c6=dict(trials=len(a), error_bound_violations=sum(not r["error_bound_holds"] for r in a),
                makespan_bound_violations=sum(not r["makespan_bound_holds"] for r in a),
                max_error_ratio=float(max(r["error_ratio"] for r in a)),
                median_error_ratio=float(np.median([r["error_ratio"] for r in a]))),
        c7={k: dict(min_second_difference=v["min_second_difference"],
                    max_slope_plus_price_abs=v["max_slope_plus_price_abs"],
                    error_reduction_full_range=v["error_reduction_full_range"]) for k, v in b.items()},
        elapsed_s=time.perf_counter() - start, hardware_jobs=0)
    (OUT / "c6_bicriteria.json").write_text(json.dumps(a, indent=1, default=float))
    (OUT / "c7_frontier.json").write_text(json.dumps(b, indent=1, default=float))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
