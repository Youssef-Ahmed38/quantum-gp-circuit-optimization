"""Deterministic free-solver validation of fixed-DAG GP global optima."""

import json
from pathlib import Path

import cvxpy
import numpy as np
import scipy

from qcgp.generalized import GeneralizedProblem, solve_generalized
from qcgp.validation import certify_duration_solution


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evidence/generalized_certificate/results.json"
SEED = 20261010


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for count in (3, 5, 8):
        for instance in range(4):
            lower = rng.uniform(.7, 1.2, count)
            upper = lower + rng.uniform(1.5, 3., count)
            inverse = rng.uniform(.5, 4., count)
            linear = rng.uniform(.02, .4, count)
            edges = tuple((i, j) for i in range(count) for j in range(i + 1, count)
                          if rng.random() < .35)
            power = (1., 2.)[instance % 2]
            weight = float(rng.uniform(.1, .8))
            # A deadline is feasible by construction and active for some cases.
            def longest(values):
                finish = values.copy()
                for parent, child in edges:
                    finish[child] = max(finish[child], finish[parent] + values[child])
                return float(max(finish))
            deadline = None if instance < 2 else (longest(lower) + longest(upper)) / 2
            problem = GeneralizedProblem(lower, upper, inverse, linear, edges,
                                         weight, power, deadline)
            solution = solve_generalized(problem)
            certificate = certify_duration_solution(problem, solution["durations"])
            tolerance = 2e-5 * max(1., certificate["objective"])
            if certificate["absolute_gap"] > tolerance:
                raise RuntimeError(f"Instance {count}-{instance} exceeded certificate tolerance")
            if abs(solution["objective"] - certificate["objective"]) > tolerance:
                raise RuntimeError("Independent objective does not match solver report")
            rows.append(dict(gates=count, instance=instance, inverse_power=power,
                             deadline=deadline, edges=[list(e) for e in edges],
                             lower=lower.tolist(), upper=upper.tolist(),
                             inverse=inverse.tolist(), linear=linear.tolist(),
                             makespan_weight=weight, durations=solution["durations"],
                             solver_status=solution["status"], **certificate))
    payload = dict(method="Independent supporting-plane LP lower bounds for convex GP objective",
                   all_free_local=True, hardware_jobs=0, seed=SEED,
                   cvxpy_version=cvxpy.__version__, scipy_version=scipy.__version__,
                   instances=rows,
                   max_absolute_gap=max(row["absolute_gap"] for row in rows),
                   max_relative_gap=max(row["absolute_gap"] / row["objective"] for row in rows))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(instances=len(rows),
                          max_absolute_gap=payload["max_absolute_gap"],
                          max_relative_gap=payload["max_relative_gap"]), indent=2))


if __name__ == "__main__":
    main()
