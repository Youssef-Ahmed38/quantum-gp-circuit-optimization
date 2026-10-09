"""Independent supporting-plane certificates for duration-allocation results.

The bound uses SciPy/HiGHS on a linear timing polytope. It does not call the
CVXPY model, its solver, or the project's critical-path implementation.
"""

import numpy as np
from scipy.optimize import linprog


def certify_duration_solution(problem, durations):
    """Bound a proposed convex objective between a HiGHS LP and its value.

    The objective is sum_i A_i/t_i**p + C_i*t_i + w*m, where m is the DAG
    makespan. A supporting hyperplane at the proposed duration vector is a
    global lower bound on this convex objective over the full timing polytope.
    """
    problem.validate()
    lower = np.asarray(problem.lower, dtype=float)
    upper = np.asarray(problem.upper, dtype=float)
    inverse = np.asarray(problem.inverse, dtype=float)
    linear = np.asarray(problem.linear, dtype=float)
    duration = np.asarray(durations, dtype=float)
    n = len(lower)
    if (duration.shape != (n,) or not np.isfinite(duration).all() or
            np.any(duration < lower - 1e-8) or np.any(duration > upper + 1e-8)):
        raise ValueError("Durations must be finite and within the declared box")

    # Independent longest-path implementation: the input edges are topological.
    finish = duration.copy()
    for parent, child in problem.edges:
        finish[child] = max(finish[child], finish[parent] + duration[child])
    makespan = float(max(finish))
    if problem.deadline is not None and makespan > problem.deadline + 1e-7:
        raise ValueError("Proposed durations exceed the deadline")

    p = float(problem.inverse_power)
    weight = float(problem.makespan_weight)
    error = float(np.sum(inverse / duration ** p + linear * duration))
    objective = error + weight * makespan
    gradient = -p * inverse / duration ** (p + 1) + linear

    # LP variables: [durations, gate finish times, makespan].
    rows, rhs = [], []
    for i in range(n):
        row = np.zeros(2 * n + 1)
        row[i], row[n + i] = 1., -1.
        rows.append(row)
        rhs.append(0.)
        row = np.zeros(2 * n + 1)
        row[n + i], row[-1] = 1., -1.
        rows.append(row)
        rhs.append(0.)
    for parent, child in problem.edges:
        row = np.zeros(2 * n + 1)
        row[n + parent], row[child], row[n + child] = 1., 1., -1.
        rows.append(row)
        rhs.append(0.)
    coefficients = np.zeros(2 * n + 1)
    coefficients[:n] = gradient
    coefficients[-1] = weight
    bounds = [(float(a), float(b)) for a, b in zip(lower, upper)]
    bounds += [(0., None)] * n
    bounds.append((0., None if problem.deadline is None else float(problem.deadline)))
    result = linprog(coefficients, A_ub=np.asarray(rows), b_ub=np.asarray(rhs),
                     bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"Supporting-plane LP failed: {result.message}")
    # f(y) >= f(x) + grad f(x)·(y-x). The finish-time gradient is zero.
    lower_bound = objective - float(np.dot(gradient, duration)) - weight * makespan
    lower_bound += float(result.fun)
    gap = objective - lower_bound
    if gap < -1e-7 * max(1., abs(objective)):
        raise RuntimeError("Independent lower bound exceeds the proposed value")
    return dict(objective=objective, lower_bound=float(lower_bound),
                absolute_gap=float(max(0., gap)), makespan=makespan,
                error_proxy=error, lp_status=int(result.status))
