"""C7: latency-fidelity Pareto frontier and its dual price.

For ``E*(D) = min { sum_j a_j tau_j^-p + g_j tau_j : makespan(tau) <= D, l <= tau <= u }``
the function ``d -> log E*(e^d)`` is convex (GP sensitivity: the optimal value of
the log-transformed convex program is convex in the log right-hand side), and
its slope is ``-nu(D)``, the optimal multiplier of the log-form deadline
constraint. ``nu`` is the *fidelity price*: the relative error reduction bought
by one unit of relative latency, which a compiler can threshold.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np

from .core import critical_path


def frontier_point(lower, upper, inverse, linear, edges, deadline, power=1.):
    m = len(lower)
    scale = np.sqrt(lower * upper)
    horizon = critical_path(scale, edges).makespan
    width = scale / horizon
    x = cp.Variable(m, pos=True)
    finish = cp.Variable(m, pos=True)
    end = cp.Variable(pos=True)
    unit = float(np.sum(inverse * scale ** -power + linear * scale))
    terms = [inverse[j] / unit * scale[j] ** -power * cp.power(x[j], -power) for j in range(m) if inverse[j] > 0]
    terms += [linear[j] / unit * scale[j] * x[j] for j in range(m) if linear[j] > 0]
    deadline_constraint = end <= deadline / horizon
    constraints = [lower / scale <= x, x <= upper / scale, cp.multiply(width, x) <= finish,
                   finish <= end, deadline_constraint]
    constraints += [(finish[i] + width[j] * x[j]) / finish[j] <= 1 for i, j in edges]
    problem = cp.Problem(cp.Minimize(sum(terms)), constraints)
    problem.solve(gp=True, solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
    tau = np.clip(x.value * scale, lower, upper)
    return dict(deadline=float(deadline), error=float(problem.value) * unit, durations=tau,
                price=float(deadline_constraint.dual_value), status=problem.status)


def pareto_frontier(lower, upper, inverse, linear, edges, deadlines, power=1.):
    points = [frontier_point(lower, upper, inverse, linear, edges, d, power) for d in deadlines]
    logd = np.log([p["deadline"] for p in points])
    loge = np.log([p["error"] for p in points])
    second = np.diff(loge, 2) / (np.diff(logd)[1:] * np.diff(logd)[:-1]) if len(points) > 2 else np.array([])
    slope = np.gradient(loge, logd)
    return dict(points=points, log_deadline=logd.tolist(), log_error=loge.tolist(),
                finite_difference_slope=slope.tolist(), dual_price=[p["price"] for p in points],
                min_second_difference=float(second.min()) if len(second) else None)
