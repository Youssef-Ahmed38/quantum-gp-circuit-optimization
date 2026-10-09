"""Convex/GP duration allocation with an inverse-power gate-error law.

The exponent is declared before optimization.  Fitting it on a target circuit's
evaluation outcomes would invalidate a held-out comparison.
"""
from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from .core import critical_path


@dataclass(frozen=True)
class GeneralizedProblem:
    lower: np.ndarray
    upper: np.ndarray
    inverse: np.ndarray
    linear: np.ndarray
    edges: tuple[tuple[int, int], ...]
    makespan_weight: float
    inverse_power: float = 2.
    deadline: float | None = None

    def validate(self):
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        inverse = np.asarray(self.inverse, dtype=float)
        linear = np.asarray(self.linear, dtype=float)
        if lower.ndim != 1 or len(lower) == 0:
            raise ValueError("Expected a nonempty one-dimensional duration box")
        if any(x.shape != lower.shape or not np.isfinite(x).all()
               for x in (upper, inverse, linear)):
            raise ValueError("Parameters must be matching finite vectors")
        if np.any(lower <= 0) or np.any(upper < lower):
            raise ValueError("Duration bounds must be positive and ordered")
        if np.any(inverse < 0) or np.any(linear < 0):
            raise ValueError("Gate-error coefficients must be nonnegative")
        if not np.isfinite(self.makespan_weight) or self.makespan_weight <= 0:
            raise ValueError("Makespan weight must be positive")
        if not np.isfinite(self.inverse_power) or self.inverse_power <= 0:
            raise ValueError("Inverse power must be positive")
        critical_path(lower, self.edges)
        if self.deadline is not None:
            if not np.isfinite(self.deadline) or self.deadline <= 0:
                raise ValueError("Deadline must be positive and finite")
            if critical_path(lower, self.edges).makespan > self.deadline + 1e-9:
                raise ValueError("Deadline is shorter than the minimum critical path")


def solve_generalized(problem: GeneralizedProblem):
    """Return a verified DGP optimum for the declared positive-power law."""
    problem.validate()
    lower = np.asarray(problem.lower, dtype=float)
    upper = np.asarray(problem.upper, dtype=float)
    inverse = np.asarray(problem.inverse, dtype=float)
    linear = np.asarray(problem.linear, dtype=float)
    scale = float(np.max(upper))
    x = cp.Variable(len(lower), pos=True)
    finish = cp.Variable(len(lower), pos=True)
    makespan = cp.Variable(pos=True)
    constraints = [lower / scale <= x, x <= upper / scale,
                   x <= finish, finish <= makespan]
    constraints += [(finish[i] + x[j]) / finish[j] <= 1
                    for i, j in problem.edges]
    if problem.deadline is not None:
        constraints.append(makespan <= problem.deadline / scale)
    # Zero is not a positive monomial in CVXPY's DGP grammar. Omit zero terms.
    terms = [problem.makespan_weight * scale * makespan]
    for i in range(len(lower)):
        if inverse[i] > 0:
            terms.append((inverse[i] / scale ** problem.inverse_power)
                         * cp.power(x[i], -problem.inverse_power))
        if linear[i] > 0:
            terms.append(linear[i] * scale * x[i])
    objective = sum(terms)
    model = cp.Problem(cp.Minimize(objective), constraints)
    if not model.is_dgp():
        raise RuntimeError("Generalized model is not a geometric program")
    model.solve(gp=True, solver="CLARABEL", tol_gap_abs=1e-9,
                tol_gap_rel=1e-9, tol_feas=1e-9, max_iter=400)
    if x.value is None or model.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Generalized GP did not return a solution: {model.status}")
    durations = np.asarray(x.value).reshape(-1) * scale
    durations = np.clip(durations, lower, upper)
    path = critical_path(durations, problem.edges)
    if problem.deadline is not None and path.makespan > problem.deadline + 1e-6 * max(1., problem.deadline):
        raise RuntimeError("Returned durations exceed the deadline")
    value = float(np.sum(inverse / durations ** problem.inverse_power
                         + linear * durations)
                  + problem.makespan_weight * path.makespan)
    if not np.isfinite(value) or abs(value - model.value) > 1e-6 * max(1., value):
        raise RuntimeError("Independent objective check failed")
    return dict(durations=durations.tolist(), makespan=path.makespan,
                objective=value, status=model.status, inverse_power=problem.inverse_power)


def solve_robust_generalized(problem: GeneralizedProblem, inverse_scenarios,
                             linear_scenarios):
    """Minimize worst-case declared scenario error plus makespan pressure.

    Scenario coefficients are inputs, not inferred confidence limits. Each row
    defines one complete error law across all gates; correlations can therefore
    be represented without independently maximizing every coefficient.
    """
    problem.validate()
    lower = np.asarray(problem.lower, dtype=float)
    upper = np.asarray(problem.upper, dtype=float)
    inverse = np.asarray(inverse_scenarios, dtype=float)
    linear = np.asarray(linear_scenarios, dtype=float)
    if (inverse.ndim != 2 or inverse.shape != linear.shape or
            inverse.shape[1] != len(lower) or inverse.shape[0] == 0 or
            not np.isfinite(inverse).all() or not np.isfinite(linear).all() or
            np.any(inverse < 0) or np.any(linear < 0) or
            np.any((inverse + linear).sum(axis=1) <= 0)):
        raise ValueError("Nonnegative, nonzero matching scenario matrices required")
    scale = float(np.max(upper))
    x = cp.Variable(len(lower), pos=True)
    finish = cp.Variable(len(lower), pos=True)
    makespan = cp.Variable(pos=True)
    worst_error = cp.Variable(pos=True)
    constraints = [lower / scale <= x, x <= upper / scale,
                   x <= finish, finish <= makespan]
    constraints += [(finish[i] + x[j]) / finish[j] <= 1
                    for i, j in problem.edges]
    if problem.deadline is not None:
        constraints.append(makespan <= problem.deadline / scale)
    for scenario in range(inverse.shape[0]):
        terms = []
        for gate in range(len(lower)):
            if inverse[scenario, gate] > 0:
                terms.append(inverse[scenario, gate] / scale ** problem.inverse_power
                             * cp.power(x[gate], -problem.inverse_power))
            if linear[scenario, gate] > 0:
                terms.append(linear[scenario, gate] * scale * x[gate])
        constraints.append(sum(terms) / worst_error <= 1)
    model = cp.Problem(cp.Minimize(worst_error + problem.makespan_weight * scale * makespan),
                       constraints)
    if not model.is_dgp():
        raise RuntimeError("Robust scenario model is not a geometric program")
    model.solve(gp=True, solver="CLARABEL", tol_gap_abs=1e-9,
                tol_gap_rel=1e-9, tol_feas=1e-9, max_iter=400)
    if x.value is None or model.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Robust GP did not return a solution: {model.status}")
    durations = np.clip(np.asarray(x.value).reshape(-1) * scale, lower, upper)
    path = critical_path(durations, problem.edges)
    scenario_errors = np.sum(inverse / durations[None, :] ** problem.inverse_power
                             + linear * durations[None, :], axis=1)
    robust_error = float(np.max(scenario_errors))
    value = robust_error + problem.makespan_weight * path.makespan
    if (problem.deadline is not None and path.makespan > problem.deadline + 1e-6 * max(1., problem.deadline)):
        raise RuntimeError("Robust duration solution exceeds deadline")
    if (not np.isfinite(value) or abs(value - model.value) > 1e-6 * max(1., value) or
            robust_error > float(worst_error.value) + 1e-6 * max(1., robust_error)):
        raise RuntimeError("Independent robust objective check failed")
    return dict(durations=durations.tolist(), makespan=path.makespan,
                scenario_errors=scenario_errors.tolist(), worst_error=robust_error,
                objective=float(value), status=model.status,
                inverse_power=problem.inverse_power)


def dual_duration_response(inverse, linear, pressure, lower, upper, power):
    """Coordinate minimizer of A/t^p + (C+rho)t on the duration box."""
    inverse, linear, pressure, lower, upper = (
        np.asarray(x, dtype=float) for x in (inverse, linear, pressure, lower, upper))
    if any(x.shape != inverse.shape for x in (linear, pressure, lower, upper)):
        raise ValueError("Matching coefficient vectors required")
    if power <= 0 or np.any(inverse < 0) or np.any(linear + pressure < 0):
        raise ValueError("Positive power and nonnegative coefficients required")
    denominator = linear + pressure
    result = np.full_like(inverse, upper)
    mask = denominator > 0
    result[mask] = np.power(power * inverse[mask] / denominator[mask],
                            1 / (power + 1))
    return np.clip(result, lower, upper)
