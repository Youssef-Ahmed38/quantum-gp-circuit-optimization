"""C4: certified robust duration allocation (exact GP).

Two guarantees, both conditional on the declared error-law family:

* Calibration coverage. Gate laws ``A_g alpha_j / t^p + G_g beta_j t`` have
  Gaussian estimates of ``theta_g = (A_g, G_g)`` with covariance ``S_g``. The
  worst case over the joint ellipsoid ``sum_g (theta_g - hat)' S_g^-1 (.) <= kappa^2``
  is ``nominal + kappa * sqrt(sum_g v_g' S_g v_g)``. With ``S_g <= c diag(S_g)``
  the square root bounds a posynomial, so the robust program is an exact GP.
  Choosing ``kappa`` by :func:`coverage_radius` makes the returned certificate
  hold with probability at least ``1 - delta`` simultaneously for all schedules.

* Baseline dominance. Constraints ``E_s(t) <= E_s(t_base)`` for declared laws
  ``E_s`` keep the problem feasible (the baseline satisfies them) and, because
  each ``E_s`` is linear in its coefficients, the result never loses to the
  baseline on any law in the convex hull of the declared laws.

Neither guarantee extends to laws outside the declared family.
"""
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from scipy.stats import chi2

from .core import critical_path
from .generalized import GeneralizedProblem


def coverage_radius(delta, parameters):
    """Scheffe radius for a ``parameters``-dimensional Gaussian ellipsoid."""
    if not 0 < delta < 1 or int(parameters) < 1:
        raise ValueError("Need 0 < delta < 1 and a positive parameter count")
    return float(np.sqrt(chi2.ppf(1 - delta, int(parameters))))


def fit_duration_law(durations, errors, sigma, power=1.):
    """Weighted least squares for ``e(t) = A t^-p + G t`` and its covariance.

    The unconstrained Gaussian estimate is returned; clipping negative values
    would bias the covariance that the coverage statement relies on.
    """
    t = np.asarray(durations, dtype=float)
    e = np.asarray(errors, dtype=float)
    s = np.asarray(sigma, dtype=float) * np.ones_like(e)
    if t.ndim != 1 or t.shape != e.shape or len(t) < 2 or np.any(t <= 0) or np.any(s <= 0):
        raise ValueError("Need matching positive durations, errors and sigmas")
    design = np.column_stack([t ** -power, t]) / s[:, None]
    covariance = np.linalg.inv(design.T @ design)
    theta = covariance @ design.T @ (e / s)
    return theta, covariance


def law_value(durations, law):
    """Evaluate a declared law ``[(coefficients, exponent), ...]``."""
    t = np.asarray(durations, dtype=float)
    return float(sum(np.sum(np.asarray(c, dtype=float) * t ** e) for c, e in law))


def _terms(x, scale, coefficients, exponent):
    return [value * scale[j] ** exponent * cp.power(x[j], exponent)
            for j, value in enumerate(np.asarray(coefficients, dtype=float)) if value > 0]


def _solve_gp(model):
    """Clarabel with tight, then relaxed, tolerances; SCS as a last resort."""
    attempts = [dict(solver="CLARABEL", tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9, max_iter=500),
                dict(solver="CLARABEL", tol_gap_abs=1e-7, tol_gap_rel=1e-7, tol_feas=1e-8, max_iter=1000),
                dict(solver="SCS", eps=1e-8, max_iters=20000)]
    for options in attempts:
        try:
            model.solve(gp=True, **options)
        except cp.error.SolverError:
            continue
        if model.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return options["solver"]
    raise RuntimeError(f"All free GP solvers failed: {model.status}")


@dataclass(frozen=True)
class RobustSpec:
    """Calibration uncertainty for grouped duration laws.

    ``groups[j]`` names the parameter group of gate ``j``; ``alpha``/``beta`` are
    known per-gate multipliers; ``theta`` holds group estimates ``(A_g, G_g)``
    and ``covariance`` their 2x2 covariances.
    """
    groups: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    theta: np.ndarray
    covariance: np.ndarray

    @classmethod
    def independent(cls, inverse, linear, covariance):
        m = len(inverse)
        return cls(np.arange(m), np.ones(m), np.ones(m),
                   np.column_stack([inverse, linear]), np.asarray(covariance, dtype=float))

    def validate(self, m):
        g = np.asarray(self.groups)
        count = len(self.theta)
        if g.shape != (m,) or g.min() < 0 or g.max() >= count:
            raise ValueError("Every gate needs a valid group")
        if np.asarray(self.covariance).shape != (count, 2, 2):
            raise ValueError("Need one 2x2 covariance per group")
        if np.any(np.linalg.eigvalsh(self.covariance) < -1e-12):
            raise ValueError("Covariances must be positive semidefinite")
        if np.any(np.asarray(self.alpha) < 0) or np.any(np.asarray(self.beta) < 0):
            raise ValueError("Multipliers must be nonnegative")

    def nominal(self):
        g = np.asarray(self.groups)
        theta = np.maximum(np.asarray(self.theta, dtype=float), 0.)
        return self.alpha * theta[g, 0], self.beta * theta[g, 1]

    def spread(self, durations, power):
        """Exact ``sum_g v_g' S_g v_g`` (no diagonal relaxation)."""
        t = np.asarray(durations, dtype=float)
        total = 0.
        for g, cov in enumerate(self.covariance):
            members = self.groups == g
            v = np.array([np.sum(self.alpha[members] * t[members] ** -power),
                          np.sum(self.beta[members] * t[members])])
            total += float(v @ cov @ v)
        return total


def solve_certified_robust(lower, upper, edges, spec: RobustSpec, kappa, *, power=1.,
                           deadline=None, makespan_weight=1e-6, dominance_laws=(),
                           baseline=None):
    """Exact-GP worst-case allocation; see the module docstring."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    m = len(lower)
    spec.validate(m)
    GeneralizedProblem(lower, upper, np.ones(m), np.ones(m), tuple(edges),
                       max(makespan_weight, 1e-12), power, deadline).validate()
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and nonnegative")
    if dominance_laws and baseline is None:
        raise ValueError("Dominance constraints need a baseline schedule")
    if baseline is not None:
        baseline = np.asarray(baseline, dtype=float)
        if baseline.shape != (m,) or np.any(baseline < lower - 1e-9) or np.any(baseline > upper + 1e-9):
            raise ValueError("Baseline must lie in the duration box")
        if deadline is not None and critical_path(baseline, edges).makespan > deadline * (1 + 1e-9):
            raise ValueError("Baseline must meet the deadline")
    cov = np.asarray(spec.covariance, dtype=float)
    factor = 1. if np.allclose(cov[:, 0, 1], 0) else 2.
    inverse, linear = spec.nominal()
    # Per-gate duration scaling and a global time scale keep the GP well
    # conditioned when 1q and 2q gate parameters differ by orders of magnitude.
    scale = np.sqrt(lower * upper)
    horizon = critical_path(scale, edges).makespan
    unit = float(np.sum(inverse * scale ** -power + linear * scale))
    unit = unit if unit > 0 else 1.
    inverse, linear, cov = inverse / unit, linear / unit, cov / unit ** 2
    x = cp.Variable(m, pos=True)
    finish = cp.Variable(m, pos=True)
    makespan = cp.Variable(pos=True)
    radius = cp.Variable(pos=True)
    width = scale / horizon
    constraints = [lower / scale <= x, x <= upper / scale,
                   cp.multiply(width, x) <= finish, finish <= makespan]
    constraints += [(finish[i] + width[j] * x[j]) / finish[j] <= 1 for i, j in edges]
    if deadline is not None:
        constraints.append(makespan <= deadline / horizon)
    variance = []
    for g in range(len(cov)):
        members = np.flatnonzero(spec.groups == g)
        inv = [spec.alpha[j] * scale[j] ** -power * cp.power(x[j], -power)
               for j in members if spec.alpha[j] > 0]
        lin = [spec.beta[j] * scale[j] * x[j] for j in members if spec.beta[j] > 0]
        if cov[g, 0, 0] > 0 and inv:
            variance.append(factor * cov[g, 0, 0] * cp.power(sum(inv), 2))
        if cov[g, 1, 1] > 0 and lin:
            variance.append(factor * cov[g, 1, 1] * cp.power(sum(lin), 2))
    objective = _terms(x, scale, inverse, -power) + _terms(x, scale, linear, 1.)
    if kappa > 0 and variance:
        constraints.append(sum(variance) / cp.power(radius, 2) <= 1)
        objective.append(kappa * radius)
    if makespan_weight > 0:
        objective.append(makespan_weight / unit * horizon * makespan)
    for law in dominance_laws:
        bound = law_value(baseline, law)
        law_terms = []
        for coefficients, exponent in law:
            law_terms += _terms(x, scale, np.asarray(coefficients, dtype=float) / bound, exponent)
        if law_terms and bound > 0:
            constraints.append(sum(law_terms) <= 1)
    model = cp.Problem(cp.Minimize(sum(objective)), constraints)
    if not model.is_dgp():
        raise RuntimeError("Certified robust model is not a geometric program")
    _solve_gp(model)
    if x.value is None or model.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Certified robust GP failed: {model.status}")
    durations = np.clip(np.asarray(x.value).reshape(-1) * scale, lower, upper)
    inverse, linear, cov = inverse * unit, linear * unit, cov * unit ** 2
    nominal = float(np.sum(inverse * durations ** -power + linear * durations))
    exact_radius = float(np.sqrt(spec.spread(durations, power)))
    slack = [law_value(baseline, law) - law_value(durations, law) for law in dominance_laws]
    span = critical_path(durations, edges).makespan
    repaired = False
    if deadline is not None and span > deadline * (1 + 1e-9):
        # Solver residue: contract toward the lower box (which meets the deadline).
        if span > deadline * (1 + 1e-3):
            raise RuntimeError("Returned schedule exceeds the deadline")
        low, high = 0., 1.
        for _ in range(60):
            mid = (low + high) / 2
            trial = lower + mid * (durations - lower)
            if critical_path(trial, edges).makespan <= deadline:
                low = mid
            else:
                high = mid
        durations = lower + low * (durations - lower)
        span = critical_path(durations, edges).makespan
        repaired = True
        nominal = float(np.sum(inverse * durations ** -power + linear * durations))
        exact_radius = float(np.sqrt(spec.spread(durations, power)))
        slack = [law_value(baseline, law) - law_value(durations, law) for law in dominance_laws]
    for law, s in zip(dominance_laws, slack):
        if s < -1e-6 * law_value(baseline, law):
            raise RuntimeError("Returned schedule violates a dominance constraint")
    diagonal = 0.
    for g in range(len(cov)):
        members = spec.groups == g
        diagonal += cov[g, 0, 0] * np.sum(spec.alpha[members] * durations[members] ** -power) ** 2
        diagonal += cov[g, 1, 1] * np.sum(spec.beta[members] * durations[members]) ** 2
    certified = nominal + kappa * float(np.sqrt(factor * diagonal))
    if nominal + kappa * exact_radius > certified * (1 + 1e-9):
        raise RuntimeError("Diagonal relaxation failed to bound the exact worst case")
    return dict(durations=durations, makespan=span, nominal_error=nominal,
                exact_worst_case=nominal + kappa * exact_radius, certified_error=certified,
                kappa=float(kappa), dominance_slack=slack, status=model.status, repaired=repaired)
