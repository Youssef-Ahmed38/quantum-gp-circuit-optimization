"""C6: bicriteria approximation for discrete pulse libraries.

Exact selection from arbitrary finite pulse libraries under a deadline is
NP-hard (reduction from the deadline discrete time-cost trade-off problem; see
docs/THEORY.md). This module implements the polynomial-time alternative:

1. solve the continuous GP with exposure coefficients inflated to ``q g``;
2. round every duration *up* onto the grid ``l_j q^k`` (clamped at ``u_j``).

Guarantee, for any library contained in the duration box ``[l, u]``:

    makespan(rounded) <= q D     and     error(rounded) <= q OPT_library(D).
"""
from __future__ import annotations

import itertools

import numpy as np

from .core import critical_path
from .robust import RobustSpec, solve_certified_robust


def law_error(inverse, linear, tau, power=1.):
    tau = np.asarray(tau, dtype=float)
    return float(np.sum(inverse / tau ** power + linear * tau))


def geometric_round_up(tau, lower, upper, q):
    tau = np.clip(np.asarray(tau, dtype=float), lower, upper)
    steps = np.ceil(np.log(tau / lower) / np.log(q) - 1e-12)
    return np.minimum(lower * q ** np.maximum(steps, 0), upper)


def bicriteria(lower, upper, inverse, linear, edges, deadline, q=1.05, power=1.):
    """Continuous solve with ``q``-inflated exposure, then upward grid rounding."""
    m = len(lower)
    spec = RobustSpec.independent(np.asarray(inverse, float), q * np.asarray(linear, float), np.zeros((m, 2, 2)))
    relaxed = solve_certified_robust(lower, upper, edges, spec, 0., power=power, deadline=deadline)
    rounded = geometric_round_up(relaxed["durations"], lower, upper, q)
    return dict(continuous=relaxed["durations"], continuous_inflated_error=relaxed["nominal_error"],
                durations=rounded, makespan=critical_path(rounded, edges).makespan,
                error=law_error(inverse, linear, rounded, power))


def library_optimum(library, inverse, linear, edges, deadline, power=1.):
    """Exhaustive optimum over a finite per-gate library (tiny instances only)."""
    best, choice = np.inf, None
    for combo in itertools.product(*[range(len(options)) for options in library]):
        tau = np.array([library[j][k] for j, k in enumerate(combo)])
        if critical_path(tau, edges).makespan <= deadline * (1 + 1e-12):
            value = law_error(inverse, linear, tau, power)
            if value < best:
                best, choice = value, tau
    return best, choice
