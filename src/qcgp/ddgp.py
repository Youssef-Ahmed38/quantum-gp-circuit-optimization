"""C2: joint gate-duration, placement and dynamical-decoupling allocation.

Declared model (fitted from spectroscopy, never from the evaluation circuits):

* timed gate ``j``: hazard ``a_j / tau_j + g_j tau_j`` on ``[l_j, u_j]``;
* idle window ``k`` of length ``w`` on qubit ``q`` with ``n`` refocusing pulses:
  ``c_q w^(1+b_q) (n+1)^(-b_q) + eta_q w + eps_q n``  and  ``n * t_pulse <= w``.

``w`` is affine in start/duration variables, and ``w^(1+b) (n+1)^(-b)`` is the
perspective of ``x -> x^(1+b)`` evaluated at ``(w, n+1)``; it is jointly convex.
The relaxation over real ``n >= 0`` is therefore a convex program solved to
global optimality, and its value lower-bounds every integer DD assignment.
Rounding ``n`` to an even integer and re-solving yields an executable schedule
with a certified relaxation gap.

Idle windows are counted between consecutive timed gates on a qubit and from a
qubit's last gate to the common makespan. Time before a qubit's first gate is
excluded (the qubit is in its initial computational state). Idle-aware timing
contains ``M - sum(tau)`` with a negative coefficient, so it is convex but not
a geometric program in ``(tau, M)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import cvxpy as cp
import numpy as np

from .noisesim import Gate, ScheduledCircuit


@dataclass(frozen=True)
class IdleModel:
    """Fitted idle hazard for one qubit (model side)."""
    amplitude: float      # c
    beta: float           # b >= 0
    markov: float         # eta, per us (unital part: white dephasing, twirled decay)
    pulse_error: float    # eps, hazard per pulse
    relaxation: float = 0.  # non-unital T1 rate, weighted by excited population
    static: float = 0.      # quasi-static/coherent detuning: s w^2 unrefocused, 0 under a balanced block


@dataclass(frozen=True)
class DDProblem:
    n: int
    gates: tuple          # noisesim.Gate objects, program order
    base: np.ndarray      # nominal duration per timed gate (program order of timed gates)
    lower: np.ndarray
    upper: np.ndarray
    inverse: np.ndarray   # a_j
    linear: np.ndarray    # g_j
    idle: tuple           # IdleModel per qubit
    pulse_duration: float = 0.036

    @property
    def timed(self):
        return [i for i, g in enumerate(self.gates) if g.timed]

    def fixed(self):
        """Indices (timed order) whose duration is not controllable (blocks)."""
        return [k for k, i in enumerate(self.timed) if self.gates[i].block]

    def windows(self):
        """List of (qubit, previous timed-order index, next index or None)."""
        timed = self.timed
        sequence = {q: [] for q in range(self.n)}
        for k, i in enumerate(timed):
            for q in self.gates[i].qubits:
                sequence[q].append(k)
        result = []
        for q, ks in sequence.items():
            for a, b in zip(ks, ks[1:]):
                result.append((q, a, b))
            if ks:
                result.append((q, ks[-1], None))
        return result

    def excited_populations(self):
        """Ideal P(|1>) of each qubit after each timed gate (state-aware T1 weights).

        The reduced state of a qubit is unchanged by gates on other qubits, so the
        population after gate ``a`` holds for the whole idle window that follows it.
        """
        state = np.zeros(2 ** self.n, dtype=complex)
        state[0] = 1
        tensor = state.reshape((2,) * self.n)
        after = {}
        position = {i: k for k, i in enumerate(self.timed)}
        for index, g in enumerate(self.gates):
            k = len(g.qubits)
            u = g.matrix.reshape((2,) * (2 * k))
            tensor = np.tensordot(u, tensor, axes=(list(range(k, 2 * k)), list(g.qubits)))
            tensor = np.moveaxis(tensor, list(range(k)), list(g.qubits))
            if index in position:
                probabilities = np.abs(tensor) ** 2
                for q in g.qubits:
                    after[(position[index], q)] = float(np.take(probabilities, 1, axis=q).sum())
        return [after[(a, q)] for q, a, b in self.windows()]

    def validate(self):
        m = len(self.timed)
        for name in ("base", "lower", "upper", "inverse", "linear"):
            v = np.asarray(getattr(self, name), dtype=float)
            if v.shape != (m,) or not np.isfinite(v).all():
                raise ValueError(f"{name} must have one finite entry per timed gate")
        if np.any(self.lower <= 0) or np.any(self.lower > self.upper):
            raise ValueError("Duration bounds must be positive and ordered")
        if np.any(self.inverse < 0) or np.any(self.linear < 0):
            raise ValueError("Gate coefficients must be nonnegative")
        if len(self.idle) != self.n:
            raise ValueError("One idle model per qubit is required")
        for model in self.idle:
            if model.amplitude < 0 or model.beta < 0 or model.markov < 0 or model.pulse_error < 0:
                raise ValueError("Idle model parameters must be nonnegative")


def _geo_weights(beta):
    first = Fraction(1 / (1 + beta)).limit_denominator(64)
    return [first, 1 - first]


def _relaxation_weight(p1, n):
    """Excited fraction of a window: p1 without pulses; 1/2 with an even CPMG block."""
    return p1 if n <= 0 else 0.5


def _window_hazard_numeric(model: IdleModel, w, n, p1=0.5):
    w = max(w, 0.)
    value = model.markov * w + model.pulse_error * n + model.relaxation * _relaxation_weight(p1, n) * w
    if n <= 0:
        value += model.static * w ** 2
    if model.amplitude > 0:
        value += model.amplitude * w ** (1 + model.beta) * (n + 1) ** (-model.beta)
    return value


def model_objective(problem: DDProblem, durations, starts, makespan, pulses_per_window):
    """Declared-model hazard of any schedule (used for every policy)."""
    gate = np.sum(problem.inverse / durations + problem.linear * durations)
    idle = 0.
    population = problem.excited_populations()
    for k, (q, a, b) in enumerate(problem.windows()):
        end = makespan if b is None else starts[b]
        w = end - starts[a] - durations[a]
        idle += _window_hazard_numeric(problem.idle[q], w, pulses_per_window[k], population[k])
    return float(gate + idle)


def _solve(problem: DDProblem, deadline, *, optimize_durations, allow_dd, fixed_pulses=None,
           makespan_weight=0., max_pulses=None):
    problem.validate()
    m = len(problem.timed)
    windows = problem.windows()
    fixed = set(problem.fixed())
    if optimize_durations:
        tau = cp.Variable(m)
        free = [k for k in range(m) if k not in fixed]
        constraints = [tau >= problem.lower, tau <= problem.upper]
        if fixed:
            constraints.append(tau[list(fixed)] == problem.base[list(fixed)])
    else:
        tau = cp.Constant(problem.base)
        free = []
        constraints = []
    start = cp.Variable(m)
    makespan = cp.Variable()
    constraints += [start >= 0, makespan <= deadline]
    terms = []
    if free:
        idx = np.array(free)
        terms.append(cp.sum(cp.multiply(problem.inverse[idx], cp.inv_pos(tau[idx]))))
        terms.append(problem.linear[idx] @ tau[idx])
    pulses = cp.Variable(len(windows)) if allow_dd and fixed_pulses is None else None
    if pulses is not None:
        constraints.append(pulses >= 0)
        if max_pulses is not None:
            constraints.append(pulses <= max_pulses)
    widths = []
    population = problem.excited_populations()
    for k, (q, a, b) in enumerate(windows):
        end = makespan if b is None else start[b]
        w = end - start[a] - tau[a]
        widths.append(w)
        model = problem.idle[q]
        if fixed_pulses is not None:
            count = float(fixed_pulses[k])
            constraints.append(w >= count * problem.pulse_duration)
            if model.amplitude > 0:
                factor = model.amplitude * (count + 1) ** (-model.beta)
                terms.append(factor * cp.power(w, 1 + model.beta) if model.beta > 0
                             else factor * w)
            terms.append(model.markov * w + model.pulse_error * count)
            terms.append(model.relaxation * _relaxation_weight(population[k], count) * w)
            constraints.append(w >= 0)
            continue
        constraints.append(w >= 0)
        if pulses is not None and model.beta > 0 and model.amplitude > 0:
            hazard = cp.Variable(nonneg=True)
            weights = _geo_weights(model.beta)
            # c w^(1+b) <= h (n+1)^b  <=>  w <= (h/c)^(1/(1+b)) (n+1)^(b/(1+b))
            constraints.append(w <= cp.geo_mean(cp.hstack([hazard / model.amplitude, pulses[k] + 1]),
                                                [float(x) for x in weights]))
            constraints.append(pulses[k] * problem.pulse_duration <= w)
            terms += [hazard, model.markov * w, model.pulse_error * pulses[k],
                      # min(p1, 1/2) lower-bounds the weight of every integer count.
                      model.relaxation * min(population[k], 0.5) * w]
        else:
            if pulses is not None:
                constraints.append(pulses[k] == 0)
            if model.amplitude > 0:
                terms.append(model.amplitude * (cp.power(w, 1 + model.beta)
                                                if model.beta > 0 else w))
            terms.append(model.markov * w)
            terms.append(model.relaxation * population[k] * w)
    if makespan_weight > 0:
        terms.append(makespan_weight * makespan)
    objective = cp.sum(cp.hstack(terms)) if terms else cp.Constant(0)
    program = cp.Problem(cp.Minimize(objective), constraints)
    if not program.is_dcp():
        raise RuntimeError("DD allocation is not DCP")
    for options in (dict(solver="CLARABEL", max_iter=500), dict(solver="SCS", eps=1e-9, max_iters=100000)):
        try:
            program.solve(**options)
        except cp.error.SolverError:
            continue
        if program.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            break
    if start.value is None:
        raise RuntimeError(f"DD allocation failed: {program.status}")
    durations = np.asarray(tau.value, dtype=float).reshape(-1) if optimize_durations else problem.base.copy()
    durations = np.clip(durations, problem.lower, problem.upper)
    counts = (np.zeros(len(windows)) if pulses is None and fixed_pulses is None
              else np.asarray(fixed_pulses if fixed_pulses is not None else pulses.value, dtype=float))
    counts = np.maximum(counts, 0.)
    return dict(durations=durations, starts=np.maximum(np.asarray(start.value), 0.),
                makespan=float(makespan.value), pulses=counts, value=float(program.value),
                status=program.status)


def _repair(problem, result):
    """Re-derive a consistent schedule: earliest-feasible starts preserving gaps."""
    starts = result["starts"].copy()
    durations = result["durations"]
    for q, a, b in problem.windows():
        if b is not None and starts[b] < starts[a] + durations[a]:
            starts[b] = starts[a] + durations[a]
    result["starts"] = starts
    result["makespan"] = max(result["makespan"], float(np.max(starts + durations)))
    return result


def solve_dd_allocation(problem: DDProblem, deadline, *, optimize_durations=True, allow_dd=True,
                        parity=2, makespan_weight=0., max_pulses=None):
    """Relaxed optimum, even-integer rounding, re-solve, and certified gap."""
    relaxed = _repair(problem, _solve(problem, deadline, optimize_durations=optimize_durations,
                                       allow_dd=allow_dd, makespan_weight=makespan_weight,
                                       max_pulses=max_pulses))
    if not allow_dd:
        relaxed["relaxation_value"] = relaxed["value"]
        relaxed["integer_value"] = model_objective(problem, relaxed["durations"], relaxed["starts"],
                                                   relaxed["makespan"], relaxed["pulses"])
        relaxed["relative_gap"] = 0.
        return relaxed
    windows = problem.windows()
    population = problem.excited_populations()
    counts = []
    for k, (q, a, b) in enumerate(windows):
        end = relaxed["makespan"] if b is None else relaxed["starts"][b]
        w = end - relaxed["starts"][a] - relaxed["durations"][a]
        n = relaxed["pulses"][k]
        low = parity * np.floor(n / parity)
        candidates = [0.] + [c for c in {low, low + parity} if c > 0 and c * problem.pulse_duration <= w + 1e-9
                             and (max_pulses is None or c <= max_pulses)]
        counts.append(min(candidates, key=lambda c: _window_hazard_numeric(problem.idle[q], w, c, population[k])))
    final = _repair(problem, _solve(problem, deadline, optimize_durations=optimize_durations,
                                     allow_dd=True, fixed_pulses=np.array(counts),
                                     makespan_weight=makespan_weight))
    final["pulses"] = np.array(counts)
    final["relaxation_value"] = relaxed["value"]
    final["integer_value"] = model_objective(problem, final["durations"], final["starts"],
                                             final["makespan"], final["pulses"]) + \
        makespan_weight * final["makespan"]
    final["relative_gap"] = max(0., final["integer_value"] - relaxed["value"]) / max(final["integer_value"], 1e-15)
    return final


# ---------------------------------------------------------------------------
# Baseline schedules and conversion to simulator input
# ---------------------------------------------------------------------------

def asap(problem: DDProblem, durations=None):
    durations = problem.base.copy() if durations is None else np.asarray(durations, dtype=float)
    timed = problem.timed
    starts = np.zeros(len(timed))
    ready = np.zeros(problem.n)
    for k, i in enumerate(timed):
        qs = problem.gates[i].qubits
        starts[k] = max(ready[q] for q in qs)
        for q in qs:
            ready[q] = starts[k] + durations[k]
    return durations, starts, float(np.max(starts + durations))


def alap(problem: DDProblem, deadline, durations=None):
    durations = problem.base.copy() if durations is None else np.asarray(durations, dtype=float)
    timed = problem.timed
    starts = np.zeros(len(timed))
    latest = np.full(problem.n, float(deadline))
    for k in reversed(range(len(timed))):
        qs = problem.gates[timed[k]].qubits
        starts[k] = min(latest[q] for q in qs) - durations[k]
        for q in qs:
            latest[q] = starts[k]
    if np.min(starts) < -1e-9:
        raise ValueError("Deadline shorter than the critical path")
    return durations, starts, float(deadline)


def fixed_pattern(problem, durations, starts, makespan, pulses_per_window):
    """Pulse counts for a standard sequence placed only where it fits."""
    counts = []
    for q, a, b in problem.windows():
        end = makespan if b is None else starts[b]
        w = end - starts[a] - durations[a]
        counts.append(pulses_per_window if pulses_per_window * problem.pulse_duration <= w + 1e-12 else 0)
    return np.array(counts, dtype=float)


def to_schedule(problem: DDProblem, durations, starts, makespan, pulses_per_window, sequence="XX"):
    """Place CPMG-spaced pulses (centres at (2i+1)w/(2n)) in each window."""
    timed = problem.timed
    pulses = {q: [] for q in range(problem.n)}
    letters = {"XX": "X", "XY4": "XY"}[sequence]
    for k, (q, a, b) in enumerate(problem.windows()):
        count = int(round(pulses_per_window[k]))
        if count <= 0:
            continue
        t0 = starts[a] + durations[a]
        end = makespan if b is None else starts[b]
        w = end - t0
        for i in range(count):
            pulses[q].append((t0 + w * (2 * i + 1) / (2 * count), letters[i % len(letters)]))
    for q in pulses:
        pulses[q].sort()
    return ScheduledCircuit(problem.n, list(problem.gates),
                            {i: float(starts[k]) for k, i in enumerate(timed)},
                            {i: float(durations[k]) for k, i in enumerate(timed)},
                            float(makespan), pulses)


def fit_idle_model(rows, pulse_duration=0.036, spam_offset=False, static_term=False):
    """Least-squares fit of the idle law to spectroscopy hazards (log residuals).

    With ``spam_offset`` a constant hazard (state preparation and measurement)
    is fitted as well and returned separately; it does not depend on the
    schedule, so it is not part of the idle model.
    """
    from scipy.optimize import least_squares
    w = np.array([r["window"] for r in rows])
    n = np.array([r["pulses"] for r in rows], dtype=float)
    h = np.array([r["hazard"] for r in rows])
    keep = (h > 1e-6) & (h < 3.0)
    w, n, h = w[keep], n[keep], h[keep]

    def predict(p):
        c, b, eta, eps = np.exp(p[0]), p[1], np.exp(p[2]), np.exp(p[3])
        value = c * w ** (1 + b) * (n + 1) ** (-b) + eta * w + eps * n
        if static_term:
            value = value + np.exp(p[-1]) * w ** 2 * (n == 0)
        return value + (np.exp(p[4]) if spam_offset else 0.)

    best = None
    starts = [(b0, s0) for b0 in (0.25, 1.0, 2.0) for s0 in ((1e-4, 1e-3, 1e-2) if static_term else (None,))]
    for b0, s0 in starts:
        guess = [np.log(max(np.median(h / w ** (1 + b0)), 1e-12)), b0, np.log(1e-4), np.log(1e-4)]
        low, high = [-40, 0, -30, -30], [10, 4, 5, 5]
        if spam_offset:
            guess, low, high = guess + [np.log(max(h.min() / 2, 1e-6))], low + [-30], high + [2]
        if static_term:
            guess, low, high = guess + [np.log(s0)], low + [-30], high + [2]
        fit = least_squares(lambda p: np.log(predict(p)) - np.log(h), np.array(guess), bounds=(low, high))
        if best is None or fit.cost < best.cost:
            best = fit
    p = best.x
    model = IdleModel(float(np.exp(p[0])), float(p[1]), float(np.exp(p[2])), float(np.exp(p[3])),
                      static=float(np.exp(p[-1])) if static_term else 0.)
    if spam_offset:
        return model, float(best.cost), float(np.exp(p[4]))
    return model, float(best.cost)


def correlated_protection(problem: DDProblem, durations, starts, makespan, *, max_pulses=32, parity=2,
                          enumerate_limit=16):
    """Choose which windows to protect when low-frequency noise couples windows.

    For a fixed schedule, each qubit's unrefocused phase accumulates over the
    windows left without a balanced CPMG block, so its low-frequency hazard is
    ``c_q (sum_k w_k (1 - z_k))^(1+b_q)`` with ``z_k = 1`` for a protected window
    (convex in z). A protected window keeps the residual per-window hazard
    ``c w^(1+b) (n+1)^-b + eps n`` with the best even ``n >= parity``. Markov and
    state-aware relaxation terms are per window. The objective separates by
    qubit, so qubits with at most ``enumerate_limit`` windows are solved exactly
    by enumeration; larger ones use the convex relaxation rounded at 1/2.
    Returns pulse counts per window (``problem.windows()`` order).
    """
    windows = problem.windows()
    population = problem.excited_populations()
    widths = []
    for q, a, b in windows:
        end = makespan if b is None else starts[b]
        widths.append(max(end - starts[a] - durations[a], 0.))
    counts = np.zeros(len(windows))
    for q in range(problem.n):
        model = problem.idle[q]
        ks = [k for k, (qq, a, b) in enumerate(windows) if qq == q]
        if not ks:
            continue
        protected_cost, protected_n, open_cost = {}, {}, {}
        for k in ks:
            w = widths[k]
            options = [n for n in range(parity, max_pulses + 1, parity) if n * problem.pulse_duration <= w]
            if options and (model.amplitude > 0 or model.static > 0):
                best = min(options, key=lambda n: model.amplitude * w ** (1 + model.beta) * (n + 1) ** (-model.beta)
                           + model.pulse_error * n)
                protected_n[k] = best
                protected_cost[k] = (model.amplitude * w ** (1 + model.beta) * (best + 1) ** (-model.beta)
                                     + model.pulse_error * best + model.relaxation * 0.5 * w)
            open_cost[k] = model.relaxation * population[k] * w
        candidates = [k for k in ks if k in protected_n]

        def total(chosen):
            unrefocused = sum(widths[k] for k in ks if k not in chosen)
            value = model.amplitude * unrefocused ** (1 + model.beta) if model.amplitude > 0 else 0.
            # Quasi-static phase is coherent across windows; any balanced block cancels it.
            value += model.static * unrefocused ** 2
            value += model.markov * sum(widths[k] for k in ks)
            return value + sum(protected_cost[k] if k in chosen else open_cost[k] for k in ks)

        if len(candidates) <= enumerate_limit:
            best_set, best_value = set(), total(set())
            for mask in range(1, 2 ** len(candidates)):
                chosen = {candidates[i] for i in range(len(candidates)) if mask >> i & 1}
                value = total(chosen)
                if value < best_value:
                    best_set, best_value = chosen, value
        else:
            z = cp.Variable(len(candidates))
            unrefocused = sum(widths[k] for k in ks if k not in protected_n) + \
                sum(widths[k] * (1 - z[i]) for i, k in enumerate(candidates))
            objective = model.amplitude * cp.power(unrefocused, 1 + model.beta) + \
                sum(z[i] * (protected_cost[k] - open_cost[k]) for i, k in enumerate(candidates))
            cp.Problem(cp.Minimize(objective), [z >= 0, z <= 1]).solve(solver="CLARABEL")
            best_set = {k for i, k in enumerate(candidates) if z.value[i] > 0.5}
        for k in best_set:
            counts[k] = protected_n[k]
    return counts
