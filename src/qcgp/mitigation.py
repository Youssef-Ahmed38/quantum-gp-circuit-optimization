"""C3: runtime-optimal error-mitigated estimation.

Under the declared post-gate Pauli-Lindblad model with total gate intensity
``lambda_j(tau) = a_j / tau_j + g_j tau_j``, probabilistic error cancellation
(PEC) has sampling overhead ``gamma = exp(2 Lambda)`` with ``Lambda = sum_j lambda_j``.
Reaching standard error ``eps`` needs about ``gamma^2 / eps^2`` shots, and each
shot occupies ``M(tau) + t_rep`` of QPU time. The total QPU time

    T(tau) = eps^-2 exp(4 Lambda(tau)) (M(tau) + t_rep)

is convex in ``y = log tau``: ``Lambda(e^y)`` is a sum of exponentials and
``log(M + t_rep)`` is a maximum of log-sum-exps. At an interior optimum

    tau_j = sqrt(a_j / (g_j + p_j / (4 (M + t_rep)))),

with ``p_j`` the optimal critical-path flow through gate ``j``. Off-critical
gates take their error-optimal duration; large ``t_rep`` removes all latency
pressure.
"""
from __future__ import annotations

import time

import cvxpy as cp
import numpy as np

from .core import critical_path
from .observables import ObservableModel, _binary_gate, local_paulis


def intensities(model: ObservableModel, tau):
    tau = np.asarray(tau, dtype=float)
    return model.a / tau + model.gamma * tau


def pec_overhead(model, tau):
    return float(np.exp(2 * np.sum(intensities(model, tau))))


def qpu_time(model, tau, *, epsilon=0.01, rep_delay=0.):
    """Shots x (makespan + repetition delay), in the model's time unit."""
    span = critical_path(np.asarray(tau, dtype=float), model.edges).makespan
    shots = pec_overhead(model, tau) ** 2 / epsilon ** 2
    return float(shots * (span + rep_delay)), float(shots), float(span)


def optimize_runtime(model: ObservableModel, *, rep_delay=0., epsilon=0.01, deadline=None):
    """Global minimiser of PEC-mitigated QPU time over the duration box."""
    begin = time.perf_counter()
    model.validate()
    m = len(model.base)
    scale = critical_path(model.base, model.edges).makespan
    relative = model.base / scale
    y = cp.Variable(m)
    finish = cp.Variable(m)
    end = cp.Variable()
    constraints = [y >= np.log(model.lower / model.base), y <= np.log(model.upper / model.base),
                   np.log(relative) + y <= finish, finish <= end]
    for i, j in model.edges:
        constraints.append(cp.log_sum_exp(cp.hstack([finish[i], np.log(relative[j]) + y[j]])) <= finish[j])
    if deadline is not None:
        constraints.append(end <= np.log(deadline / scale))
    total = (cp.sum(cp.multiply(model.a / model.base, cp.exp(-y)))
             + cp.sum(cp.multiply(model.gamma * model.base, cp.exp(y))))
    latency = (cp.log_sum_exp(cp.hstack([end + np.log(scale), np.log(rep_delay)]))
               if rep_delay > 0 else end + np.log(scale))
    problem = cp.Problem(cp.Minimize(4 * total + latency), constraints)
    if not problem.is_dcp():
        raise RuntimeError("Runtime objective is not convex in log coordinates")
    for options in (dict(solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10, max_iter=500),
                    dict(solver="CLARABEL", tol_gap_abs=1e-8, tol_gap_rel=1e-8, tol_feas=1e-8, max_iter=1000),
                    dict(solver="SCS", eps=1e-9, max_iters=200000)):
        try:
            problem.solve(**options)
        except cp.error.SolverError:
            continue
        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            break
    if y.value is None:
        raise RuntimeError(f"Runtime optimisation failed: {problem.status}")
    tau = np.clip(model.base * np.exp(y.value), model.lower, model.upper)
    value, shots, span = qpu_time(model, tau, epsilon=epsilon, rep_delay=rep_delay)
    return dict(durations=tau, makespan=span, shots=shots, qpu_time=value,
                stationarity=stationarity(model, tau, span, rep_delay),
                status=problem.status, elapsed=time.perf_counter() - begin)


def stationarity(model, tau, span, rep_delay, tolerance=1e-3):
    """Recover implied path flows ``p_j`` and check the closed form.

    For interior gates ``p_j = 4 (M + t_rep)(a_j / tau_j^2 - g_j)`` must lie in
    ``[0, 1]`` and vanish off every critical path. Recovering ``p`` multiplies
    solver error in ``tau`` by ``4 (M + t_rep)``, so the tolerance is scaled by
    ``1 + t_rep / M``.
    """
    tau = np.asarray(tau, dtype=float)
    tolerance = tolerance * (1 + rep_delay / span)
    interior = (tau > model.lower * (1 + 1e-6)) & (tau < model.upper * (1 - 1e-6))
    flow = 4 * (span + rep_delay) * (model.a / tau ** 2 - model.gamma)
    # Slack of each gate: latest start minus earliest start at these durations.
    schedule = critical_path(tau, model.edges)
    successors = [[] for _ in tau]
    for i, j in model.edges:
        successors[i].append(j)
    latest = np.full(len(tau), span)
    for j in reversed(range(len(tau))):
        finish = min((latest[k] for k in successors[j]), default=span)
        latest[j] = finish - tau[j]
    slack = latest - schedule.starts
    critical = slack <= 1e-4 * span
    return dict(interior_gates=int(interior.sum()),
                min_flow=float(flow[interior].min()) if interior.any() else None,
                max_flow=float(flow[interior].max()) if interior.any() else None,
                max_offcritical_flow=float(np.abs(flow[interior & ~critical]).max()) if (interior & ~critical).any() else 0.,
                valid=bool(not interior.any() or (flow[interior].min() > -tolerance and flow[interior].max() < 1 + tolerance)))


def error_optimal(model):
    return np.clip(np.sqrt(model.a / model.gamma), model.lower, model.upper)


def best_uniform(model, *, rep_delay=0., epsilon=0.01, grid=4001):
    low = float(np.max(model.lower / model.base))
    high = float(np.min(model.upper / model.base))
    scales = np.geomspace(low, high, grid)
    values = [qpu_time(model, model.base * s, epsilon=epsilon, rep_delay=rep_delay)[0] for s in scales]
    best = int(np.argmin(values))
    return model.base * scales[best]


def pec_pauli_frame(circuit, model: ObservableModel, tau, *, shots=50000, seed=11):
    """Forward Pauli-frame simulation of PEC on a Clifford circuit.

    Noise: each local Pauli ``P`` of gate ``j`` independently with probability
    ``(1 - exp(-2 lambda_j / (4^k - 1))) / 2``. PEC inserts each ``P`` with the
    same probability and sign -1, weighted by ``gamma = exp(2 Lambda)``. Returns
    estimator means and variances for every stabilizer generator, whose ideal
    expectation is +1.
    """
    rng = np.random.default_rng(seed)
    n = circuit.num_qubits
    x = np.zeros((shots, n), dtype=bool)
    z = np.zeros((shots, n), dtype=bool)
    sign = np.ones(shots)
    positions = {index: j for j, index in enumerate(model.timed_indices)}
    lam = intensities(model, tau)
    for index, ins in enumerate(circuit.data):
        qs = tuple(circuit.find_bit(q).index for q in ins.qubits)
        _binary_gate(x, z, ins.operation.name, qs, ins.operation.params)
        if index not in positions:
            continue
        j = positions[index]
        paulis = local_paulis(qs)
        p = -0.5 * np.expm1(-2 * lam[j] / len(paulis))
        for px, pz in paulis:
            noise = rng.random(shots) < p
            inverse = rng.random(shots) < p
            flip = noise ^ inverse
            sign = np.where(inverse, -sign, sign)
            for q, xx, zz in zip(qs, px, pz):
                if xx:
                    x[:, q] ^= flip
                if zz:
                    z[:, q] ^= flip
    from qiskit.quantum_info import Pauli
    final = [Pauli(label) for label in model.labels]
    fx = np.stack([p.x for p in final]).astype(np.uint8)
    fz = np.stack([p.z for p in final]).astype(np.uint8)
    anti = ((x.astype(np.uint8) @ fz.T + z.astype(np.uint8) @ fx.T) % 2).astype(bool)
    gamma = float(np.exp(2 * lam.sum()))
    estimates = gamma * sign[:, None] * np.where(anti, -1.0, 1.0)
    return dict(gamma=gamma, mean=estimates.mean(axis=0), variance=estimates.var(axis=0, ddof=1),
                predicted_variance=gamma ** 2 - 1, unmitigated=np.where(anti, -1.0, 1.0).mean(axis=0))
