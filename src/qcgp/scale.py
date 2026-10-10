"""C8: level-vectorised path-flow Frank-Wolfe for very large circuit DAGs.

Same algorithm and certificate as :func:`qcgp.pathflow.optimize_pathflow`
(primal-dual gap from the path-flow dual), with a NumPy longest-path oracle
that processes one topological level at a time. The objective is

    sum_j (a_j / tau_j + c_j tau_j) + b * makespan(tau)   on  l <= tau <= u.
"""
from __future__ import annotations

import time

import numpy as np


class LevelOracle:
    def __init__(self, m, edges):
        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        level = np.zeros(m, dtype=np.int64)
        order = np.argsort(edges[:, 1], kind="stable")
        edges = edges[order]
        # Program order is a topological order; one pass fixes levels.
        for i, j in edges[np.argsort(edges[:, 1], kind="stable")]:
            if level[i] + 1 > level[j]:
                level[j] = level[i] + 1
        self.m = m
        self.levels = []
        edge_level = level[edges[:, 1]]
        for value in range(int(level.max()) + 1 if m else 0):
            nodes = np.flatnonzero(level == value)
            group = edges[edge_level == value]
            self.levels.append((nodes, group[:, 0], group[:, 1]))

    def longest_path(self, durations):
        ends = np.array(durations, dtype=float)
        parent = np.full(self.m, -1, dtype=np.int64)
        start = np.zeros(self.m)
        for nodes, src, dst in self.levels:
            if len(src):
                candidate = ends[src]
                np.maximum.at(start, dst, candidate)
                hit = candidate >= start[dst] - 1e-15
                parent[dst[hit]] = src[hit]
                ends[nodes] = start[nodes] + durations[nodes]
        terminal = int(np.argmax(ends))
        incidence = np.zeros(self.m)
        node = terminal
        while node >= 0:
            incidence[node] = 1.
            node = parent[node]
        return float(ends[terminal]), incidence


def fast_pathflow(lower, upper, a, c, b, edges, *, tolerance=1e-6, max_iterations=5000):
    begin = time.perf_counter()
    lower, upper, a, c = (np.asarray(x, dtype=float) for x in (lower, upper, a, c))
    oracle = LevelOracle(len(lower), edges)
    setup = time.perf_counter() - begin

    def response(pressure):
        return np.clip(np.sqrt(a / (c + pressure)), lower, upper)

    _, path = oracle.longest_path(response(np.zeros_like(a)))
    pressure = b * path
    history = []
    for iteration in range(max_iterations + 1):
        tau = response(pressure)
        span, path = oracle.longest_path(tau)
        primal = float(np.sum(a / tau + c * tau) + b * span)
        dual = float(np.sum(a / tau + (c + pressure) * tau))
        gap = max(0., primal - dual) / max(abs(primal), 1e-300)
        if iteration % 50 == 0:
            history.append((iteration, gap))
        if gap <= tolerance or iteration == max_iterations:
            break
        direction = b * path - pressure
        if np.dot(response(pressure + direction), direction) >= 0:
            step = 1.
        else:
            left, right = 0., 1.
            for _ in range(40):
                middle = (left + right) / 2
                if np.dot(response(pressure + middle * direction), direction) > 0:
                    left = middle
                else:
                    right = middle
            step = (left + right) / 2
        pressure = (1 - step) * pressure + step * b * path
    return dict(durations=tau, makespan=span, primal=primal, dual=dual, relative_gap=gap,
                iterations=iteration, certified=gap <= tolerance, setup_s=setup,
                elapsed_s=time.perf_counter() - begin, history=history)


def random_circuit_dag(n, depth, seed=0, two_qubit_fraction=0.5):
    """Brickwork-like random circuit DAG (program order is topological)."""
    rng = np.random.default_rng(seed)
    last = np.full(n, -1, dtype=np.int64)
    edges, arity = [], []
    gate = 0
    for layer in range(depth):
        perm = rng.permutation(n)
        q = 0
        while q < n:
            if q + 1 < n and rng.random() < two_qubit_fraction:
                qubits = (perm[q], perm[q + 1])
                q += 2
            else:
                qubits = (perm[q],)
                q += 1
            for x in qubits:
                if last[x] >= 0:
                    edges.append((last[x], gate))
                last[x] = gate
            arity.append(len(qubits))
            gate += 1
    return np.array(arity), np.array(edges, dtype=np.int64).reshape(-1, 2)


def sparse_interior_point(lower, upper, a, c, b, edges):
    """Vectorised convex formulation solved by Clarabel (primal-dual certificate).

    min sum(a/tau + c tau) + b M  s.t.  s_j >= s_i + tau_i (edges),  M >= s_j + tau_j,
    s >= 0, l <= tau <= u. Constraints are built as sparse matrices, so model
    construction is linear in the circuit size.
    """
    import cvxpy as cp
    from scipy import sparse
    begin = time.perf_counter()
    lower, upper, a, c = (np.asarray(x, dtype=float) for x in (lower, upper, a, c))
    m = len(lower)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    r = len(edges)
    rows = np.arange(r)
    # (s_j - s_i - tau_i) >= 0
    S = sparse.csr_matrix((np.concatenate([np.ones(r), -np.ones(r)]),
                           (np.concatenate([rows, rows]), np.concatenate([edges[:, 1], edges[:, 0]]))), shape=(r, m))
    T = sparse.csr_matrix((-np.ones(r), (rows, edges[:, 0])), shape=(r, m))
    scale = float(np.median(lower))
    tau = cp.Variable(m)
    start = cp.Variable(m)
    span = cp.Variable()
    unit = float(np.sum(a / lower + c * lower))
    objective = (cp.sum(cp.multiply(a / unit / scale, cp.inv_pos(tau))) + (c * scale / unit) @ tau
                 + b * scale / unit * span)
    constraints = [S @ start + T @ tau >= 0, span >= start + tau, start >= 0,
                   tau >= lower / scale, tau <= upper / scale]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    build = time.perf_counter() - begin
    problem.solve(solver="CLARABEL", tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9, max_iter=500)
    durations = np.clip(np.asarray(tau.value) * scale, lower, upper)
    oracle = LevelOracle(m, edges)
    makespan, _ = oracle.longest_path(durations)
    primal = float(np.sum(a / durations + c * durations) + b * makespan)
    stats = problem.solver_stats
    return dict(durations=durations, makespan=makespan, primal=primal, status=problem.status,
                build_s=build, solve_s=stats.solve_time, elapsed_s=time.perf_counter() - begin,
                iterations=stats.num_iters)
