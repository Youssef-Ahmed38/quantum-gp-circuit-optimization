"""Executable-candidate compilation and fixed-order GP scheduling.

Research prototype; synthetic hardware; no discrete compilation optimality claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, DensityMatrix, Statevector, Kraus
from qiskit.transpiler import CouplingMap

BASIS = ("rz", "sx", "x", "cx")
DURATION = {"rz": 0.1, "sx": 1.0, "x": 2.0, "cx": 10.0}


@dataclass(frozen=True)
class Schedule:
    starts: np.ndarray
    makespan: float
    elapsed: float
    status: str
    violation: float
    solver_used: str = "none"


def topology(n: int, name: str) -> CouplingMap:
    if n < 2 or name not in ("line", "ring"):
        raise ValueError("Expected n >= 2 and topology line or ring")
    edges = {(i, i + 1) for i in range(n - 1)}
    if name == "ring":
        edges.add((n - 1, 0))
    return CouplingMap(sorted(edges | {(b, a) for a, b in edges}))


def benchmark(family: str, n: int, seed: int) -> QuantumCircuit:
    """Generate fixed, bound-parameter circuits, without measurements/barriers."""
    if n < 2:
        raise ValueError("Expected at least two qubits")
    rng = np.random.default_rng(seed)
    c = QuantumCircuit(n, name=f"{family}_{n}_{seed}")
    if family == "qft":
        for i in reversed(range(n)):
            c.h(i)
            for j in reversed(range(i)):
                c.cp(math.pi / 2 ** (i - j), i, j)
        for i in range(n // 2):
            c.swap(i, n - i - 1)
    elif family == "grover":
        c.h(range(n))
        for _ in range(1 + seed % 2):
            c.h(n - 1)
            c.mcx(list(range(n - 1)), n - 1)
            c.h(n - 1)
            c.h(range(n))
            c.x(range(n))
            c.h(n - 1)
            c.mcx(list(range(n - 1)), n - 1)
            c.h(n - 1)
            c.x(range(n))
            c.h(range(n))
    elif family == "vqe_ansatz":
        for _ in range(2):
            for q in range(n):
                c.ry(float(rng.uniform(-math.pi, math.pi)), q)
                c.rz(float(rng.uniform(-math.pi, math.pi)), q)
            for q in range(n - 1):
                c.cx(q, q + 1)
    elif family == "qaoa":
        c.h(range(n))
        edges = [(q, (q + 1) % n) for q in range(n)]
        for _ in range(2):
            gamma, beta = rng.uniform(0.1, 1.0, 2)
            for a, b in edges:
                c.cx(a, b)
                c.rz(float(2 * gamma), b)
                c.cx(a, b)
            for q in range(n):
                c.rx(float(2 * beta), q)
    else:
        raise ValueError(f"Unknown family: {family}")
    return c


def compile_candidate(logical: QuantumCircuit, coupling: CouplingMap,
                      seed: int) -> tuple[QuantumCircuit, float]:
    start = time.perf_counter()
    circuit = transpile(logical, basis_gates=list(BASIS), coupling_map=coupling,
                        optimization_level=3, layout_method="sabre",
                        routing_method="sabre", seed_transpiler=seed,
                        num_processes=1, qubits_initially_zero=False)
    return circuit, time.perf_counter() - start


def extract(circuit: QuantumCircuit, coupling: CouplingMap | None = None):
    """Preserve per-qubit gate order; disjoint gates may execute concurrently."""
    last: dict[int, int] = {}
    edges: list[tuple[int, int]] = []
    durations, qubits, errors = [], [], []
    allowed = set(coupling.get_edges()) if coupling is not None else None
    for i, ins in enumerate(circuit.data):
        name = ins.operation.name
        if name not in BASIS or ins.clbits or ins.operation.num_clbits:
            raise ValueError(f"Unsupported instruction {name}; native unitary gates only")
        qs = tuple(circuit.find_bit(q).index for q in ins.qubits)
        if len(qs) == 2 and allowed is not None and qs not in allowed:
            raise ValueError(f"Gate {i} violates connectivity at {qs}")
        for predecessor in sorted({last[q] for q in qs if q in last}):
            edges.append((predecessor, i))
        for q in qs:
            last[q] = i
        duration = DURATION[name]
        if name == "cx":
            duration *= 1 + 0.05 * sum(qs)
            error = 0.005 + 0.002 * sum(qs) + 0.001 * abs(qs[0] - qs[1])
        else:
            error = {"rz": 0.0, "sx": 0.0002, "x": 0.0004}[name]
        durations.append(duration)
        qubits.append(qs)
        errors.append(error)
    return np.asarray(durations), edges, qubits, np.asarray(errors)


def _validate_problem(durations, edges):
    d = np.asarray(durations, dtype=float)
    if d.ndim != 1 or not np.isfinite(d).all() or np.any(d <= 0):
        raise ValueError("Durations must be a finite positive one-dimensional array")
    for i, j in edges:
        if not isinstance(i, int) or not isinstance(j, int) or not 0 <= i < j < len(d):
            raise ValueError("Edges must use integer indices in forward topological order")
    return d


def constraint_violation(starts, makespan, durations, edges) -> float:
    violations = [0.0, *(-np.asarray(starts)).tolist(),
                  *(np.asarray(starts) + durations - makespan).tolist()]
    violations.extend(starts[i] + durations[i] - starts[j] for i, j in edges)
    return float(max(violations))


def critical_path(durations, edges) -> Schedule:
    begin = time.perf_counter()
    d = _validate_problem(durations, edges)
    predecessors = [[] for _ in d]
    for i, j in edges:
        predecessors[j].append(i)
    starts = np.zeros(len(d))
    for j, parents in enumerate(predecessors):
        starts[j] = max((starts[i] + d[i] for i in parents), default=0.0)
    makespan = float(np.max(starts + d)) if len(d) else 0.0
    return Schedule(starts, makespan, time.perf_counter() - begin, "exact", 0.0)


def gp_schedule(durations, edges, offset=1.0, solver="CLARABEL") -> Schedule:
    """Minimize shifted makespan using a genuine disciplined geometric program."""
    begin = time.perf_counter()
    d = _validate_problem(durations, edges)
    if not math.isfinite(offset) or offset <= 0:
        raise ValueError("GP offset must be finite and positive")
    if not len(d):
        return Schedule(np.empty(0), 0.0, time.perf_counter() - begin, "empty", 0.0)
    if solver not in ("CLARABEL", "ECOS", "SCS"):
        raise ValueError("This project permits only free solvers: CLARABEL, ECOS, SCS")
    if solver not in cp.installed_solvers():
        raise RuntimeError(f"Free solver {solver} is not installed")
    # Normalize by a serial-schedule upper bound to avoid badly scaled exponentials.
    scale = float(np.sum(d))
    scaled_d = d / scale
    shifted_start = cp.Variable(len(d), pos=True)
    shifted_finish = cp.Variable(pos=True)
    constraints = [offset / shifted_start <= 1,
                   (shifted_start + scaled_d) / shifted_finish <= 1,
                   shifted_start <= offset + 2,
                   shifted_finish <= offset + 2]
    constraints += [(shifted_start[i] + scaled_d[i]) / shifted_start[j] <= 1
                    for i, j in edges]
    problem = cp.Problem(cp.Minimize(shifted_finish), constraints)
    if not problem.is_dgp():
        raise RuntimeError("Scheduling model failed the DGP grammar check")
    options = {"CLARABEL": dict(tol_gap_abs=1e-8, tol_gap_rel=1e-8,
                                tol_feas=1e-8, max_iter=200),
               "ECOS": dict(abstol=1e-8, reltol=1e-8, feastol=1e-8),
               "SCS": dict(eps=1e-7, max_iters=100000)}
    exact = critical_path(d, edges).makespan
    attempts = [solver]
    if solver == "CLARABEL" and "ECOS" in cp.installed_solvers():
        attempts.append("ECOS")
    failures = []
    for solver_name in attempts:
        try:
            problem.solve(gp=True, solver=solver_name, **options[solver_name])
        except cp.error.SolverError as error:
            failures.append(f"{solver_name}: {error}")
            continue
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            failures.append(f"{solver_name}: status {problem.status}")
            continue
        starts = scale * (np.asarray(shifted_start.value).reshape(-1) - offset)
        makespan = scale * (float(shifted_finish.value) - offset)
        violation = constraint_violation(starts, makespan, d, edges)
        if (violation > 1e-5 * max(1.0, makespan) or
                abs(makespan - exact) > 1e-5 * max(1.0, exact)):
            failures.append(f"{solver_name}: failed independent feasibility/optimum check")
            continue
        return Schedule(starts, makespan, time.perf_counter() - begin,
                        problem.status, violation, solver_name)
    raise RuntimeError("No independently certified GP solution: " + "; ".join(failures))


def lp_schedule(durations, edges) -> Schedule:
    """Independent LP formulation, solved by HiGHS through SciPy."""
    begin = time.perf_counter()
    d = _validate_problem(durations, edges)
    if not len(d):
        return Schedule(np.empty(0), 0.0, time.perf_counter() - begin, "empty", 0.0)
    matrix, rhs = [], []
    for i, j in edges:
        row = np.zeros(len(d) + 1)
        row[i], row[j] = 1, -1
        matrix.append(row)
        rhs.append(-d[i])
    for i in range(len(d)):
        row = np.zeros(len(d) + 1)
        row[i], row[-1] = 1, -1
        matrix.append(row)
        rhs.append(-d[i])
    objective = np.zeros(len(d) + 1)
    objective[-1] = 1
    result = linprog(objective, A_ub=np.asarray(matrix), b_ub=rhs,
                     bounds=(0, None), method="highs")
    if not result.success:
        raise RuntimeError(f"LP scheduling failed: {result.message}")
    violation = constraint_violation(result.x[:-1], result.x[-1], d, edges)
    return Schedule(result.x[:-1], float(result.x[-1]),
                    time.perf_counter() - begin, "optimal", violation)


def unitary_error(logical: QuantumCircuit, physical: QuantumCircuit) -> float:
    """Compare complete operators, accounting for transpiler layouts/global phase."""
    expected = Operator(logical).data
    actual = Operator.from_circuit(physical).data
    overlap = np.vdot(expected, actual)
    phase = overlap / abs(overlap) if abs(overlap) > 0 else 1.0
    return float(np.max(np.abs(actual - phase * expected)))


def candidate_metrics(circuit, coupling):
    d, edges, qubits, errors = extract(circuit, coupling)
    schedule = critical_path(d, edges)
    two = sum(len(qs) == 2 for qs in qubits)
    loss = float(np.sum(-np.log1p(-errors)))
    return {"gates": len(d), "two_qubit": two,
            "depth": circuit.depth(), "gate_cost": len(d) + 9 * two,
            "makespan": schedule.makespan, "error_loss": loss,
            "survival_proxy": math.exp(-loss)}, (d, edges)


def normalized_score(metrics, reference, weights=(0.4, 0.3, 0.3)):
    if len(weights) != 3 or min(weights) < 0 or sum(weights) <= 0:
        raise ValueError("Need three nonnegative objective weights with positive sum")
    return float(sum(w * metrics[k] / max(reference[k], 1e-12)
                     for w, k in zip(weights, ("makespan", "gate_cost", "error_loss"))))


def certified_selection(metrics, gp_makespans, weights=(0.4, 0.3, 0.3)):
    """Rank actual GP costs, resolving numerical uncertainty with exact DAG costs.

    Also return the independently computed critical-path comparator choice.
    The score tolerance is bounded by the scheduling certificate tolerance;
    raw (unresolved) GP rankings are retained for reproducibility.
    """
    if not metrics or len(metrics) != len(gp_makespans):
        raise ValueError("Need equally sized nonempty candidate metrics and GP costs")
    exact_scores = [normalized_score(m, metrics[0], weights) for m in metrics]
    gp_metrics = [dict(m, makespan=t) for m, t in zip(metrics, gp_makespans)]
    gp_scores = [normalized_score(m, gp_metrics[0], weights) for m in gp_metrics]
    tolerance = 4e-5 * max(1.0, abs(min(gp_scores))) * max(1.0, sum(weights))
    eligible = [i for i,score in enumerate(gp_scores) if score <= min(gp_scores)+tolerance]
    selected = min(eligible, key=lambda i:(exact_scores[i],i))
    matched = min(range(len(metrics)), key=lambda i:(exact_scores[i],i))
    raw = min(range(len(metrics)), key=lambda i:(gp_scores[i],i))
    return selected, matched, raw, gp_scores


def _depolarizing_channel(arity: int, probability: float) -> Kraus:
    paulis = [np.eye(2), np.array([[0, 1], [1, 0]]),
              np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]
    matrices = [np.eye(1)]
    for _ in range(arity):
        matrices = [np.kron(a, b) for a in matrices for b in paulis]
    dimension = 2 ** arity
    terms = [math.sqrt(1 - probability) * np.eye(dimension)]
    terms += [math.sqrt(probability) / dimension * p for p in matrices]
    return Kraus(terms)


def noisy_fidelity(circuit, coupling):
    """Synthetic local depolarizing simulation on the physical all-zero input.

    No T1/T2, readout, crosstalk, pulse or hardware calibration model is included.
    """
    _, _, qubits, errors = extract(circuit, coupling)
    state = DensityMatrix.from_label("0" * circuit.num_qubits)
    cache = {}
    for ins, qs, probability in zip(circuit.data, qubits, errors):
        state = state.evolve(ins.operation, qargs=list(qs))
        if probability:
            key = len(qs), float(probability)
            if key not in cache:
                cache[key] = _depolarizing_channel(*key)
            state = state.evolve(cache[key], qargs=list(qs))
    ideal = Statevector.from_instruction(circuit).data
    fidelity = float(np.real(np.vdot(ideal, state.data @ ideal)))
    trace_error = float(abs(np.trace(state.data) - 1))
    minimum_eigenvalue = float(np.linalg.eigvalsh(state.data).min())
    if trace_error > 1e-8 or minimum_eigenvalue < -1e-8 or not -1e-8 <= fidelity <= 1+1e-8:
        raise RuntimeError("Invalid density matrix after simulation")
    return fidelity, trace_error, minimum_eigenvalue
