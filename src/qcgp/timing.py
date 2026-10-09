"""Joint timing/control optimization with an explicitly conditional noise certificate.

Free local research model; no hardware pulse synthesis or measured calibration.
GP and ordinary convex formulations solve the same fixed-DAG problem.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
import numpy as np
import cvxpy as cp
from scipy.linalg import schur, expm
from qiskit.quantum_info import Operator, DensityMatrix, Statevector, partial_trace
from .core import extract, critical_path, constraint_violation, _validate_problem


@dataclass(frozen=True)
class TimingModel:
    base: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    amplitude_noise: np.ndarray  # a_i; integrated exposure a_i / duration
    gate_noise_rate: np.ndarray  # c_i; integrated exposure c_i * duration
    generator_squared: np.ndarray  # kappa_i^2; control effort kappa_i^2/duration
    background_rate: float  # globally covariant depolarizing rate, not T1/T2
    edges: tuple[tuple[int, int], ...]

    def __post_init__(self):
        for name in ("base","lower","upper","amplitude_noise","gate_noise_rate","generator_squared"):
            values=np.array(getattr(self,name),dtype=float,copy=True)
            values.setflags(write=False)
            object.__setattr__(self,name,values)
        object.__setattr__(self,'edges',tuple(tuple(edge) for edge in self.edges))

    def validate(self):
        base = _validate_problem(self.base, self.edges)
        if not len(base):
            raise ValueError("Joint timing requires a nonempty circuit")
        for name in ("lower", "upper", "amplitude_noise", "gate_noise_rate", "generator_squared"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != base.shape or not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError(f"{name} must match base and be finite and strictly positive")
        if np.any(self.lower > self.base) or np.any(self.base > self.upper):
            raise ValueError("Require lower <= base <= upper")
        if not math.isfinite(self.background_rate) or self.background_rate <= 0:
            raise ValueError("background_rate must be finite and positive")


@dataclass(frozen=True)
class TimingResult:
    durations: np.ndarray
    starts: np.ndarray
    makespan: float
    risk: float
    effort: float
    fidelity_lower_bound: float
    objective: float
    elapsed: float
    method: str
    solver: str
    status: str
    raw_violation: float
    path_pressure: np.ndarray | None = None
    kkt_relative_residual: float | None = None
    solver_failures: tuple[str, ...] = ()


def minimal_generator(unitary):
    """Hermitian principal-arc generator, minimized in spectral norm modulo phase.

    Return K, phase with exp(-i K) = exp(-i phase) U. The largest circular
    eigenphase gap identifies a shortest enclosing arc. No control locality or
    pulse-bandwidth realizability follows from existence of this generator.
    """
    u = np.asarray(unitary, dtype=complex)
    if u.ndim != 2 or u.shape[0] != u.shape[1] or not np.allclose(u.conj().T @ u, np.eye(len(u)), atol=1e-10,rtol=1e-10):
        raise ValueError("Expected a square unitary matrix")
    triangular, basis = schur(u, output="complex")
    angles = np.mod(np.angle(np.diag(triangular)), 2 * np.pi)
    ordered = np.sort(angles)
    gaps = np.diff(np.r_[ordered, ordered[0] + 2 * np.pi])
    cut = int(np.argmax(gaps))
    start = ordered[(cut + 1) % len(ordered)]
    unwrapped = start + np.mod(angles - start, 2 * np.pi)
    center = float((unwrapped.min() + unwrapped.max()) / 2)
    k = basis @ np.diag(center - unwrapped) @ basis.conj().T
    k = (k + k.conj().T) / 2
    if not np.allclose(expm(-1j * k), np.exp(-1j * center) * u, atol=1e-9,rtol=1e-9):
        raise RuntimeError("Generator reconstruction failed")
    return k, center


def synthetic_model(circuit, coupling=None, *, strength=1.0, background=0.00015):
    """Construct a declared synthetic model, not a fit to device calibration.

    Gate strength-dependent isotropic Lindblad noise has rate c_i + a_i/tau^2.
    Coefficients deliberately vary by operation and position. Rz is treated as
    a finite-duration controllable rotation, not an IBM virtual-Z instruction.
    """
    if not math.isfinite(strength) or strength <= 0:
        raise ValueError("strength must be finite and positive")
    base, edges, qubits, _ = extract(circuit, coupling)
    generator = []
    for instruction in circuit.data:
        k, _ = minimal_generator(Operator(instruction.operation).data)
        generator.append(float(np.linalg.norm(k, 2) ** 2))
    generator = np.asarray(generator)
    # Identity rotations have no controllable generator; remove them before modeling.
    if np.any(generator < 1e-14):
        raise ValueError("Remove exact identity gates before joint timing")
    rate = np.array([0.00006 if len(q) == 1 else 0.00022 for q in qubits]) * strength
    position = np.array([1 + 0.11 * sum(q) + 0.07 * (i % 5) for i, q in enumerate(qubits)])
    # Reference gate-local minimizers are between 1.2 and 2.0 times base duration.
    target_ratio = 1.2 + 0.2 * (np.arange(len(base)) % 5)
    amplitude = rate * (base * target_ratio) ** 2 * position
    model = TimingModel(base, 0.5 * base, 3.0 * base, amplitude, rate,
                        generator, background * strength, tuple(edges))
    model.validate()
    return model


def evaluate_timing(model, durations, weights=(1.0, 0.0, 0.0), references=None,
                    *, method="evaluation", elapsed=0.0, solver="none", status="exact",
                    raw_violation=0.0, pressure=None, kkt=None):
    model.validate()
    durations = np.asarray(durations, dtype=float)
    if durations.shape != model.base.shape or not np.isfinite(durations).all() or np.any(durations <= 0):
        raise ValueError("Durations must be finite, positive and match model")
    schedule = critical_path(durations, model.edges)
    risk = float(model.background_rate * schedule.makespan +
                 np.sum(model.amplitude_noise / durations + model.gate_noise_rate * durations))
    effort = float(np.sum(model.generator_squared / durations))
    if references is None:
        references = reference_metrics(model)
    w = np.asarray(weights, dtype=float)
    if w.shape != (3,) or not np.isfinite(w).all() or np.any(w < 0) or np.sum(w) <= 0:
        raise ValueError("Need three finite nonnegative weights with positive sum")
    objective = float(np.dot(w, np.array([risk, schedule.makespan, effort]) / references))
    return TimingResult(durations.copy(), schedule.starts, schedule.makespan, risk,
                        effort, math.exp(-risk), objective, elapsed, method, solver,
                        status, raw_violation, pressure, kkt)


def reference_metrics(model):
    makespan = critical_path(model.base, model.edges).makespan
    risk = model.background_rate * makespan + np.sum(model.amplitude_noise / model.base + model.gate_noise_rate * model.base)
    effort = np.sum(model.generator_squared / model.base)
    return np.array([risk, makespan, effort], dtype=float)


def optimize_timing(model, weights=(1.0, 0.0, 0.0), *, method="gp",
                    solver="CLARABEL", deadline=None, effort_limit=None, references=None):
    """Solve exact fixed-DAG multi-objective timing GP or its convex counterpart.

    Positive weights on risk or effort imply strictly convex duration cost.
    Raw numerical feasibility is retained; returned starts are earliest starts
    for returned durations. A solver label is not a scientific certificate.
    """
    begin = time.perf_counter()
    model.validate()
    w = np.asarray(weights, dtype=float)
    if w.shape != (3,) or not np.isfinite(w).all() or np.any(w < 0) or w[0] <= 0:
        raise ValueError("Need nonnegative finite weights and strictly positive risk weight")
    if method not in ("gp", "dgp", "convex"):
        raise ValueError("method must be gp (explicit log form), dgp (automatic conversion) or convex")
    if solver not in ("CLARABEL", "ECOS", "SCS") or solver not in cp.installed_solvers():
        raise ValueError("Choose an installed free solver: CLARABEL, ECOS or SCS")
    for name, value in (("deadline", deadline), ("effort_limit", effort_limit)):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be positive and finite")
    n = len(model.base)
    scale = float(np.sum(model.upper))
    ref = reference_metrics(model) if references is None else np.asarray(references, dtype=float)
    if ref.shape != (3,) or not np.isfinite(ref).all() or np.any(ref <= 0):
        raise ValueError("References must be three finite strictly positive values")
    q = cp.Variable(n, pos=True)
    finish = cp.Variable(n, pos=True)
    end = cp.Variable(pos=True)
    scaled_base = model.base / scale
    lower = model.lower / model.base
    upper = model.upper / model.base
    a = model.amplitude_noise / model.base
    c = model.gate_noise_rate * model.base
    e = model.generator_squared / model.base
    sources = np.array([i for i, _ in model.edges], dtype=int)
    targets = np.array([j for _, j in model.edges], dtype=int)
    inverse = 1 / q if method in ("gp", "dgp") else cp.inv_pos(q)
    risk_expr = model.background_rate * scale * end + cp.sum(cp.multiply(a, inverse) + cp.multiply(c, q))
    effort_expr = cp.sum(cp.multiply(e, inverse))
    terms = [w[0] * risk_expr / ref[0]]
    if w[1] > 0:
        terms.append(w[1] * scale * end / ref[1])
    if w[2] > 0:
        terms.append(w[2] * effort_expr / ref[2])
    if method in ("gp", "dgp"):
        constraints = [lower / q <= 1, q / upper <= 1,
                       cp.multiply(scaled_base, q) / finish <= 1,
                       finish / end <= 1, end <= 1]
        if len(sources):
            constraints.append((finish[sources] + cp.multiply(scaled_base[targets], q[targets])) / finish[targets] <= 1)
        if deadline is not None:
            constraints.append(end / (deadline / scale) <= 1)
        if effort_limit is not None:
            constraints.append(effort_expr / effort_limit <= 1)
    else:
        constraints = [q >= lower, q <= upper, cp.multiply(scaled_base, q) <= finish,
                       finish <= end, end <= 1]
        if len(sources):
            constraints.append(finish[sources] + cp.multiply(scaled_base[targets], q[targets]) <= finish[targets])
        if deadline is not None:
            constraints.append(end <= deadline / scale)
        if effort_limit is not None:
            constraints.append(effort_expr <= effort_limit)
    problem = cp.Problem(cp.Minimize(sum(terms)), constraints)
    if not (problem.is_dgp() if method in ("gp", "dgp") else problem.is_dcp()):
        raise RuntimeError("Model failed its declared convex grammar")
    if method == "gp":
        # Exact vectorized logarithmic GP transformation. Automatic DGP conversion
        # is retained as an independently tested interface, but expands many scalar
        # expressions in this CVXPY release. No approximation is introduced here.
        log_q=cp.Variable(n)
        log_finish=cp.Variable(n)
        log_end=cp.Variable()
        effective_inverse=w[0]*a/ref[0]+w[2]*e/ref[2]
        effective_linear=w[0]*c/ref[0]
        effective_time=scale*(w[0]*model.background_rate/ref[0]+w[1]/ref[1])
        log_terms=cp.hstack([np.log(effective_inverse)-log_q,
                            np.log(effective_linear)+log_q,
                            np.log(effective_time)+log_end])
        constraints=[log_q>=np.log(lower),log_q<=np.log(upper),
                     np.log(scaled_base)+log_q<=log_finish,
                     log_finish<=log_end,log_end<=0]
        if len(sources):
            pair=cp.vstack([log_finish[sources],np.log(scaled_base[targets])+log_q[targets]])
            constraints.append(cp.log_sum_exp(pair,axis=0)<=log_finish[targets])
        if deadline is not None:
            constraints.append(log_end<=math.log(deadline/scale))
        if effort_limit is not None:
            constraints.append(cp.log_sum_exp(np.log(e)-log_q)<=math.log(effort_limit))
        # Minimize the posynomial itself after variable transformation. Taking
        # its logarithm has the same minimizer, but an additional exponential
        # cone for that epigraph can be poorly conditioned on these long DAGs.
        problem=cp.Problem(cp.Minimize(cp.sum(cp.exp(log_terms))),constraints)
        if not problem.is_dcp():raise RuntimeError("Log GP failed convex grammar")
    options = {"CLARABEL": dict(tol_gap_abs=1e-7, tol_gap_rel=1e-7, tol_feas=1e-8, max_iter=300),
               "ECOS": dict(abstol=1e-8, reltol=1e-8, feastol=1e-8, max_iters=300),
               "SCS": dict(eps=1e-7, max_iters=100000)}
    attempts=[solver]
    if solver=='CLARABEL' and 'ECOS' in cp.installed_solvers():attempts.append('ECOS')
    if solver=='CLARABEL' and 'SCS' in cp.installed_solvers():attempts.append('SCS')
    solver_failures=[]
    for actual_solver in attempts:
        try:
            problem.solve(gp=(method == "dgp"), solver=actual_solver, **options[actual_solver])
        except cp.error.SolverError as error:
            solver_failures.append(f'{actual_solver}: {error}')
            continue
        if problem.status in (cp.OPTIMAL,cp.OPTIMAL_INACCURATE):
            solver=actual_solver
            break
        # A certified infeasibility is an input outcome, not a reason to relax constraints.
        raise RuntimeError(f"Joint timing {method}/{actual_solver} failed: {problem.status}")
    else:
        raise RuntimeError('All free solvers failed: '+'; '.join(solver_failures))
    if method == "gp":
        q_value=np.exp(np.asarray(log_q.value).reshape(-1))
        finish_value=np.exp(np.asarray(log_finish.value).reshape(-1))
        end_value=math.exp(float(log_end.value))
    else:
        q_value=np.asarray(q.value).reshape(-1)
        finish_value=np.asarray(finish.value).reshape(-1)
        end_value=float(end.value)
    durations = q_value * model.base
    raw_starts = finish_value * scale - durations
    raw_end = end_value * scale
    violation = max(constraint_violation(raw_starts, raw_end, durations, model.edges),
                    float(np.max(model.lower - durations)), float(np.max(durations - model.upper)), 0.0)
    if deadline is not None:
        violation = max(violation, raw_end - deadline)
    if violation > 2e-5 * max(1.0, raw_end):
        raise RuntimeError(f"Numerically infeasible timing: residual {violation}")
    if effort_limit is not None and np.sum(model.generator_squared / durations) > effort_limit * (1 + 2e-5):
        raise RuntimeError("Numerically infeasible control effort")
    pressure = kkt = None
    # Dual characterization is reported for the ordinary convex, unconstrained-budget case.
    if method == "convex" and deadline is None and effort_limit is None:
        pressure = np.asarray(constraints[2].dual_value).reshape(-1) / scale
        if len(sources):
            np.add.at(pressure, targets, np.asarray(constraints[5].dual_value).reshape(-1) / scale)
        effective_a = w[0] * model.amplitude_noise / ref[0] + w[2] * model.generator_squared / ref[2]
        effective_c = w[0] * model.gate_noise_rate / ref[0]
        stationarity = (-effective_a / durations ** 2 + effective_c + pressure +
                        (np.asarray(constraints[1].dual_value) - np.asarray(constraints[0].dual_value)) / model.base)
        magnitude = max(float(np.max(effective_a / durations ** 2)), float(np.max(effective_c)), 1e-12)
        kkt = float(np.max(np.abs(stationarity)) / magnitude)
    result = evaluate_timing(model, durations, w, ref, method=method,
                             elapsed=time.perf_counter()-begin, solver=solver,
                             status=problem.status, raw_violation=violation, pressure=pressure, kkt=kkt)
    if solver_failures:
        from dataclasses import replace
        result=replace(result,solver_failures=tuple(solver_failures))
    solved_objective=float(problem.value)
    if abs(result.objective - solved_objective) > 2e-5 * max(1.0, abs(solved_objective)):
        raise RuntimeError("Canonical schedule and numerical objective disagree")
    return result


def robust_upper_model(model, fractional_radius):
    """Exact worst-case risk for independent coefficient intervals; not arbitrary noise."""
    model.validate()
    if not math.isfinite(fractional_radius) or not 0 <= fractional_radius < 1:
        raise ValueError("Interval radius must be in [0,1)")
    # Deliberately unequal uncertainty by coefficient class; a uniform scaling would
    # leave risk-only optimal timing unchanged and be an uninformative experiment.
    return TimingModel(model.base, model.lower, model.upper,
                       model.amplitude_noise * (1 + fractional_radius),
                       model.gate_noise_rate * (1 + 0.5 * fractional_radius),
                       model.generator_squared,
                       model.background_rate * (1 + 2 * fractional_radius), model.edges)


def quantize_timing(model, result, ticks):
    """Round to feasible gate-duration grids, recompute schedule and loss bound.

    Deadlines and effort budgets must be rechecked by the caller after rounding.
    Continuous optimality does not transfer to a discrete grid.
    """
    tick = np.broadcast_to(np.asarray(ticks, dtype=float), model.base.shape)
    if not np.isfinite(tick).all() or np.any(tick <= 0):
        raise ValueError("Timing ticks must be finite and positive")
    lowest = np.ceil(model.lower / tick - 1e-12)
    highest = np.floor(model.upper / tick + 1e-12)
    if np.any(lowest > highest):
        raise ValueError("No feasible timing-grid point within gate bounds")
    rounded = np.clip(np.rint(result.durations / tick), lowest, highest) * tick
    difference = np.abs(rounded - result.durations)
    bound = float(np.dot(model.amplitude_noise / model.lower ** 2 + model.gate_noise_rate + model.background_rate, difference))
    evaluated = evaluate_timing(model, rounded, method="quantized")
    if abs(evaluated.risk-result.risk) > bound + 1e-9:
        raise RuntimeError("Quantization risk bound failed")
    return evaluated, bound


def simulate_timing(circuit, model, result, initial=None, *, amplitude_scale=1.0,
                    gate_rate_scale=1.0, background_scale=1.0):
    """Exact channel composition for the stated local/global depolarizing model.

    Supports mixed density matrices and spectator/reference qubits. Local
    depolarization commutes with its gate and disjoint operations, so fixed DAG
    linearizations produce the same channel. Global depolarization commutes
    with all unital circuit channels and can be applied at the end.
    """
    model.validate()
    _,edges,_,_=extract(circuit)
    if tuple(edges)!=model.edges:
        raise ValueError('Circuit dependency graph and timing model disagree')
    if result.makespan+1e-8*max(1,result.makespan)<critical_path(result.durations,model.edges).makespan:
        raise ValueError('Reported makespan is shorter than required gate schedule')
    n = circuit.num_qubits
    if initial is None:
        initial = Statevector.from_label("0" * n)
    state = DensityMatrix(initial)
    ideal = state
    for name, value in (("amplitude_scale", amplitude_scale), ("gate_rate_scale", gate_rate_scale), ("background_scale", background_scale)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if len(circuit.data) != len(result.durations):
        raise ValueError("Circuit and timing result disagree")
    total = state.num_qubits
    if total < n:
        raise ValueError("Initial state has fewer qubits than circuit")
    for i, instruction in enumerate(circuit.data):
        qargs = [circuit.find_bit(q).index for q in instruction.qubits]
        state = state.evolve(instruction.operation, qargs=qargs)
        ideal = ideal.evolve(instruction.operation, qargs=qargs)
        exposure = amplitude_scale * model.amplitude_noise[i] / result.durations[i] + gate_rate_scale * model.gate_noise_rate[i] * result.durations[i]
        state = (math.exp(-exposure) * state.data + (-math.expm1(-exposure)) * replace_subsystem(state.data, qargs, total))
        state = DensityMatrix(state)
    exposure = background_scale * model.background_rate * result.makespan
    state = DensityMatrix(math.exp(-exposure) * state.data + (-math.expm1(-exposure)) * replace_subsystem(state.data, list(range(n)), total))
    if abs(np.trace(state.data)-1)>1e-9 or np.linalg.eigvalsh(state.data).min() < -1e-9:
        raise RuntimeError("Invalid noisy density matrix")
    return state, ideal


def replace_subsystem(density, qargs, total_qubits):
    """T_A(rho)=I_A/d_A tensor Tr_A(rho), with explicit Qiskit little-endian order."""
    selected = tuple(sorted(qargs))
    if len(set(selected)) != len(selected) or any(q < 0 or q >= total_qubits for q in selected):
        raise ValueError("Invalid subsystem qubits")
    kept = tuple(q for q in range(total_qubits) if q not in selected)
    reduced = partial_trace(DensityMatrix(density), list(selected)).data
    dimension = 2 ** total_qubits
    indices = np.arange(dimension)
    selected_index = sum((((indices >> q) & 1) << j for j, q in enumerate(selected)), np.zeros_like(indices))
    kept_index = sum((((indices >> q) & 1) << j for j, q in enumerate(kept)), np.zeros_like(indices))
    return reduced[np.ix_(kept_index, kept_index)] * (selected_index[:, None] == selected_index[None, :]) / (2 ** len(selected))
