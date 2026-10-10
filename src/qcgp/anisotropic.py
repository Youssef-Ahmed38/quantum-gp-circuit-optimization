"""C1/C5: anisotropic, state-aware observable constraints (still exact GPs).

Generalising the isotropic model, every local Pauli ``P`` of timed gate ``j``
has its own intensity ``a_jP / tau_j + g_jP tau_j``. For a Clifford circuit and
a stabilizer generator ``S_k`` of the ideal output,

    <S_k> = exp(-2 sum_j sum_{P anticommuting at j} (a_jP / tau_j + g_jP tau_j)),

so ``<S_k> >= t_k`` is a posynomial constraint with coefficient matrices
``A_kj = sum_P anti_kjP a_jP`` and ``G_kj = sum_P anti_kjP g_jP``.

Proposition (basis invariance under isotropy). If every gate's channel is
isotropic, every weight-one Pauli observable decays by the same factor for a
given schedule, so the ranking of schedules by single-qubit Pauli survival is
identical in the X, Y and Z bases. Any measured basis-dependent ranking
therefore falsifies isotropy, and anisotropic intensities are the minimal
convex extension able to represent it.

State awareness. Amplitude damping is non-unital: a qubit in |0> does not decay.
For Clifford circuits the ideal Bloch vector of each qubit in each idle window
is known exactly from stabilizer tracking, so the T1 exposure of a window can
be weighted by its excited-state population ``(1 - <Z_q>)/2``. The weight is a
nonnegative constant, so the program remains a GP.
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from qiskit.quantum_info import Clifford, Pauli

from .core import critical_path
from .observables import _binary_gate, local_paulis


@dataclass(frozen=True)
class AnisotropicModel:
    base: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    A: np.ndarray        # (K, m) inverse-duration coefficients per observable
    G: np.ndarray        # (K, m) linear coefficients per observable
    edges: tuple
    labels: tuple

    def expectations(self, tau):
        tau = np.asarray(tau, dtype=float)
        return np.exp(-2 * (self.A / tau[None, :] + self.G * tau[None, :]).sum(axis=1))


def anticommutation(circuit):
    """Per timed gate: (timed indices, list of (K x L_j) anticommutation arrays, labels)."""
    clifford = Clifford(circuit)
    labels = tuple(clifford.to_labels(mode="S"))
    observables = [Pauli(label) for label in labels]
    x = np.stack([p.x for p in observables])
    z = np.stack([p.z for p in observables])
    timed = [i for i, ins in enumerate(circuit.data) if ins.operation.name != "rz"]
    tables = {}
    for index in reversed(range(len(circuit.data))):
        ins = circuit.data[index]
        qs = tuple(circuit.find_bit(q).index for q in ins.qubits)
        if index in timed:
            columns = []
            for px, pz in local_paulis(qs):
                columns.append(np.logical_xor.reduce((x[:, qs] & pz) ^ (z[:, qs] & px), axis=1))
            tables[index] = np.stack(columns, axis=1)
        _binary_gate(x, z, ins.operation.name, qs, ins.operation.params)
    return timed, [tables[i] for i in timed], labels


def build_model(circuit, base, lower, upper, per_pauli_inverse, per_pauli_linear):
    """``per_pauli_*[j]`` are arrays over the local Paulis of timed gate ``j``."""
    timed, tables, labels = anticommutation(circuit)
    K, m = len(labels), len(timed)
    A = np.zeros((K, m))
    G = np.zeros((K, m))
    for j, table in enumerate(tables):
        A[:, j] = table @ np.asarray(per_pauli_inverse[j], dtype=float)
        G[:, j] = table @ np.asarray(per_pauli_linear[j], dtype=float)
    last, edges = {}, set()
    for j, index in enumerate(timed):
        for q in circuit.data[index].qubits:
            q = circuit.find_bit(q).index
            if q in last:
                edges.add((last[q], j))
            last[q] = j
    return AnisotropicModel(np.asarray(base, float), np.asarray(lower, float), np.asarray(upper, float),
                            A, G, tuple(sorted(edges)), labels)


def isotropic_split(total_inverse, total_linear, arity):
    """Isotropic per-Pauli coefficients reproducing a total gate intensity."""
    count = 4 ** arity - 1
    return np.full(count, total_inverse / count), np.full(count, total_linear / count)


def optimize_anisotropic(model: AnisotropicModel, targets):
    """Minimum makespan with every ``<S_k> >= targets[k]`` (exact GP)."""
    targets = np.broadcast_to(np.asarray(targets, dtype=float), (len(model.labels),))
    if np.any(targets <= 0) or np.any(targets >= 1):
        raise ValueError("Targets must lie in (0, 1)")
    m = len(model.base)
    scale = model.base
    horizon = critical_path(model.base, model.edges).makespan
    x = cp.Variable(m, pos=True)
    finish = cp.Variable(m, pos=True)
    end = cp.Variable(pos=True)
    width = scale / horizon
    constraints = [model.lower / scale <= x, x <= model.upper / scale,
                   cp.multiply(width, x) <= finish, finish <= end]
    constraints += [(finish[i] + width[j] * x[j]) / finish[j] <= 1 for i, j in model.edges]
    for k, target in enumerate(targets):
        budget = -0.5 * np.log(target)
        terms = [model.A[k, j] / scale[j] / budget * cp.power(x[j], -1) for j in range(m) if model.A[k, j] > 0]
        terms += [model.G[k, j] * scale[j] / budget * x[j] for j in range(m) if model.G[k, j] > 0]
        if terms:
            constraints.append(sum(terms) <= 1)
    problem = cp.Problem(cp.Minimize(end), constraints)
    problem.solve(gp=True, solver="CLARABEL")
    if x.value is None:
        return dict(feasible=False, status=problem.status)
    tau = np.clip(x.value * scale, model.lower, model.upper)
    return dict(feasible=True, durations=tau, makespan=critical_path(tau, model.edges).makespan,
                expectations=model.expectations(tau), status=problem.status)


# ---------------------------------------------------------------------------
# Single-qubit driven/idle physics for the native-X experiment (spin locking)
# ---------------------------------------------------------------------------

def _su2_step(omega_x, delta, dt):
    """Batched exp(-i dt (omega_x X + delta Z) / 2)."""
    norm = np.sqrt(omega_x ** 2 + delta ** 2)
    angle = 0.5 * norm * dt
    c = np.cos(angle)
    s = np.where(norm > 0, np.sin(angle) / np.where(norm > 0, norm, 1), 0.5 * dt)
    u = np.empty(omega_x.shape + (2, 2), dtype=complex)
    u[..., 0, 0] = c - 1j * s * delta
    u[..., 1, 1] = c + 1j * s * delta
    u[..., 0, 1] = -1j * s * omega_x
    u[..., 1, 0] = -1j * s * omega_x
    return u


@dataclass(frozen=True)
class SingleQubitPhysics:
    t1: float = 300.0            # us, true (non-unital) amplitude damping
    white_dephasing: float = 1 / 600.0
    quasi_static: float = 0.03   # rad/us
    spectral_s0: float = 3e-4
    spectral_beta: float = 1.0
    amplitude_noise: float = 0.002  # quasi-static fractional Rabi error


def run_x_sequence(policy, basis, gates, physics: SingleQubitPhysics, *, slow=1.0, fast=0.036,
                   dt=0.004, trajectories=400, seed=0):
    """Lindblad + classical-noise integration of the four native-X policies.

    Every policy has the same total duration ``gates * slow``. Returns the
    probability of returning to the prepared basis state.
    """
    total = gates * slow

    def drive(t):
        if policy == "slow":
            return np.full_like(t, np.pi / slow)
        if policy == "distributed":
            return np.where((t % slow) < fast, np.pi / fast, 0.)
        burst = (t < gates * fast)
        if policy == "tail":
            return np.where(burst, np.pi / fast, 0.)
        if policy == "dd_tail":
            tail = total - gates * fast - 2 * fast
            p1 = gates * fast + tail / 4
            p2 = gates * fast + 2 * fast + 3 * tail / 4 - fast
            dd = ((t >= p1) & (t < p1 + fast)) | ((t >= p2) & (t < p2 + fast))
            return np.where(burst | dd, np.pi / fast, 0.)
        raise ValueError(policy)
    return simulate_drive(drive, total, {"Z": "0", "X": "+"}[basis], physics, dt=dt,
                          trajectories=trajectories, seed=seed)


def simulate_drive(drive, total, state, physics: SingleQubitPhysics, *, dt=0.004, trajectories=400, seed=0):
    """Return probability of the prepared state ('0', '1' or '+') after ``total``.

    ``drive(t)`` gives the X-drive Rabi rate (rad/us); the ideal sequence must be
    the identity for the result to be a survival probability.
    """
    rng = np.random.default_rng(seed)
    steps = int(round(total / dt))
    times = (np.arange(steps) + 0.5) * dt
    amplitude = drive(times)
    # Classical detuning realisations.
    offset = rng.normal(0, physics.quasi_static, trajectories)
    freq = np.geomspace(1e-4, 20., 120)
    edges = np.sqrt(freq[:-1] * freq[1:])
    width = np.diff(np.concatenate([[1e-4], edges, [20.]]))
    sd = np.sqrt(physics.spectral_s0 * freq ** -physics.spectral_beta * width)
    a = rng.normal(size=(trajectories, len(freq))) * sd
    b = rng.normal(size=(trajectories, len(freq))) * sd
    rabi_error = 1 + rng.normal(0, physics.amplitude_noise, trajectories)
    if state in ("0", "1"):
        rho = np.zeros((trajectories, 2, 2), complex)
        rho[:, int(state), int(state)] = 1
    else:
        rho = np.full((trajectories, 2, 2), 0.5, complex)
    g1 = 1 / physics.t1
    gphi = physics.white_dephasing
    decay = np.exp(-g1 * dt)
    for k in range(steps):
        t = times[k]
        delta = offset + a @ np.cos(2 * np.pi * freq * t) + b @ np.sin(2 * np.pi * freq * t)
        u = _su2_step(amplitude[k] * rabi_error, delta, dt)
        rho = u @ rho @ np.conj(np.swapaxes(u, 1, 2))
        # Exact amplitude damping and pure dephasing for one step.
        p1 = rho[:, 1, 1].real
        rho[:, 0, 0] += (1 - decay) * p1
        rho[:, 1, 1] *= decay
        coherence = np.sqrt(decay) * np.exp(-gphi * dt)
        rho[:, 0, 1] *= coherence
        rho[:, 1, 0] *= coherence
    if state in ("0", "1"):
        return float(np.mean(rho[:, int(state), int(state)].real))
    return float(np.mean(0.5 + rho[:, 0, 1].real))
