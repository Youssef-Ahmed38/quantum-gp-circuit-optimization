"""Independent time-correlated noise simulator for scheduled circuits.

This simulator is deliberately *not* the optimisation model. It evolves batched
density matrices under:

* classical detuning noise per qubit: a quasi-static Gaussian offset plus a
  spectral sum approximating ``S(f) = s0 / f^beta`` on ``[f_min, f_max]``
  (common random numbers across policies; one realisation per trajectory);
* Markovian amplitude damping and white dephasing, Pauli-twirled;
* duration-dependent isotropic Pauli gate errors (a declared *truth* law);
* imperfect DD pulses (depolarising error per pulse).

Pauli channels are applied exactly to each trajectory's density matrix, so the
only sampling noise is over classical detuning realisations. Dephasing during a
driven gate is applied after that gate (an approximation stated in the paper).
Units: microseconds and rad/us.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
S = np.diag([1, 1j])
SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}


def rz(theta):
    return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)])


def ry(theta):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
CZ = np.diag([1, 1, 1, -1]).astype(complex)


@dataclass(frozen=True)
class Gate:
    """One instruction. ``timed=False`` marks virtual (zero-duration) frame gates.

    Two-qubit matrices use the ordering ``|q0 q1>`` with ``q0 = qubits[0]`` most
    significant. ``block=True`` marks a fixed-duration identity stand-in (e.g. the
    latency of a mid-circuit measurement on that qubit).
    """
    name: str
    qubits: tuple
    matrix: np.ndarray
    timed: bool = True
    block: bool = False


def gate(name, *qubits, theta=None):
    table = {"h": H, "x": X, "y": Y, "z": Z, "s": S, "sx": SX, "cx": CX, "cz": CZ}
    if name == "rz":
        return Gate(name, qubits, rz(theta), timed=False)
    if name == "ry":
        return Gate(name, qubits, ry(theta))
    if name == "block":
        return Gate(name, qubits, I2, block=True)
    return Gate(name, qubits, table[name])


@dataclass(frozen=True)
class QubitPhysics:
    """Truth-side noise for one qubit (not visible to the optimiser)."""
    t1: float = 250.0                # us
    white_dephasing: float = 1 / 400.0  # 1/us, Markovian
    quasi_static: float = 0.02       # rad/us standard deviation
    spectral_s0: float = 0.0         # rad^2/us^2 per MHz at f = 1 MHz
    spectral_beta: float = 1.0
    f_min: float = 1e-4              # MHz
    f_max: float = 10.0              # MHz
    components: int = 160
    pulse_error: float = 2e-4        # depolarising probability per DD pulse
    exact_damping: bool = False      # True: non-unital amplitude damping (|0> does not decay)


@dataclass(frozen=True)
class GateLaw:
    """Truth gate law: total Pauli intensity ``a / t^p + g t`` for each timed gate."""
    inverse: np.ndarray
    linear: np.ndarray
    power: float = 1.0

    def intensity(self, index, duration):
        return self.inverse[index] / duration ** self.power + self.linear[index] * duration


@dataclass
class ScheduledCircuit:
    """Gates with start times for timed gates and DD pulses per qubit.

    ``pulses[q]`` is a sorted list of ``(time, 'X' | 'Y')``.
    """
    n: int
    gates: list
    starts: dict          # gate index -> start (timed gates only)
    durations: dict       # gate index -> duration (timed gates only)
    makespan: float
    pulses: dict = field(default_factory=dict)


class DetuningNoise:
    """Per-trajectory classical detuning realisations with analytic phase integrals."""

    def __init__(self, physics, trajectories, rng):
        self.physics = physics
        self.offset, self.omega, self.cos, self.sin = [], [], [], []
        for p in physics:
            self.offset.append(rng.normal(0.0, p.quasi_static, size=trajectories))
            if p.spectral_s0 > 0:
                f = np.geomspace(p.f_min, p.f_max, p.components)
                edges = np.sqrt(f[:-1] * f[1:])
                width = np.diff(np.concatenate([[p.f_min], edges, [p.f_max]]))
                # One-sided S(f) df per component: delta = sum a cos(wt) + b sin(wt).
                sd = np.sqrt(p.spectral_s0 * f ** -p.spectral_beta * width)
                self.omega.append(2 * np.pi * f)
                self.cos.append(rng.normal(size=(trajectories, len(f))) * sd)
                self.sin.append(rng.normal(size=(trajectories, len(f))) * sd)
            else:
                self.omega.append(None)
                self.cos.append(None)
                self.sin.append(None)

    def phase_integral(self, q, t0, t1):
        """Integral of the detuning over [t0, t1] for each trajectory (radians)."""
        value = self.offset[q] * (t1 - t0)
        if self.omega[q] is not None:
            w = self.omega[q]
            value = value + self.cos[q] @ ((np.sin(w * t1) - np.sin(w * t0)) / w)
            value = value + self.sin[q] @ ((np.cos(w * t0) - np.cos(w * t1)) / w)
        return value


class BatchDensity:
    """Batched n-qubit density matrices with shape (T, 2^n, 2^n)."""

    def __init__(self, n, trajectories):
        self.n = n
        self.t = trajectories
        d = 2 ** n
        self.rho = np.zeros((trajectories, d, d), dtype=complex)
        self.rho[:, 0, 0] = 1.0

    def _tensor(self):
        return self.rho.reshape((self.t,) + (2,) * (2 * self.n))

    def apply(self, matrix, qubits):
        """Apply a common unitary on ``qubits`` (qubits[0] most significant)."""
        k = len(qubits)
        tensor = self._tensor()
        u = matrix.reshape((2,) * (2 * k))
        rows = [1 + q for q in qubits]
        cols = [1 + self.n + q for q in qubits]
        tensor = np.tensordot(u, tensor, axes=(list(range(k, 2 * k)), rows))
        tensor = np.moveaxis(tensor, list(range(k)), rows)
        tensor = np.tensordot(u.conj(), tensor, axes=(list(range(k, 2 * k)), cols))
        tensor = np.moveaxis(tensor, list(range(k)), cols)
        self.rho = tensor.reshape(self.rho.shape)

    def zphase(self, q, phi):
        """Per-trajectory exp(-i phi Z / 2) on qubit q."""
        bit = (np.arange(2 ** self.n) >> (self.n - 1 - q)) & 1
        sign = 1 - 2 * bit  # +1 for |0>, -1 for |1>
        factor = np.exp(-0.5j * phi[:, None] * sign[None, :])
        self.rho = self.rho * factor[:, :, None] * np.conj(factor)[:, None, :]

    def pauli_channel(self, qubits, probabilities):
        """rho -> (1 - sum p) rho + sum_P p_P P rho P for Pauli strings on qubits."""
        total = sum(probabilities.values())
        if total <= 0:
            return
        original = self.rho
        result = (1 - total) * original
        for label, p in probabilities.items():
            if p <= 0:
                continue
            self.rho = original
            for q, letter in zip(qubits, label):
                if letter != "I":
                    self.apply(PAULI[letter], (q,))
            result = result + p * self.rho
        self.rho = result

    def kraus(self, q, operators):
        """rho -> sum_k K rho K^dagger on qubit q."""
        original = self.rho
        result = 0.
        for k in operators:
            self.rho = original
            self.apply(k, (q,))
            result = result + self.rho
        self.rho = result

    def fidelity(self, psi):
        return np.real(np.einsum("i,tij,j->t", psi.conj(), self.rho, psi))


def _pauli_labels(k):
    letters = "IXYZ"
    labels = []
    for code in range(1, 4 ** k):
        labels.append("".join(letters[(code // 4 ** i) % 4] for i in range(k)))
    return labels


def isotropic(intensity, k):
    """Independent-generator isotropic Pauli channel with total intensity."""
    labels = _pauli_labels(k)
    p = -0.5 * np.expm1(-2 * intensity / len(labels))
    # Composing independent single-Pauli channels; first-order exact enough is not
    # acceptable here, so compose exactly by sequential application in the caller.
    return labels, p


def markov_probabilities(physics, duration):
    """Pauli-twirled amplitude damping + white dephasing over ``duration``."""
    px = (1 - math.exp(-duration / physics.t1)) / 4
    t2 = 1 / (1 / (2 * physics.t1) + physics.white_dephasing)
    pz = max(0.0, (1 - math.exp(-duration / t2)) / 2 - px)
    return {"X": px, "Y": px, "Z": pz}


def ideal_state(n, gates):
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1
    tensor = psi.reshape((2,) * n)
    for g in gates:
        k = len(g.qubits)
        u = g.matrix.reshape((2,) * (2 * k))
        tensor = np.tensordot(u, tensor, axes=(list(range(k, 2 * k)), list(g.qubits)))
        tensor = np.moveaxis(tensor, list(range(k)), list(g.qubits))
    return tensor.reshape(-1)


def simulate(schedule: ScheduledCircuit, physics, law: GateLaw, *, trajectories=256, seed=0):
    """Return per-trajectory state fidelity with the ideal output."""
    n = schedule.n
    rng = np.random.default_rng(seed)
    noise = DetuningNoise(physics, trajectories, rng)
    state = BatchDensity(n, trajectories)
    clock = np.zeros(n)  # time up to which each qubit's idle evolution is applied
    timed = sorted(schedule.starts)
    order = {g: i for i, g in enumerate(timed)}

    def markov(q, duration):
        if physics[q].exact_damping:
            p = 1 - math.exp(-duration / physics[q].t1)
            state.kraus(q, [np.diag([1, math.sqrt(1 - p)]).astype(complex),
                            np.array([[0, math.sqrt(p)], [0, 0]], dtype=complex)])
            state.pauli_channel((q,), {"Z": -0.5 * math.expm1(-physics[q].white_dephasing * duration)})
        else:
            state.pauli_channel((q,), markov_probabilities(physics[q], duration))

    def advance(q, t):
        if t > clock[q] + 1e-12:
            state.zphase(q, noise.phase_integral(q, clock[q], t))
            markov(q, t - clock[q])
            clock[q] = t

    events = []
    for index, g in enumerate(schedule.gates):
        if g.timed:
            events.append((schedule.starts[index], 0, index))
    for q, sequence in schedule.pulses.items():
        for t, kind in sequence:
            events.append((t, 1, (q, kind)))
    events.sort(key=lambda e: (e[0], e[1], e[2] if e[1] == 0 else -1))
    # A virtual (frame) gate is applied immediately before the next timed gate on
    # its qubit, or at the end. DD blocks are identities up to phase, so moving a
    # frame change past a complete DD block does not change the ideal operation.
    attached = {index: [] for index in range(len(schedule.gates))}
    trailing = []
    for index, g in enumerate(schedule.gates):
        if g.timed:
            continue
        (q,) = g.qubits
        later = [j for j in range(index + 1, len(schedule.gates))
                 if schedule.gates[j].timed and q in schedule.gates[j].qubits]
        (attached[later[0]] if later else trailing).append(index)

    for time, kind, payload in events:
        if kind == 1:
            q, letter = payload
            advance(q, time)
            state.apply(PAULI[letter], (q,))
            p = physics[q].pulse_error
            state.pauli_channel((q,), {"X": p / 3, "Y": p / 3, "Z": p / 3})
            continue
        index = payload
        g = schedule.gates[index]
        start, duration = schedule.starts[index], schedule.durations[index]
        for q in g.qubits:
            advance(q, start)
        for virtual in attached[index]:
            state.apply(schedule.gates[virtual].matrix, schedule.gates[virtual].qubits)
        if not g.block:
            state.apply(g.matrix, g.qubits)
            labels, p = isotropic(law.intensity(order[index], duration), len(g.qubits))
            for label in labels:
                state.pauli_channel(g.qubits, {label: p})
        for q in g.qubits:
            # Dephasing and Markov decay during the gate, applied after it.
            state.zphase(q, noise.phase_integral(q, start, start + duration))
            markov(q, duration)
            clock[q] = start + duration
    for virtual in trailing:
        state.apply(schedule.gates[virtual].matrix, schedule.gates[virtual].qubits)
    for q in range(n):
        advance(q, schedule.makespan)
    return state.fidelity(ideal_state(n, schedule.gates))


def relaxation_spectroscopy(physics: QubitPhysics, windows, *, trajectories=200, seed=2):
    """Simulated T1 experiment: X, wait w, X; returns -ln P(return) / w per window."""
    rows = []
    for w in windows:
        gates = [gate("x", 0), gate("x", 0)]
        sched = ScheduledCircuit(1, gates, {0: 0.0, 1: 0.05 + w}, {0: 0.05, 1: 0.05}, 0.1 + w, {})
        fidelity = float(np.mean(simulate(sched, [physics], GateLaw(np.zeros(2), np.zeros(2)),
                                          trajectories=trajectories, seed=seed)))
        rows.append(dict(window=w, survival=fidelity, rate=-math.log(max(fidelity, 1e-12)) / w))
    return rows


def spectroscopy(physics: QubitPhysics, windows, pulse_counts, *, trajectories=2000, seed=1,
                 pulse_duration=0.036):
    """Simulated CPMG/Ramsey decay of |+> for one qubit: returns hazard -ln C."""
    rows = []
    for w in windows:
        for count in pulse_counts:
            if count * pulse_duration > w:
                continue
            gates = [gate("h", 0), gate("h", 0)]
            pulses = {0: [(0.05 + w * (2 * i + 1) / (2 * count), "X") for i in range(count)]} if count else {}
            sched = ScheduledCircuit(1, gates, {0: 0.0, 1: 0.05 + w}, {0: 0.05, 1: 0.05},
                                     0.1 + w, pulses)
            law = GateLaw(np.zeros(2), np.zeros(2))
            fidelity = simulate(sched, [physics], law, trajectories=trajectories, seed=seed)
            # Ideal output |0>; for odd counts the ideal is still |0> since X^2k = I only
            # for even counts. Odd counts are not used.
            contrast = 2 * float(np.mean(fidelity)) - 1
            rows.append(dict(window=w, pulses=count, contrast=contrast,
                             hazard=-math.log(max(contrast, 1e-9))))
    return rows
