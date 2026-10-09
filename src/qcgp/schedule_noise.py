"""Small-circuit, local density-matrix stress tests for scheduled circuits.

This is a declared simulation model, not a device prediction.  A detuning draw
is constant throughout each circuit execution; Markovian idle decay is separate.
"""
from dataclasses import dataclass
from itertools import product
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import XGate
from qiskit.quantum_info import DensityMatrix, Kraus, Pauli, Statevector
from qiskit.transpiler import InstructionDurations, PassManager
from qiskit.transpiler.passes import (
    ALAPScheduleAnalysis, ASAPScheduleAnalysis, PadDelay, PadDynamicalDecoupling,
)


@dataclass(frozen=True)
class NoiseScenario:
    name: str
    t1_ns: float
    tphi_ns: float
    detuning_std_rad_per_ns: float
    px: float
    py: float
    pz: float


def example_circuit():
    circuit = QuantumCircuit(3)
    circuit.sx(0)
    circuit.cz(0, 1)
    circuit.sx(2)
    circuit.cz(1, 2)
    circuit.sx(0)
    return circuit


def instruction_durations():
    # 1 dt = 1 ns; these values match the saved Torino calibration's native
    # sx/x and CZ durations on the selected three-qubit chain.
    entries = [(gate, [q], 32, "dt") for gate in ("sx", "x") for q in range(3)]
    entries += [("cz", [0, 1], 68, "dt"), ("cz", [1, 2], 68, "dt")]
    return InstructionDurations(entries, dt=1e-9)


def schedule(circuit, policy, durations=None):
    if policy not in ("ASAP", "ALAP", "ASAP+DD", "ALAP+DD"):
        raise ValueError("Unknown scheduling policy")
    durations = instruction_durations() if durations is None else durations
    analysis = ASAPScheduleAnalysis if policy.startswith("ASAP") else ALAPScheduleAnalysis
    if policy.endswith("+DD"):
        padding = PadDynamicalDecoupling(
            durations, dd_sequence=[XGate(), XGate()], skip_reset_qubits=False,
            pulse_alignment=1,
        )
    else:
        padding = PadDelay(durations=durations)
    return PassManager([analysis(durations), padding]).run(circuit)


def timeline(circuit, durations=None):
    """Extract per-qubit start/finish times from a fully padded circuit."""
    durations = instruction_durations() if durations is None else durations
    clocks = [0] * circuit.num_qubits
    events = []
    for inst in circuit.data:
        qubits = tuple(circuit.find_bit(q).index for q in inst.qubits)
        start = max(clocks[q] for q in qubits)
        name = inst.operation.name
        length = int(inst.operation.duration) if name == "delay" else int(durations.get(name, qubits))
        for q in qubits:
            if clocks[q] != start:
                raise ValueError("Circuit is not fully padded")
            clocks[q] = start + length
        events.append((name, qubits, start, length))
    if len(set(clocks)) != 1:
        raise ValueError("Unequal qubit makespans")
    return events, clocks[0]


def _pauli_channel(state, qubit, px, py, pz):
    if min(px, py, pz) < 0 or px + py + pz > 1:
        raise ValueError("Invalid Pauli probabilities")
    result = (1 - px - py - pz) * state.data
    for label, p in (("X", px), ("Y", py), ("Z", pz)):
        if p:
            result += p * state.evolve(Pauli(label), qargs=[qubit]).data
    return DensityMatrix(result)


def _idle(state, qubit, length_ns, detuning, noise):
    if not length_ns:
        return state
    angle = detuning * length_ns
    rotation = np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])
    state = state.evolve(rotation, qargs=[qubit])
    if math.isfinite(noise.t1_ns):
        gamma = -math.expm1(-length_ns / noise.t1_ns)
        k0 = np.diag([1., math.sqrt(1 - gamma)])
        k1 = np.array([[0., math.sqrt(gamma)], [0., 0.]])
        state = state.evolve(Kraus([k0, k1]), qargs=[qubit])
    if math.isfinite(noise.tphi_ns):
        pz = -0.5 * math.expm1(-length_ns / noise.tphi_ns)
        state = _pauli_channel(state, qubit, 0., 0., pz)
    return state


def state_fidelity(circuit, scheduled, noise, quadrature_order=7):
    """Exact density propagation, Gaussian quadrature over static Z detuning."""
    if circuit.num_qubits > 4 or scheduled.num_qubits != circuit.num_qubits:
        raise ValueError("At most four matching qubits supported")
    if min(noise.t1_ns, noise.tphi_ns) <= 0 or noise.detuning_std_rad_per_ns < 0:
        raise ValueError("Invalid noise time or detuning")
    ideal = Statevector.from_instruction(circuit).data
    events, _ = timeline(scheduled)
    if noise.detuning_std_rad_per_ns:
        nodes, weights = np.polynomial.hermite.hermgauss(quadrature_order)
        draws = math.sqrt(2) * noise.detuning_std_rad_per_ns * nodes
        weights = weights / math.sqrt(math.pi)
    else:
        draws, weights = np.array([0.]), np.array([1.])
    average = np.zeros((len(ideal), len(ideal)), dtype=complex)
    for choices in product(range(len(draws)), repeat=circuit.num_qubits):
        detuning = draws[list(choices)]
        weight = float(np.prod(weights[list(choices)]))
        state = DensityMatrix.from_label("0" * circuit.num_qubits)
        # Qiskit preserves the operation order of this padded DAG. Operations
        # at overlapping wall times act on disjoint qubits and commute.
        for (name, qubits, _, length), inst in zip(events, scheduled.data):
            if name == "delay":
                state = _idle(state, qubits[0], length, detuning[qubits[0]], noise)
            else:
                state = state.evolve(inst.operation, qargs=list(qubits))
                if name != "rz":
                    for qubit in qubits:
                        state = _pauli_channel(state, qubit, noise.px, noise.py, noise.pz)
        average += weight * state.data
    return float(np.real(np.vdot(ideal, average @ ideal)))
