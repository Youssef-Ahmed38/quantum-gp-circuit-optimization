"""Equal-deadline, local comparison of idealized duration GP and Qiskit schedulers.

Variable-duration gates are hypothetical: this module creates no pulse shapes.
The calibrated snapshot supplies only nominal error and duration anchors.
"""
from dataclasses import dataclass
from itertools import product
import math

import numpy as np
from qiskit.quantum_info import DensityMatrix, Statevector

from .generalized import GeneralizedProblem, solve_generalized
from .schedule_noise import _idle, _pauli_channel, timeline, NoiseScenario


@dataclass(frozen=True)
class TimedEvent:
    name: str
    qubits: tuple[int, ...]
    start_ns: float
    duration_ns: float
    operation: object | None


def native_edges(circuit):
    last = {}
    edges = set()
    for index, inst in enumerate(circuit.data):
        qubits = tuple(circuit.find_bit(q).index for q in inst.qubits)
        for qubit in qubits:
            if qubit in last:
                edges.add((last[qubit], index))
            last[qubit] = index
    return tuple(sorted(edges))


def variable_schedule(circuit, durations_ns, deadline_ns, placement="ASAP"):
    """Place fixed-order variable-length gates and pad all qubit timelines."""
    durations = np.asarray(durations_ns, dtype=float)
    if durations.shape != (len(circuit.data),) or not np.isfinite(durations).all() or np.any(durations <= 0):
        raise ValueError("One positive finite duration is required per gate")
    if not np.isfinite(deadline_ns) or deadline_ns <= 0 or placement not in ("ASAP", "ALAP"):
        raise ValueError("Invalid deadline or placement")
    qubits = [tuple(circuit.find_bit(q).index for q in inst.qubits) for inst in circuit.data]
    starts = np.zeros(len(durations))
    if placement == "ASAP":
        available = np.zeros(circuit.num_qubits)
        for i, qs in enumerate(qubits):
            starts[i] = max(available[q] for q in qs)
            for q in qs:
                available[q] = starts[i] + durations[i]
    else:
        next_start = np.full(circuit.num_qubits, float(deadline_ns))
        for i in reversed(range(len(durations))):
            starts[i] = min(next_start[q] for q in qubits[i]) - durations[i]
            for q in qubits[i]:
                next_start[q] = starts[i]
    if np.min(starts) < -1e-6 or np.max(starts + durations) > deadline_ns + 1e-6:
        raise ValueError("Duration vector cannot meet deadline")
    gates = [TimedEvent(inst.operation.name, qubits[i], float(starts[i]), float(durations[i]), inst.operation)
             for i, inst in enumerate(circuit.data)]
    padded = list(gates)
    for q in range(circuit.num_qubits):
        occupied = sorted((event for event in gates if q in event.qubits), key=lambda e: e.start_ns)
        cursor = 0.
        for event in occupied:
            gap = event.start_ns - cursor
            if gap < -1e-7:
                raise ValueError("Overlapping gates on one qubit")
            if gap > 1e-9:
                padded.append(TimedEvent("delay", (q,), cursor, gap, None))
            cursor = event.start_ns + event.duration_ns
        if deadline_ns - cursor > 1e-9:
            padded.append(TimedEvent("delay", (q,), cursor, deadline_ns - cursor, None))
    padded.sort(key=lambda e: (e.start_ns, e.name == "delay", e.qubits))
    return padded


def qiskit_events(scheduled):
    records, deadline = timeline(scheduled)
    events = [TimedEvent(name, qubits, float(start), float(length),
                         None if name == "delay" else inst.operation)
              for (name, qubits, start, length), inst in zip(records, scheduled.data)]
    events.sort(key=lambda e: (e.start_ns, e.name == "delay", e.qubits))
    return events, float(deadline)


def nominal_snapshot(circuit, snapshot):
    """Extract matched nominal anchors; no duration-curve parameters inferred."""
    table = {(g["gate"], tuple(g["qubits"])): {p["name"]: p for p in g["parameters"]}
             for g in snapshot["gates"]}
    base, error = [], []
    for inst in circuit.data:
        name = inst.operation.name
        qs = tuple(circuit.find_bit(q).index for q in inst.qubits)
        entry = table.get((name, qs))
        if entry is None and name == "cz":
            entry = table.get((name, qs[::-1]))
        if entry is None:
            raise ValueError(f"Missing nominal calibration for {name} on {qs}")
        length = entry["gate_length"]
        if length["unit"] != "ns":
            raise ValueError("This study requires nanosecond nominal lengths")
        base.append(float(length["value"]))
        error.append(float(entry["gate_error"]["value"]))
    return np.asarray(base), np.asarray(error)


def snapshot_anchors(snapshot, selected_qubits=(0, 1, 2)):
    selected = set(selected_qubits)
    anchors = {}
    for gate in snapshot["gates"]:
        qs = tuple(gate["qubits"])
        if gate["gate"] not in ("sx", "x", "cz") or not set(qs) <= selected:
            continue
        entry = {p["name"]: p for p in gate["parameters"]}
        if "gate_error" not in entry or "gate_length" not in entry:
            continue
        if entry["gate_length"]["unit"] != "ns":
            continue
        anchors[(gate["gate"], qs)] = (float(entry["gate_length"]["value"]),
                                       float(entry["gate_error"]["value"]))
    return anchors


def gp_policy(circuit, base_ns, nominal_error, deadline_ns, assumed_power=1., alpha=.8):
    if not 0 < alpha < 1:
        raise ValueError("Drive fraction must lie strictly between zero and one")
    base = np.asarray(base_ns, dtype=float)
    error = np.asarray(nominal_error, dtype=float)
    if base.shape != error.shape or np.any(base <= 0) or np.any(error <= 0):
        raise ValueError("Positive matching nominal anchors required")
    model = GeneralizedProblem(lower=.5 * base, upper=1.5 * base,
                               inverse=alpha * error * base ** assumed_power,
                               linear=(1 - alpha) * error / base,
                               edges=native_edges(circuit), makespan_weight=1e-9,
                               inverse_power=assumed_power, deadline=deadline_ns)
    result = solve_generalized(model)
    return result


def _gate_probability(event, anchors, true_power, alpha):
    key = (event.name, event.qubits)
    if key not in anchors:
        raise ValueError(f"No error anchor for {key}")
    nominal_duration, nominal_error = anchors[key]
    if true_power == "flat":
        return nominal_error
    if not isinstance(true_power, (int, float)) or true_power <= 0:
        raise ValueError("True power must be positive or 'flat'")
    duration = event.duration_ns
    return nominal_error * (alpha * (nominal_duration / duration) ** true_power
                            + (1 - alpha) * duration / nominal_duration)


def event_fidelity(circuit, events, deadline_ns, anchors, *, true_power,
                   alpha, noise: NoiseScenario, anisotropy=(1 / 3, 1 / 3, 1 / 3),
                   quadrature_order=7):
    """Independent density simulation for an explicit timed operation list."""
    if circuit.num_qubits > 4:
        raise ValueError("Dense comparison supports at most four qubits")
    weights_pauli = np.asarray(anisotropy, dtype=float)
    if weights_pauli.shape != (3,) or np.any(weights_pauli < 0) or not np.isclose(weights_pauli.sum(), 1):
        raise ValueError("Pauli anisotropy must be a probability vector")
    if not 0 < alpha < 1:
        raise ValueError("Invalid drive fraction")
    # Check that every qubit is covered from zero to the common deadline.
    for q in range(circuit.num_qubits):
        segments = sorted((event for event in events if q in event.qubits), key=lambda e: e.start_ns)
        cursor = 0.
        for event in segments:
            if abs(event.start_ns - cursor) > 1e-6 or event.duration_ns < 0:
                raise ValueError("Events do not form a nonoverlapping padded schedule")
            cursor += event.duration_ns
        if abs(cursor - deadline_ns) > 1e-6:
            raise ValueError("Schedule does not meet common deadline")
    ideal = Statevector.from_instruction(circuit).data
    if noise.detuning_std_rad_per_ns:
        nodes, weights = np.polynomial.hermite.hermgauss(quadrature_order)
        draws = np.sqrt(2) * noise.detuning_std_rad_per_ns * nodes
        weights = weights / np.sqrt(np.pi)
    else:
        draws, weights = np.array([0.]), np.array([1.])
    average = np.zeros((len(ideal), len(ideal)), dtype=complex)
    for choice in product(range(len(draws)), repeat=circuit.num_qubits):
        detuning = draws[list(choice)]
        ensemble_weight = float(np.prod(weights[list(choice)]))
        state = DensityMatrix.from_label("0" * circuit.num_qubits)
        for event in events:
            if event.name == "delay":
                state = _idle(state, event.qubits[0], event.duration_ns,
                              detuning[event.qubits[0]], noise)
            else:
                state = state.evolve(event.operation, qargs=list(event.qubits))
                probability = _gate_probability(event, anchors, true_power, alpha)
                per_qubit = probability / len(event.qubits)
                for q in event.qubits:
                    state = _pauli_channel(state, q, *(weights_pauli * per_qubit))
        average += ensemble_weight * state.data
    return float(np.real(np.vdot(ideal, average @ ideal)))
