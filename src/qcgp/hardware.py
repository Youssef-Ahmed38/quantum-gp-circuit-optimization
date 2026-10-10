"""Convert DD-GP schedules into Qiskit circuits with explicit delays.

Hardware executes instructions on each qubit in order; two-qubit gates wait for
both qubits. Inserting ``delay(start - ready)`` before every gate therefore
reproduces the optimised start times exactly, and DD pulses are inserted inside
the delays at CPMG positions. Fixed native durations come from the backend at
submission time; the defaults below are only for dry runs.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from .ddgp import DDProblem


def _gate_on(circuit, g):
    name = g.name
    if name == "block":
        return
    if name.endswith("dg") and name[:-2] in ("s", "sx"):
        getattr(circuit, name)(*g.qubits)
        return
    if name == "rz":
        raise ValueError("Use explicit RZ angles via circuits built with qiskit directly")
    getattr(circuit, name)(*g.qubits)


def schedule_to_circuit(problem: DDProblem, durations, starts, makespan, pulses_per_window,
                        *, pulse_duration=None, measure=True, time_unit="us"):
    """Qiskit circuit whose per-qubit timeline matches the schedule (in us)."""
    tp = problem.pulse_duration if pulse_duration is None else pulse_duration
    circuit = QuantumCircuit(problem.n, problem.n if measure else 0)
    ready = np.zeros(problem.n)
    windows = problem.windows()
    counts = {(q, a): int(round(n)) for (q, a, b), n in zip(windows, pulses_per_window)}
    timed = problem.timed
    position = {i: k for k, i in enumerate(timed)}

    def fill(q, t0, t1, count):
        w = t1 - t0
        if w <= 1e-9:
            return
        if count <= 0:
            circuit.delay(w, q, unit=time_unit)
            return
        # Pulse centres at (2i+1) w / (2n): gaps of w/(2n) - tp/2 at the ends.
        edges = [t0 + w * (2 * i + 1) / (2 * count) for i in range(count)]
        cursor = t0
        for centre in edges:
            gap = centre - tp / 2 - cursor
            if gap > 1e-9:
                circuit.delay(gap, q, unit=time_unit)
            circuit.x(q)
            cursor = centre + tp / 2
        if t1 - cursor > 1e-9:
            circuit.delay(t1 - cursor, q, unit=time_unit)

    last_gate = {q: None for q in range(problem.n)}
    for index, g in enumerate(problem.gates):
        if not g.timed:
            raise ValueError("Virtual gates are not supported in hardware conversion")
        k = position[index]
        for q in g.qubits:
            previous = last_gate[q]
            count = counts.get((q, previous), 0) if previous is not None else 0
            fill(q, ready[q], starts[k], count)
            ready[q] = starts[k]
        if g.block:
            for q in g.qubits:
                circuit.delay(durations[k], q, unit=time_unit)
        else:
            _gate_on(circuit, g)
        for q in g.qubits:
            ready[q] = starts[k] + durations[k]
            last_gate[q] = k
    for q in range(problem.n):
        if last_gate[q] is not None:
            fill(q, ready[q], makespan, counts.get((q, last_gate[q]), 0))
    if measure:
        circuit.measure(range(problem.n), range(problem.n))
    return circuit
