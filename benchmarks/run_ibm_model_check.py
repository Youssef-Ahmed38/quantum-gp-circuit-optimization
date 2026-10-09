"""Free IBM QPU check of the paper's fixed-duration Pauli noise model.

This does not execute GP-optimized gate durations. It compares pre-submission
predictions from the model at nominal native durations with held-out QPU counts.
Use --fake to validate the exact circuit/analysis pipeline without credentials.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random

import numpy as np
from qiskit import QuantumCircuit, qasm3, qpy
from qiskit.quantum_info import DensityMatrix, Operator, Statevector, Pauli
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

from qcgp.observables import local_paulis, snapshot_model


SEED = 20261010
LAYOUT = (0, 1, 2)
NAMES = ("single_return", "bell", "bell_return", "ghz")


def save_payload(path, payload):
    path.write_text(json.dumps(payload, indent=2,
                               default=lambda value: value.isoformat()
                               if isinstance(value, datetime) else _reject_json(value)) + "\n",
                    encoding="utf-8")


def _reject_json(value):
    raise TypeError(f"Unexpected non-JSON protocol value: {type(value).__name__}")


def logical_circuit(name):
    width = {"single_return": 1, "bell": 2, "bell_return": 2, "ghz": 3}[name]
    circuit = QuantumCircuit(width, width, name=name)
    if name == "single_return":
        circuit.h(0)
        circuit.barrier()
        circuit.h(0)
    else:
        circuit.h(0)
        circuit.h(1)
        circuit.cz(0, 1)
        circuit.h(1)
        if name == "bell_return":
            circuit.barrier()
            circuit.h(1)
            circuit.cz(0, 1)
            circuit.h(1)
            circuit.h(0)
        elif name == "ghz":
            circuit.h(2)
            circuit.cz(1, 2)
            circuit.h(2)
    circuit.measure(range(width), range(width))
    return circuit


def native_local_circuit(isa, width):
    """Copy only mapped active native gates for small-state prediction."""
    local = QuantumCircuit(width)
    for instruction in isa.data:
        name = instruction.operation.name
        if name in ("measure", "barrier"):
            continue
        physical = tuple(isa.find_bit(q).index for q in instruction.qubits)
        if name not in ("rz", "sx", "x", "cz", "id") or any(q >= width for q in physical):
            raise RuntimeError(f"Unexpected compiled operation: {name} on {physical}")
        local.append(instruction.operation, physical)
    return local


def compact_snapshot(properties, widths):
    raw = properties.to_dict()
    active = set(range(max(widths)))
    gates = [gate for gate in raw["gates"]
             if gate["gate"] in ("sx", "x", "cz", "id")
             and set(gate["qubits"]).issubset(active)]
    return dict(backend_name=raw["backend_name"],
                last_update_date=raw["last_update_date"],
                qubits=raw["qubits"][:max(widths)], gates=gates)


def readout_probabilities(snapshot, width):
    result = []
    for q in range(width):
        parameters = {entry["name"]: float(entry["value"])
                      for entry in snapshot["qubits"][q]}
        one_given_zero = parameters["prob_meas1_prep0"]
        zero_given_one = parameters["prob_meas0_prep1"]
        if not (0 <= one_given_zero <= 1 and 0 <= zero_given_one <= 1):
            raise RuntimeError("Invalid readout calibration")
        result.append((one_given_zero, zero_given_one))
    return result


def predicted_distribution(circuit, snapshot):
    """Density matrix with the paper's post-gate Pauli channel and readout map."""
    width = circuit.num_qubits
    model = snapshot_model(circuit, tuple(range(width)), snapshot, alpha=.5)
    state = DensityMatrix(Statevector.from_label("0" * width))
    timed = {index: position for position, index in enumerate(model.timed_indices)}
    for index, instruction in enumerate(circuit.data):
        qubits = tuple(circuit.find_bit(q).index for q in instruction.qubits)
        state = state.evolve(instruction.operation, qargs=qubits)
        if index not in timed:
            continue
        j = timed[index]
        hazard = (model.a[j] / model.base[j] + model.gamma[j] * model.base[j])
        flip = -.5 * np.expm1(-2 * hazard / (4 ** len(qubits) - 1))
        for x, z in local_paulis(qubits):
            changed = state.evolve(Operator(Pauli((z, x))), qargs=qubits)
            state = DensityMatrix((1 - flip) * state.data + flip * changed.data)
    true_probabilities = np.real(np.diag(state.data))
    readout = readout_probabilities(snapshot, width)
    observed = np.zeros(2 ** width)
    for prepared, mass in enumerate(true_probabilities):
        for measured in range(2 ** width):
            likelihood = 1.
            for q, (one_given_zero, zero_given_one) in enumerate(readout):
                truth, outcome = (prepared >> q) & 1, (measured >> q) & 1
                likelihood *= ((1 - one_given_zero) if outcome == 0 else one_given_zero) if truth == 0 else (zero_given_one if outcome == 0 else (1 - zero_given_one))
            observed[measured] += mass * likelihood
    if not np.isclose(observed.sum(), 1., atol=1e-9) or np.min(observed) < -1e-9:
        raise RuntimeError("Invalid predicted probability distribution")
    return {format(index, f"0{width}b"): float(value)
            for index, value in enumerate(observed)}


def build_protocol(backend, shots, blocks):
    if "cz" not in backend.target.operation_names:
        raise RuntimeError("Backend lacks native CZ")
    for edge in ((0, 1), (1, 2)):
        if edge not in backend.target["cz"]:
            raise RuntimeError(f"Predeclared physical edge {edge} unavailable")
    logical = {name: logical_circuit(name) for name in NAMES}
    properties = backend.properties()
    snapshot = compact_snapshot(properties, [c.num_qubits for c in logical.values()])
    native, isa, predictions = {}, {}, {}
    for name, source in logical.items():
        width = source.num_qubits
        pass_manager = generate_preset_pass_manager(
            backend=backend, optimization_level=0,
            initial_layout=list(LAYOUT[:width]), seed_transpiler=SEED)
        compiled = pass_manager.run(source)
        local = native_local_circuit(compiled, width)
        ideal = source.remove_final_measurements(inplace=False)
        expected, actual = Operator(ideal).data, Operator(local).data
        overlap = np.vdot(expected, actual)
        phase = overlap / abs(overlap) if abs(overlap) else 1.
        error = np.max(np.abs(actual - phase * expected))
        if error > 1e-8:
            raise RuntimeError(f"Compiled circuit changed the unitary for {name}: {error}")
        native[name], isa[name] = local, compiled
        predictions[name] = predicted_distribution(local, snapshot)
    rng = random.Random(SEED)
    order = []
    for block in range(blocks):
        names = list(NAMES)
        rng.shuffle(names)
        order.extend(names)
    payload = dict(purpose="Nominal fixed-gate noise-model check; not GP hardware improvement",
                   backend=backend.name, physical_qubits=list(LAYOUT),
                   calibration_snapshot=snapshot,
                   created_utc=datetime.now(timezone.utc).isoformat(),
                   shots_per_circuit=shots, blocks=blocks, seed=SEED, order=order,
                   predicted_observed_distributions=predictions,
                   native_qasm3={name: qasm3.dumps(c) for name, c in native.items()},
                   compiled_gate_counts={name: dict(isa[name].count_ops()) for name in NAMES})
    return payload, [isa[name] for name in order]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake", action="store_true", help="Compile only; no account or QPU use")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--instance", help="IBM Open instance CRN; never written to output")
    parser.add_argument("--backend", default="ibm_marrakesh")
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.shots < 1 or args.blocks < 1:
        parser.error("Positive shots and blocks required")
    if args.fake:
        if args.token_file or args.instance:
            parser.error("--fake cannot use credentials")
        backend = FakeMarrakesh()
    else:
        if args.token_file is None or not args.instance:
            parser.error("Actual IBM run requires --token-file and --instance")
        token = args.token_file.read_text(encoding="utf-8").strip()
        service = QiskitRuntimeService(channel="ibm_quantum_platform",
                                       token=token, instance=args.instance)
        backend = service.backend(args.backend)
    payload, circuits = build_protocol(backend, args.shots, args.blocks)
    payload["dry_run"] = args.fake
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_payload(args.output, payload)
    with args.output.with_suffix(".qpy").open("wb") as handle:
        qpy.dump(circuits, handle)
    if not args.fake:
        job = SamplerV2(mode=backend).run(circuits, shots=args.shots)
        payload["job_id"] = job.job_id()
        save_payload(args.output, payload)
        result = job.result()
        payload["counts"] = [pub.data.c.get_counts() for pub in result]
        payload["job_status"] = str(job.status())
        if any(sum(counts.values()) != args.shots for counts in payload["counts"]):
            raise RuntimeError("Returned counts do not sum to requested shots")
        save_payload(args.output, payload)
        print(f"IBM job ID: {payload['job_id']}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
