"""Prespecified native X versus XSlow characterization on a free IBM QPU."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random

from qiskit import QuantumCircuit, qasm3, qpy
from qiskit.transpiler import InstructionProperties
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.circuit.library import XSlowGate
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

from run_ibm_readout_holdout import configured_sampler


QUBITS = (0, 1)
BASES = ("Z", "X")
MODES = ("fast", "padded", "slow")
LENGTHS = (0, 8, 32, 64)
BLOCKS = 3
SHOTS = 1024
SEED = 20261012
MAX_JOB_SECONDS = 90


def prepare_fake_backend():
    backend = FakeMarrakesh()
    backend.target.add_instruction(
        XSlowGate(),
        {(qubit,): InstructionProperties(duration=1e-6,
                                          error=backend.target["x"][(qubit,)].error)
         for qubit in range(backend.num_qubits)},
    )
    return backend


def make_circuit(mode, length, basis, pad_dt):
    if mode not in MODES or length not in LENGTHS or basis not in BASES:
        raise ValueError("Unknown frozen condition")
    circuit = QuantumCircuit(1, 1)
    if basis == "X":
        circuit.h(0)
    for _ in range(length):
        if mode == "slow":
            circuit.append(XSlowGate(), [0])
        else:
            circuit.x(0)
            if mode == "padded":
                circuit.delay(pad_dt, 0, unit="dt")
    if basis == "X":
        circuit.h(0)
    circuit.measure(0, 0)
    return circuit


def compile_condition(backend, qubit, basis, mode, length, pad_dt):
    source = make_circuit(mode, length, basis, pad_dt)
    pass_manager = generate_preset_pass_manager(
        backend=backend, optimization_level=0,
        initial_layout=[qubit], seed_transpiler=SEED,
    )
    compiled = pass_manager.run(source)
    mapped = compiled.layout.initial_layout[source.qubits[0]]
    if mapped != qubit:
        raise RuntimeError("Physical qubit mapping changed")
    operations = compiled.count_ops()
    expected = {"fast": ("x", length), "padded": ("x", length),
                "slow": ("xslow", length)}[mode]
    if operations.get(expected[0], 0) != expected[1]:
        raise RuntimeError("Intended native X implementation was not retained")
    if mode == "padded" and operations.get("delay", 0) != length:
        raise RuntimeError("Duration-matching delays were not retained")
    if mode != "padded" and operations.get("delay", 0):
        raise RuntimeError("Unexpected delay in unpadded circuit")
    return compiled


def build_protocol(backend):
    for name in ("x", "xslow", "delay"):
        if name not in backend.target.operation_names:
            raise RuntimeError(f"Backend does not expose {name}")
    durations = {}
    for qubit in QUBITS:
        x = backend.target["x"][(qubit,)]
        slow = backend.target["xslow"][(qubit,)]
        if x.duration is None or slow.duration is None or slow.duration <= x.duration:
            raise RuntimeError("Unexpected native X durations")
        pad_dt_float = (slow.duration - x.duration) / backend.dt
        pad_dt = round(pad_dt_float)
        if abs(pad_dt_float - pad_dt) > 1e-6:
            raise RuntimeError("Cannot exactly match slow duration in dt units")
        durations[qubit] = {
            "x_seconds": x.duration, "xslow_seconds": slow.duration,
            "x_target_error": x.error, "xslow_target_error": slow.error,
            "pad_dt": pad_dt,
        }
    conditions = []
    for qubit in QUBITS:
        for basis in BASES:
            conditions.append((qubit, basis, "fast", 0))
            for length in LENGTHS[1:]:
                for mode in MODES:
                    conditions.append((qubit, basis, mode, length))
    compiled = {}
    for condition in conditions:
        qubit, basis, mode, length = condition
        compiled[condition] = compile_condition(
            backend, qubit, basis, mode, length, durations[qubit]["pad_dt"]
        )
    rng = random.Random(SEED)
    order = []
    for _ in range(BLOCKS):
        block = conditions[:]
        rng.shuffle(block)
        order.extend(block)
    circuits = [compiled[item] for item in order]
    payload = {
        "purpose": "Native two-duration X characterization; no GP improvement claim",
        "backend": backend.name, "created_utc": datetime.now(timezone.utc).isoformat(),
        "qubits": list(QUBITS), "bases": list(BASES), "modes": list(MODES),
        "lengths": list(LENGTHS), "blocks": BLOCKS, "shots_per_circuit": SHOTS,
        "seed": SEED, "max_job_seconds": MAX_JOB_SECONDS,
        "durations": durations,
        "order": [{"qubit": q, "basis": b, "mode": mode, "length": m}
                  for q, b, mode, m in order],
        "compiled_counts": [dict(circuit.count_ops()) for circuit in circuits],
        "qasm3": [qasm3.dumps(circuit, basis_gates=("xslow",))
                  for circuit in circuits],
        "sampler_options": {"dynamical_decoupling": False,
                            "gate_twirling": False, "measurement_twirling": False},
    }
    return payload, circuits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--instance")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.fake:
        backend = prepare_fake_backend()
    else:
        if args.token_file is None or not args.instance:
            parser.error("Live run requires --token-file and --instance")
        credential = json.loads(args.token_file.read_text(encoding="utf-8"))["apikey"]
        service = QiskitRuntimeService(channel="ibm_quantum_platform",
                                       token=credential, instance=args.instance)
        backend = service.backend("ibm_marrakesh")
    payload, circuits = build_protocol(backend)
    payload["dry_run"] = args.fake
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with args.output.with_suffix(".qpy").open("wb") as handle:
        qpy.dump(circuits, handle)
    if args.fake:
        print(f"Prepared {len(circuits)} frozen circuits")
        return
    sampler = configured_sampler(backend)
    sampler.options.max_execution_time = MAX_JOB_SECONDS
    job = sampler.run(circuits, shots=SHOTS)
    payload["job_id"] = job.job_id()
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    result = job.result()
    payload["counts"] = [pub.data.c.get_counts() for pub in result]
    payload["job_status"] = str(job.status())
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Job {payload['job_id']}: {payload['job_status']}; {len(circuits)} circuits")


if __name__ == "__main__":
    main()
