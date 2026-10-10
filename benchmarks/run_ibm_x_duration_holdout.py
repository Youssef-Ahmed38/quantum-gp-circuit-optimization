"""Equal-duration XSlow holdout versus delayed X and standard two-X DD."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import random

from qiskit import QuantumCircuit, qasm3, qpy
from qiskit.circuit.library import XGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.circuit.library import XSlowGate
from qiskit_ibm_runtime.transpiler.passes.scheduling import (
    ALAPScheduleAnalysis, PadDynamicalDecoupling,
)

from run_ibm_readout_holdout import configured_sampler
from run_ibm_x_duration_characterization import prepare_fake_backend


QUBIT = 1
BASES = ("X", "Z")
LENGTHS = (16, 48)
MODES = ("slow", "padded", "fast_tail", "dd_tail")
BLOCKS = 3
SHOTS = 2048
SEED = 20261013
MAX_JOB_SECONDS = 90


def make_source(basis, length, mode, pad_dt, x_dt):
    if basis not in BASES or length not in LENGTHS or mode not in MODES:
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
    if mode in ("fast_tail", "dd_tail"):
        tail_dt = length * pad_dt - (2 * x_dt if mode == "dd_tail" else 0)
        if tail_dt <= 0:
            raise ValueError("Insufficient tail for two-X DD")
        circuit.delay(tail_dt, 0, unit="dt")
    if basis == "X":
        circuit.h(0)
    circuit.measure(0, 0)
    return circuit


def nominal_ticks(circuit, backend):
    ticks = 0
    for inst in circuit.data:
        operation = inst.operation
        if operation.name == "delay":
            ticks += operation.duration if operation.unit == "dt" else round(
                operation.duration * 1e-9 / backend.dt
            )
        else:
            properties = backend.target[operation.name][(QUBIT,)]
            if properties and properties.duration:
                ticks += round(properties.duration / backend.dt)
    return ticks


def compile_condition(backend, basis, length, mode, pad_dt, x_dt):
    source = make_source(basis, length, mode, pad_dt, x_dt)
    compiled = generate_preset_pass_manager(
        backend=backend, optimization_level=0,
        initial_layout=[QUBIT], seed_transpiler=SEED,
    ).run(source)
    if compiled.layout.initial_layout[source.qubits[0]] != QUBIT:
        raise RuntimeError("Physical layout changed")
    if mode == "dd_tail":
        dd_pass = PassManager([
            ALAPScheduleAnalysis(target=backend.target),
            PadDynamicalDecoupling(target=backend.target,
                                    dd_sequences=[XGate(), XGate()],
                                    qubits=[QUBIT]),
        ])
        compiled = dd_pass.run(compiled)
    counts = compiled.count_ops()
    if mode == "slow" and counts.get("xslow", 0) != length:
        raise RuntimeError("Slow native gates were lost")
    if mode == "padded" and (counts.get("x", 0) != length
                              or counts.get("delay", 0) != length):
        raise RuntimeError("Distributed padding changed")
    if mode == "fast_tail" and (counts.get("x", 0) != length
                                 or counts.get("delay", 0) != 1):
        raise RuntimeError("Tail padding changed")
    if mode == "dd_tail" and counts.get("x", 0) != length + 2:
        raise RuntimeError("Two-X dynamical decoupling was not inserted")
    return compiled


def build_protocol(backend, training_sha256):
    if any(name not in backend.target.operation_names for name in ("x", "xslow", "delay")):
        raise RuntimeError("Required native choices unavailable")
    fast = backend.target["x"][(QUBIT,)].duration
    slow = backend.target["xslow"][(QUBIT,)].duration
    pad_dt = round((slow - fast) / backend.dt)
    x_dt = round(fast / backend.dt)
    if abs(fast + pad_dt * backend.dt - slow) > 1e-12:
        raise RuntimeError("Cannot match target duration")
    conditions = [(basis, length, mode) for basis in BASES
                  for length in LENGTHS for mode in MODES]
    compiled = {condition: compile_condition(backend, *condition, pad_dt, x_dt)
                for condition in conditions}
    for basis in BASES:
        for length in LENGTHS:
            ticks = {mode: nominal_ticks(compiled[basis, length, mode], backend)
                     for mode in MODES}
            if len(set(ticks.values())) != 1:
                raise RuntimeError(f"Unmatched nominal durations: {ticks}")
    rng = random.Random(SEED)
    order = []
    for _ in range(BLOCKS):
        block = conditions[:]
        rng.shuffle(block)
        order.extend(block)
    circuits = [compiled[condition] for condition in order]
    payload = {
        "purpose": "Held-out native-X choice versus equal-duration controls; not GP validation",
        "backend": backend.name, "created_utc": datetime.now(timezone.utc).isoformat(),
        "physical_qubit": QUBIT, "bases": list(BASES), "lengths": list(LENGTHS),
        "modes": list(MODES), "blocks": BLOCKS, "shots_per_circuit": SHOTS,
        "seed": SEED, "max_job_seconds": MAX_JOB_SECONDS,
        "training_data_sha256": training_sha256,
        "target_x_seconds": fast, "target_xslow_seconds": slow,
        "pad_dt": pad_dt, "x_dt": x_dt, "dt_seconds": backend.dt,
        "order": [{"basis": basis, "length": length, "mode": mode}
                  for basis, length, mode in order],
        "compiled_counts": [dict(circuit.count_ops()) for circuit in circuits],
        "nominal_ticks": [nominal_ticks(circuit, backend) for circuit in circuits],
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
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.fake:
        backend = prepare_fake_backend()
    else:
        if args.token_file is None or not args.instance:
            parser.error("Live run requires --token-file and --instance")
        token = json.loads(args.token_file.read_text(encoding="utf-8"))["apikey"]
        service = QiskitRuntimeService(channel="ibm_quantum_platform",
                                       token=token, instance=args.instance)
        backend = service.backend("ibm_marrakesh")
    training_sha = sha256(args.training.read_bytes()).hexdigest()
    payload, circuits = build_protocol(backend, training_sha)
    payload["dry_run"] = args.fake
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with args.output.with_suffix(".qpy").open("wb") as handle:
        qpy.dump(circuits, handle)
    if args.fake:
        print(f"Prepared {len(circuits)} held-out circuits")
        return
    lock = {"training_data_sha256": training_sha,
            "conditions": payload["order"], "nominal_ticks": payload["nominal_ticks"],
            "frozen_utc": datetime.now(timezone.utc).isoformat()}
    lock_path = args.output.with_name(args.output.stem + "_protocol_lock.json")
    lock_bytes = (json.dumps(lock, indent=2, sort_keys=True) + "\n").encode()
    lock_path.write_bytes(lock_bytes)
    payload["protocol_lock_sha256"] = sha256(lock_bytes).hexdigest()
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
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
