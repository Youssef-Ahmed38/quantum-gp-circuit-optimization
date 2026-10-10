"""Prospective free-QPU diagnostic of reported versus measured readout errors.

Readout-calibration circuits run first. Their counts alone determine a second
prediction for held-out target circuits; predictions are saved before the
target job is submitted. No GP gate-duration choice is executed.
"""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np
from qiskit import QuantumCircuit, qasm3, qpy
from qiskit.quantum_info import Operator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

from run_ibm_model_check import (
    SEED,
    compact_snapshot,
    logical_circuit,
    native_local_circuit,
    predicted_distribution,
    save_payload,
)


TARGET_NAMES = (
    "single_return", "single_one_return", "bell", "bell_return",
    "ghz", "ghz_return",
)
EVENTS = {
    "single_return": ("0",),
    "single_one_return": ("1",),
    "bell": ("00", "11"),
    "bell_return": ("00",),
    "ghz": ("000", "111"),
    "ghz_return": ("000",),
}
LAYOUT = (0, 1, 2)
BLOCKS = 3
SHOTS = 2048
MAX_JOB_SECONDS = 60


def target_source(name):
    if name in ("single_return", "bell", "bell_return", "ghz"):
        return logical_circuit(name)
    if name == "single_one_return":
        circuit = QuantumCircuit(1, 1, name=name)
        circuit.x(0)
        circuit.h(0)
        circuit.barrier()
        circuit.h(0)
        circuit.measure(0, 0)
        return circuit
    if name == "ghz_return":
        prepared = logical_circuit("ghz").remove_final_measurements(inplace=False)
        circuit = QuantumCircuit(3, 3, name=name)
        circuit.compose(prepared, inplace=True)
        circuit.barrier()
        circuit.h(2)
        circuit.cz(1, 2)
        circuit.h(2)
        circuit.h(1)
        circuit.cz(0, 1)
        circuit.h(1)
        circuit.h(0)
        circuit.measure(range(3), range(3))
        return circuit
    raise KeyError(name)


def calibration_name(qubit, prepared):
    return f"readout_q{qubit}_{prepared}"


def calibration_source(qubit, prepared):
    circuit = QuantumCircuit(1, 1, name=calibration_name(qubit, prepared))
    if prepared:
        circuit.x(0)
    circuit.measure(0, 0)
    return circuit


def _compile(source, backend, physical):
    manager = generate_preset_pass_manager(
        backend=backend, optimization_level=0,
        initial_layout=list(physical), seed_transpiler=SEED,
    )
    isa = manager.run(source)
    mapped = tuple(isa.layout.initial_layout[source.qubits[q]]
                   for q in range(source.num_qubits))
    if mapped != tuple(physical):
        raise RuntimeError(f"Transpiler changed physical mapping: {mapped}")
    return isa


def build_protocol(backend):
    if "cz" not in backend.target.operation_names or any(
        edge not in backend.target["cz"] for edge in ((0, 1), (1, 2))
    ):
        raise RuntimeError("Predeclared physical CZ edges are unavailable")
    snapshot = compact_snapshot(backend.properties(), [3])
    cal_isa = {}
    for qubit in LAYOUT:
        for prepared in (0, 1):
            source = calibration_source(qubit, prepared)
            name = source.name
            compiled = _compile(source, backend, (qubit,))
            if name.endswith("_0") and any(
                inst.operation.name not in ("measure", "barrier") for inst in compiled.data
            ):
                raise RuntimeError("Zero-state calibrator gained a nonmeasurement gate")
            cal_isa[name] = compiled
    target_isa, target_local, baseline = {}, {}, {}
    for name in TARGET_NAMES:
        source = target_source(name)
        width = source.num_qubits
        compiled = _compile(source, backend, LAYOUT[:width])
        local = native_local_circuit(compiled, width)
        ideal = source.remove_final_measurements(inplace=False)
        expected, actual = Operator(ideal).data, Operator(local).data
        overlap = np.vdot(expected, actual)
        phase = overlap / abs(overlap) if abs(overlap) else 1.0
        if np.max(np.abs(actual - phase * expected)) > 1e-8:
            raise RuntimeError(f"Compiled circuit changed the unitary: {name}")
        target_isa[name], target_local[name] = compiled, local
        baseline[name] = predicted_distribution(local, snapshot)

    rng = random.Random(SEED + 1)
    cal_order, target_order = [], []
    for _ in range(BLOCKS):
        names = list(cal_isa)
        rng.shuffle(names)
        cal_order.extend(names)
    for _ in range(BLOCKS):
        names = list(TARGET_NAMES)
        rng.shuffle(names)
        target_order.extend(names)
    payload = {
        "purpose": "Readout-model diagnosis with held-out native circuits; not GP hardware improvement",
        "backend": backend.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "physical_qubits": list(LAYOUT),
        "calibration_snapshot": snapshot,
        "calibration_order": cal_order,
        "target_order": target_order,
        "blocks": BLOCKS,
        "shots_per_circuit": SHOTS,
        "seed": SEED + 1,
        "max_job_seconds": MAX_JOB_SECONDS,
        "target_events": {name: list(EVENTS[name]) for name in TARGET_NAMES},
        "snapshot_predictions": baseline,
        "native_target_qasm3": {name: qasm3.dumps(target_local[name]) for name in TARGET_NAMES},
        "target_gate_counts": {name: dict(target_isa[name].count_ops()) for name in TARGET_NAMES},
        "calibrator_gate_counts": {name: dict(cal_isa[name].count_ops()) for name in cal_isa},
        "sampler_options": {"dynamical_decoupling": False,
                            "gate_twirling": False, "measurement_twirling": False},
    }
    return payload, [cal_isa[name] for name in cal_order], [target_isa[name] for name in target_order], target_local


def fit_operational_readout(calibration_order, counts, shots):
    if len(calibration_order) != len(counts) or not calibration_order:
        raise ValueError("Calibration order and counts disagree")
    errors = {(qubit, prepared): 0 for qubit in LAYOUT for prepared in (0, 1)}
    totals = {key: 0 for key in errors}
    for name, distribution in zip(calibration_order, counts):
        if sum(distribution.values()) != shots or set(distribution) - {"0", "1"}:
            raise ValueError("Invalid calibration counts")
        match = name.split("_")
        if len(match) != 3 or match[0] != "readout" or not match[1].startswith("q"):
            raise ValueError("Unknown calibration circuit")
        key = int(match[1][1:]), int(match[2])
        if key not in errors:
            raise ValueError("Unknown physical qubit or prepared state")
        errors[key] += distribution.get(str(1 - key[1]), 0)
        totals[key] += shots
    if any(total == 0 for total in totals.values()):
        raise ValueError("Missing readout-calibration condition")
    return [[errors[q, 0] / totals[q, 0], errors[q, 1] / totals[q, 1]] for q in LAYOUT]


def configured_sampler(backend):
    sampler = SamplerV2(mode=backend)
    sampler.options.dynamical_decoupling.enable = False
    sampler.options.twirling.enable_gates = False
    sampler.options.twirling.enable_measure = False
    sampler.options.max_execution_time = MAX_JOB_SECONDS
    return sampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--instance", help="IBM Open instance CRN, never saved")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.fake:
        if args.token_file or args.instance:
            parser.error("--fake cannot use credentials")
        backend = FakeMarrakesh()
    else:
        if args.token_file is None or not args.instance:
            parser.error("Live run requires --token-file and --instance")
        credential = args.token_file.read_text(encoding="utf-8").strip()
        if args.token_file.suffix.lower() == ".json":
            credential = json.loads(credential)["apikey"]
        service = QiskitRuntimeService(
            channel="ibm_quantum_platform",
            token=credential,
            instance=args.instance,
        )
        backend = service.backend("ibm_marrakesh")
    payload, calibrators, targets, local = build_protocol(backend)
    payload["dry_run"] = args.fake
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_payload(args.output, payload)
    with args.output.with_suffix(".qpy").open("wb") as handle:
        qpy.dump(calibrators + targets, handle)
    if args.fake:
        print(f"Prepared {len(calibrators)} calibrators and {len(targets)} held-out circuits")
        return

    sampler = configured_sampler(backend)
    cal_job = sampler.run(calibrators, shots=SHOTS)
    payload["calibration_job_id"] = cal_job.job_id()
    save_payload(args.output, payload)
    cal_result = cal_job.result()
    payload["calibration_counts"] = [pub.data.c.get_counts() for pub in cal_result]
    payload["calibration_job_status"] = str(cal_job.status())
    empirical = fit_operational_readout(payload["calibration_order"],
                                         payload["calibration_counts"], SHOTS)
    payload["operational_readout"] = empirical
    payload["calibrated_predictions"] = {
        name: predicted_distribution(local[name], payload["calibration_snapshot"],
                                     readout_override=empirical[:local[name].num_qubits])
        for name in TARGET_NAMES
    }
    lock = {
        "calibration_job_id": payload["calibration_job_id"],
        "calibration_counts": payload["calibration_counts"],
        "operational_readout": empirical,
        "snapshot_predictions": payload["snapshot_predictions"],
        "calibrated_predictions": payload["calibrated_predictions"],
        "target_order": payload["target_order"],
        "target_events": payload["target_events"],
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
    }
    lock_bytes = (json.dumps(lock, sort_keys=True, indent=2) + "\n").encode("utf-8")
    lock_path = args.output.with_name(args.output.stem + "_prediction_lock.json")
    lock_path.write_bytes(lock_bytes)
    payload["prediction_lock_sha256"] = sha256(lock_bytes).hexdigest()
    save_payload(args.output, payload)

    target_job = sampler.run(targets, shots=SHOTS)
    payload["target_job_id"] = target_job.job_id()
    save_payload(args.output, payload)
    target_result = target_job.result()
    payload["target_counts"] = [pub.data.c.get_counts() for pub in target_result]
    payload["target_job_status"] = str(target_job.status())
    if any(sum(counts.values()) != SHOTS for counts in payload["target_counts"]):
        raise RuntimeError("Target counts do not sum to requested shots")
    save_payload(args.output, payload)
    print(f"Calibration job: {payload['calibration_job_id']}")
    print(f"Held-out target job: {payload['target_job_id']}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
