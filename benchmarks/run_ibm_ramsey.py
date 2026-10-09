"""Optional, free-plan IBM QPU Ramsey/echo pilot; not a GP hardware benchmark.

Uses a temporary API key supplied by the user outside the repository. The script
never writes or prints the key. A dry run compiles the circuits without QPU usage.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random

from qiskit import QuantumCircuit, qasm3
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2


CONDITIONS = ("ramsey_1us", "ramsey_20us", "ramsey_80us", "echo_80us")
SEED = 20261010


def make_circuit(condition: str) -> QuantumCircuit:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    circuit = QuantumCircuit(1, 1, name=condition)
    circuit.h(0)
    if condition == "ramsey_1us":
        circuit.delay(1000, 0, unit="ns")
    elif condition == "ramsey_20us":
        circuit.delay(20000, 0, unit="ns")
    elif condition == "ramsey_80us":
        circuit.delay(80000, 0, unit="ns")
    else:
        circuit.delay(40000, 0, unit="ns")
        circuit.x(0)
        circuit.delay(40000, 0, unit="ns")
        circuit.x(0)
    circuit.h(0)
    circuit.measure(0, 0)
    return circuit


def sequence() -> list[str]:
    rng = random.Random(SEED)
    order = []
    for _ in range(2):
        block = list(CONDITIONS)
        rng.shuffle(block)
        order.extend(block)
    return order


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--instance", required=True, help="IBM Quantum instance CRN")
    parser.add_argument("--backend", default="ibm_marrakesh")
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.shots < 1:
        parser.error("--shots must be positive")
    token = args.token_file.read_text(encoding="utf-8").strip()
    service = QiskitRuntimeService(
        channel="ibm_quantum_platform", token=token, instance=args.instance
    )
    backend = service.backend(args.backend)
    if "delay" not in backend.target.operation_names:
        raise RuntimeError(f"{backend.name} does not advertise delay support")
    properties = backend.properties()
    order = sequence()
    logical = [make_circuit(name) for name in order]
    pass_manager = generate_preset_pass_manager(
        backend=backend, optimization_level=0, initial_layout=[0],
        seed_transpiler=SEED,
    )
    isa = [pass_manager.run(circuit) for circuit in logical]
    expected_delays = {"ramsey_1us": 1, "ramsey_20us": 1,
                       "ramsey_80us": 1, "echo_80us": 2}
    for name, circuit in zip(order, isa):
        if circuit.count_ops().get("delay", 0) != expected_delays[name]:
            raise RuntimeError(f"Transpilation changed delay count for {name}")
    payload = {
        "purpose": "Measured idle-duration and echo pilot; not a GP hardware benchmark",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend.name,
        "processor_qubits": backend.num_qubits,
        "physical_qubit": 0,
        "instance_plan": "open",
        "shots_per_circuit": args.shots,
        "seed": SEED,
        "order": order,
        "source_qasm3": [qasm3.dumps(circuit) for circuit in logical],
        "compiled_operation_counts": [dict(circuit.count_ops()) for circuit in isa],
        "backend_target_has_delay": True,
        "backend_dt_seconds": backend.dt,
        "calibration_snapshot": {
            "last_update_date": properties.last_update_date.isoformat(),
            "qubit_0_t1_seconds": properties.t1(0),
            "qubit_0_t2_seconds": properties.t2(0),
            "qubit_0_readout_error": properties.readout_error(0),
        },
        "dry_run": args.dry_run,
    }
    if not args.dry_run:
        job = SamplerV2(mode=backend).run(isa, shots=args.shots)
        payload["job_id"] = job.job_id()
        result = job.result()
        payload["counts"] = [pub.data.c.get_counts() for pub in result]
        payload["job_status"] = str(job.status())
        for counts in payload["counts"]:
            if sum(counts.values()) != args.shots:
                raise RuntimeError("Counts do not sum to requested shots")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    if not args.dry_run:
        print(f"IBM job ID: {payload['job_id']}")


if __name__ == "__main__":
    main()
