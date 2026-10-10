"""Decompose saved QPU model predictions into ideal, readout, and gate terms."""

import argparse
import json
from pathlib import Path

from qiskit import qasm3
from qiskit.quantum_info import Statevector

from analyze_ibm_model_check import SUCCESS
from run_ibm_model_check import readout_probabilities


def readout_only_distribution(circuit, snapshot):
    width = circuit.num_qubits
    true_probabilities = Statevector.from_instruction(circuit).probabilities()
    assignment = readout_probabilities(snapshot, width)
    result = {}
    for measured in range(1 << width):
        mass = 0.0
        for prepared, prior in enumerate(true_probabilities):
            conditional = 1.0
            for q, (p10, p01) in enumerate(assignment):
                truth = (prepared >> q) & 1
                outcome = (measured >> q) & 1
                conditional *= ((1 - p10) if outcome == 0 else p10) if truth == 0 else (p01 if outcome == 0 else 1 - p01)
            mass += float(prior) * conditional
        result[format(measured, f"0{width}b")] = mass
    return result


def audit(payload):
    if payload.get("dry_run") or "counts" not in payload:
        raise ValueError("Completed QPU counts required")
    result = {}
    for name, qasm in payload["native_qasm3"].items():
        if name not in SUCCESS:
            raise ValueError(f"Unknown saved circuit: {name}")
        local = qasm3.loads(qasm)
        ideal = Statevector.from_instruction(local).probabilities_dict()
        readout = readout_only_distribution(local, payload["calibration_snapshot"])
        event = SUCCESS[name]
        selected = [counts for circuit, counts in zip(payload["order"], payload["counts"])
                    if circuit == name]
        shots = sum(sum(counts.values()) for counts in selected)
        observed = sum(sum(counts.get(bitstring, 0) for bitstring in event)
                       for counts in selected) / shots
        ideal_probability = sum(ideal.get(bitstring, 0.0) for bitstring in event)
        readout_probability = sum(readout[bitstring] for bitstring in event)
        full_probability = sum(payload["predicted_observed_distributions"][name][bitstring]
                               for bitstring in event)
        result[name] = {
            "observed": observed,
            "ideal": ideal_probability,
            "readout_only": readout_probability,
            "full_model": full_probability,
            "predicted_loss_from_readout": ideal_probability - readout_probability,
            "additional_predicted_loss_from_gate_model": readout_probability - full_probability,
            "observed_minus_readout_only": observed - readout_probability,
        }
    return {"job_id": payload["job_id"], "backend": payload["backend"], "families": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["families"], indent=2))


if __name__ == "__main__":
    main()
