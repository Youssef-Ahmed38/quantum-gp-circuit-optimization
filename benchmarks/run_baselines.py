from pathlib import Path
import argparse
import csv
import importlib.metadata
import json
from qiskit import qpy
from qcgp.core import topology, candidate_metrics, unitary_error, noisy_fidelity
from qcgp.baselines import compile_baseline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evidence", type=Path, default=Path(__file__).resolve().parents[1]/"evidence")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    out = args.output or args.evidence
    out.mkdir(parents=True, exist_ok=True)
    cases = json.loads((args.evidence/"comparisons.json").read_text(encoding="utf8"))
    rows = []
    methods = ["qiskit_l0", "qiskit_l1", "qiskit_l2", "qiskit_l3",
               "tket_common_route", "cirq_common_route"]
    for case in cases:
        with (args.evidence/"circuits"/f"{case['case']}.qpy").open("rb") as stream:
            logical = qpy.load(stream)[0]
        coupling = topology(case["qubits"], case["topology"])
        for method in methods:
            physical, elapsed = compile_baseline(logical, coupling, method,
                                                 1000*case["instance"]+7)
            error = unitary_error(logical, physical)
            if error > 1e-8:
                raise RuntimeError(f"{method}/{case['case']}: unitary error {error}")
            metrics, _ = candidate_metrics(physical, coupling)
            fidelity = None
            if case["qubits"] <= 4:
                fidelity, _, _ = noisy_fidelity(physical, coupling)
            row = {k:case[k] for k in ("case", "family", "qubits", "topology", "instance")}
            row.update(method=method, **metrics, compile_seconds=elapsed,
                       unitary_max_error=error, simulated_fidelity=fidelity)
            rows.append(row)
            with (out/f"baseline_{case['case']}_{method}.qpy").open("wb") as stream:
                qpy.dump(physical, stream)
        (out/"baselines.partial.json").write_text(json.dumps(rows), encoding="utf8")
        print(f"{case['case']}: six baseline pipelines verified", flush=True)
    (out/"baselines.json").write_text(json.dumps(rows,indent=2),encoding="utf8")
    with (out/"baselines.csv").open("w",newline="",encoding="utf8") as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out/"baseline_environment.json").write_text(json.dumps({m:importlib.metadata.version(m)
        for m in ("cirq-core", "pytket", "pytket-qiskit", "qiskit")},indent=2),encoding="utf8")
    (out/"baselines.partial.json").unlink()
    print(f"Verified {len(rows)} baseline circuits",flush=True)


if __name__ == "__main__": main()
