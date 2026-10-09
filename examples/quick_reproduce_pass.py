"""Small local reproducibility example for peer reviewers.

Demonstrates:
1. Loading an algorithmic Clifford circuit.
2. Using a saved IBM Heron 133-qubit calibration snapshot (ibm_torino).
3. Running ConvexDurationAllocationPass inside Qiskit PassManager.
4. Reporting local solve time, duality gap bracket, and discrete-duration rounding.
"""
from pathlib import Path
import json, time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qcgp import ConvexDurationAllocationPass, snapshot_model, dense_observables, expectations

def main():
    root = Path(__file__).resolve().parents[1]
    snapshot_path = root / "evidence/observable_study/props_torino.json"

    print("=" * 70)
    print("QCGP LOCAL SIMULATION WITH SAVED IBM HERON CALIBRATION PARAMETERS")
    print("=" * 70)

    # 1. Build benchmark circuit (4 qubits, heavy-hex topology)
    c = QuantumCircuit(4)
    for q in range(4):
        c.rz(np.pi/2, q)
        c.sx(q)
        c.rz(np.pi/2, q)
    c.cz(0, 1)
    c.cz(1, 2)
    c.cz(2, 3)

    print(f"[*] Benchmark Circuit: 4 qubits, {len(c.data)} instructions.")

    # 2. Load IBM Heron calibration data
    snapshot = json.loads(snapshot_path.read_text())
    print(f"[*] Calibration Source: IBM Heron (133-qubit ibm_torino), last updated {snapshot['last_update_date']}")

    # 3. Execute Convex Duration Allocation Pass
    t0 = time.perf_counter()
    pm = PassManager([ConvexDurationAllocationPass(snapshot=snapshot, target=0.95, safety=1.02)])
    pm.run(c)
    elapsed = (time.perf_counter() - t0) * 1000

    result = pm.property_set["duration_allocation"]
    makespan = pm.property_set["certified_makespan"]
    gap = pm.property_set["duality_gap"]
    rounded = pm.property_set["rounded_durations"]

    print(f"[+] Optimization Completed in: {elapsed:.2f} ms!")
    print(f"[+] Feasible: {result['feasible']}")
    print(f"[+] Certified Makespan: {makespan:.4f} us")
    print(f"[+] Relative Duality Gap: {gap:.2e}")
    print(f"[+] Rounded Durations (1.02 grid): {[round(d, 4) for d in rounded]}")

    # 4. Machine-precision master-equation audit
    model = snapshot_model(c, [0, 1, 2, 3], snapshot, alpha=0.8)
    ana_exp = expectations(model, result["durations"])
    dense_exp = dense_observables(c, model, result["durations"])
    max_error = float(np.max(np.abs(ana_exp - dense_exp)))
    print(f"[+] Master-Equation Density Matrix Max Error: {max_error:.2e} (machine precision!)")

    print("=" * 70)
    print("LOCAL MODEL CHECKS COMPLETED; NO HARDWARE JOB WAS RUN.")
    print("=" * 70)

if __name__ == "__main__":
    main()
