"""Free local baselines with a common native-gate/routing stage.

TKET/Cirq comparisons are logical-optimizer adapters, not claims about the
complete native hardware pipelines of either compiler.
"""
import time
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from .core import BASIS


def compile_baseline(logical, coupling, method, seed):
    begin = time.perf_counter()
    options = dict(basis_gates=list(BASIS), coupling_map=coupling,
                   layout_method="sabre", routing_method="sabre",
                   seed_transpiler=seed, num_processes=1, qubits_initially_zero=False)
    if method.startswith("qiskit_l"):
        level = int(method[-1])
        if level not in range(4):
            raise ValueError("Qiskit level must be 0..3")
        result = transpile(logical, optimization_level=level, **options)
    else:
        intermediate = transpile(logical, basis_gates=["u3", "cx"],
                                  optimization_level=0, num_processes=1,
                                  qubits_initially_zero=False)
        if method == "tket_common_route":
            from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
            from pytket.passes import FullPeepholeOptimise
            tket_circuit = qiskit_to_tk(intermediate)
            FullPeepholeOptimise().apply(tket_circuit)
            optimized = tk_to_qiskit(tket_circuit, replace_implicit_swaps=True)
        elif method == "cirq_common_route":
            import cirq
            qs = cirq.LineQubit.range(logical.num_qubits)
            operations = []
            for ins in intermediate.data:
                indices = [intermediate.find_bit(q).index for q in ins.qubits]
                operations.append(cirq.MatrixGate(Operator(ins.operation).data).on(
                    *(qs[i] for i in reversed(indices))))
            cirq_circuit = cirq.Circuit(operations)
            optimized_cirq = cirq.optimize_for_target_gateset(cirq_circuit,
                gateset=cirq.CZTargetGateset(atol=1e-10), ignore_failures=False,
                max_num_passes=2)
            optimized = QuantumCircuit(logical.num_qubits)
            for op in optimized_cirq.all_operations():
                indices = [q.x for q in op.qubits]
                optimized.unitary(np.asarray(cirq.unitary(op)), list(reversed(indices)))
        else:
            raise ValueError(f"Unknown baseline {method}")
        result = transpile(optimized, optimization_level=0, **options)
    return result, time.perf_counter()-begin
