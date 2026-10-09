import numpy as np
import pytest
from qiskit import QuantumCircuit, qasm3
from qiskit.transpiler import PassManager

from qcgp.generalized import GeneralizedProblem, solve_generalized
from qcgp.passes import ConvexDurationAllocationPass
from qcgp.validation import certify_duration_solution


def test_supporting_plane_certifies_chain_optimum():
    problem = GeneralizedProblem(np.array([1., 1.]), np.array([4., 4.]),
                                 np.array([4., 9.]), np.array([.1, .2]),
                                 ((0, 1),), .3, 1.)
    solution = solve_generalized(problem)
    certificate = certify_duration_solution(problem, solution["durations"])
    assert certificate["lower_bound"] <= certificate["objective"] + 1e-8
    assert certificate["absolute_gap"] < 1e-5
    assert certificate["objective"] == pytest.approx(solution["objective"], rel=1e-7)


def test_supporting_plane_rejects_deadline_violation():
    problem = GeneralizedProblem(np.array([1., 1.]), np.array([3., 3.]),
                                 np.array([1., 1.]), np.array([1., 1.]),
                                 ((0, 1),), .1, 1., deadline=3.)
    with pytest.raises(ValueError, match="deadline"):
        certify_duration_solution(problem, [2., 2.])


def test_duration_pass_does_not_emit_executable_gate_changes():
    circuit = QuantumCircuit(2)
    circuit.sx(0)
    circuit.cx(0, 1)
    analysis = ConvexDurationAllocationPass()
    output = PassManager(analysis).run(circuit)
    assert analysis.property_set["duration_allocation"]["feasible"]
    assert qasm3.dumps(output) == qasm3.dumps(circuit)
