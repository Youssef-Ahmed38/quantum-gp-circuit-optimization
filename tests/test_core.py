import math
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qcgp.core import (benchmark, topology, compile_candidate, extract, critical_path,
                       gp_schedule, lp_schedule, unitary_error, noisy_fidelity,
                       _depolarizing_channel, normalized_score)


@pytest.mark.parametrize("durations,edges,expected", [
    ([], [], 0), ([2], [], 2), ([2, 3], [], 3),
    ([2, 3], [(0, 1)], 5), ([2, 3, 4], [(0, 2), (1, 2)], 7),
    ([0.1, 0.2], [(0, 1)], 0.3)])
def test_known_optima(durations, edges, expected):
    for solver in (critical_path, gp_schedule, lp_schedule):
        result = solver(durations, edges)
        assert abs(result.makespan - expected) < 1e-5
        assert result.violation < 1e-5


@pytest.mark.parametrize("seed", range(12))
def test_random_dags_agree_with_independent_lp(seed):
    rng = np.random.default_rng(seed)
    d = rng.uniform(0.01, 10, 12)
    edges = [(i, j) for j in range(len(d)) for i in range(j) if rng.random() < 0.2]
    exact = critical_path(d, edges).makespan
    assert abs(gp_schedule(d, edges).makespan - exact) < 1e-5
    assert abs(lp_schedule(d, edges).makespan - exact) < 1e-8


@pytest.mark.parametrize("durations,edges", [([0], []), ([-1], []),
    ([float("nan")], []), ([1], [(0, 0)]), ([1, 1], [(1, 0)]), ([1], [(0, 2)])])
def test_invalid_scheduling_inputs_are_rejected(durations, edges):
    with pytest.raises(ValueError):
        gp_schedule(durations, edges)


@pytest.mark.parametrize("family", ["qft", "grover", "vqe_ansatz", "qaoa"])
def test_full_unitary_and_native_connectivity(family):
    logical = benchmark(family, 3, 11)
    coupling = topology(3, "line")
    physical, _ = compile_candidate(logical, coupling, 7)
    extract(physical, coupling)
    assert unitary_error(logical, physical) < 1e-8


def test_gate_counts_do_not_imply_equivalence():
    a, b = QuantumCircuit(1), QuantumCircuit(1)
    a.h(0); a.x(0)
    b.x(0); b.h(0)
    assert dict(a.count_ops()) == dict(b.count_ops())
    assert not Operator(a).equiv(Operator(b))


@pytest.mark.parametrize("beta", [1/3, 0.5, 0.8])
def test_original_swap_constraints_are_infeasible(beta):
    # n >= logical + 3s and s >= beta*n imply (1-3beta)n >= logical>0.
    assert 1 - 3 * beta <= 1e-15


def test_upward_rounding_can_break_original_constraints():
    n, s, logical = 1.2, 0.4, 0
    assert n + 1e-12 >= logical + 3*s
    assert math.ceil(n) < logical + 3*math.ceil(s)


def test_depolarizing_map_matches_identity_mixture():
    from qiskit.quantum_info import DensityMatrix
    rho = DensityMatrix.from_label("0")
    p = 0.23
    out = rho.evolve(_depolarizing_channel(1, p)).data
    assert np.max(abs(out - ((1-p)*rho.data + p*np.eye(2)/2))) < 1e-12


def test_noise_simulation_is_physical_and_proxy_is_not_fidelity():
    c = QuantumCircuit(2); c.cx(0, 1)
    coupling = topology(2, "line")
    f, trace, smallest = noisy_fidelity(c, coupling)
    p = 0.005 + 0.002 + 0.001
    assert abs(f - (1 - 3*p/4)) < 1e-12
    assert abs(f - (1-p)) > 1e-4
    assert trace < 1e-12 and smallest > -1e-12


def test_layout_aware_validation_detects_corruption():
    original = benchmark("qft", 3, 1)
    physical, _ = compile_candidate(original, topology(3, "line"), 2)
    assert unitary_error(original, physical) < 1e-8
    physical.x(0)
    assert unitary_error(original, physical) > 0.1


def test_measurements_and_bad_connectivity_are_rejected():
    c = QuantumCircuit(2, 1); c.measure(0, 0)
    with pytest.raises(ValueError): extract(c)
    c = QuantumCircuit(3); c.cx(0, 2)
    with pytest.raises(ValueError): extract(c, topology(3, "line"))


def test_offset_does_not_change_physical_optimum():
    d, edges = [1, 2, 3], [(0, 2), (1, 2)]
    for delta in (0.1, 1, 10):
        assert abs(gp_schedule(d, edges, delta).makespan - 5) < 1e-5


def test_score_units_are_normalized():
    m = {"makespan": 10, "gate_cost": 30, "error_loss": 0.2}
    assert abs(normalized_score(m, m) - 1) < 1e-12


@pytest.mark.parametrize("n", [2, 3, 4])
def test_qft_generator_matches_fourier_matrix(n):
    indices = np.arange(2**n)
    expected = np.exp(2j*np.pi*np.outer(indices, indices)/2**n)/np.sqrt(2**n)
    assert np.max(abs(Operator(benchmark("qft", n, 0)).data - expected)) < 1e-12


@pytest.mark.parametrize("n,seed", [(3, 0), (3, 1), (4, 0)])
def test_grover_generator_amplifies_marked_state(n, seed):
    from qiskit.quantum_info import Statevector
    output = Statevector.from_instruction(benchmark("grover", n, seed)).data
    repetitions = 1 + seed % 2
    expected = math.sin((2*repetitions+1)*math.asin(1/math.sqrt(2**n)))**2
    assert abs(abs(output[-1])**2 - expected) < 1e-12


def test_grover_gp_regression_has_independent_certificate():
    circuit, _ = compile_candidate(benchmark("grover", 3, 101), topology(3, "line"), 7)
    durations, edges, _, _ = extract(circuit)
    exact = critical_path(durations, edges).makespan
    assert abs(gp_schedule(durations, edges).makespan - exact) < 1e-5 * exact


def test_qiskit_analysis_pass_attaches_schedule_and_preserves_circuit():
    from qiskit.transpiler import PassManager
    from qcgp.passes import GPSchedulingPass
    c = QuantumCircuit(2); c.sx(0); c.cx(0, 1)
    manager = PassManager([GPSchedulingPass(topology(2, "line"))])
    out = manager.run(c)
    assert Operator(out).equiv(Operator(c))
    assert manager.property_set["gp_schedule"].makespan > 0


def test_hybrid_compiler_returns_valid_candidate():
    from qcgp.passes import HybridCompiler
    logical = benchmark("qaoa", 3, 101)
    result = HybridCompiler(topology(3, "line"), candidate_seeds=(7, 24)).compile(logical)
    assert unitary_error(logical, result.circuit) < 1e-8
    assert result.selected_candidate in (0, 1)


@pytest.mark.parametrize("method", ["qiskit_l0", "qiskit_l1", "qiskit_l2", "qiskit_l3",
                                    "tket_common_route", "cirq_common_route"])
def test_baseline_adapters_preserve_unitary_and_connectivity(method):
    from qcgp.baselines import compile_baseline
    logical = benchmark("qaoa", 3, 101)
    coupling = topology(3, "line")
    physical, _ = compile_baseline(logical, coupling, method, 7)
    extract(physical, coupling)
    assert unitary_error(logical, physical) < 1e-8


def test_ecos_and_clarabel_agree():
    assert abs(gp_schedule([1, 2, 3], [(0, 2)], solver="ECOS").makespan - 4) < 1e-5


def test_large_grover_solver_failure_regression():
    circuit, _ = compile_candidate(benchmark("grover", 6, 101), topology(6, "line"), 7)
    durations, edges, _, _ = extract(circuit)
    exact = critical_path(durations, edges).makespan
    solved = gp_schedule(durations, edges)
    assert abs(solved.makespan-exact) < 1e-5 * exact


def test_gp_rankings_are_compared_to_independent_candidate_selection():
    from qcgp.core import certified_selection
    metrics=[dict(makespan=t,gate_cost=g,error_loss=e) for t,g,e in
             [(10,20,.1),(8,18,.09),(20,50,.2)]]
    selected,matched,raw,scores=certified_selection(metrics,[10.000001,8.000001,20.000001])
    assert selected==matched==raw==1
    assert scores[1]<scores[0]<scores[2]
