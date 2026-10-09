import numpy as np
import pytest
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector, DensityMatrix, random_statevector
from qcgp.timing import (TimingModel, minimal_generator, synthetic_model, optimize_timing,
                         evaluate_timing, reference_metrics, robust_upper_model,
                         quantize_timing, replace_subsystem, simulate_timing)
from qcgp import JointGPTimingPass,DurationAwareCompiler
from qiskit.transpiler import PassManager
from qcgp.core import topology,benchmark
from qcgp.pulse import simulate_pulses,embed_matrix
from qcgp.pathflow import optimize_pathflow


def model(base, edges=(), a=None, c=None, b=0.3):
    base=np.asarray(base,dtype=float)
    return TimingModel(base,0.2*base,4*base,
        np.asarray(a if a is not None else base**2,dtype=float),
        np.asarray(c if c is not None else np.ones(len(base)),dtype=float),
        np.ones(len(base)),b,tuple(edges))


@pytest.mark.parametrize('method',['gp','dgp','convex'])
def test_serial_chain_matches_closed_form_and_pressure(method):
    m=model([1,2,3],[(0,1),(1,2)],a=[1,3,5],c=[0.7,0.9,1.1])
    r=optimize_timing(m,method=method)
    expected=np.sqrt(m.amplitude_noise/(m.gate_noise_rate+m.background_rate))
    # Objective stopping tolerance does not imply the same relative coordinate
    # tolerance; at a smooth optimum coordinate error is second order in cost.
    assert abs(r.objective-evaluate_timing(m,expected).objective)<1e-6
    assert np.allclose(r.durations,expected,rtol=1e-3)
    if method=='convex':
        assert np.allclose(r.path_pressure,m.background_rate/reference_metrics(m)[0],rtol=1e-4)
        assert r.kkt_relative_residual<1e-3


@pytest.mark.parametrize('method',['gp','convex'])
def test_parallel_slack_is_allocated_without_changing_critical_duration(method):
    m=model([1,5],a=[1,25],c=[1,1],b=1)
    r=optimize_timing(m,method=method)
    assert np.allclose(r.durations,[1,np.sqrt(12.5)],rtol=3e-4)
    assert np.isclose(r.makespan,r.durations[1])


@pytest.mark.parametrize('weights',[(1,0,0),(0.5,0.3,0.2),(0.1,0.1,0.8)])
def test_gp_and_direct_convex_agree_on_branching_problem(weights):
    m=model([1,2,1,3],[(0,2),(1,2),(2,3)])
    gp=optimize_timing(m,weights)
    direct=optimize_timing(m,weights,method='convex')
    assert abs(gp.objective-direct.objective)<1e-6
    assert np.allclose(gp.durations,direct.durations,rtol=2e-3,atol=1e-5)
    automatic=optimize_timing(m,weights,method='dgp')
    assert abs(gp.objective-automatic.objective)<1e-6


@pytest.mark.parametrize('method',['gp','convex'])
def test_deadline_and_control_budget_are_enforced(method):
    m=model([1,2],[(0,1)])
    r=optimize_timing(m,method=method,deadline=2.5,effort_limit=2)
    assert r.makespan<=2.5+1e-5
    assert r.effort<=2+1e-5


@pytest.mark.parametrize('method',['gp','convex'])
def test_infeasible_deadline_is_rejected(method):
    with pytest.raises(RuntimeError,match='infeasible'):
        optimize_timing(model([1,2],[(0,1)]),method=method,deadline=0.1)


def test_box_uncertainty_envelope_dominates_interior_coefficients():
    m=model([1,2],[(0,1)])
    upper=robust_upper_model(m,0.3)
    r=optimize_timing(upper)
    rng=np.random.default_rng(541)
    for _ in range(20):
        inside=TimingModel(m.base,m.lower,m.upper,
            m.amplitude_noise*rng.uniform(0.7,1.3,2),
            m.gate_noise_rate*rng.uniform(0.85,1.15,2),
            m.generator_squared,m.background_rate*rng.uniform(0.4,1.6),m.edges)
        assert evaluate_timing(inside,r.durations).risk<=r.risk+1e-10


def test_quantization_bound_controls_risk_and_preserves_gate_bounds():
    m=model([1,2,1],[(0,2)])
    r=optimize_timing(m)
    rounded,bound=quantize_timing(m,r,0.15)
    assert np.all(rounded.durations>=m.lower)
    assert np.all(rounded.durations<=m.upper)
    assert abs(rounded.risk-r.risk)<=bound+1e-10
    with pytest.raises(ValueError,match='No feasible'):
        quantize_timing(m,r,100)


@pytest.mark.parametrize('gate',['sx','x','cx','rz'])
def test_minimum_arc_generator_preserves_gate_up_to_phase(gate):
    c=QuantumCircuit(2 if gate=='cx' else 1)
    if gate=='cx': c.cx(0,1)
    elif gate=='rz': c.rz(5.3,0)
    else: getattr(c,gate)(0)
    u=Operator(c).data
    k,phase=minimal_generator(u)
    assert np.allclose(expm(-1j*k),np.exp(-1j*phase)*u,atol=1e-10)
    assert np.linalg.norm(k,2)<=np.pi/2+1e-10


def test_subsystem_replacement_has_correct_little_endian_orientation():
    rho=DensityMatrix.from_label('01').data
    assert np.allclose(replace_subsystem(rho,[0],2),np.diag([.5,.5,0,0]))
    assert np.allclose(replace_subsystem(rho,[1],2),np.diag([0,.5,0,.5]))
    assert np.allclose(replace_subsystem(rho,[0,1],2),np.eye(4)/4)


def test_single_pulse_channel_matches_independent_lindblad_exponential():
    c=QuantumCircuit(1); c.sx(0)
    m=synthetic_model(c)
    r=optimize_timing(m)
    initial=random_statevector(2,seed=831)
    simulated,_=simulate_timing(c,m,r,initial)
    k,_=minimal_generator(Operator(c).data)
    duration=r.durations[0]
    h=k/duration
    identity=np.eye(2)
    replacement=np.outer((identity/2).reshape(-1,order='F'),identity.reshape(-1,order='F'))
    rate=m.gate_noise_rate[0]+m.amplitude_noise[0]/duration**2+m.background_rate
    liouvillian=-1j*(np.kron(identity,h)-np.kron(h.T,identity))+rate*(replacement-np.eye(4))
    density=DensityMatrix(initial).data
    expected=(expm(liouvillian*duration)@density.reshape(-1,order='F')).reshape(2,2,order='F')
    assert np.allclose(simulated.data,expected,atol=1e-10)


@pytest.mark.parametrize('seed',[91,92,93])
def test_certificate_holds_with_entangled_reference_and_random_inputs(seed):
    c=QuantumCircuit(2);c.sx(0);c.cx(0,1);c.rz(.6,1)
    m=synthetic_model(c);r=optimize_timing(m)
    initial=random_statevector(16,seed=seed)
    actual,ideal=simulate_timing(c,m,r,initial)
    fidelity=float(np.real(np.trace(actual.data@ideal.data)))
    assert fidelity>=r.fidelity_lower_bound-1e-10
    trace_distance=.5*np.linalg.norm(actual.data-ideal.data,ord='nuc')
    assert trace_distance<=1-r.fidelity_lower_bound+1e-10


@pytest.mark.parametrize('method',['gp','convex'])
def test_second_free_solver_matches(method):
    m=model([1,2],[(0,1)])
    a=optimize_timing(m,method=method,solver='CLARABEL')
    b=optimize_timing(m,method=method,solver='ECOS')
    assert abs(a.objective-b.objective)<2e-6


@pytest.mark.parametrize('weights',[(0,1,0),(-1,1,0),(1,float('nan'),0)])
def test_invalid_weights_are_rejected(weights):
    with pytest.raises(ValueError): optimize_timing(model([1]),weights)


def test_paid_solver_and_zero_generator_are_rejected():
    with pytest.raises(ValueError,match='free solver'): optimize_timing(model([1]),solver='MOSEK')
    c=QuantumCircuit(1);c.rz(0,0)
    with pytest.raises(ValueError,match='identity'): synthetic_model(c)


def test_unmodeled_coherent_error_can_violate_certificate():
    c=QuantumCircuit(1);c.sx(0)
    m=synthetic_model(c);r=optimize_timing(m)
    actual,ideal=simulate_timing(c,m,r)
    unwanted=QuantumCircuit(1);unwanted.rz(.9,0)
    perturbed=actual.evolve(unwanted)
    fidelity=float(np.real(np.trace(perturbed.data@ideal.data)))
    assert fidelity<r.fidelity_lower_bound


def test_joint_analysis_pass_preserves_unitary_and_records_model():
    c=QuantumCircuit(2);c.sx(0);c.cx(0,1);c.rz(.6,1)
    manager=PassManager([JointGPTimingPass(synthetic_model)])
    out=manager.run(c)
    assert np.allclose(Operator(out).data,Operator(c).data)
    assert manager.property_set['joint_gp_timing'].risk>0
    assert isinstance(manager.property_set['joint_gp_model'],TimingModel)


def test_finite_search_uses_shared_normalizers_and_valid_circuits():
    coupling=topology(3,'line')
    compiler=DurationAwareCompiler(coupling,lambda c:synthetic_model(c,coupling),seeds=(7,8))
    result=compiler.compile(benchmark('qaoa',3,101))
    assert max(result.unitary_errors)<1e-8
    assert result.scores[result.candidate]<=min(result.scores)+2e-5*max(1,min(result.scores))
    assert len(result.timing.durations)==len(result.circuit.data)


def test_independent_concurrent_pulse_integrator_matches_covariant_channels():
    c=QuantumCircuit(2);c.sx(0);c.x(1);c.cx(0,1);c.rz(.4,0)
    m=synthetic_model(c);r=optimize_timing(m)
    initial=random_statevector(4,seed=593)
    composed,_=simulate_timing(c,m,r,initial)
    pulses,_,metadata=simulate_pulses(c,m,r,initial,background='global')
    assert np.allclose(composed.data,pulses.data,atol=1e-9)
    assert metadata['intervals']>0


def test_noncommuting_local_memory_noise_obeys_general_no_jump_bound():
    c=QuantumCircuit(2);c.sx(0);c.cx(0,1);c.x(1);c.rz(.4,0)
    m=synthetic_model(c);r=optimize_timing(m)
    initial=random_statevector(4,seed=731)
    actual,ideal,_=simulate_pulses(c,m,r,initial,background='local_dephasing')
    fidelity=float(np.real(np.trace(actual.data@ideal.data)))
    assert fidelity>=r.fidelity_lower_bound-1e-9
    zero,expected,_=simulate_pulses(c,m,r,initial,noise_scale=0)
    assert np.allclose(zero.data,expected.data,atol=1e-9)


def test_embedding_matches_qiskit_for_reversed_control_target_order():
    c=QuantumCircuit(3);c.cx(2,0)
    local=QuantumCircuit(2);local.cx(0,1)
    assert np.allclose(embed_matrix(Operator(local).data,[2,0],3),Operator(c).data)


@pytest.mark.parametrize('weights',[(1,0,0),(.5,.3,.2)])
def test_pathflow_bounds_independent_convex_optimum(weights):
    m=model([1,2,1,3],[(0,2),(1,2),(2,3)])
    flow=optimize_pathflow(m,weights,relative_tolerance=1e-7)
    direct=optimize_timing(m,weights,method='convex')
    assert flow.certified
    assert flow.dual_bound<=direct.objective+1e-6
    assert direct.objective<=flow.timing.objective+1e-6
    assert flow.relative_gap<=1e-7


def test_pathflow_reports_unfinished_optimization_and_keeps_valid_bounds():
    m=model([1,1,2,2],[(0,2),(1,3)],a=[1,2,5,1],c=[1,1,1,1],b=3)
    flow=optimize_pathflow(m,relative_tolerance=1e-15,max_iterations=1)
    assert flow.dual_bound<=flow.timing.objective+1e-10
    assert flow.certified==(flow.relative_gap<=1e-15)
