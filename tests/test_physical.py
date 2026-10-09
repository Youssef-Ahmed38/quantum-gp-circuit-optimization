import itertools
import numpy as np
import pytest
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator,SuperOp,Choi,Statevector,DensityMatrix,diamond_norm
from qcgp.core import benchmark,critical_path,unitary_error
from qcgp.physical import PhysicalParameters,compile_pauli,physical_model,physical_channel,controls,dissipator,LOWERING,general_certificate
from qcgp.library import duration_library,optimize_library,nearest_policy,slack_policy
from qcgp.pathflow import optimize_pathflow
from qcgp.timing import TimingModel,evaluate_timing,reference_metrics
from qcgp.characterization import choi_error_upper,characterize_library,library_certificate,transfer_gp_lower_bound


def example():
    c=QuantumCircuit(2);c.rx(.7,0);c.rz(.4,1);c.rxx(.6,0,1);c.rx(-.3,1)
    p=PhysicalParameters((120,140),(180,160))
    return c,p,physical_model(c,p)


@pytest.mark.parametrize('family',['qft','grover','vqe_ansatz','qaoa'])
def test_pauli_translation_preserves_full_logical_operator(family):
    logical=benchmark(family,3,101);native=compile_pauli(logical)
    assert unitary_error(logical,native)<1e-8
    matrices,_,_=controls(native)
    for k,instruction in zip(matrices,native.data):assert np.allclose(expm(-1j*k),Operator(instruction.operation).data,atol=1e-10)


def test_relaxation_liouvillian_matches_analytic_amplitude_damping():
    t=2.3;rate=.07
    channel=SuperOp(expm(dissipator(np.sqrt(rate)*LOWERING)*t))
    excited=DensityMatrix(Statevector.from_label('1')).evolve(channel).data
    assert np.allclose(np.diag(excited),[1-np.exp(-rate*t),np.exp(-rate*t)])
    coherence=DensityMatrix(Statevector.from_label('+')).evolve(channel).data[0,1]
    assert np.isclose(coherence,.5*np.exp(-rate*t/2))


def test_nonunitary_channel_has_valid_general_diamond_certificate():
    c,p,m=example();flow=optimize_pathflow(m)
    channel,info=physical_channel(c,p,flow.timing.durations)
    ideal=SuperOp(Operator(c))
    distance=.5*diamond_norm(Choi(channel)-Choi(ideal),solver='CLARABEL')
    assert distance<=info['certificate']['half_diamond_upper']+1e-7
    assert info['tp_error']<1e-9


def test_adaptive_ode_and_matrix_exponentials_agree_under_relaxation():
    c,p,m=example()
    a,_=physical_channel(c,p,m.base,integrator='expm')
    b,_=physical_channel(c,p,m.base,integrator='ode')
    assert np.max(abs(a.data-b.data))<1e-8


@pytest.mark.parametrize('epsilon',[0,.03,-.03])
def test_bound_includes_signed_coherent_control_errors(epsilon):
    c,p,m=example()
    p=PhysicalParameters(p.t1,p.tphi,coherent_fraction=epsilon)
    actual,info=physical_channel(c,p,m.base)
    initial=Statevector.from_label('++')
    state=DensityMatrix(initial).evolve(actual).data;ideal=initial.evolve(c).data
    fidelity=float(np.real(ideal.conj()@state@ideal))
    assert fidelity>=info['certificate']['fidelity_floor']-1e-8


def small_model():
    base=np.array([1.,1.,1.])
    return TimingModel(base,.5*base,3*base,np.array([.1,.8,.3]),np.array([.1,.1,.1]),base,.3,((0,2),(1,2)))


def test_finite_library_matches_exhaustive_optimization_and_lp_lower_bound():
    m=small_model();lib=duration_library(m,ratios=(.5,1.,2.))
    policy=optimize_library(m,lib);lp=optimize_library(m,lib,relax=True)
    brute=min(evaluate_timing(m,lib[np.arange(3),choice]).objective for choice in itertools.product(range(3),repeat=3))
    assert policy.optimal
    assert abs(policy.objective-brute)<1e-8
    assert lp.lower_bound<=brute+1e-8
    assert policy.lower_bound<=brute+1e-8


def test_published_slack_policy_preserves_fastest_latency_and_uses_available_pulses():
    m=small_model();lib=duration_library(m)
    tau=slack_policy(m,lib,np.ones(3))
    assert np.isclose(critical_path(tau,m.edges).makespan,critical_path(lib[:,0],m.edges).makespan)
    assert all(np.any(abs(lib[i]-t)<1e-12) for i,t in enumerate(tau))


def test_deadline_library_solution_is_an_exact_admissible_policy():
    m=small_model();lib=duration_library(m)
    deadline=critical_path(lib[:,0],m.edges).makespan
    result=optimize_library(m,lib,deadline=deadline)
    assert critical_path(result.durations,m.edges).makespan<=deadline+1e-8
    with pytest.raises(RuntimeError,match='No feasible'):optimize_library(m,lib,deadline=deadline/2)


def test_nearest_policy_returns_only_library_controls():
    m=small_model();lib=duration_library(m);flow=optimize_pathflow(m)
    tau=nearest_policy(m,flow.timing,lib)
    assert all(np.any(abs(lib[i]-t)<1e-12) for i,t in enumerate(tau))


@pytest.mark.parametrize('bad',[0,-1,float('nan')])
def test_invalid_physical_coherence_times_are_rejected(bad):
    c,p,_=example()
    with pytest.raises(ValueError):physical_model(c,PhysicalParameters((bad,140),p.tphi))


def test_vacuous_general_bound_is_explicit_not_negative_fidelity():
    m=small_model();cert=general_certificate(m,m.base,coherent_fraction=2)
    assert cert['fidelity_floor']==0 and cert['half_diamond_upper']==1


def test_choi_dual_upper_bound_brackets_independent_diamond_sdp():
    channel=SuperOp(expm(dissipator(.3*LOWERING)*2))
    ideal=SuperOp(Operator(np.eye(2)))
    bracket=choi_error_upper(channel,ideal)
    exact=.5*diamond_norm(Choi(channel)-Choi(ideal),solver='CLARABEL')
    assert bracket['choi_trace_lower']<=exact+1e-7
    assert exact<=bracket['half_diamond_upper']+1e-7


def test_characterized_library_bound_covers_concurrent_relaxation_and_controls():
    c,p,m=example();lib=duration_library(m,ratios=(.5,1.,2.))
    bounds,rates,_=characterize_library(c,p,lib)
    chosen=optimize_library(m,lib,pulse_bounds=bounds,support_rates=rates)
    lp=optimize_library(m,lib,relax=True,pulse_bounds=bounds,support_rates=rates)
    cert=library_certificate(m,chosen.durations,lib,bounds,rates)
    actual,_=physical_channel(c,p,chosen.durations)
    exact=.5*diamond_norm(Choi(actual)-Choi(SuperOp(Operator(c))),solver='CLARABEL')
    assert exact<=cert['half_diamond_upper']+1e-7
    assert lp.lower_bound<=chosen.objective+1e-7
    assert abs(chosen.objective-cert['integrated_library_bound']/reference_metrics(m)[0])<1e-8


def test_gp_dual_transfer_brackets_nonposynomial_finite_costs():
    m=small_model();lib=duration_library(m,ratios=(.5,1.,2.))
    pulse=np.array([[.1,.4,.3],[.5,.2,.6],[.2,.3,.1]])
    rates=np.zeros(3);flow=optimize_pathflow(m)
    finite=optimize_library(m,lib,pulse_bounds=pulse,support_rates=rates)
    transfer=transfer_gp_lower_bound(m,lib,pulse,rates,flow.dual_bound,reference_metrics(m))
    assert transfer['transferred_lower_bound']<=finite.lower_bound+1e-6


def test_general_no_jump_floor_strengthens_fidelity_but_not_diamond_bound():
    m=small_model();cert=general_certificate(m,m.base)
    assert cert['no_jump_fidelity_floor']>=cert['trace_distance_fidelity_floor']
    assert cert['half_diamond_upper']==min(1,cert['noise_bound'])


def test_tight_fastest_deadline_retains_feasible_qaoa_library_policy():
    # Regression: presolve reported infeasible despite the explicit fastest witness.
    c=compile_pauli(benchmark('qaoa',3,103))
    rng=np.random.default_rng(2503)
    p=PhysicalParameters(tuple(rng.uniform(80,160,3)),tuple(rng.uniform(120,240,3)))
    m=physical_model(c,p);lib=duration_library(m)
    bounds,rates,_=characterize_library(c,p,lib)
    deadline=critical_path(lib[:,0],m.edges).makespan
    chosen=optimize_library(m,lib,pulse_bounds=bounds,support_rates=rates,deadline=deadline+1e-9)
    assert critical_path(chosen.durations,m.edges).makespan<=deadline+1e-7
    assert chosen.lower_bound<=chosen.objective+1e-7


def test_physical_parameter_json_roundtrip_preserves_characterization():
    import json
    from dataclasses import asdict
    c,p,m=example()
    restored=PhysicalParameters(**json.loads(json.dumps(asdict(p))))
    lib=duration_library(m,ratios=(.5,1.))
    original,_,_=characterize_library(c,p,lib)
    replay,_,_=characterize_library(c,restored,lib)
    assert restored==p
    assert np.allclose(original,replay)
