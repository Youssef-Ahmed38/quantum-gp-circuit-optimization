import itertools
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SuperOp,Operator,Pauli,average_gate_fidelity
from qcgp.observables import snapshot_model,expectations,dense_observables,optimize_observables,optimal_uniform,upward_round,sample_observables,geometric_library
from qcgp.core import critical_path


def fixture_snapshot(n=3,error=.006):
    gates=[]
    for name in ['sx','x','rz','cz']:
        supports=itertools.combinations(range(n),2) if name=='cz' else [(q,) for q in range(n)]
        for qs in supports:
            gates.append(dict(gate=name,qubits=list(qs),parameters=[dict(name='gate_error',value=0. if name=='rz' else error,unit=''),dict(name='gate_length',value=0. if name=='rz' else 150 if name=='cz' else 40,unit='ns')]))
    return dict(gates=gates,qubits=[[] for _ in range(n)])


def example(seed=7):
    rng=np.random.default_rng(seed);c=QuantumCircuit(3)
    for _ in range(3):
        for q in range(3):c.rz(int(rng.integers(0,4))*np.pi/2,q);c.sx(q)
        c.cz(0,1);c.cz(1,2)
    return c,snapshot_model(c,[0,1,2],fixture_snapshot(),.8)


@pytest.mark.parametrize('seed',[3,7,11,29])
def test_signed_observable_predictions_match_independent_density_channels(seed):
    c,m=example(seed);tau=m.base*np.linspace(.5,2,len(m.base))
    assert np.max(abs(expectations(m,tau)-dense_observables(c,m,tau)))<1e-10


def test_snapshot_average_error_anchor_matches_independent_channel_fidelity():
    c=QuantumCircuit(1);c.sx(0);m=snapshot_model(c,[0],fixture_snapshot(1),.8)
    total=float(m.a[0]/m.base[0]+m.gamma[0]*m.base[0]);noise=SuperOp(Operator(np.eye(2)))
    p=-.5*np.expm1(-2*total/3)
    for label in ['X','Y','Z']:
        channel=(1-p)*SuperOp(Operator(np.eye(2)))+p*SuperOp(Operator(Pauli(label)))
        noise=channel.compose(noise)
    assert abs(1-average_gate_fidelity(noise)-.006)<1e-12


def test_exact_gp_convex_and_uniform_satisfy_same_operational_targets():
    _,m=example();target=.85
    gp=optimize_observables(m,target);direct=optimize_observables(m,target,method='convex')
    assert gp['feasible'] and direct['feasible']
    assert abs(gp['makespan']-direct['makespan'])<1e-6
    assert gp['normalized_dual_lower']<=direct['normalized_objective']+1e-6
    uniform=optimal_uniform(m,target)
    assert uniform is not None and gp['makespan']<=critical_path(uniform,m.edges).makespan+1e-7


def test_analytic_infeasibility_uses_common_gatewise_noise_minimum():
    _,m=example();result=optimize_observables(m,.999999)
    assert not result['feasible'] and result['status']=='analytic_infeasibility'
    assert optimal_uniform(m,.999999) is None


def test_quantization_protected_gp_meets_targets_after_available_pulse_rounding():
    _,m=example();q=1.07;result=optimize_observables(m,.85,safety=q)
    assert result['feasible']
    rounded=upward_round(m,result['durations'],q);lib=geometric_library(m,q)
    assert np.max(np.min(abs(lib-rounded[:,None]),axis=1))<1e-12
    assert np.min(expectations(m,rounded))>=.85-1e-7
    assert critical_path(rounded,m.edges).makespan<=q*result['makespan']+1e-9


def test_forward_frame_sampling_agrees_with_density_and_exact_predictions():
    c,m=example();sample=sample_observables(c,m,m.base,shots=32768,seed=882)
    exact=expectations(m,m.base)
    assert np.max(abs(np.asarray(sample['expectations'])-exact))<.025
    assert np.max(abs(dense_observables(c,m,m.base)-exact))<1e-10


@pytest.mark.parametrize('angle',[np.pi/4,float('nan'),float('inf')])
def test_nonclifford_and_nonfinite_virtual_angles_are_rejected(angle):
    c=QuantumCircuit(1);c.sx(0);c.rz(angle,0)
    with pytest.raises(ValueError,match='Clifford'):snapshot_model(c,[0],fixture_snapshot(1))


def test_missing_device_edge_cannot_be_emitted_as_native_instruction():
    c=QuantumCircuit(3);c.cz(0,2);snapshot=fixture_snapshot()
    snapshot['gates']=[g for g in snapshot['gates'] if not (g['gate']=='cz' and g['qubits']==[0,2])]
    with pytest.raises(ValueError,match='No snapshot'):snapshot_model(c,[0,1,2],snapshot)


def test_global_envelope_feasible_set_is_conservative_for_selected_observables():
    _,m=example();global_result=optimize_observables(m,.85,global_envelope=True,compute_dual=False)
    if global_result['feasible']:assert np.min(expectations(m,global_result['durations']))>=.85-1e-7
    aware=optimize_observables(m,.85)
    assert aware['feasible']
