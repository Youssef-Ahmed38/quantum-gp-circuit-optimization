"""Declared free-local study of nonunitary noise and finite characterized pulses."""
from pathlib import Path
from dataclasses import asdict
import argparse
import json
import platform
import importlib.metadata
import time
import numpy as np
from qiskit import QuantumCircuit,qpy
from qiskit.quantum_info import Statevector,DensityMatrix,Operator,SuperOp,Choi,random_statevector,process_fidelity,diamond_norm
from qcgp.core import benchmark,unitary_error,critical_path
from qcgp.physical import PhysicalParameters,compile_pauli,physical_model,physical_channel,controls
from qcgp.library import duration_library,optimize_library,slack_policy,nearest_policy
from qcgp.characterization import characterize_library,library_certificate,transfer_gp_lower_bound
from qcgp.timing import evaluate_timing,optimize_timing,reference_metrics
from qcgp.pathflow import optimize_pathflow

ROOT=Path(__file__).resolve().parents[1]


def save(path,value):
    path.write_text(json.dumps(value,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x))+'\n',encoding='utf8')


def workload(family,n,seed):
    rng=np.random.default_rng(seed)
    if family in ['random_pauli','mirror']:
        c=QuantumCircuit(n)
        for _ in range(3):
            for q in range(n):c.rx(float(rng.uniform(.1,2.8)),q);c.rz(float(rng.uniform(-2.8,-.1)),q)
            for q in range(n-1):c.rxx(float(rng.uniform(.1,1.4)),q,q+1)
        if family=='mirror':c=c.compose(c.inverse())
        return c,c
    original=benchmark(family,n,seed)
    if family in ['qft','grover']:
        preparation=QuantumCircuit(n)
        for q in range(n):preparation.rx(float(rng.uniform(.1,2.8)),q);preparation.rz(float(rng.uniform(-2.8,-.1)),q)
        original=preparation.compose(original)
    return original,compile_pauli(original)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--quick',action='store_true');args=parser.parse_args()
    folder=ROOT/'evidence/physical_study';folder.mkdir(exist_ok=True)
    families=['qft','grover','vqe_ansatz','qaoa','random_pauli','mirror']
    sizes=[2,3];seeds=[101,102,103,104];regimes={'low_drive':.002,'balanced':.02,'high_drive':.2}
    if args.quick:families=['qft'];sizes=[2];seeds=[101];regimes={'balanced':.02}
    protocol=dict(families=families,sizes=sizes,seeds=seeds,amplitude_eta_regimes=regimes,
        duration_ratios=[.5,.75,1,1.5,2,3],t1_uniform_us=[80,160],tphi_uniform_us=[120,240],
        spin_model=True,all_to_all=True,hardware_calibrated=False,free_local_only=True,
        active_rate_per_us=.0005,omega_single_radians_per_us=1.5,omega_pair_radians_per_us=.35,
        two_qubit_eta_multiplier=4,coefficients_chosen_before_experiments=True,
        input_preparation='Seeded RX/RZ prefix for QFT/Grover; seeded variational/random circuits; mirror inverse pairs kept intentionally',
        mirror_compiler_cancellation_disabled=True,primary_metrics=['risk','process_fidelity','zero_fidelity','haar_fidelity','observable_error','latency','finite_optimality_gap'],
        interpretation='Exploratory declared simulator model; no physical calibration, population significance or universal compiler claim')
    protocol_path=folder/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:raise RuntimeError('Existing study has a different protocol')
    save(protocol_path,protocol)
    checkpoint=folder/'checkpoint.json';done=json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    for family in families:
        for n in sizes:
            for seed in seeds:
                case=f'{family}_{n}_{seed}'
                original,c=workload(family,n,seed)
                error=unitary_error(original,c)
                if error>1e-8:raise RuntimeError('Native control circuit changed logical unitary')
                with (folder/(case+'.qpy')).open('wb') as f:qpy.dump([original,c],f)
                rng=np.random.default_rng(seed+800*n)
                t1=tuple(rng.uniform(80,160,n));tphi=tuple(rng.uniform(120,240,n))
                matrices,_,_=controls(c);angles=np.array([2*np.linalg.norm(k,2) for k in matrices])
                for regime,eta in regimes.items():
                    key=case+'_'+regime
                    if key in done:continue
                    p=PhysicalParameters(t1,tphi,amplitude_eta=eta)
                    model=physical_model(c,p);lib=duration_library(model);ref=reference_metrics(model)
                    flow=optimize_pathflow(model)
                    if not flow.certified:
                        print(f'Iteration limit retained for {key}: continuous relative gap={flow.relative_gap:.3e}',flush=True)
                    direct=optimize_timing(model,method='convex')
                    if not flow.dual_bound-2e-6<=direct.objective<=flow.timing.objective+2e-6:raise RuntimeError('Independent convex optimum outside bracket')
                    pulse,rates,characterized=characterize_library(c,p,lib)
                    finite=optimize_library(model,lib,time_limit=10);lp=optimize_library(model,lib,relax=True)
                    measured=optimize_library(model,lib,pulse_bounds=pulse,support_rates=rates,time_limit=10)
                    measured_lp=optimize_library(model,lib,relax=True,pulse_bounds=pulse,support_rates=rates)
                    transfer=transfer_gp_lower_bound(model,lib,pulse,rates,flow.dual_bound,ref)
                    if transfer['transferred_lower_bound']>measured.objective+1e-6:raise RuntimeError('GP certificate transfer exceeds feasible characterized score')
                    uniform_choices=[lib[:,k] for k in range(lib.shape[1])]
                    uniform=min(uniform_choices,key=lambda tau:evaluate_timing(model,tau).risk)
                    measured_uniform=min(uniform_choices,key=lambda tau:library_certificate(model,tau,lib,pulse,rates)['integrated_library_bound'])
                    policies=dict(fixed=model.base,fastest=lib[:,0],uniform=uniform,
                        published_slack=slack_policy(model,lib,angles),gp_nearest=nearest_policy(model,flow.timing,lib),
                        finite_milp=finite.durations,continuous_gp=flow.timing.durations,
                        characterized_milp=measured.durations,characterized_uniform=measured_uniform)
                    rows=[];channels={};input_states={'zero':Statevector.from_label('0'*n),'haar':random_statevector(2**n,seed=seed+2001)}
                    ideal_channel=SuperOp(Operator(c));d=2**n
                    for method,tau in policies.items():
                        begin=time.perf_counter();channel,metadata=physical_channel(c,p,tau)
                        row=dict(case=case,key=key,family=family,qubits=n,seed=seed,regime=regime,method=method,
                            gates=len(c.data),depth=c.depth(),unitary_error=error,durations=tau.tolist(),
                            risk=evaluate_timing(model,tau).risk,makespan=metadata['makespan'],
                            process_fidelity=float(process_fidelity(channel,ideal_channel)),general_certificate=metadata['certificate'],
                            simulation_seconds=time.perf_counter()-begin,tp_error=metadata['tp_error'],minimum_choi_eigenvalue=metadata['minimum_choi_eigenvalue'])
                        if row['process_fidelity']<metadata['certificate']['fidelity_floor']-1e-7:raise RuntimeError('Purified-channel general certificate failed')
                        if method!='continuous_gp':
                            row['library_certificate']=library_certificate(model,tau,lib,pulse,rates)
                            if row['process_fidelity']<row['library_certificate']['fidelity_floor']-1e-7:raise RuntimeError('Finite-library channel certificate failed')
                        for label,initial in input_states.items():
                            actual=DensityMatrix(initial).evolve(channel).data;ideal=initial.evolve(c).data
                            fidelity=float(np.real(ideal.conj()@actual@ideal))
                            distance=float(.5*np.linalg.norm(actual-np.outer(ideal,ideal.conj()),ord='nuc'))
                            floor=metadata['certificate']['fidelity_floor'];upper=metadata['certificate']['half_diamond_upper']
                            if fidelity<floor-1e-7 or distance>upper+1e-7:raise RuntimeError('General physical state bound failed')
                            if 'library_certificate' in row:
                                cert=row['library_certificate']
                                if fidelity<cert['fidelity_floor']-1e-7 or distance>cert['half_diamond_upper']+1e-7:raise RuntimeError('Characterized state bound failed')
                            observable=np.diag([(-1)**(i&1) for i in range(d)])
                            energy_error=float(abs(np.trace(observable@actual)-ideal.conj()@observable@ideal))
                            row[label]=dict(fidelity=fidelity,trace_distance=distance,observable_error=energy_error,probability_zero=float(np.real(actual[0,0])),ideal_probability_zero=float(abs(ideal[0])**2))
                        rows.append(row);channels[method]=channel
                    extra={}
                    if n==2 and seed==101 and regime=='balanced':
                        actual=channels['characterized_milp'];ode,_=physical_channel(c,p,measured.durations,integrator='ode')
                        difference=float(np.max(abs(actual.data-ode.data)))
                        if difference>1e-7:raise RuntimeError('Independent physical integrators disagree')
                        diamond=.5*diamond_norm(Choi(actual)-Choi(ideal_channel),solver='CLARABEL')
                        cert=library_certificate(model,measured.durations,lib,pulse,rates)
                        if diamond>cert['half_diamond_upper']+1e-6:raise RuntimeError('Independent diamond SDP exceeds characterized bound')
                        extra=dict(independent_ode_channel_difference=difference,independent_half_diamond=diamond,certificate=cert)
                    np.savez_compressed(folder/(key+'_characterized_channel.npz'),channel=channels['characterized_milp'].data)
                    done[key]=dict(rows=rows,parameters=asdict(p),pulse_library=lib.tolist(),pulse_bounds=pulse.tolist(),support_rates=rates.tolist(),
                        characterization=characterized,continuous=asdict(flow),direct_objective=direct.objective,
                        finite=asdict(finite),finite_lp=asdict(lp),characterized_finite=asdict(measured),characterized_lp=asdict(measured_lp),transfer=transfer,extra=extra)
                    save(checkpoint,done)
                    print(f'Validated {key}: finite gap={finite.relative_gap:.2e}; characterized gap={measured.relative_gap:.2e}',flush=True)
    save(folder/'results.json',[row for item in done.values() for row in item['rows']])
    save(folder/'cases.json',done)
    save(folder/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),packages={name:importlib.metadata.version(name) for name in ['qiskit','cvxpy','numpy','scipy','clarabel']},all_free_local=True,paid_services=False,hardware_jobs=0))
    print(json.dumps(dict(configurations=len(done),policies=sum(len(v['rows']) for v in done.values()),independent_integrator_and_diamond_checks=sum(bool(v['extra']) for v in done.values())),indent=2),flush=True)


if __name__=='__main__':main()
