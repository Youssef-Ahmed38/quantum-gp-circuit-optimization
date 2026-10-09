"""Free local joint-timing study. Saved original circuits are reused, not retuned.

All coefficient choices, comparator definitions and seeds are in this script.
Uncertainty is specified by the experimenter, not learned from hardware data.
"""
from pathlib import Path
from dataclasses import asdict
import argparse
import csv
import json
import time
import platform
import importlib.metadata
import numpy as np
from qiskit import qpy, QuantumCircuit
from qiskit.quantum_info import Statevector, random_statevector, Pauli
from qcgp.core import topology, benchmark, compile_candidate, unitary_error, extract
from qcgp.timing import (synthetic_model, optimize_timing, evaluate_timing,
    reference_metrics, robust_upper_model, quantize_timing, simulate_timing)
from qcgp.pathflow import optimize_pathflow


def solve_joint(model,weights=(1,0,0),references=None):
    if len(model.base)<=300 and tuple(weights)==(1,0,0):
        return optimize_timing(model,weights,references=references)
    flow=optimize_pathflow(model,weights,references=references)
    if not flow.certified:raise RuntimeError('Large GP-equivalent subproblem lacked a dual-gap certificate')
    return flow.timing


def serialize(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    raise TypeError(type(value).__name__)


def save(path, data):
    path.write_text(json.dumps(data,indent=2,default=serialize)+'\n',encoding='utf8')


def csv_save(path, rows):
    fields=sorted(set().union(*(row.keys() for row in rows)))
    with path.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,default=serialize) if isinstance(v,(list,dict,np.ndarray)) else v for k,v in row.items()})


def strategies(model):
    upper=robust_upper_model(model,0.3)
    fixed=evaluate_timing(model,model.base,method='fixed')
    fastest=evaluate_timing(model,model.lower,method='fastest')
    local=evaluate_timing(model,np.clip(np.sqrt(model.amplitude_noise/model.gate_noise_rate),model.lower,model.upper),method='local')
    a=float(np.sum(model.amplitude_noise/model.base))
    c=float(np.sum(model.gate_noise_rate*model.base)+model.background_rate*fixed.makespan)
    ratio=float(np.clip(np.sqrt(a/c),0.5,3.0))
    uniform=evaluate_timing(model,model.base*ratio,method='uniform')
    # This is a continuous shortest-latency comparator inspired by, not a
    # reproduction of, Lussi et al.'s discrete fast/robust pulse selection.
    deadline=optimize_timing(model,method='convex',deadline=fastest.makespan*(1+1e-8))
    joint=solve_joint(model)
    direct=optimize_timing(model,method='convex')
    robust=solve_joint(upper)
    quantized,qbound=quantize_timing(model,joint,0.05*model.base)
    outputs={'fixed':fixed,'fastest':fastest,'local':local,'uniform':uniform,
             'fast_deadline':deadline,'joint_gp':joint,'joint_convex':direct,
             'robust_gp':robust,'quantized_gp':quantized}
    gap=abs(joint.objective-direct.objective)/max(1,abs(direct.objective))
    if gap>2e-5: raise RuntimeError(f'Independent convex disagreement {gap}')
    for name,result in outputs.items():
        if name!='robust_gp' and name!='joint_convex' and result.risk < joint.risk-2e-5*max(1,joint.risk):
            raise RuntimeError(f'Joint risk worse than feasible {name}')
    return outputs,upper,gap,qbound


def fidelity_record(circuit,model,result,initial,scenario):
    factors={'nominal':(1,1,1),'upper_box':(1.3,1.15,1.6)}[scenario]
    actual,ideal=simulate_timing(circuit,model,result,initial,
        amplitude_scale=factors[0],gate_rate_scale=factors[1],background_scale=factors[2])
    fidelity=float(np.real(np.trace(actual.data@ideal.data)))
    trace_distance=float(.5*np.linalg.norm(actual.data-ideal.data,ord='nuc'))
    certmodel=model if scenario=='nominal' else robust_upper_model(model,0.3)
    cert=evaluate_timing(certmodel,result.durations).fidelity_lower_bound
    if fidelity<cert-1e-8 or trace_distance>1-cert+1e-8:
        raise RuntimeError('Operational channel certificate failed within model')
    n=actual.num_qubits
    observables=[Pauli('I'*(n-1)+'Z').to_matrix(),Pauli('I'*(n-2)+'XX').to_matrix()]
    errors=[float(abs(np.trace(p@(actual.data-ideal.data)))) for p in observables]
    if max(errors)>2*(1-cert)+1e-8: raise RuntimeError('Observable certificate failed')
    return {'fidelity':fidelity,'trace_distance':trace_distance,'fidelity_lower_bound':cert,
            'observable_error_z':errors[0],'observable_error_xx':errors[1],
            'trace_error':float(abs(np.trace(actual.data)-1)),
            'minimum_eigenvalue':float(np.linalg.eigvalsh(actual.data).min())}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--evidence',type=Path,default=Path(__file__).resolve().parents[1]/'evidence')
    parser.add_argument('--quick',action='store_true')
    args=parser.parse_args()
    out=args.evidence/'joint_timing';out.mkdir(parents=True,exist_ok=True)
    records=json.loads((args.evidence/'candidates.json').read_text(encoding='utf8'))
    cases=[r for r in records if r['candidate']==0]
    if args.quick: cases=[r for r in cases if r['qubits']==3 and r['instance']==0 and r['topology']=='line']
    methods=['fixed','fastest','local','uniform','fast_deadline','joint_gp','robust_gp','quantized_gp']
    rows=[];noise=[];process=[];failures=[];pareto=[];outer=[];sensitivity=[];scale=[]
    completed={}
    checkpoint=out/'case_checkpoint.json'
    if checkpoint.exists(): completed=json.loads(checkpoint.read_text(encoding='utf8'))
    start=time.perf_counter()
    for info in cases:
        case=info['case']
        if case in completed:
            data=completed[case]
            rows+=data['rows'];noise+=data['noise'];process+=data['process'];failures+=data['failures']
            continue
        with (args.evidence/'circuits'/f'{case}.qpy').open('rb') as f: circuits=qpy.load(f)
        logical,circuit=circuits[0],circuits[1]
        error=unitary_error(logical,circuit)
        if error>1e-8: raise RuntimeError('Saved circuit failed full-unitary check')
        model=synthetic_model(circuit,topology(info['qubits'],info['topology']))
        outputs,upper,gap,qbound=strategies(model)
        case_rows=[];case_noise=[];case_process=[];case_failures=[]
        for method,result in outputs.items():
            nominal=evaluate_timing(model,result.durations)
            box=evaluate_timing(upper,result.durations)
            record={k:info[k] for k in ['case','family','qubits','topology','instance','gates','two_qubit','gate_cost']}
            record.update(asdict(result));record.update(method=method,implementation=result.method,nominal_risk=nominal.risk,
                nominal_fidelity_bound=nominal.fidelity_lower_bound,worst_box_risk=box.risk,
                robust_fidelity_bound=box.fidelity_lower_bound,gp_convex_gap=gap,
                quantization_risk_bound=qbound if method=='quantized_gp' else None,unitary_error=error)
            case_rows.append(record)
            if info['qubits']<=4 and method in methods:
                inputs={'zero':Statevector.from_label('0'*info['qubits']),
                        'haar':random_statevector(2**info['qubits'],seed=1011+info['qubits']*100+info['instance'])}
                for label,initial in inputs.items():
                    record=fidelity_record(circuit,model,result,initial,'nominal')
                    case_noise.append(dict(case=case,family=info['family'],qubits=info['qubits'],topology=info['topology'],
                        instance=info['instance'],method=method,input=label,scenario='nominal',**record))
                record=fidelity_record(circuit,model,result,inputs['zero'],'upper_box')
                case_noise.append(dict(case=case,family=info['family'],qubits=info['qubits'],topology=info['topology'],
                    instance=info['instance'],method=method,input='zero',scenario='upper_box',**record))
                if method=='joint_gp' and info['instance']==0:
                    actual,ideal=simulate_timing(circuit,model,result,inputs['haar'])
                    unwanted=QuantumCircuit(info['qubits'])
                    for q in range(info['qubits']): unwanted.rz(0.7,q)
                    perturbed=actual.evolve(unwanted)
                    f=float(np.real(np.trace(perturbed.data@ideal.data)))
                    case_failures.append(dict(case=case,unmodeled_noise='terminal_coherent_z_rotation',
                        rotation_radians=0.7,fidelity=f,nominal_certificate=result.fidelity_lower_bound,
                        certificate_breached=bool(f<result.fidelity_lower_bound-1e-8)))
            if info['qubits']==3 and info['instance']==0 and method in ['fixed','joint_gp','robust_gp']:
                d=2**info['qubits']
                initial=Statevector(np.eye(d).reshape(-1)/np.sqrt(d))
                record=fidelity_record(circuit,model,result,initial,'nominal')
                case_process.append(dict(case=case,method=method,system_qubits=info['qubits'],reference_qubits=info['qubits'],**record))
        data=dict(rows=case_rows,noise=case_noise,process=case_process,failures=case_failures)
        completed[case]=data;save(checkpoint,completed)
        rows+=case_rows;noise+=case_noise;process+=case_process;failures+=case_failures
        print(f'Validated timing/noise {case}: GP/direct gap={gap:.2e}',flush=True)
    save(out/'results.json',rows);csv_save(out/'results.csv',rows)
    save(out/'noise.json',noise);csv_save(out/'noise.csv',noise)
    save(out/'process.json',process);save(out/'out_of_model.json',failures)
    weights=[(1,0,0),(.7,.3,0),(.4,.6,0),(.7,0,.3),(.4,.2,.4)]
    for info in cases:
        if info['instance']!=0: continue
        case=info['case']
        with (args.evidence/'circuits'/f'{case}.qpy').open('rb') as f: circuits=qpy.load(f)
        coupling=topology(info['qubits'],info['topology'])
        model=synthetic_model(circuits[1],coupling)
        for w in weights:
            result=solve_joint(model,w)
            record=asdict(result);record.update(case=case,family=info['family'],qubits=info['qubits'],topology=info['topology'],weights=list(w))
            pareto.append(record)
        if info['qubits']<=4:
            for background in [0.00002,0.00015,0.001]:
                changed=synthetic_model(circuits[1],coupling,background=background)
                for method in ['fixed','uniform','joint_gp']:
                    if method=='uniform':
                        fixed=evaluate_timing(changed,changed.base)
                        a=float(np.sum(changed.amplitude_noise/changed.base))
                        c=float(np.sum(changed.gate_noise_rate*changed.base)+changed.background_rate*fixed.makespan)
                        result=evaluate_timing(changed,changed.base*np.clip(np.sqrt(a/c),.5,3.0),method='uniform')
                    else:
                        result=optimize_timing(changed) if method=='joint_gp' else evaluate_timing(changed,changed.base)
                    f=fidelity_record(circuits[1],changed,result,None,'nominal')
                    sensitivity.append(dict(case=case,background_rate=background,method=method,risk=result.risk,makespan=result.makespan,**f))
            # Finite executable candidate selection with shared normalizers.
            ref=reference_metrics(model); w=(.5,.2,.1); gateweight=.2
            options=[]
            for k,candidate in enumerate(circuits[1:]):
                localmodel=synthetic_model(candidate,coupling)
                _,_,qs,_=extract(candidate,coupling)
                gatecost=len(qs)+9*sum(len(q)==2 for q in qs)
                fixed=evaluate_timing(localmodel,localmodel.base,w,ref)
                gp=solve_joint(localmodel,w,references=ref)
                direct=optimize_timing(localmodel,w,references=ref,method='convex')
                if abs(gp.objective-direct.objective)>2e-5: raise RuntimeError('Outer GP/direct gap')
                options.append(dict(candidate=k,gatecost=gatecost,
                    fixed_score=fixed.objective+gateweight*gatecost/info['gate_cost'],
                    joint_gp_score=gp.objective+gateweight*gatecost/info['gate_cost'],
                    joint_convex_score=direct.objective+gateweight*gatecost/info['gate_cost'],
                    gp_durations=gp.durations.tolist(),convex_durations=direct.durations.tolist()))
            tolerance=2e-5
            def choice(key):
                best=min(x[key] for x in options)
                return min(x['candidate'] for x in options if x[key]<=best+tolerance)
            outer.append(dict(case=case,weights_risk_time_effort_gates=[*w,gateweight],
                options=options,selected_fixed=choice('fixed_score'),selected_gp=choice('joint_gp_score'),
                selected_convex=choice('joint_convex_score'),shared_references=ref.tolist()))
        print(f'Validated Pareto/sensitivity/outer search {case}',flush=True)
    save(out/'pareto.json',pareto);csv_save(out/'pareto.csv',pareto)
    save(out/'outer_search.json',outer);save(out/'sensitivity.json',sensitivity)
    if not args.quick:
        for family in ['qaoa','vqe_ansatz']:
            for n in [8,12,20]:
                logical=benchmark(family,n,101)
                circuit,compile_seconds=compile_candidate(logical,topology(n,'line'),7)
                with (out/f'scale_{family}_{n}.qpy').open('wb') as f:qpy.dump([logical,circuit],f)
                model=synthetic_model(circuit,topology(n,'line'))
                gp=solve_joint(model);direct=optimize_timing(model,method='convex')
                gap=abs(gp.objective-direct.objective)
                if gap>2e-5:raise RuntimeError('Scaling solver disagreement')
                scale.append(dict(family=family,qubits=n,gates=len(circuit.data),compile_seconds=compile_seconds,
                    gp=asdict(gp),convex=asdict(direct),gap=gap,
                    verification='Native/connectivity and timing checks only; full unitary and noise simulation not run at this size'))
                print(f'Scaling-only validation {family} {n} qubits',flush=True)
    save(out/'scaling.json',scale)
    environment=dict(python=platform.python_version(),platform=platform.platform(),
        packages={name:importlib.metadata.version(name) for name in ['qiskit','cvxpy','numpy','scipy','clarabel','ecos','scs']},
        invocation_seconds=time.perf_counter()-start,free_local_only=True,hardware_jobs=0,paid_services=False,
        gp_interface='Exact vectorized logarithmic transformation; automatic CVXPY DGP cross-checked in unit tests',
        large_problem_policy='Risk-only at most 300 gates: logarithmic cones. Larger or multiobjective: certified GP-equivalent path-flow dual. Deadline comparator: ordinary convex.',
        uncertainty_radius=.3,synthetic_not_calibrated=True,
        noise_model='local isotropic strength-dependent depolarization plus globally covariant background depolarization')
    save(out/'environment.json',environment)
    print(json.dumps(dict(configurations=len(cases),timing_rows=len(rows),state_channel_checks=len(noise),
        entangled_process_checks=len(process),pareto_points=len(pareto),outer_cases=len(outer),scaling_cases=len(scale)),indent=2),flush=True)


if __name__=='__main__':main()
