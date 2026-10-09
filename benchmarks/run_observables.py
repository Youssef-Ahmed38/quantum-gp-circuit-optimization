"""Free calibration-anchored observable-constrained scheduling study."""
from pathlib import Path
import json,argparse,hashlib,time,platform
from collections import deque
import numpy as np
from qiskit import QuantumCircuit,qpy
from qcgp.observables import snapshot_model,expectations,optimal_uniform,optimize_observables,geometric_library,upward_round,dense_observables,sample_observables
from qcgp.core import critical_path
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'evidence/observable_study/props_torino.json'

def save(path,data):
    path.write_text(json.dumps(data,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())+'\n')


def mapped_region(snapshot,n,seed):
    neighbors={q:set() for q in range(len(snapshot['qubits']))}
    for g in snapshot['gates']:
        if g['gate']=='cz':
            i,j=g['qubits']
            err=float(next(param['value'] for param in g['parameters'] if param['name']=='gate_error'))
            if 0<err<0.75:
                neighbors[i].add(j);neighbors[j].add(i)
    active_qubits=[q for q,nbrs in neighbors.items() if len(nbrs)>0]
    start_q=active_qubits[seed%len(active_qubits)]
    region=[];queue=deque([start_q]);seen=set(queue)
    while queue and len(region)<n:
        i=queue.popleft();region.append(i)
        for j in sorted(neighbors[i]):
            if j not in seen:seen.add(j);queue.append(j)
    if len(region)!=n:raise ValueError('Requested connected region is unavailable')
    indices={q:i for i,q in enumerate(region)}
    edges=sorted({tuple(sorted((indices[i],indices[j]))) for i in region for j in neighbors[i] if j in indices})
    return region,edges


def workload(snapshot,n,seed,family):
    mapping,edges=mapped_region(snapshot,n,seed);c=QuantumCircuit(n)
    for q in range(n):c.rz(np.pi/2,q);c.sx(q);c.rz(np.pi/2,q)
    rng=np.random.default_rng(seed)
    for layer in range(1 if family=='graph' else 3):
        order=list(edges);rng.shuffle(order)
        for i,j in order:c.cz(i,j)
        if family=='scrambled':
            for q in range(n):
                c.rz(int(rng.integers(0,4))*np.pi/2,q)
                if rng.random()<.7:c.sx(q)
    return c,mapping


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--quick',action='store_true');args=parser.parse_args()
    folder=ROOT/'evidence'/('observable_quick' if args.quick else 'observable_study');folder.mkdir(exist_ok=True)
    snapshot=json.loads(SOURCE.read_text());sizes=[4] if args.quick else [4,8,16,32,64,128]
    seeds=[11] if args.quick else [11,29];alphas=[.8] if args.quick else [.2,.5,.8];targets=[.95,.99];families=['graph','scrambled'];q=1.02
    protocol=dict(sizes=sizes,seeds=seeds,alphas=alphas,targets=targets,families=families,grid_ratio=q,
        snapshot_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),snapshot_date=snapshot['last_update_date'],
        source='https://raw.githubusercontent.com/Qiskit/qiskit-ibm-runtime/refs/heads/main/qiskit_ibm_runtime/fake_provider/backends/torino/props_torino.json',
        native_graph=True,base_average_gate_errors_and_lengths_anchored=True,duration_dependence_calibrated=False,
        noise='Independent isotropic Pauli Markov channels after ideal Clifford gates; virtual RZ noiseless; no idle relaxation or crosstalk',
        duration_box=[.5,2.],all_free_local=True,hardware_jobs=0,comparison='Minimum latency with identical stabilizer-generator expectation targets; global envelope is a conservative comparator',
        predeclared_before_full_measurement=True)
    pp=folder/'protocol.json'
    if pp.exists() and json.loads(pp.read_text())!=protocol:raise RuntimeError('Different protocol in destination')
    save(pp,protocol);checkpoint=folder/'checkpoint.json';done=json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    for family in families:
        for n in sizes:
            for seed in seeds:
                c,mapping=workload(snapshot,n,seed,family);name=f'{family}_{n}_{seed}'
                with (folder/(name+'.qpy')).open('wb') as f:qpy.dump(c,f)
                for alpha in alphas:
                    model=snapshot_model(c,mapping,snapshot,alpha)
                    for target in targets:
                        key=f'{name}_{alpha}_{target}'
                        if key in done:continue
                        gp=optimize_observables(model,target);direct=optimize_observables(model,target,method='convex',compute_dual=False)
                        safe=optimize_observables(model,target,safety=q);global_result=optimize_observables(model,target,global_envelope=True,compute_dual=False)
                        if gp['feasible']!=direct['feasible']:raise RuntimeError('Matched formulations disagree on feasibility')
                        uniform=optimal_uniform(model,target);record=dict(key=key,family=family,qubits=n,seed=seed,alpha=alpha,target=target,
                            gates=len(c.data),timed_gates=len(model.base),physical_qubits=mapping,gp=gp,direct=direct,safe_gp=safe,global_gp=global_result,
                            fixed_minimum_expectation=float(np.min(expectations(model,model.base))),base_makespan=critical_path(model.base,model.edges).makespan,
                            uniform_makespan=None if uniform is None else critical_path(uniform,model.edges).makespan)
                        if gp['feasible']:
                            gap=abs(gp['makespan']-direct['makespan'])/max(gp['makespan'],1e-12)
                            if gap>2e-5:raise RuntimeError('Independent convex optimum disagrees')
                            record['formulation_relative_difference']=gap
                            if uniform is not None:record['latency_reduction_vs_uniform_percent']=100*(1-gp['makespan']/record['uniform_makespan'])
                            naive=upward_round(model,gp['durations'],q);record['unprotected_rounding_minimum_expectation']=float(np.min(expectations(model,naive)))
                        if safe['feasible']:
                            rounded=upward_round(model,safe['durations'],q);actual=expectations(model,rounded)
                            if np.min(actual)<target-1e-6:raise RuntimeError('Protected geometric rounding violates exact observable target')
                            span=critical_path(rounded,model.edges).makespan
                            if span>q*safe['makespan']+1e-8:raise RuntimeError('Protected rounding exceeds latency factor')
                            lib=geometric_library(model,q);uniform_grid=[lib[:,j] for j in range(lib.shape[1]) if np.min(expectations(model,lib[:,j]))>=target-1e-12]
                            ug=min((critical_path(t,model.edges).makespan for t in uniform_grid),default=None)
                            record['rounded']=dict(durations=rounded.tolist(),makespan=span,minimum_expectation=float(np.min(actual)),
                                uniform_grid_makespan=ug,latency_reduction_vs_grid_uniform_percent=None if ug is None else 100*(1-span/ug))
                            if n==4 and seed==11 and alpha==.8:
                                dense=dense_observables(c,model,rounded);difference=float(np.max(abs(dense-actual)))
                                if difference>1e-9:raise RuntimeError('Density-matrix expectation check failed')
                                record['dense_validation']=dict(maximum_difference=difference,minimum_dense_expectation=float(np.min(dense)))
                            if seed==11 and alpha==.8 and target==.95:
                                sample=sample_observables(c,model,rounded,shots=16384,seed=1911+n)
                                difference=float(np.max(abs(np.asarray(sample['expectations'])-actual)))
                                epsilon=float(np.sqrt(2*np.log(2*n/.001)/sample['shots']))
                                if difference>epsilon:raise RuntimeError('Forward-frame sampling exceeds declared simultaneous Hoeffding check')
                                record['sampling_validation']=dict(**sample,maximum_difference=difference,simultaneous_tolerance=epsilon,
                                    generator_fidelity_lower_bound=max(0.,1-.5*np.sum(1-actual)))
                        done[key]=record;save(checkpoint,done)
                        print(f'Validated {key}: feasible={gp["feasible"]}, protected={safe["feasible"]}',flush=True)
    save(folder/'results.json',list(done.values()));save(folder/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),hardware_jobs=0,paid_services=False))
    print('Configurations: '+str(len(done)),flush=True)
if __name__=='__main__':main()
