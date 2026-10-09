"""Matched fastest-latency check; objective and deadline selected after first study."""
from pathlib import Path
import json,time
import numpy as np
from qiskit import qpy
from qiskit.quantum_info import Operator,SuperOp,Statevector,DensityMatrix,process_fidelity,random_statevector
from qcgp.physical import PhysicalParameters,physical_model,physical_channel
from qcgp.library import optimize_library
from qcgp.characterization import library_certificate
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'evidence/physical_study'

def main():
    cases=json.loads((DATA/'cases.json').read_text());path=DATA/'deadline_results.json'
    done=json.loads(path.read_text()) if path.exists() else {}
    for key,item in cases.items():
        if key in done:continue
        slack=next(r for r in item['rows'] if r['method']=='published_slack')
        with (DATA/(slack['case']+'.qpy')).open('rb') as f:c=qpy.load(f)[1]
        params=PhysicalParameters(**item['parameters']);model=physical_model(c,params)
        lib=np.asarray(item['pulse_library']);bounds=np.asarray(item['pulse_bounds']);rates=np.asarray(item['support_rates'])
        deadline=slack['makespan'];policy=optimize_library(model,lib,pulse_bounds=bounds,support_rates=rates,deadline=deadline+1e-9,time_limit=10)
        channel,metadata=physical_channel(c,params,policy.durations)
        if metadata['makespan']>deadline+1e-7:raise RuntimeError('Matched deadline violated')
        cert=library_certificate(model,policy.durations,lib,bounds,rates)
        pf=float(process_fidelity(channel,SuperOp(Operator(c))))
        if pf<cert['fidelity_floor']-1e-7:raise RuntimeError('Deadline policy certificate failed')
        states={}
        for label,initial in [('zero',Statevector.from_label('0'*c.num_qubits)),('haar',random_statevector(2**c.num_qubits,seed=slack['seed']+2001))]:
            actual=DensityMatrix(initial).evolve(channel).data;ideal=initial.evolve(c).data
            f=float(np.real(ideal.conj()@actual@ideal))
            if f<cert['fidelity_floor']-1e-7:raise RuntimeError('Deadline state certificate failed')
            states[label]=f
        done[key]=dict(key=key,regime=slack['regime'],family=slack['family'],qubits=slack['qubits'],seed=slack['seed'],deadline=deadline,makespan=metadata['makespan'],
            durations=policy.durations.tolist(),process_fidelity=pf,state_fidelities=states,process_gain_pp=100*(pf-slack['process_fidelity']),
            relative_gap=policy.relative_gap,optimal=policy.optimal,status=policy.status,certificate=cert,tp_error=metadata['tp_error'],minimum_choi_eigenvalue=metadata['minimum_choi_eigenvalue'])
        path.write_text(json.dumps(done,indent=2)+'\n')
        print('Matched-deadline validated: '+key,flush=True)
    summary={}
    for regime in ['low_drive','balanced','high_drive']:
        rows=[r for r in done.values() if r['regime']==regime];values=np.array([r['process_gain_pp'] for r in rows])
        summary[regime]=dict(n=len(rows),mean_gain_pp=float(np.mean(values)),median_gain_pp=float(np.median(values)),minimum_gain_pp=float(np.min(values)),maximum_gain_pp=float(np.max(values)),regressions=sum(values<-1e-7),max_relative_gap=max(r['relative_gap'] for r in rows),uncertified_optimal=sum(not r['optimal'] for r in rows),max_deadline_excess=max(r['makespan']-r['deadline'] for r in rows))
    def default(value):
        if isinstance(value,np.generic):return value.item()
        raise TypeError(type(value).__name__)
    (DATA/'deadline_summary.json').write_text(json.dumps(dict(scope='Exploratory follow-up after unconstrained-duration results; same pulse libraries, fastest latency matched to published slack; no coefficient tuning',channels=len(done),state_checks=2*len(done),regimes=summary),indent=2,default=default)+'\n')
    print(json.dumps(summary,indent=2,default=default))
if __name__=='__main__':main()
