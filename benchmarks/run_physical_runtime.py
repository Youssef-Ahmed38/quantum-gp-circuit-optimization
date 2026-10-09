"""Repeated solver timings on cached, matched models; no channel simulation."""
from pathlib import Path
import json,time,platform
import numpy as np
from qiskit import qpy
from qcgp.physical import PhysicalParameters,physical_model
from qcgp.pathflow import optimize_pathflow
from qcgp.timing import optimize_timing
from qcgp.library import optimize_library
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'evidence/physical_study'

def main():
    cases=json.loads((DATA/'cases.json').read_text());records=[]
    keys=[k for k in cases if k.endswith('_2_101_balanced')]
    for key in keys:
        item=cases[key]
        with (DATA/(item['rows'][0]['case']+'.qpy')).open('rb') as f:c=qpy.load(f)[1]
        model=physical_model(c,PhysicalParameters(**item['parameters']));lib=np.asarray(item['pulse_library'])
        bounds=np.asarray(item['pulse_bounds']);rates=np.asarray(item['support_rates'])
        methods={'pathflow':lambda:optimize_pathflow(model),'direct_convex':lambda:optimize_timing(model,method='convex'),
            'finite_milp':lambda:optimize_library(model,lib,time_limit=10),
            'characterized_milp':lambda:optimize_library(model,lib,pulse_bounds=bounds,support_rates=rates,time_limit=10)}
        for repeat in range(4):
            for method,call in methods.items():
                begin=time.perf_counter();result=call();elapsed=time.perf_counter()-begin
                if repeat==0:continue
                records.append(dict(key=key,repeat=repeat,method=method,seconds=elapsed,status=getattr(result,'status',getattr(getattr(result,'timing',None),'status',None)),gap=getattr(result,'relative_gap',None)))
        print('Repeated matched runtimes: '+key,flush=True)
    summary={}
    for method in methods:
        x=np.array([r['seconds'] for r in records if r['method']==method])
        summary[method]=dict(n=len(x),median=float(np.median(x)),q25=float(np.quantile(x,.25)),q75=float(np.quantile(x,.75)),maximum=float(np.max(x)))
    (DATA/'runtime.json').write_text(json.dumps(dict(records=records,summary=summary,platform=platform.platform(),warmup_per_method_per_case=1,measured_repetitions=3,scope='Six two-qubit balanced cases; excludes characterization; single process, one BLAS/OpenMP thread; no asymptotic claim'),indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
