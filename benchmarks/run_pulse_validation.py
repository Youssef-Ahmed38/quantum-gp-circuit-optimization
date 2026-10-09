"""Independent free Lindblad validation using saved joint-timing decisions."""
from pathlib import Path
import json,time
from types import SimpleNamespace
import numpy as np
from qiskit import qpy
from qiskit.quantum_info import random_statevector
from qcgp.core import topology
from qcgp.timing import synthetic_model
from qcgp.pulse import simulate_pulses


def main():
    evidence=Path(__file__).resolve().parents[1]/'evidence'
    folder=evidence/'joint_timing'
    rows=json.loads((folder/'results.json').read_text())
    checks=[]
    for row in rows:
        if row['qubits']!=3 or row['instance']!=0 or row['method'] not in ['fixed','joint_gp','robust_gp']:continue
        case=row['case']
        with (evidence/'circuits'/f'{case}.qpy').open('rb') as f:circuit=qpy.load(f)[1]
        model=synthetic_model(circuit,topology(3,row['topology']))
        result=SimpleNamespace(durations=np.asarray(row['durations']),starts=np.asarray(row['starts']),makespan=row['makespan'])
        initial=random_statevector(8,seed=2119)
        start=time.perf_counter()
        actual,ideal,metadata=simulate_pulses(circuit,model,result,initial,background='local_dephasing')
        fidelity=float(np.real(np.trace(actual.data@ideal.data)))
        distance=float(.5*np.linalg.norm(actual.data-ideal.data,ord='nuc'))
        bound=row['nominal_fidelity_bound']
        if fidelity<bound-1e-8 or distance>1-bound+1e-8:raise RuntimeError('General random-unitary Lindblad certificate failed')
        checks.append(dict(case=case,family=row['family'],topology=row['topology'],method=row['method'],
            fidelity=fidelity,trace_distance=distance,fidelity_bound=bound,seconds=time.perf_counter()-start,**metadata))
        (folder/'pulse_validation.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf8')
        print(f'Exact local-memory pulse validation {case}/{row["method"]}: F={fidelity:.6f}, bound={bound:.6f}',flush=True)
    print(f'{len(checks)} dense Lindblad pulse simulations passed',flush=True)


if __name__=='__main__':main()
