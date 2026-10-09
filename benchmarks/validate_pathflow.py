"""Independent optimum brackets and path-flow feasibility checks on saved circuits."""
from pathlib import Path
from dataclasses import asdict
import json
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from qiskit import qpy
from qcgp.core import topology
from qcgp.timing import synthetic_model,robust_upper_model,reference_metrics
from qcgp.pathflow import optimize_pathflow


def certify_flow_graph(model,pressure,total):
    n=len(model.base);r=len(model.edges)
    parents=[[] for _ in range(n)];children=[[] for _ in range(n)]
    for k,(i,j) in enumerate(model.edges):children[i].append(k);parents[j].append(k)
    roots=[j for j in range(n) if not parents[j]];sinks=[i for i in range(n) if not children[i]]
    if abs(np.sum(pressure[roots])-total)>1e-8 or abs(np.sum(pressure[sinks])-total)>1e-8:
        raise RuntimeError('Source/sink pressure does not equal total flow')
    row_indices=[];column_indices=[];rhs=[]
    for j in range(n):
        for adjacency in (parents[j],children[j]):
            if adjacency:
                row_indices.extend([len(rhs)]*len(adjacency));column_indices.extend(adjacency)
                rhs.append(pressure[j]/total)
    if r:
        matrix=coo_matrix((np.ones(len(row_indices)),(row_indices,column_indices)),shape=(len(rhs),r)).tocsr()
        feasible=linprog(np.zeros(r),A_eq=matrix,b_eq=np.asarray(rhs),bounds=(0,None),method='highs')
        if not feasible.success:raise RuntimeError('Independent network-flow feasibility LP failed')
        residual=float(np.max(np.abs(matrix@feasible.x-np.asarray(rhs)))*total)
    else:residual=0.0
    return residual


def main():
    evidence=Path(__file__).resolve().parents[1]/'evidence'
    folder=evidence/'joint_timing'
    saved=json.loads((folder/'results.json').read_text())
    cases={r['case']:r for r in saved if r['method']=='joint_gp'}
    direct={r['case']:r for r in saved if r['method']=='joint_convex'}
    rows=[]
    for name,info in cases.items():
        with (evidence/'circuits'/f'{name}.qpy').open('rb') as f:circuit=qpy.load(f)[1]
        nominal=synthetic_model(circuit,topology(info['qubits'],info['topology']))
        for label,model in [('nominal',nominal),('upper_box',robust_upper_model(nominal,.3))]:
            flow=optimize_pathflow(model)
            if not flow.certified:raise RuntimeError('Path-flow algorithm reached limit without certificate')
            row=asdict(flow)
            row.update(case=name,family=info['family'],qubits=info['qubits'],topology=info['topology'],instance=info['instance'],coefficient_model=label)
            total=model.background_rate/reference_metrics(model)[0]
            row['flow_lp_residual']=certify_flow_graph(model,flow.pressure,total)
            if label=='nominal':
                numerical=direct[name]['objective']
                if numerical<flow.dual_bound-2e-6 or numerical>flow.timing.objective+2e-6:
                    raise RuntimeError('Independent convex solution outside certified optimum bracket')
                row['independent_convex_objective']=numerical
            rows.append(row)
        (folder/'pathflow_validation.json').write_text(json.dumps(rows,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x)+'\n',encoding='utf8')
        print(f'Primal-dual bracket and independent flow LP passed {name}',flush=True)
    print(f'{len(rows)} path-flow solutions certified and independently checked',flush=True)


if __name__=='__main__':main()
