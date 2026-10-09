"""Exact Pauli-observable constraints for a declared Clifford/Pauli channel model.

The analytical expectation formula and Clifford tracking are established tools.
This module couples them to GP/convex duration allocation and safe log-grid rounding.
It does not apply to general non-Clifford or non-Pauli noise.
"""
from dataclasses import dataclass
import math,time
import numpy as np
import cvxpy as cp
from scipy.sparse import csr_matrix
from qiskit.quantum_info import Clifford,Pauli,DensityMatrix,Statevector,Operator
from .core import critical_path
from .pathflow import longest_path_oracle


@dataclass(frozen=True)
class ObservableModel:
    base: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    a: np.ndarray
    gamma: np.ndarray
    sensitivity: np.ndarray
    edges: tuple
    timed_indices: tuple
    labels: tuple

    def __post_init__(self):
        for name in ['base','lower','upper','a','gamma','sensitivity']:
            values=np.array(getattr(self,name),dtype=float,copy=True)
            values.setflags(write=False);object.__setattr__(self,name,values)

    def validate(self):
        n=len(self.base)
        if n==0:raise ValueError('A nonempty timed circuit is required')
        for name in ['base','lower','upper','a','gamma']:
            x=np.asarray(getattr(self,name))
            if x.shape!=(n,) or not np.isfinite(x).all() or np.any(x<=0):raise ValueError('Positive finite gate parameters required')
        h=np.asarray(self.sensitivity)
        if h.ndim!=2 or h.shape[1]!=n or not len(h) or not np.isfinite(h).all() or np.any(h<0):raise ValueError('Nonnegative observable sensitivities required')
        if np.any(self.lower>self.base) or np.any(self.base>self.upper):raise ValueError('Invalid duration bounds')
        critical_path(self.base,self.edges)


def _binary_gate(x,z,name,qubits,params=()):
    """Clifford symplectic update; phases are irrelevant to commutation tests."""
    if name=='sx':x[:,qubits[0]]^=z[:,qubits[0]]
    elif name=='rz':
        angle=float(params[0])
        if not math.isfinite(angle):raise ValueError('Only finite Clifford RZ angles are supported')
        steps=round(angle/(np.pi/2))
        if abs(angle-steps*np.pi/2)>1e-9:raise ValueError('Only Clifford RZ angles are supported')
        if steps%2:z[:,qubits[0]]^=x[:,qubits[0]]
    elif name=='cz':
        i,j=qubits;z[:,i]^=x[:,j];z[:,j]^=x[:,i]
    elif name in ['x','id']:pass
    else:raise ValueError('Only native sx/x/cz and virtual Clifford rz are supported')


def local_paulis(qubits):
    """All nonidentity local Pauli bit vectors, in little-endian qubit order."""
    k=len(qubits);result=[]
    for code in range(1,4**k):
        digits=[(code//(4**j))%4 for j in range(k)]
        result.append((np.array([d in (1,2) for d in digits],dtype=bool),np.array([d in (2,3) for d in digits],dtype=bool)))
    return result


def snapshot_model(circuit,physical_qubits,snapshot,alpha=.5):
    """Anchor base error/length to a public snapshot; duration curve is assumed.

    Gate noise is an isotropic Pauli Markov channel AFTER each ideal Clifford.
    Virtual RZ has zero modeled duration/noise. No idle T1/crosstalk is added.
    alpha controls an uncalibrated inverse-duration/exposure decomposition.
    """
    if not 0<alpha<1:raise ValueError('Require 0 < alpha < 1')
    if len(physical_qubits)!=circuit.num_qubits or len(set(physical_qubits))!=len(physical_qubits):raise ValueError('Injective physical qubit mapping required')
    table={(g['gate'],tuple(g['qubits'])):{p['name']:p for p in g['parameters']} for g in snapshot['gates']}
    base=[];hazard=[];indices=[];last={};edges=set();gate_index={}
    for index,ins in enumerate(circuit.data):
        qs=tuple(circuit.find_bit(q).index for q in ins.qubits);name=ins.operation.name
        # Validate virtual angles too, before treating them as noiseless.
        _binary_gate(np.zeros((1,circuit.num_qubits),dtype=bool),np.zeros((1,circuit.num_qubits),dtype=bool),name,qs,ins.operation.params)
        if name=='rz':continue
        physical=tuple(physical_qubits[q] for q in qs);entry=table.get((name,physical))
        if entry is None and name=='cz':entry=table.get((name,physical[::-1]))
        if entry is None:raise ValueError(f'No snapshot instruction {name} on {physical}')
        error=float(entry['gate_error']['value']);length=entry['gate_length']
        units={'ns':1e-3,'us':1.,'s':1e6}
        if length['unit'] not in units:raise ValueError('Unknown gate-length unit')
        duration=float(length['value'])*units[length['unit']];d=2**len(qs)
        if duration<=0 or not 0<error<(d-1)/d:raise ValueError('Snapshot gate has invalid error/duration')
        exposure=-(d*d-1)/(d*d)*np.log1p(-error*d/(d-1))
        t=len(base);indices.append(index);gate_index[index]=t;base.append(duration);hazard.append(exposure)
        for q in qs:
            if q in last:edges.add((last[q],t))
            last[q]=t
    if not base:raise ValueError('Need a timed noisy gate')
    clifford=Clifford(circuit);labels=tuple(clifford.to_labels(mode='S'))
    observables=[Pauli(label) for label in labels]
    x=np.stack([p.x for p in observables]);z=np.stack([p.z for p in observables])
    h=np.zeros((circuit.num_qubits,len(base)))
    for index in reversed(range(len(circuit.data))):
        ins=circuit.data[index];qs=tuple(circuit.find_bit(q).index for q in ins.qubits)
        if index in gate_index:
            anticommuting=np.zeros(circuit.num_qubits)
            for px,pz in local_paulis(qs):
                # Use XOR: Boolean addition is OR, rather than GF(2) addition.
                anti=np.logical_xor.reduce((x[:,qs]&pz)^(z[:,qs]&px),axis=1)
                anticommuting+=anti
            h[:,gate_index[index]]=anticommuting/(4**len(qs)-1)
        _binary_gate(x,z,ins.operation.name,qs,ins.operation.params)
    base=np.asarray(base);hazard=np.asarray(hazard)
    model=ObservableModel(base,.5*base,2*base,alpha*hazard*base,(1-alpha)*hazard/base,h,tuple(sorted(edges)),tuple(indices),labels)
    model.validate();return model


def expectations(model,tau):
    tau=np.asarray(tau,dtype=float);model.validate()
    if tau.shape!=model.base.shape or not np.isfinite(tau).all() or np.any(tau<=0):raise ValueError('Positive matching durations required')
    return np.exp(-2*(model.sensitivity@(model.a/tau+model.gamma*tau)))


def minimum_exposures(model,safety=1.):
    tau=np.clip(np.sqrt(model.a/(safety*model.gamma)),model.lower,model.upper)
    return model.sensitivity@(model.a/tau+safety*model.gamma*tau)


def optimal_uniform(model,target):
    """Exact earliest common-scale policy satisfying all observable targets."""
    if not 0<target<1:raise ValueError('Target must be strictly between zero and one')
    budget=-.5*np.log(target);a=model.sensitivity@(model.a/model.base);c=model.sensitivity@(model.gamma*model.base)
    lower=float(np.max(model.lower/model.base));upper=float(np.min(model.upper/model.base))
    for ai,ci in zip(a,c):
        if ai==ci==0:continue
        discriminant=budget*budget-4*ai*ci
        if discriminant<-1e-14:return None
        root=np.sqrt(max(0.,discriminant))
        if ci>0:
            lower=max(lower,2*ai/(budget+root));upper=min(upper,(budget+root)/(2*ci))
        else:lower=max(lower,ai/budget)
    if lower>upper+1e-10:return None
    return model.base*lower


def _dual_lower(model,weights,safety,budget,iterations=1500):
    """Feasible nonnegative constraint multipliers plus feasible path pressure."""
    multipliers=np.maximum(np.asarray(weights),0.)/budget
    gate_weight=model.sensitivity.T@multipliers
    a=model.a*gate_weight;c=safety*model.gamma*gate_weight
    span=critical_path(model.base,model.edges).makespan;b=1/span
    previous=[[] for _ in model.base]
    for i,j in model.edges:previous[j].append(i)
    _,path=longest_path_oracle(model.lower,previous);rho=b*path
    def durations(pressure):
        denom=c+pressure;values=model.lower.copy();mask=(a>0)&(denom>0)
        values[mask]=np.sqrt(a[mask]/denom[mask]);values[(a>0)&(denom==0)]=model.upper[(a>0)&(denom==0)]
        return np.clip(values,model.lower,model.upper)
    for iteration in range(iterations):
        tau=durations(rho);makespan,path=longest_path_oracle(tau,previous)
        lagrangian=float(np.sum(a/tau+c*tau)+b*makespan)
        dual=float(np.sum(a/tau+(c+rho)*tau))
        if lagrangian-dual<=1e-7*max(1,abs(lagrangian)):break
        direction=b*path-rho
        if np.dot(durations(rho+direction),direction)>=0:step=1.
        else:
            left,right=0.,1.
            for _ in range(35):
                middle=(left+right)/2
                if np.dot(durations(rho+middle*direction),direction)>0:left=middle
                else:right=middle
            step=(left+right)/2
        rho=(1-step)*rho+step*b*path
    # Every iterate is a scaled convex combination of true DAG paths.
    tau=durations(rho)
    return max(0.,float(np.sum(a/tau+(c+rho)*tau)-np.sum(np.maximum(weights,0.))))


def optimize_observables(model,target,method='gp',*,safety=1.,global_envelope=False,compute_dual=True):
    """Minimize latency under exact positive-observable constraints.

    safety=q strengthens only exposure terms for upward geometric-grid rounding.
    global_envelope replaces observable sensitivity by ones: sufficient, not exact.
    Infeasibility is certified analytically using gatewise common noise minima.
    """
    begin=time.perf_counter();model.validate()
    if not 0<target<1 or not np.isfinite(safety) or safety<1 or method not in ['gp','convex']:raise ValueError('Invalid target, safety or formulation')
    h=np.ones((1,len(model.base))) if global_envelope else model.sensitivity
    budget=-.5*np.log(target)
    tau_min=np.clip(np.sqrt(model.a/(safety*model.gamma)),model.lower,model.upper)
    minima=h@(model.a/tau_min+safety*model.gamma*tau_min)
    if np.max(minima)>budget+1e-12:return dict(feasible=False,status='analytic_infeasibility',minimum_constraint_exposure=float(np.max(minima)),budget=budget,elapsed=time.perf_counter()-begin)
    m=len(model.base);scale=critical_path(model.base,model.edges).makespan;relative=model.base/scale
    finish=cp.Variable(m);end=cp.Variable()
    if method=='gp':
        y=cp.Variable(m);duration_exp=cp.exp(y);inverse_exp=cp.exp(-y)
        constraints=[y>=np.log(model.lower/model.base),y<=np.log(model.upper/model.base),np.log(relative)+y<=finish,finish<=end]
        for i,j in model.edges:constraints.append(cp.log_sum_exp(cp.hstack([finish[i],np.log(relative[j])+y[j]]))<=finish[j])
        exposure=cp.multiply(model.a/model.base,inverse_exp)+cp.multiply(safety*model.gamma*model.base,duration_exp)
        objective=cp.exp(end)
    else:
        y=cp.Variable(m,pos=True);duration_exp=y
        constraints=[y>=model.lower/model.base,y<=model.upper/model.base,cp.multiply(relative,y)<=finish,finish<=end]
        for i,j in model.edges:constraints.append(finish[i]+relative[j]*y[j]<=finish[j])
        exposure=cp.multiply(model.a/model.base,cp.inv_pos(y))+cp.multiply(safety*model.gamma*model.base,y)
        objective=end
    noise=csr_matrix(h/budget)@exposure<=1
    constraints.append(noise);problem=cp.Problem(cp.Minimize(objective),constraints)
    if not problem.is_dcp():raise RuntimeError('Expected a convex transformed problem')
    problem.solve(solver='CLARABEL',tol_gap_abs=1e-8,tol_gap_rel=1e-8,tol_feas=1e-9,max_iter=300)
    if y.value is None:raise RuntimeError('Solver failed on analytically feasible model: '+str(problem.status))
    tau=model.base*(np.exp(y.value) if method=='gp' else y.value)
    # Snap numerical boundary residue before constructing a finite grid policy.
    tau=np.clip(tau,model.lower,model.upper)
    tau=np.where(abs(tau-model.lower)<=1e-8*model.base,model.lower,tau)
    tau=np.where(abs(tau-model.upper)<=1e-8*model.base,model.upper,tau)
    actual=critical_path(tau,model.edges);viol=float(np.max(h@(model.a/tau+safety*model.gamma*tau)-budget))
    if viol>1e-7 or np.max(model.lower-tau)>1e-7 or np.max(tau-model.upper)>1e-7:raise RuntimeError('Observable optimization violates constraints')
    lower=None
    if compute_dual and not global_envelope:lower=_dual_lower(model,noise.dual_value,safety,budget)
    if lower is not None and lower>actual.makespan/scale+1e-6:raise RuntimeError('Invalid observable optimum bracket')
    return dict(feasible=True,status=problem.status,durations=tau.tolist(),makespan=actual.makespan,normalized_objective=actual.makespan/scale,
        normalized_dual_lower=lower,relative_gap=None if lower is None else max(0,actual.makespan/scale-lower)/max(actual.makespan/scale,1e-12),
        minimum_expectation=float(np.min(expectations(model,tau))),constraint_excess=viol,method=method,safety=safety,elapsed=time.perf_counter()-begin)


def geometric_library(model,q=1.02):
    if not np.isfinite(q) or q<=1:raise ValueError('Geometric spacing must exceed one')
    model.validate()
    count=int(np.ceil(np.max(np.log(model.upper/model.lower))/np.log(q)))
    return np.minimum(model.lower[:,None]*q**np.arange(count+1)[None,:],model.upper[:,None])


def upward_round(model,tau,q=1.02):
    lib=geometric_library(model,q);tau=np.asarray(tau)
    if np.any(tau<model.lower-1e-8) or np.any(tau>model.upper+1e-8):raise ValueError('Cannot round outside duration box')
    tau=np.clip(tau,model.lower,model.upper);choice=np.sum(lib<tau[:,None]-1e-12,axis=1)
    choice=np.minimum(choice,lib.shape[1]-1);rounded=lib[np.arange(len(tau)),choice]
    if np.any(rounded<tau-1e-10) or np.any(rounded>q*tau+1e-10):raise RuntimeError('Geometric rounding ratio failed')
    return rounded


def dense_observables(circuit,model,tau):
    """Independent Qiskit density-matrix channel application, limited to <=5 qubits."""
    if circuit.num_qubits>5:raise ValueError('Dense validation is restricted to five qubits')
    state=DensityMatrix(Statevector.from_label('0'*circuit.num_qubits));positions={index:j for j,index in enumerate(model.timed_indices)}
    for index,ins in enumerate(circuit.data):
        qs=[circuit.find_bit(q).index for q in ins.qubits];state=state.evolve(ins.operation,qargs=qs)
        if index not in positions:continue
        j=positions[index];hazard=(model.a[j]/tau[j]+model.gamma[j]*tau[j])/(4**len(qs)-1)
        p=-.5*np.expm1(-2*hazard)
        for px,pz in local_paulis(qs):
            pauli=Pauli((pz,px));changed=state.evolve(Operator(pauli),qargs=qs)
            state=DensityMatrix((1-p)*state.data+p*changed.data)
    return np.array([float(np.real(state.expectation_value(Pauli(label)))) for label in model.labels])


def sample_observables(circuit,model,tau,shots=16384,seed=7):
    """Forward Pauli-frame Monte Carlo, independent of backward sensitivity code."""
    if not isinstance(shots,int) or shots<1:raise ValueError('Positive shot count required')
    rng=np.random.default_rng(seed);x=np.zeros((shots,circuit.num_qubits),dtype=bool);z=x.copy()
    positions={index:j for j,index in enumerate(model.timed_indices)}
    for index,ins in enumerate(circuit.data):
        qs=tuple(circuit.find_bit(q).index for q in ins.qubits)
        _binary_gate(x,z,ins.operation.name,qs,ins.operation.params)
        if index not in positions:continue
        j=positions[index];rate=(model.a[j]/tau[j]+model.gamma[j]*tau[j])/(4**len(qs)-1)
        p=-.5*np.expm1(-2*rate)
        for px,pz in local_paulis(qs):
            flip=rng.random(shots)<p
            for q,xx,zz in zip(qs,px,pz):
                if xx:x[:,q]^=flip
                if zz:z[:,q]^=flip
    final=[Pauli(label) for label in model.labels]
    fx=np.stack([p.x for p in final]).astype(np.uint8);fz=np.stack([p.z for p in final]).astype(np.uint8)
    syndrome=((x.astype(np.uint8)@fz.T+z.astype(np.uint8)@fx.T)%2).astype(bool)
    # uint8 overflow subtracts multiples of 256, preserving GF(2) parity.
    return dict(expectations=(1-2*np.mean(syndrome,axis=0)).tolist(),fidelity=float(np.mean(~np.any(syndrome,axis=1))),shots=shots,seed=seed)
