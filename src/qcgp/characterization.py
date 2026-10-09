"""Simulated finite-pulse channels and Choi-derived operational upper bounds.

No device calibration is inferred. Small matrices characterize the declared
RX/RZ/XX model; only independently available pulses are emitted by the MILP.
"""
from dataclasses import replace
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator,SuperOp,Choi
from .physical import physical_channel,controls
from .core import critical_path


def choi_error_upper(actual,ideal):
    """1/2 diamond upper bound from |J(actual-ideal)| via an SDP dual feasible point.

    Choi matrices use Qiskit's unnormalized input-tensor-output convention.
    For Hermiticity-preserving Delta, Y=|J(Delta)| dominates +/-J(Delta),
    and ||Tr_output Y||_infinity bounds its diamond norm.
    """
    delta=Choi(actual).data-Choi(ideal).data
    if not np.allclose(delta,delta.conj().T,atol=1e-9):raise ValueError('Hermitian channel difference required')
    eigenvalues,eigenvectors=np.linalg.eigh(delta)
    absolute=(eigenvectors*abs(eigenvalues))@eigenvectors.conj().T
    d=round(np.sqrt(len(delta)))
    if d*d!=len(delta):raise ValueError('Equal input/output dimension required')
    marginal=np.trace(absolute.reshape(d,d,d,d),axis1=1,axis2=3)
    upper=.5*float(np.linalg.eigvalsh(marginal).max())
    lower=.5*float(np.sum(abs(eigenvalues)))/d
    return dict(half_diamond_upper=min(1.,upper),choi_trace_lower=lower)


def characterize_library(circuit,parameters,library):
    matrices,supports,_=controls(circuit)
    bounds=np.empty(library.shape);rates=np.empty(len(matrices));records=[];cache={}
    for i,(instruction,qs) in enumerate(zip(circuit.data,supports)):
        local=QuantumCircuit(len(qs));local.append(instruction.operation,range(len(qs)))
        p=replace(parameters,t1=tuple(parameters.t1[q] for q in qs),tphi=tuple(parameters.tphi[q] for q in qs))
        rates[i]=sum(1/t for t in p.t1)+sum(1/(2*t) for t in p.tphi)
        ideal=SuperOp(Operator(local))
        for k,tau in enumerate(library[i]):
            key=(instruction.operation.name,float(instruction.operation.params[0]),p,float(tau))
            if key not in cache:
                channel,info=physical_channel(local,p,np.array([tau]))
                cert=choi_error_upper(channel,ideal)
                cache[key]=(channel,info,cert)
            channel,info,cert=cache[key]
            bounds[i,k]=cert['half_diamond_upper']
            records.append(dict(gate=i,choice=k,duration=float(tau),qargs=list(qs),
                **cert,tp_error=info['tp_error'],minimum_choi_eigenvalue=info['minimum_choi_eigenvalue']))
    return bounds,rates,records


def library_certificate(model,durations,library,pulse_bounds,support_rates):
    tau=np.asarray(durations,dtype=float)
    choice=np.argmin(abs(library-tau[:,None]),axis=1)
    if np.max(abs(library[np.arange(len(tau)),choice]-tau))>1e-8:raise ValueError('A finite-library policy is required')
    span=critical_path(tau,model.edges).makespan
    idle=model.background_rate*span-float(np.dot(support_rates,tau))
    if idle<-1e-8:raise ValueError('Negative idle exposure: incompatible support rates/schedule')
    pulse=float(np.sum(pulse_bounds[np.arange(len(tau)),choice]))
    radius=pulse+max(0.,idle)
    return dict(integrated_library_bound=radius,pulse_error_sum=pulse,idle_exposure=max(0.,idle),
        fidelity_floor=max(0.,1-radius),half_diamond_upper=min(1.,radius))


def transfer_gp_lower_bound(model,library,pulse_bounds,support_rates,gp_dual,references):
    """Transfer a valid risk-only GP dual bound to arbitrary characterized costs.

    The gatewise worst residual makes this a conservative perturbation bound,
    even if the characterized costs have no continuous convex interpolation.
    Inputs use the shared GP reference R0. Approximate primal values are invalid.
    """
    reference=float(references[0])
    if not np.isfinite(gp_dual) or not np.isfinite(reference) or reference<=0:raise ValueError('Finite dual bound and positive reference required')
    residue=(pulse_bounds-support_rates[:,None]*library-model.amplitude_noise[:,None]/library-model.gate_noise_rate[:,None]*library)/reference
    minima=np.min(residue,axis=1)
    lower=max(0.,float(gp_dual+np.sum(minima)))
    return dict(transferred_lower_bound=lower,gatewise_minimum_residuals=minima.tolist(),
        residual_span=float(np.sum(np.max(residue,axis=1)-minima)))
