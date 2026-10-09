"""Independent dense Lindblad simulation of scheduled constant-angle pulses.

This CPU simulator is limited to small circuits. It integrates each interval
between gate start/end events, including simultaneously active disjoint gates.
Local Z memory noise need not commute with the control Hamiltonian.
"""
import itertools
import math
import numpy as np
from scipy.linalg import expm
from qiskit.quantum_info import Operator,Statevector,DensityMatrix
from .timing import minimal_generator
from .core import extract,critical_path


def embed_matrix(matrix,qargs,n):
    """Embed a local linear operator, using qargs order and little-endian indices."""
    qargs=tuple(qargs)
    if len(set(qargs))!=len(qargs) or any(q<0 or q>=n for q in qargs):
        raise ValueError('Invalid embedding qubits')
    matrix=np.asarray(matrix,dtype=complex)
    if matrix.shape!=(2**len(qargs),2**len(qargs)):raise ValueError('Matrix dimension disagrees with qubits')
    indices=np.arange(2**n)
    local=sum((((indices>>q)&1)<<j for j,q in enumerate(qargs)),np.zeros_like(indices))
    mask=sum(1<<q for q in range(n) if q not in qargs)
    spectator=indices&mask
    return matrix[np.ix_(local,local)]*(spectator[:,None]==spectator[None,:])


def replacement_superoperator(qargs,n):
    paulis=[np.eye(2),np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.diag([1,-1])]
    dimension=2**n
    superoperator=np.zeros((dimension**2,dimension**2),dtype=complex)
    for choices in itertools.product(paulis,repeat=len(qargs)):
        local=np.eye(1)
        for p in choices:local=np.kron(p,local)
        full=embed_matrix(local,qargs,n)
        superoperator+=np.kron(full.conj(),full)/(4**len(qargs))
    return superoperator


def simulate_pulses(circuit,model,result,initial=None,*,background='local_dephasing',noise_scale=1.0):
    """Integrate a declared Markovian model, not an IBM/transmon pulse backend.

    Gate-local generator: -i[K/tau,rho]+(c+a/tau^2)(T_A-I)rho.
    Local memory generator: (b/n) sum_q (Z_q rho Z_q-rho).
    Its total unitary-jump hazard is b, so the same R certificate applies.
    """
    n=circuit.num_qubits
    if n>4:raise ValueError('Dense pulse validation is limited to four system qubits')
    if background not in ('global','local_dephasing'):raise ValueError('Unknown background model')
    if not math.isfinite(noise_scale) or noise_scale<0:raise ValueError('noise_scale must be finite and nonnegative')
    model.validate()
    _,edges,qubits,_=extract(circuit)
    if tuple(edges)!=model.edges or len(circuit.data)!=len(result.durations):raise ValueError('Circuit and timing disagree')
    schedule=critical_path(result.durations,model.edges)
    if not np.allclose(schedule.starts,result.starts,atol=1e-7,rtol=1e-9):raise ValueError('Expected canonical earliest-start schedule')
    if initial is None:initial=Statevector.from_label('0'*n)
    density=DensityMatrix(initial)
    if density.num_qubits!=n:raise ValueError('Pulse simulator expects system-only initial state')
    dimension=2**n;identity=np.eye(dimension);identity_super=np.eye(dimension**2)
    if background=='global':
        replacement=np.outer((identity/dimension).reshape(-1,order='F'),identity.reshape(-1,order='F'))
        memory=np.asarray(model.background_rate*(replacement-identity_super),dtype=complex)
    else:
        memory=np.zeros_like(identity_super,dtype=complex)
        for q in range(n):
            z=embed_matrix(np.diag([1,-1]),[q],n)
            memory+=model.background_rate/n*(np.kron(z.conj(),z)-identity_super)
    generators=[];noise=[];cache={}
    for i,(instruction,qargs) in enumerate(zip(circuit.data,qubits)):
        k,_=minimal_generator(Operator(instruction.operation).data)
        h=embed_matrix(k,qargs,n)/result.durations[i]
        generators.append(-1j*(np.kron(identity,h)-np.kron(h.T,identity)))
        support=tuple(sorted(qargs))
        if support not in cache:cache[support]=replacement_superoperator(support,n)-identity_super
        rate=model.gate_noise_rate[i]+model.amplitude_noise[i]/result.durations[i]**2
        noise.append(rate*cache[support])
    ends=result.starts+result.durations
    events=np.unique(np.r_[0,result.starts,ends,result.makespan])
    vector=density.data.reshape(-1,order='F')
    intervals=0
    for left,right in zip(events[:-1],events[1:]):
        if right-left<1e-12:continue
        middle=(left+right)/2
        active=np.flatnonzero((result.starts<=middle)&(ends>middle))
        liouvillian=noise_scale*memory.copy()
        for i in active:liouvillian+=generators[i]+noise_scale*noise[i]
        vector=expm(liouvillian*(right-left))@vector
        intervals+=1
    actual=DensityMatrix(vector.reshape(dimension,dimension,order='F'))
    ideal=DensityMatrix(initial).evolve(circuit)
    trace_error=float(abs(np.trace(actual.data)-1))
    minimum=float(np.linalg.eigvalsh(actual.data).min())
    if trace_error>1e-8 or minimum < -1e-8:raise RuntimeError('Pulse density matrix failed physicality check')
    return actual,ideal,dict(intervals=intervals,trace_error=trace_error,minimum_eigenvalue=minimum,
                            background=background,noise_scale=noise_scale)
