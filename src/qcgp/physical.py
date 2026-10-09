"""Explicit Pauli controls with relaxation, dephasing and bounded control errors.

This is a declared spin-qubit simulator, not calibrated IBM/transmon hardware.
Time is in microseconds, Hamiltonians in radians per microsecond (hbar=1).
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from qiskit import transpile
from qiskit.quantum_info import Operator, SuperOp, Choi
from .timing import TimingModel
from .core import critical_path, unitary_error
from .pulse import embed_matrix

X=np.array([[0,1],[1,0]],dtype=complex)
Z=np.diag([1,-1]).astype(complex)
LOWERING=np.array([[0,1],[0,0]],dtype=complex)


@dataclass(frozen=True)
class PhysicalParameters:
    t1: tuple[float,...]
    tphi: tuple[float,...]
    amplitude_eta: float = .02
    active_rate: float = .0005
    omega_single: float = 1.5
    omega_pair: float = .35
    coherent_fraction: float = 0.0

    def __post_init__(self):
        # JSON round-trips tuples as lists; normalize the immutable cache key.
        object.__setattr__(self,'t1',tuple(self.t1))
        object.__setattr__(self,'tphi',tuple(self.tphi))

    def validate(self,n):
        if len(self.t1)!=n or len(self.tphi)!=n:raise ValueError('One T1/Tphi per qubit required')
        positive=(*self.t1,*self.tphi,self.amplitude_eta,self.active_rate,self.omega_single,self.omega_pair)
        if any(not math.isfinite(v) or v<=0 for v in positive):raise ValueError('Physical rates and scales must be positive')
        if not math.isfinite(self.coherent_fraction):raise ValueError('Control error must be finite')


def compile_pauli(logical,seed=7):
    """Translate to RX/RZ/RXX on an all-to-all declared spin-control model."""
    circuit=transpile(logical,basis_gates=['rx','rz','rxx'],optimization_level=3,seed_transpiler=seed)
    # Zero-angle controls have no exposure; remove only numerical zero rotations.
    remove=[i for i,ins in enumerate(circuit.data) if abs(float(ins.operation.params[0]))<1e-12]
    for i in reversed(remove):del circuit.data[i]
    if logical.num_qubits<=6 and unitary_error(logical,circuit)>1e-8:
        raise RuntimeError('Native Pauli translation changed the logical unitary')
    return circuit


def controls(circuit):
    matrices=[];supports=[];edges=set();last={}
    for i,ins in enumerate(circuit.data):
        name=ins.operation.name
        if name not in ['rx','rz','rxx']:raise ValueError('Only finite RX/RZ/RXX controls are supported')
        theta=float(ins.operation.params[0]);qs=tuple(circuit.find_bit(q).index for q in ins.qubits)
        if not math.isfinite(theta) or abs(theta)<1e-12:raise ValueError('Nonzero bound rotation required')
        pauli={'rx':X,'rz':Z,'rxx':np.kron(X,X)}[name]
        k=theta/2*pauli
        if not np.allclose(expm(-1j*k),Operator(ins.operation).data,atol=1e-10):
            raise RuntimeError('Explicit generator does not realize gate')
        matrices.append(k);supports.append(qs)
        for q in qs:
            if q in last:edges.add((last[q],i))
            last[q]=i
    if not matrices:raise ValueError('A nonempty unitary circuit is required')
    return matrices,supports,tuple(sorted(edges))


def physical_model(circuit,parameters):
    parameters.validate(circuit.num_qubits)
    matrices,supports,edges=controls(circuit)
    kappa2=np.array([np.linalg.norm(k,2)**2 for k in matrices])
    theta=np.sqrt(kappa2)*2
    base=theta/np.array([parameters.omega_pair if len(qs)==2 else parameters.omega_single for qs in supports])
    eta=parameters.amplitude_eta*np.array([4 if len(qs)==2 else 1 for qs in supports])
    rate=np.full(len(base),parameters.active_rate)
    background=sum(1/t for t in parameters.t1)+sum(1/(2*t) for t in parameters.tphi)
    model=TimingModel(base,.5*base,3*base,eta*kappa2,rate,kappa2,background,edges)
    model.validate()
    return model


def general_certificate(model,durations,coherent_fraction=0):
    schedule=critical_path(durations,model.edges)
    exposure=float(model.background_rate*schedule.makespan+np.sum(model.amplitude_noise/durations+model.gate_noise_rate*durations))
    # Integral norm of a relative coherent amplitude error is |eps|*||K_i||.
    coherent=abs(coherent_fraction)*float(np.sum(np.sqrt(model.generator_squared)))
    radius=exposure+coherent
    trace_floor=max(0.,1-radius)
    no_jump_floor=max(0.,1-coherent-exposure/2)**2
    return dict(integrated_generator_bound=radius,noise_bound=exposure,coherent_bound=coherent,
        fidelity_floor=max(trace_floor,no_jump_floor),trace_distance_fidelity_floor=trace_floor,
        no_jump_fidelity_floor=no_jump_floor,half_diamond_upper=min(1.,radius),observable_upper=min(2.,2*radius))


def dissipator(jump):
    dimension=jump.shape[0];identity=np.eye(dimension);gram=jump.conj().T@jump
    return np.kron(jump.conj(),jump)-.5*(np.kron(identity,gram)+np.kron(gram.T,identity))


def physical_channel(circuit,parameters,durations,*,integrator='expm'):
    """Return a whole-system channel; integrate concurrent controls and always-on T1.

    `ode` independently integrates the superoperator ODE using adaptive RK45.
    Dense channel evolution is intentionally limited to at most three qubits.
    """
    n=circuit.num_qubits
    if n>3:raise ValueError('Dense physical-channel validation is limited to three qubits')
    if integrator not in ['expm','ode']:raise ValueError('Unknown integrator')
    model=physical_model(circuit,parameters)
    tau=np.asarray(durations,dtype=float)
    if tau.shape!=model.base.shape or not np.isfinite(tau).all() or np.any(tau<model.lower-1e-9) or np.any(tau>model.upper+1e-9):
        raise ValueError('Pulse durations must lie in the declared control range')
    matrices,supports,_=controls(circuit)
    schedule=critical_path(tau,model.edges)
    d=2**n;identity=np.eye(d);memory=np.zeros((d*d,d*d),dtype=complex)
    for q in range(n):
        memory+=dissipator(embed_matrix(LOWERING,[q],n)/np.sqrt(parameters.t1[q]))
        memory+=dissipator(embed_matrix(Z,[q],n)/np.sqrt(2*parameters.tphi[q]))
    generators=[]
    for i,(k,qs) in enumerate(zip(matrices,supports)):
        full=embed_matrix(k,qs,n);h=(1+parameters.coherent_fraction)*full/tau[i]
        eta=parameters.amplitude_eta*(4 if len(qs)==2 else 1)
        amplitude=dissipator(np.sqrt(eta)*full/tau[i])
        p=full/np.sqrt(model.generator_squared[i])
        active=dissipator(np.sqrt(parameters.active_rate)*p)
        generators.append(-1j*(np.kron(identity,h)-np.kron(h.T,identity))+amplitude+active)
    ends=schedule.starts+tau;events=np.unique(np.r_[0,schedule.starts,ends,schedule.makespan])
    channel=np.eye(d*d,dtype=complex)
    for left,right in zip(events[:-1],events[1:]):
        if right-left<1e-12:continue
        middle=(left+right)/2
        active=np.flatnonzero((schedule.starts<=middle)&(ends>middle))
        liouvillian=memory.copy()
        for i in active:liouvillian+=generators[i]
        interval=right-left
        if integrator=='expm':channel=expm(liouvillian*interval)@channel
        else:
            shape=channel.shape
            solution=solve_ivp(lambda t,y:(liouvillian@y.reshape(shape)).reshape(-1),(0,interval),channel.reshape(-1),rtol=1e-9,atol=1e-11)
            if not solution.success:raise RuntimeError('Independent ODE integration failed')
            channel=solution.y[:,-1].reshape(shape)
    output=SuperOp(channel)
    choi=Choi(output).data
    # Trace over output in input-tensor-output ordering verifies TP independently.
    reduced=np.trace(choi.reshape(d,d,d,d),axis1=1,axis2=3)
    tp=float(np.max(np.abs(reduced-np.eye(d))))
    minimum=float(np.linalg.eigvalsh(choi).min())
    if tp>1e-7 or minimum<-1e-7:raise RuntimeError('Simulated channel is not numerically CPTP')
    return output,dict(makespan=schedule.makespan,events=len(events),tp_error=tp,
        minimum_choi_eigenvalue=minimum,integrator=integrator,
        certificate=general_certificate(model,tau,parameters.coherent_fraction))
