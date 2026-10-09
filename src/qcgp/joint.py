"""Qiskit interfaces for model-dependent continuous timing and finite circuit search."""
from dataclasses import dataclass
import numpy as np
from qiskit.converters import dag_to_circuit
from qiskit.transpiler.basepasses import AnalysisPass
from .core import compile_candidate, extract, unitary_error
from .timing import (TimingModel, TimingResult, optimize_timing, reference_metrics,
                     robust_upper_model)


class JointGPTimingPass(AnalysisPass):
    """Annotate a native unitary circuit with optimized durations and model bounds.

    The caller supplies a model factory. This avoids silently interpreting a
    synthetic coefficient set as a backend calibration. No hardware pulses or
    device delay instructions are synthesized by this analysis pass.
    """
    def __init__(self, model_factory, weights=(1,0,0), uncertainty_radius=0.0,
                 solver='CLARABEL', deadline=None, effort_limit=None):
        super().__init__()
        if not callable(model_factory):raise TypeError('model_factory must be callable')
        self.model_factory=model_factory
        self.weights=weights
        self.uncertainty_radius=uncertainty_radius
        self.solver=solver
        self.deadline=deadline
        self.effort_limit=effort_limit

    def run(self, dag):
        circuit=dag_to_circuit(dag)
        model=self.model_factory(circuit)
        if not isinstance(model,TimingModel):raise TypeError('Factory must return TimingModel')
        if len(model.base)!=len(circuit.data):raise ValueError('Model and circuit instruction counts disagree')
        if self.uncertainty_radius:
            model=robust_upper_model(model,self.uncertainty_radius)
        result=optimize_timing(model,self.weights,solver=self.solver,
                               deadline=self.deadline,effort_limit=self.effort_limit)
        self.property_set['joint_gp_timing']=result
        self.property_set['joint_gp_model']=model
        self.property_set['joint_gp_guarantee']='Conditional continuous optimum for the supplied fixed-DAG model'
        return dag


@dataclass(frozen=True)
class JointCompilation:
    circuit: object
    timing: TimingResult
    model: TimingModel
    candidate: int
    scores: tuple[float, ...]
    unitary_errors: tuple[float, ...]


class DurationAwareCompiler:
    """Optimize risk, latency, control effort and gate cost over finite candidates.

    Continuous weights are normalized by candidate zero's fixed-duration values.
    Global optimality is restricted to each continuous subproblem and the
    enumerated candidate set. Caller supplies noise-model factory explicitly.
    """
    def __init__(self, coupling, model_factory, seeds=(7,8,9,10),
                 weights=(.5,.2,.1,.2), uncertainty_radius=0.0, solver='CLARABEL'):
        if not callable(model_factory):raise TypeError('model_factory must be callable')
        w=np.asarray(weights,dtype=float)
        if w.shape!=(4,) or not np.isfinite(w).all() or np.any(w<0) or w[0]<=0:
            raise ValueError('Four nonnegative weights required, with positive risk weight')
        if not seeds or any(not isinstance(s,int) or s<0 for s in seeds):
            raise ValueError('Need nonnegative integer compiler seeds')
        self.coupling=coupling;self.model_factory=model_factory;self.seeds=tuple(seeds)
        self.weights=tuple(w);self.radius=uncertainty_radius;self.solver=solver

    def compile(self, logical):
        if logical.num_qubits > 8:
            raise ValueError('Full-unitary candidate verification is limited to eight qubits')
        circuits=[compile_candidate(logical,self.coupling,s)[0] for s in self.seeds]
        models=[self.model_factory(c) for c in circuits]
        if self.radius:models=[robust_upper_model(m,self.radius) for m in models]
        references=reference_metrics(models[0])
        costs=[];errors=[];results=[]
        for circuit,model in zip(circuits,models):
            _,_,qs,_=extract(circuit,self.coupling)
            costs.append(len(qs)+9*sum(len(q)==2 for q in qs))
            error=unitary_error(logical,circuit)
            if error>1e-8:raise RuntimeError('Candidate failed layout-aware unitary equivalence')
            errors.append(error)
            results.append(optimize_timing(model,self.weights[:3],solver=self.solver,references=references))
        scores=[r.objective+self.weights[3]*cost/max(costs[0],1) for r,cost in zip(results,costs)]
        tolerance=2e-5*max(1,min(scores))
        chosen=min(i for i,s in enumerate(scores) if s<=min(scores)+tolerance)
        return JointCompilation(circuits[chosen],results[chosen],models[chosen],chosen,tuple(scores),tuple(errors))
