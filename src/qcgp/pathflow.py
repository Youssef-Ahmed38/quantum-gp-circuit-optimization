"""Certified path-flow conditional-gradient solver for the joint-timing GP.

This specializes established convex time-cost duality and Frank-Wolfe methods;
it does not claim a new general-purpose optimization algorithm.
"""
from dataclasses import dataclass
import math,time
import numpy as np
from .timing import TimingModel,TimingResult,evaluate_timing,reference_metrics


@dataclass(frozen=True)
class PathFlowResult:
    timing: TimingResult
    dual_bound: float
    absolute_gap: float
    relative_gap: float
    iterations: int
    pressure: np.ndarray
    certified: bool
    history: tuple[dict, ...]


def longest_path_oracle(durations,predecessors):
    ends=np.empty(len(durations));parents=np.full(len(durations),-1,dtype=int)
    for j,previous in enumerate(predecessors):
        if previous:
            parent=max(previous,key=lambda i:ends[i]);parents[j]=parent
            ends[j]=durations[j]+ends[parent]
        else:ends[j]=durations[j]
    terminal=int(np.argmax(ends));makespan=float(ends[terminal])
    incidence=np.zeros(len(durations));current=terminal
    while current>=0:
        incidence[current]=1;current=parents[current]
    return makespan,incidence


def optimize_pathflow(model:TimingModel,weights=(1,0,0),*,references=None,
                      relative_tolerance=1e-6,max_iterations=10000):
    """Optimize unconstrained-budget timing, with a primal/dual error certificate.

    No deadline/effort-budget constraints are accepted here; use the general GP
    interface for those. Each iteration uses one longest-path oracle plus a
    one-dimensional concave line search. Stop only on the computed gap.
    """
    start=time.perf_counter();model.validate()
    w=np.asarray(weights,dtype=float)
    if w.shape!=(3,) or not np.isfinite(w).all() or np.any(w<0) or w[0]<=0:
        raise ValueError('Need three nonnegative weights and positive risk weight')
    if not math.isfinite(relative_tolerance) or relative_tolerance<=0 or not isinstance(max_iterations,int) or max_iterations<1:
        raise ValueError('Positive tolerance and positive integer iteration limit required')
    ref=reference_metrics(model) if references is None else np.asarray(references,dtype=float)
    if ref.shape!=(3,) or not np.isfinite(ref).all() or np.any(ref<=0):raise ValueError('Invalid objective references')
    a=w[0]*model.amplitude_noise/ref[0]+w[2]*model.generator_squared/ref[2]
    c=w[0]*model.gate_noise_rate/ref[0]
    b=w[0]*model.background_rate/ref[0]+w[1]/ref[1]
    predecessors=[[] for _ in model.base]
    for i,j in model.edges:predecessors[j].append(i)
    def durations(pressure):return np.clip(np.sqrt(a/(c+pressure)),model.lower,model.upper)
    initial=np.clip(np.sqrt(a/c),model.lower,model.upper)
    _,path=longest_path_oracle(initial,predecessors)
    pressure=b*path
    history=[]
    for iteration in range(max_iterations+1):
        tau=durations(pressure)
        makespan,path=longest_path_oracle(tau,predecessors)
        primal=float(np.sum(a/tau+c*tau)+b*makespan)
        dual=float(np.sum(a/tau+(c+pressure)*tau))
        raw_gap=primal-dual
        if raw_gap < -1e-9*max(1,abs(primal)):raise RuntimeError('Path-flow dual feasibility failed')
        gap=max(0.0,raw_gap)
        relative=gap/max(abs(primal),1e-12)
        if iteration%100==0 or relative<=relative_tolerance or iteration==max_iterations:
            history.append(dict(iteration=iteration,primal=primal,dual=dual,gap=gap,relative_gap=relative))
        if relative<=relative_tolerance or iteration==max_iterations:break
        direction=b*path-pressure
        # Envelope theorem: derivative of the dual along direction is dot(tau,d).
        derivative_at_one=float(np.dot(durations(pressure+direction),direction))
        if derivative_at_one>=0:step=1.0
        else:
            left,right=0.0,1.0
            for _ in range(40):
                middle=(left+right)/2
                derivative=float(np.dot(durations(pressure+middle*direction),direction))
                if derivative>0:left=middle
                else:right=middle
            step=(left+right)/2
        pressure=(1-step)*pressure+step*b*path
    timing=evaluate_timing(model,tau,w,ref,method='pathflow',elapsed=time.perf_counter()-start,
        solver='DAG_oracle_Frank_Wolfe',status='certified_dual_gap' if relative<=relative_tolerance else 'iteration_limit',pressure=pressure)
    if abs(timing.objective-primal)>1e-9*max(1,abs(primal)):raise RuntimeError('Primal objective reconstruction failed')
    return PathFlowResult(timing,dual,gap,relative,iteration,pressure,relative<=relative_tolerance,tuple(history))
