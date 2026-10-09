"""Free finite-pulse optimization and independently inspectable lower bounds.

The MILP is not a GP. Its LP relaxation and the continuous GP bracket different
feasible sets; neither fractional pulse mixtures nor aggregate counts are emitted.
"""
from dataclasses import dataclass
import heapq
import math
import time
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import lil_matrix
from .core import critical_path
from .timing import evaluate_timing,reference_metrics


@dataclass(frozen=True)
class PulsePolicy:
    durations: np.ndarray
    objective: float
    lower_bound: float
    relative_gap: float
    elapsed: float
    status: str
    method: str
    optimal: bool


def duration_library(model,ratios=(.5,.75,1.,1.5,2.,3.)):
    model.validate();ratio=np.asarray(ratios,dtype=float)
    if ratio.ndim!=1 or not len(ratio) or not np.isfinite(ratio).all() or np.any(ratio<=0):raise ValueError('Positive duration ratios required')
    if len(np.unique(ratio))!=len(ratio):raise ValueError('Duplicate pulse durations')
    library=np.sort(model.base[:,None]*ratio[None,:],axis=1)
    if np.any(library<model.lower[:,None]-1e-10) or np.any(library>model.upper[:,None]+1e-10):raise ValueError('Library violates pulse bounds')
    return library


def nearest_policy(model,continuous,library):
    library=np.asarray(library)
    choice=np.argmin(abs(library-continuous.durations[:,None]),axis=1)
    return library[np.arange(len(model.base)),choice]


def optimize_library(model,library,weights=(1,0,0),*,relax=False,deadline=None,time_limit=10,
                     pulse_bounds=None,support_rates=None):
    """Fixed-order one-hot pulse MILP or its continuous LP relaxation.

    A timed-out integer solve reports its feasible incumbent and certified solver
    lower bound. No incumbent is an error. Fractional LP durations are never
    described as a realizable finite-library pulse policy.
    """
    begin=time.perf_counter();model.validate();lib=np.asarray(library,dtype=float)
    m=len(model.base)
    if lib.ndim!=2 or lib.shape[0]!=m or not lib.shape[1] or not np.isfinite(lib).all() or np.any(lib<model.lower[:,None]-1e-10) or np.any(lib>model.upper[:,None]+1e-10):raise ValueError('Invalid pulse library')
    if not math.isfinite(time_limit) or time_limit<=0:raise ValueError('Positive solver time limit required')
    w=np.asarray(weights,dtype=float)
    if w.shape!=(3,) or not np.isfinite(w).all() or np.any(w<0) or w[0]<=0:raise ValueError('Invalid weights')
    ref=reference_metrics(model);k=lib.shape[1];count=m*k+m+1;finish=m*k;makespan=count-1
    a=w[0]*model.amplitude_noise/ref[0]+w[2]*model.generator_squared/ref[2]
    c=w[0]*model.gate_noise_rate/ref[0];b=w[0]*model.background_rate/ref[0]+w[1]/ref[1]
    costs=a[:,None]/lib+c[:,None]*lib
    measured=pulse_bounds is not None or support_rates is not None
    if measured:
        pulse=np.asarray(pulse_bounds,dtype=float);support=np.asarray(support_rates,dtype=float)
        if pulse.shape!=lib.shape or support.shape!=(m,) or not np.isfinite(pulse).all() or not np.isfinite(support).all() or np.any(pulse<0) or np.any(support<0):raise ValueError('Valid measured pulse bounds and support rates required together')
        costs=w[0]*(pulse-support[:,None]*lib)/ref[0]+w[2]*model.generator_squared[:,None]/(ref[2]*lib)
    objective=np.zeros(count);objective[:m*k]=costs.reshape(-1);objective[-1]=b
    constraints=lil_matrix((3*m+len(model.edges),count));lower=[];upper=[];row=0
    for i in range(m):
        constraints[row,i*k:(i+1)*k]=1;lower.append(1);upper.append(1);row+=1
        constraints[row,i*k:(i+1)*k]=lib[i];constraints[row,finish+i]=-1;lower.append(-np.inf);upper.append(0);row+=1
        constraints[row,finish+i]=1;constraints[row,makespan]=-1;lower.append(-np.inf);upper.append(0);row+=1
    for i,j in model.edges:
        constraints[row,finish+i]=1;constraints[row,finish+j]=-1
        constraints[row,j*k:(j+1)*k]=lib[j];lower.append(-np.inf);upper.append(0);row+=1
    lo=np.zeros(count);hi=np.full(count,np.inf);hi[:m*k]=1
    if deadline is not None:
        if not math.isfinite(deadline) or deadline<=0:raise ValueError('Positive deadline required')
        hi[-1]=deadline
    integral=np.zeros(count,dtype=int)
    if not relax:integral[:m*k]=1
    result=milp(objective,integrality=integral,bounds=Bounds(lo,hi),
        constraints=LinearConstraint(constraints.tocsc(),lower,upper),
        options={'time_limit':time_limit,'mip_rel_gap':1e-7})
    # A shortest-library schedule is an explicit feasibility witness. On tight
    # deadlines, presolve can incorrectly report infeasibility near a degenerate
    # boundary; retry without it, retaining the original total time budget.
    remaining=time_limit-(time.perf_counter()-begin)
    witnessed=deadline is not None and critical_path(np.min(lib,axis=1),model.edges).makespan<=deadline+1e-9
    if result.x is None and witnessed and remaining>0:
        result=milp(objective,integrality=integral,bounds=Bounds(lo,hi),
            constraints=LinearConstraint(constraints.tocsc(),lower,upper),
            options={'time_limit':remaining,'mip_rel_gap':1e-7,'presolve':False})
        result.message='Retried without presolve after feasible deadline witness; '+result.message
    if result.x is None:raise RuntimeError('No feasible finite-pulse solution: '+result.message)
    x=result.x[:m*k].reshape(m,k)
    if np.max(abs(x.sum(axis=1)-1))>1e-6:raise RuntimeError('Pulse simplex violation')
    if relax:
        tau=np.sum(x*lib,axis=1);value=float(result.fun);bound=value
    else:
        if np.max(abs(x-np.round(x)))>1e-6:raise RuntimeError('Nonintegral emitted pulse policy')
        tau=lib[np.arange(m),np.argmax(x,axis=1)]
        value=float(np.sum(costs[np.arange(m),np.argmax(x,axis=1)])+b*critical_path(tau,model.edges).makespan) if measured else evaluate_timing(model,tau,w,ref).objective
        if abs(value-result.fun)>1e-6*max(1,abs(value)):raise RuntimeError('MILP objective reconstruction failed')
        bound=float(result.mip_dual_bound)
    span=critical_path(tau,model.edges).makespan
    if deadline is not None and span>deadline+1e-6:raise RuntimeError('Pulse policy violates deadline')
    if bound>value+1e-6:raise RuntimeError('Invalid optimum bracket')
    gap=max(0,value-bound)/max(abs(value),1e-12)
    return PulsePolicy(tau,value,bound,gap,time.perf_counter()-begin,result.message,
        ('measured_lp_fractional' if relax else 'measured_milp') if measured else ('finite_lp_fractional' if relax else 'finite_milp'),bool(result.success and gap<=1e-6))


def slack_policy(model,library,angles):
    """Algorithm-2 scheduling reimplementation of Lussi et al. (arXiv v1).

    Max angle/duration queue; extend to next available pulse within late finish;
    recompute CPM globally instead of their incremental update. The common
    simulated pulse library differs from their IBM calibration experiments.
    """
    lib=np.asarray(library,dtype=float);angles=np.asarray(angles,dtype=float)
    if lib.ndim!=2 or lib.shape[0]!=len(model.base) or angles.shape!=model.base.shape or not np.isfinite(angles).all():raise ValueError('Invalid library or angles')
    tau=lib[:,0].copy();m=len(tau);choice=np.zeros(m,dtype=int)
    initial=critical_path(tau,model.edges).makespan
    children=[[] for _ in tau]
    for i,j in model.edges:children[i].append(j)
    def cpm():
        schedule=critical_path(tau,model.edges);late=np.full(m,initial)
        for i in reversed(range(m)):
            if children[i]:late[i]=min(late[j]-tau[j] for j in children[i])
        return schedule.starts,late
    starts,late=cpm();queue=[]
    for i in range(m):
        if late[i]-starts[i]-tau[i]>1e-9:heapq.heappush(queue,(-abs(angles[i])/tau[i],i))
    while queue:
        _,i=heapq.heappop(queue);next_choice=choice[i]+1
        if next_choice>=lib.shape[1]:continue
        starts,late=cpm();candidate=lib[i,next_choice]
        if starts[i]+candidate<=late[i]+1e-9:
            tau[i]=candidate;choice[i]=next_choice
            if next_choice+1<lib.shape[1]:heapq.heappush(queue,(-abs(angles[i])/tau[i],i))
    if critical_path(tau,model.edges).makespan>initial+1e-7:raise RuntimeError('Slack reimplementation changed minimum latency')
    return tau
