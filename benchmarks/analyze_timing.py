"""Compute reported joint-framework results from preserved raw experiments."""
from pathlib import Path
import json
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'evidence'/'joint_timing'
FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)


def read(name):return json.loads((DATA/(name+'.json')).read_text())
def distribution(values):
    x=np.asarray(values,dtype=float)
    if not len(x) or not np.isfinite(x).all():raise ValueError('Empty/nonfinite analysis')
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),minimum=float(x.min()),maximum=float(x.max()))


def paired(rows,baseline,target='joint_gp',field='nominal_risk',percent=True):
    a={r['case']:r for r in rows if r['method']==baseline}
    b={r['case']:r for r in rows if r['method']==target}
    if set(a)!=set(b):raise ValueError('Unmatched cases')
    out=[]
    for case,r in a.items():
        delta=r[field]-b[case][field]
        value=100*delta/r[field] if percent else delta
        out.append(dict(case=case,family=r['family'],qubits=r['qubits'],topology=r['topology'],instance=r['instance'],value=value))
    return out


def clustered_interval(rows):
    # Exploratory resampling of size/topology strata, not independent hardware trials.
    strata={}
    for r in rows:strata.setdefault((r['qubits'],r['topology']),[]).append(r['value'])
    x=np.asarray([np.mean(v) for v in strata.values()])
    rng=np.random.default_rng(9041)
    means=x[rng.integers(0,len(x),size=(10000,len(x)))].mean(axis=1)
    return dict(strata=len(x),low=float(np.quantile(means,.025)),high=float(np.quantile(means,.975)),interpretation='Exploratory size/topology-stratum bootstrap; no population or hardware inference')


def main():
    rows=read('results');noise=read('noise');process=read('process');pulse=read('pulse_validation')
    flow=read('pathflow_validation');pareto=read('pareto');outer=read('outer_search');scale=read('scaling');sensitivity=read('sensitivity')
    families=list(dict.fromkeys(r['family'] for r in rows))
    summary=dict(counts=dict(configurations=len({r['case'] for r in rows}),timing_rows=len(rows),
        state_channel_checks=len(noise),entangled_process_checks=len(process),pulse_checks=len(pulse),
        pathflow_checks=len(flow),pareto_points=len(pareto),outer_cases=len(outer),scaling_cases=len(scale),sensitivity_rows=len(sensitivity)),families={})
    for family in families:
        subset=[r for r in rows if r['family']==family]
        item={}
        for baseline in ['fixed','fastest','local','uniform','fast_deadline']:
            values=paired(subset,baseline)
            item['risk_reduction_vs_'+baseline]=dict(**distribution([r['value'] for r in values]),interval=clustered_interval(values))
        item['latency_change_vs_fixed']=distribution([-r['value'] for r in paired(subset,'fixed',field='makespan')])
        joint=[r for r in subset if r['method']=='joint_gp']
        item['fidelity_floor']=distribution([r['nominal_fidelity_bound'] for r in joint])
        gains=[];regressions=0
        for case in {r['case'] for r in noise if r['family']==family}:
            selected=[r for r in noise if r['case']==case and r['scenario']=='nominal']
            fixed={r['input']:r['fidelity'] for r in selected if r['method']=='fixed'}
            optimized={r['input']:r['fidelity'] for r in selected if r['method']=='joint_gp'}
            g=np.mean([100*(optimized[k]-v) for k,v in fixed.items()])
            gains.append(g);regressions+=int(g < -1e-8)
        item['simulated_fidelity_gain_percentage_points']=dict(**distribution(gains),regressions=regressions)
        summary['families'][family]=item
    summary['max_gp_convex_objective_gap']=max(r['gp_convex_gap'] for r in rows)
    summary['unitary_error_max']=max(r['unitary_error'] for r in rows)
    summary['state_min_certificate_slack']=min(r['fidelity']-r['fidelity_lower_bound'] for r in noise)
    summary['process_min_certificate_slack']=min(r['fidelity']-r['fidelity_lower_bound'] for r in process)
    summary['pulse_min_certificate_slack']=min(r['fidelity']-r['fidelity_bound'] for r in pulse)
    summary['flow_max_relative_gap']=max(r['relative_gap'] for r in flow)
    summary['flow_max_lp_residual']=max(r['flow_lp_residual'] for r in flow)
    summary['flow_iterations']=distribution([r['iterations'] for r in flow])
    summary['flow_seconds']=distribution([r['timing']['elapsed'] for r in flow])
    failures=read('out_of_model');summary['coherent_counterexamples']=dict(tested=len(failures),breaches=sum(r['certificate_breached'] for r in failures))
    impl=Counter(r.get('implementation', 'gp' if r['method'] in ['joint_gp','robust_gp','fast_deadline'] else 'convex' if r['method']=='joint_convex' else r['method']) for r in rows)
    summary['implementation_counts']=dict(impl)
    summary['risk_solver_runtimes']={}
    for method in ['joint_gp','joint_convex']:
        selected=[r for r in rows if r['method']==method and (method=='joint_convex' or r.get('implementation','gp')=='gp')]
        summary['risk_solver_runtimes'][method]=distribution([r['elapsed'] for r in selected])
    summary['robust_worst_risk_reduction']=distribution([r['value'] for r in paired(rows,'joint_gp',target='robust_gp',field='worst_box_risk')])
    index={(r['case'],r['method']):r for r in rows}
    quant=[]
    for r in rows:
        if r['method']=='quantized_gp':
            gap=abs(r['nominal_risk']-index[(r['case'],'joint_gp')]['nominal_risk'])
            if gap>r['quantization_risk_bound']+1e-9:raise ValueError('Quantization bound violated')
            quant.append(dict(actual=gap,bound=r['quantization_risk_bound']))
    summary['quantization']=dict(checks=len(quant),max_absolute_risk_change=max(r['actual'] for r in quant),mean_bound=float(np.mean([r['bound'] for r in quant])))
    gains=[];changes=0;scoregaps=[]
    for r in outer:
        a=min(x['fixed_score'] for x in r['options']);b=min(x['joint_gp_score'] for x in r['options'])
        gains.append(100*(a-b)/a);changes+=int(r['selected_fixed']!=r['selected_gp'])
        scoregaps.extend(abs(x['joint_gp_score']-x['joint_convex_score']) for x in r['options'])
    summary['finite_search']=dict(improvement=distribution(gains),candidate_choice_changes=changes,max_direct_gap=max(scoregaps))
    summary['scale']=[dict(family=r['family'],qubits=r['qubits'],gates=r['gates'],method=r['gp']['method'],seconds=r['gp']['elapsed'],direct_seconds=r['convex']['elapsed'],gap=r['gap']) for r in scale]
    summary['sensitivity']={}
    for b in sorted({r['background_rate'] for r in sensitivity}):
        part=[dict(index[(r['case'],'fixed')],**r) for r in sensitivity if r['background_rate']==b]
        summary['sensitivity'][str(b)]=distribution([r['value'] for r in paired(part,'uniform',field='risk')])
    (DATA/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (DATA/'paired_risk_vs_uniform.json').write_text(json.dumps(paired(rows,'uniform'),indent=2)+'\n')
    plt.rcParams.update({'font.size':9,'figure.dpi':160})
    fig,ax=plt.subplots(figsize=(6.5,3))
    ax.boxplot([[r['value'] for r in paired([x for x in rows if x['family']==f],'uniform')] for f in families],tick_labels=['QFT','Grover','VQE ansatz','QAOA'])
    ax.set_ylabel('Risk reduction versus optimal uniform scaling (%)');ax.axhline(0,color='gray',linewidth=.7);fig.tight_layout()
    savefig(fig,'joint_risk')
    case='qaoa_4_ring_0';part=[r for r in pareto if r['case']==case]
    fig,ax=plt.subplots(figsize=(6.5,3))
    points=ax.scatter([r['makespan'] for r in part],[r['risk'] for r in part],c=[r['effort'] for r in part],cmap='viridis',s=65)
    for i,r in enumerate(part):ax.annotate(str(i+1),(r['makespan'],r['risk']),xytext=(4,4),textcoords='offset points')
    fig.colorbar(points,ax=ax,label='Integrated squared Hamiltonian norm');ax.set_xlabel('Latency (synthetic time units)');ax.set_ylabel('Noise-hazard envelope');ax.set_title('Five scalarizations: QAOA, 4 qubits, ring');fig.tight_layout();savefig(fig,'joint_tradeoffs')
    fig,axes=plt.subplots(1,2,figsize=(6.5,3))
    for data,bound,title,ax in [(noise,'fidelity_lower_bound','Gate-channel simulation',axes[0]),(pulse,'fidelity_bound','Noncommuting Lindblad simulation',axes[1])]:
        ax.scatter([r[bound] for r in data],[r['fidelity'] for r in data],s=9,alpha=.4)
        ax.plot([0,1],[0,1],color='black',lw=.7);ax.set_xlabel('Certified fidelity floor');ax.set_ylabel('Simulated fidelity');ax.set_title(title)
    fig.tight_layout();savefig(fig,'joint_certificates')
    fig,ax=plt.subplots(figsize=(6.5,3))
    for f in ['qaoa','vqe_ansatz']:
        p=[r for r in summary['scale'] if r['family']==f]
        ax.plot([r['gates'] for r in p],[r['seconds'] for r in p],'o-',label=f+' joint optimizer')
        ax.plot([r['gates'] for r in p],[r['direct_seconds'] for r in p],'x--',label=f+' convex')
    ax.set_xlabel('Native gates');ax.set_ylabel('Single-call wall time (s)');ax.legend(fontsize=7);fig.tight_layout();savefig(fig,'joint_scaling')
    lines=['# Joint GP framework: measured validation','', 'All results are free local simulations with constructed coefficients. No hardware calibration or journal-readiness claim.','', '## Coverage','',json.dumps(summary['counts'],indent=2),'','## Paired risk reduction versus optimized uniform scaling','', '| Family | Mean (%) | Range (%) | Simulated fidelity gain (percentage points) |','|---|---:|---:|---:|']
    for f,item in summary['families'].items():
        d=item['risk_reduction_vs_uniform'];g=item['simulated_fidelity_gain_percentage_points']
        lines.append(f"| {f} | {d['mean']:.3f} | {d['minimum']:.3f}–{d['maximum']:.3f} | {g['mean']:.3f} |")
    lines+=['','Risk improvements optimize the assumed hazard envelope; actual state fidelity is a separate measurement. Comparators solve different restrictions except the ordinary convex solver, which solves the identical model. Repeated QFT/Grover instances are not independent logical circuits. Intervals are exploratory stratum bootstraps.','', '## Numerical and physical checks','', f"Maximum GP/direct normalized objective disagreement: {summary['max_gp_convex_objective_gap']:.3e}.",f"Maximum independent path-flow relative gap: {summary['flow_max_relative_gap']:.3e}.",f"Independent flow LP residual: {summary['flow_max_lp_residual']:.3e}.",f"Coherent out-of-model errors breach the certificate in {summary['coherent_counterexamples']['breaches']} of {len(failures)} checks.",'','See summary.json for every aggregate, denominator, runtime, regression count, and limitation.']
    (ROOT/'docs'/'joint_validation_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(summary,indent=2))


def savefig(fig,name):
    fig.savefig(FIG/(name+'.pdf'));fig.savefig(FIG/(name+'.png'),dpi=220);plt.close(fig)


if __name__=='__main__':main()
