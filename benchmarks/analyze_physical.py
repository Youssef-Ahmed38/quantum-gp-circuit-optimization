"""Audit and analyze raw finite-pulse physical simulations; retain all regressions."""
from pathlib import Path
import json
import math
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qcgp.characterization import transfer_gp_lower_bound
from qcgp.physical import PhysicalParameters,physical_model,general_certificate
from qcgp.timing import reference_metrics
from qiskit import qpy

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'evidence/physical_study';FIG=ROOT/'figures'


def stats(values):
    x=np.asarray(values,dtype=float)
    if not len(x) or not np.isfinite(x).all():raise ValueError('Invalid analysis values')
    return dict(n=len(x),mean=float(np.mean(x)),median=float(np.median(x)),minimum=float(np.min(x)),maximum=float(np.max(x)))


def paired(index,keys,baseline,target,metric):
    result=[]
    for key in keys:
        a=index[(key,baseline)];b=index[(key,target)]
        if metric=='process_fidelity':value=100*(b[metric]-a[metric])
        elif metric=='state_fidelity':value=100*np.mean([b[label]['fidelity']-a[label]['fidelity'] for label in ['zero','haar']])
        elif metric=='latency':value=100*(b['makespan']/a['makespan']-1)
        else:raise ValueError(metric)
        result.append(dict(key=key,qubits=a['qubits'],seed=a['seed'],family=a['family'],value=float(value)))
    return result


def intervals(records):
    groups={}
    for r in records:groups.setdefault((r['qubits'],r['seed']),[]).append(r['value'])
    x=np.array([np.mean(v) for v in groups.values()]);rng=np.random.default_rng(51097)
    bootstrap=x[rng.integers(0,len(x),size=(10000,len(x)))].mean(axis=1)
    return dict(clusters=len(groups),low=float(np.quantile(bootstrap,.025)),high=float(np.quantile(bootstrap,.975)),scope='Exploratory size/seed clusters; physical parameters shared across families; not hardware inference')


def main():
    cases=json.loads((DATA/'cases.json').read_text());rows=json.loads((DATA/'results.json').read_text())
    index={(r['key'],r['method']):r for r in rows}
    if len(index)!=len(rows):raise ValueError('Duplicate policy keys')
    if set(r['key'] for r in rows)!=set(cases):raise ValueError('Cases/results join mismatch')
    for r in rows:
        for field in ['risk','process_fidelity','makespan','tp_error','minimum_choi_eigenvalue']:
            if not isinstance(r[field],(int,float)) or not math.isfinite(r[field]):raise ValueError('Missing/nonfinite measured metric')
        if not -1e-8<=r['process_fidelity']<=1+1e-8:raise ValueError('Unphysical process fidelity')
    assert all(len(item['rows'])==9 for item in cases.values())
    summary=dict(counts=dict(configurations=len(cases),native_circuits=len({r['case'] for r in rows}),policy_channels=len(rows),state_checks=2*len(rows),
        pulse_entries=sum(len(v['characterization']) for v in cases.values()),independent_ode_diamond_checks=sum(bool(v['extra']) for v in cases.values())),
        inspection=dict(duplicate_keys=0,missing_numeric_metrics=0,excluded_cases=0,all_negative_results_retained=True),regimes={},numerics={})
    for regime in sorted({r['regime'] for r in rows}):
        keys=sorted({r['key'] for r in rows if r['regime']==regime});item={}
        for base in ['fixed','fastest','uniform','published_slack','gp_nearest','characterized_uniform']:
            d=paired(index,keys,base,'characterized_milp','process_fidelity')
            item[base]=dict(process_gain_pp=stats([r['value'] for r in d]),interval=intervals(d),
                process_regressions=sum(r['value']<-1e-7 for r in d),
                state_gain_pp=stats([r['value'] for r in paired(index,keys,base,'characterized_milp','state_fidelity')]),
                latency_change_percent=stats([r['value'] for r in paired(index,keys,base,'characterized_milp','latency')]))
        summary['regimes'][regime]=item
    summary['family_regime']={}
    for family in sorted({r['family'] for r in rows}):
        summary['family_regime'][family]={}
        for regime in summary['regimes']:
            keys=[k for k in cases if index[(k,'fixed')]['family']==family and index[(k,'fixed')]['regime']==regime]
            p=paired(index,keys,'published_slack','characterized_milp','process_fidelity')
            summary['family_regime'][family][regime]=dict(**stats([r['value'] for r in p]),regressions=sum(r['value']<-1e-7 for r in p))
    transfers=[];stronger=[];solver_gaps=[];measured_gaps=[];lp_gaps=[];certificate_slack=[]
    ode=[];sdp=[]
    for key,item in cases.items():
        info=index[(key,'fixed')]
        with (DATA/(info['case']+'.qpy')).open('rb') as f:c=qpy.load(f)[1]
        model=physical_model(c,PhysicalParameters(**item['parameters']))
        lib=np.asarray(item['pulse_library']);bounds=np.asarray(item['pulse_bounds']);rates=np.asarray(item['support_rates'])
        transfer=transfer_gp_lower_bound(model,lib,bounds,rates,item['continuous']['dual_bound'],reference_metrics(model))
        primal=item['characterized_finite']['objective'];lower=item['characterized_finite']['lower_bound'];lp=item['characterized_lp']['lower_bound']
        if max(transfer['transferred_lower_bound'],lower,lp)>primal+1e-6:raise ValueError('Invalid characterized-cost bracket')
        nearest=index[(key,'gp_nearest')]['library_certificate']['integrated_library_bound']/reference_metrics(model)[0]
        transfers.append(dict(key=key,**transfer,characterized_lp_lower=lp,characterized_milp_lower=lower,
            characterized_primal=primal,gp_emitted_score=nearest,
            gp_transfer_relative_gap=max(0,nearest-transfer['transferred_lower_bound'])/max(nearest,1e-12)))
        solver_gaps.append(item['finite']['relative_gap']);measured_gaps.append(item['characterized_finite']['relative_gap'])
        lp_gaps.append(max(0,primal-lp)/max(primal,1e-12))
        for row in item['rows']:
            cert=general_certificate(model,np.asarray(row['durations']),item['parameters']['coherent_fraction'])
            fidelities=[row['process_fidelity'],row['zero']['fidelity'],row['haar']['fidelity']]
            slack=min(fidelities)-cert['fidelity_floor']
            if slack<-1e-7:raise ValueError('Stronger general no-jump floor violated by saved simulation')
            stronger.append(dict(key=key,method=row['method'],stronger_fidelity_floor=cert['fidelity_floor'],
                linear_fidelity_floor=cert['trace_distance_fidelity_floor'],minimum_fidelity=min(fidelities),slack=slack))
            if 'library_certificate' in row:certificate_slack.append(min(fidelities)-row['library_certificate']['fidelity_floor'])
        if item['extra']:
            ode.append(item['extra']['independent_ode_channel_difference']);sdp.append(item['extra']['independent_half_diamond'])
    summary['numerics']=dict(unitary_error_max=max(r['unitary_error'] for r in rows),tp_error_max=max(r['tp_error'] for r in rows),
        minimum_choi_eigenvalue=min(r['minimum_choi_eigenvalue'] for r in rows),finite_mip_gap=stats(solver_gaps),
        characterized_mip_gap=stats(measured_gaps),characterized_lp_relaxation_gap=stats(lp_gaps),
        finite_nonoptimal_count=sum(not v['finite']['optimal'] for v in cases.values()),characterized_nonoptimal_count=sum(not v['characterized_finite']['optimal'] for v in cases.values()),
        ode_difference=stats(ode),sdp_half_diamond=stats(sdp),minimum_general_fidelity_slack=min(r['slack'] for r in stronger),
        minimum_library_fidelity_slack=min(certificate_slack))
    summary['certificate_coverage']=dict(total=len(stronger),linear_nonvacuous=sum(r['linear_fidelity_floor']>0 for r in stronger),
        stronger_nonvacuous=sum(r['stronger_fidelity_floor']>0 for r in stronger),
        transfer_nonvacuous=sum(r['transferred_lower_bound']>0 for r in transfers),transfer_gap=stats([r['gp_transfer_relative_gap'] for r in transfers]))
    for name,data in [('summary',summary),('transfer_validation',transfers),('stronger_certificate_validation',stronger)]:
        (DATA/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n',encoding='utf8')
    FIG.mkdir(exist_ok=True);plt.rcParams.update({'font.size':8})
    regimes=['low_drive','balanced','high_drive']
    fig,ax=plt.subplots(figsize=(3.4,2.1))
    values=[ [r['value'] for r in paired(index,[k for k in cases if index[(k,'fixed')]['regime']==regime],'published_slack','characterized_milp','process_fidelity')] for regime in regimes]
    ax.boxplot(values,tick_labels=['Low','Balanced','High']);ax.axhline(0,color='gray',lw=.8)
    ax.set_ylabel('Process fidelity change (pp)');fig.tight_layout();plot(fig,'physical_gains')
    fig,ax=plt.subplots(figsize=(3.4,2.1))
    for label,field in [('General no-jump','stronger_fidelity_floor'),('Trace-distance floor','linear_fidelity_floor')]:
        ax.scatter([r[field] for r in stronger],[r['minimum_fidelity'] for r in stronger],s=7,alpha=.25,label=label)
    ax.plot([0,1],[0,1],color='black',lw=.7);ax.set_xlabel('Conditional fidelity floor');ax.set_ylabel('Minimum simulated fidelity');ax.legend(fontsize=7);fig.tight_layout();plot(fig,'physical_certificates')
    fig,ax=plt.subplots(figsize=(3.4,2.1))
    ax.scatter([r['characterized_primal'] for r in transfers],[r['transferred_lower_bound'] for r in transfers],s=10,label='Transferred GP lower bound')
    ax.scatter([r['characterized_primal'] for r in transfers],[r['characterized_lp_lower'] for r in transfers],s=10,label='Finite-library LP lower bound')
    upper=max(r['characterized_primal'] for r in transfers);ax.plot([0,upper],[0,upper],color='black',lw=.7)
    ax.set_xlabel('Characterized MILP score');ax.set_ylabel('Certified lower score');ax.legend(fontsize=7);fig.tight_layout();plot(fig,'physical_transfer')
    lines=['# Physical and finite-library validation','', 'Declared spin-control simulation, not hardware calibration. All negative results retained.','',
        '| Regime | Mean process fidelity gain vs slack (pp) | Regressions | Mean latency change (%) |', '|---|---:|---:|---:|']
    for regime in regimes:
        r=summary['regimes'][regime]['published_slack']
        lines.append(f"| {regime} | {r['process_gain_pp']['mean']:.3f} | {r['process_regressions']} / {r['process_gain_pp']['n']} | {r['latency_change_percent']['mean']:.2f} |")
    lines += ['',json.dumps(summary['counts'],indent=2),'','The characterized MILP optimizes an operational error upper bound, not actual state/process fidelity. The GP transfer bound can be loose; LP/MILP bounds remain separate. General-noise floors can be vacuous. These findings do not establish a field-changing result or Q1 readiness.','', 'See raw cases/results, transfer_validation, stronger_certificate_validation and summary for all checks and denominators.']
    (ROOT/'docs/physical_validation_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(summary,indent=2))


def plot(fig,name):
    fig.savefig(FIG/(name+'.pdf'));fig.savefig(FIG/(name+'.png'),dpi=220);plt.close(fig)


if __name__=='__main__':main()
