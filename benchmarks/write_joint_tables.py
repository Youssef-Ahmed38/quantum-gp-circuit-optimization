"""Generate manuscript numbers/tables only from measured analysis output."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]


def scientific(x):
    a,b=f'{x:.2e}'.split('e')
    return '$'+a+r'\times10^{'+str(int(b))+'}$'


def main():
    s=json.loads((ROOT/'evidence/joint_timing/summary.json').read_text())
    tests=(ROOT/'evidence/joint_timing/tests.txt').read_text()
    resultline=[line for line in tests.splitlines() if ' passed' in line][-1]
    count=int(resultline.split(' passed')[0].split()[-1])
    c=s['counts'];families=s['families']
    means=[x['risk_reduction_vs_uniform']['mean'] for x in families.values()]
    values=dict(ConfigurationCount=c['configurations'],StateCount=c['state_channel_checks'],PulseCount=c['pulse_checks'],
        FlowCount=c['pathflow_checks'],ProcessCount=c['entangled_process_checks'],TestCount=count,
        UniformMin=f'{min(means):.2f}',UniformMax=f'{max(means):.2f}',
        SolverGap=scientific(s['max_gp_convex_objective_gap']),FlowGap=scientific(s['flow_max_relative_gap']),
        FlowResidual=scientific(s['flow_max_lp_residual']),QuantizationGap=scientific(s['quantization']['max_absolute_risk_change']),
        RobustGain=f"{s['robust_worst_risk_reduction']['mean']:.3f}",BreachCount=s['coherent_counterexamples']['breaches'],
        CounterexampleCount=s['coherent_counterexamples']['tested'],ParetoCount=c['pareto_points'],OuterCount=c['outer_cases'],
        OuterGain=f"{s['finite_search']['improvement']['mean']:.2f}",OuterChanges=s['finite_search']['candidate_choice_changes'],
        OuterGap=scientific(s['finite_search']['max_direct_gap']))
    (ROOT/'docs/joint_results.tex').write_text('\n'.join('\\newcommand{\\'+key+'}{'+str(value)+'}' for key,value in values.items())+'\n',encoding='utf8')
    labels={'qft':'QFT','grover':'Grover','vqe_ansatz':'VQE ansatz','qaoa':'QAOA'}
    lines=[r'\begin{table}[t]',r'\caption{Paired Risk and Simulated Fidelity Changes}',r'\label{tab:results}',r'\centering\small',r'\begin{tabular}{lrrr}',r'\toprule',r'Family & Risk (\%) & Stratum interval & Fidelity (pp)\\',r'\midrule']
    for f,x in families.items():
        r=x['risk_reduction_vs_uniform'];g=x['simulated_fidelity_gain_percentage_points']
        lines.append(f"{labels[f]} & {r['mean']:.2f} & [{r['interval']['low']:.2f}, {r['interval']['high']:.2f}] & {g['mean']:.3f}"+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\vspace{2pt}',r'\parbox{.95\columnwidth}{\footnotesize Risk: reduction versus optimized uniform scaling, 24 configurations per family. Fidelity: joint minus base timing, averaged over zero/Haar inputs, 12 small-circuit configurations per family. Intervals: exploratory size/topology-stratum bootstrap; not hardware inference.}',r'\end{table}',r'\begin{table}[t]',r'\caption{Latency Tradeoff and Certificate Tightness}',r'\label{tab:tradeoff}',r'\centering\small',r'\begin{tabular}{lrr}',r'\toprule',r'Family & Mean latency change (\%) & Minimum floor\\',r'\midrule']
    for f,x in families.items():lines.append(f"{labels[f]} & {x['latency_change_vs_fixed']['mean']:.2f} & "+scientific(x['fidelity_floor']['minimum'])+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}',r'\begin{table}[t]',r'\caption{Larger Timing-Only Cases: Single-Call CPU Measurements}',r'\label{tab:scale}',r'\centering\small',r'\begin{tabular}{lrrrr}',r'\toprule',r'Family & Qubits & Gates & Joint (s) & Convex (s)\\',r'\midrule']
    for r in s['scale']:lines.append(f"{labels[r['family']]} & {r['qubits']} & {r['gates']} & {r['seconds']:.3f} & {r['direct_seconds']:.3f}"+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\vspace{2pt}',r'\parbox{.95\columnwidth}{\footnotesize Joint method recorded per case: small risk-only instances use logarithmic cones; larger instances use path-flow. No state simulation or full-unitary check at these sizes.}',r'\end{table}']
    (ROOT/'docs/joint_tables.tex').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('Generated measured manuscript commands and three tables')


if __name__=='__main__':main()
