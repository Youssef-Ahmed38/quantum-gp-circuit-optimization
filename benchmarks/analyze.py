"""Validate raw datasets, compute clustered summaries, and generate plots."""
from pathlib import Path
from collections import defaultdict, Counter
import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1]
FAMILIES=["qft","grover","vqe_ansatz","qaoa"]
LABELS={"qft":"QFT","grover":"Grover","vqe_ansatz":"VQE ansatz","qaoa":"QAOA"}


def clustered_summary(rows, metric):
    groups=defaultdict(list)
    for r in rows:
        if r[metric] is not None:
            groups[(r["qubits"],r["topology"])].append(float(r[metric]))
    values=np.array([np.mean(v) for _,v in sorted(groups.items())])
    rng=np.random.default_rng(20261009)
    resamples=rng.choice(values,size=(5000,len(values)),replace=True).mean(axis=1)
    return dict(mean=float(values.mean()),ci_low=float(np.quantile(resamples,.025)),
                ci_high=float(np.quantile(resamples,.975)),clusters=len(values),
                observations=sum(map(len,groups.values())))


def savefig(fig,name):
    folder=ROOT/"figures";folder.mkdir(exist_ok=True)
    fig.savefig(folder/f"{name}.pdf",bbox_inches="tight")
    fig.savefig(folder/f"{name}.png",dpi=300,bbox_inches="tight")
    plt.close(fig)


def main():
    evidence=ROOT/"evidence"
    raw=json.loads((evidence/"candidates.json").read_text(encoding="utf8"))
    pairs=json.loads((evidence/"comparisons.json").read_text(encoding="utf8"))
    baselines=json.loads((evidence/"baselines.json").read_text(encoding="utf8"))
    ml=json.loads((evidence/"ml_baseline.json").read_text(encoding="utf8"))
    selections=json.loads((evidence/"selections.json").read_text(encoding="utf8"))
    keys={(r["case"],r["candidate"]) for r in raw}
    assert len(keys)==len(raw) and len(raw)==4*len(pairs)
    assert len({(r["case"],r["method"]) for r in baselines})==len(baselines)
    assert len(baselines)==6*len(pairs)
    assert len({r["case"] for r in ml})==len(pairs)
    for r in raw+baselines:
        for k in ("gates","depth","makespan","error_loss","unitary_max_error"):
            assert np.isfinite(r[k]) and r[k]>=0, (r["case"],k)
        assert r["unitary_max_error"]<=1e-8
        assert (r["simulated_fidelity"] is None)==(r["qubits"]>4)
    assert all(r["matched_hybrid_selection_equal"] for r in pairs)
    for r in raw:
        assert r["gp_relative_gap"]<=1e-5
        assert r["gp_violation"]<=1e-5*max(1,r["makespan"])
    summary={"cases":len(pairs),"candidates":len(raw),"baseline_circuits":len(baselines),
        "noise_candidates":sum(r["simulated_fidelity"] is not None for r in raw),
        "noise_baselines":sum(r["simulated_fidelity"] is not None for r in baselines),
        "unitary_max_error":max(r["unitary_max_error"] for r in raw+baselines),
        "gp_max_relative_gap":max(r["gp_relative_gap"] for r in raw),
        "lp_max_absolute_gap":max(abs(r["lp_makespan"]-r["makespan"]) for r in raw),
        "gp_max_violation":max(r["gp_violation"] for r in raw),
        "gp_solver_counts":dict(Counter(r["gp_solver"] for r in raw)),
        "gp_status_counts":dict(Counter(r["gp_status"] for r in raw)),
        "matched_selection_equal_cases":sum(r["matched_hybrid_selection_equal"] for r in pairs),
        "median_gp_seconds":float(np.median([r["gp_seconds"] for r in raw])),
        "median_cp_seconds":float(np.median([r["cp_seconds"] for r in raw])),
        "median_lp_seconds":float(np.median([r["lp_seconds"] for r in raw])),
        "family_results":{},"pipeline_results":{},
        "data_cleaning":"No exclusions, imputations or outlier removal. Exact-key uniqueness checked.",
        "bootstrap":"5000 resamples of size/topology stratum means, seed 20261009; exploratory",
        "paid_services_used":False,"hardware_jobs":0}
    for family in FAMILIES:
        rows=[r for r in pairs if r["family"]==family]
        assert len(rows)==24
        summary["family_results"][family]={k:clustered_summary(rows,k) for k in
            ("depth_reduction_pct","gates_reduction_pct","makespan_reduction_pct",
             "score_improvement","simulated_fidelity_gain_pp")}
    reference={r["case"]:r for r in raw if r["candidate"]==0}
    chosen={r["case"]:r["chosen_candidate"] for r in pairs}
    bykey={(r["case"],r["candidate"]):r for r in raw}
    pipeline_rows=defaultdict(list)
    for r in baselines:
        pipeline_rows[r["method"]].append(r)
    pipeline_rows["hybrid_4seeds"]=[bykey[(case,k)] for case,k in chosen.items()]
    pipeline_rows["matched_4seeds"]=pipeline_rows["hybrid_4seeds"]
    pipeline_rows["ml_seed_predictor"]=ml
    for name,rows in pipeline_rows.items():
        reductions={k:[100*(reference[r["case"]][k]-r[k])/reference[r["case"]][k]
                       for r in rows] for k in ("depth","gates","makespan")}
        summary["pipeline_results"][name]={k:float(np.mean(v)) for k,v in reductions.items()}
    summary["ml_mean_score"]=float(np.mean([r["normalized_score"] for r in ml]))
    summary["oracle_mean_score"]=float(np.mean([r["oracle_score"] for r in ml]))
    summary["ml_choice_accuracy_pct"]=100*float(np.mean([r["predicted_candidate"]==r["oracle_candidate"] for r in ml]))
    noise=[r for r in raw if r["simulated_fidelity"] is not None]
    summary["proxy_state_fidelity_mean_absolute_gap"]=float(np.mean([abs(r["survival_proxy"]-r["simulated_fidelity"]) for r in noise]))
    summary["density_max_trace_error"]=max(r["density_trace_error"] for r in noise)
    summary["density_min_eigenvalue"]=min(r["density_min_eigenvalue"] for r in noise)
    choices=defaultdict(set)
    for r in selections:choices[r["case"]].add(r["chosen_candidate"])
    summary["weight_changed_cases"]=sum(len(v)>1 for v in choices.values())
    summary["raw_gp_ranking_differences"]=sum(r["raw_gp_candidate"]!=r["matched_candidate"]
        for r in selections if r["weight_scheme"]=="balanced")
    # Independent aggregate recomputation from equally sized strata equals the simple mean.
    for family in FAMILIES:
        rows=[r for r in pairs if r["family"]==family]
        direct=sum(r["score_improvement"] for r in rows)/len(rows)
        assert abs(direct-summary["family_results"][family]["score_improvement"]["mean"])<1e-10
    (evidence/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf8")
    with (evidence/"family_summary.csv").open("w",newline="",encoding="utf8") as f:
        w=csv.writer(f);w.writerow(["family","metric","mean","ci_low","ci_high","clusters","observations"])
        for family,metrics in summary["family_results"].items():
            for metric,s in metrics.items():w.writerow([family,metric,*[s[k] for k in ("mean","ci_low","ci_high","clusters","observations")]])
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    fig,ax=plt.subplots(figsize=(3.4,2.4))
    for metric,label,color in [("gp_seconds","GP","#1b4f72"),("lp_seconds","LP","#a04000"),("cp_seconds","Critical path","#196f3d")]:
        sizes=sorted({r["qubits"] for r in raw});med=[];lo=[];hi=[]
        for n in sizes:
            vals=[r[metric] for r in raw if r["qubits"]==n]
            med.append(np.median(vals));lo.append(np.quantile(vals,.25));hi.append(np.quantile(vals,.75))
        ax.plot(sizes,med,"o-",label=label,color=color);ax.fill_between(sizes,lo,hi,color=color,alpha=.15)
    ax.set_yscale("log");ax.set_xticks(sizes);ax.set_xlabel("Qubits");ax.set_ylabel("Scheduling time (s)");ax.legend();fig.tight_layout();savefig(fig,"scheduler_timing")
    fig,ax=plt.subplots(figsize=(3.4,2.4))
    x=np.arange(4)
    for j,(metric,label,color) in enumerate([("depth_reduction_pct","Depth","#1b4f72"),("gates_reduction_pct","Gates","#a04000"),("score_improvement","Score","#196f3d")]):
        s=[summary["family_results"][fam][metric] for fam in FAMILIES];means=np.array([v["mean"] for v in s])
        yerr=np.array([means-[v["ci_low"] for v in s],[v["ci_high"] for v in s]-means])
        ax.errorbar(x+(j-1)*.13,means,yerr=yerr,fmt="o",capsize=3,color=color,label=label)
    ax.axhline(0,color="black",linewidth=.8);ax.set_xticks(x,["QFT","Grover","VQE","QAOA"]);ax.set_ylabel("Paired reduction (%)");ax.legend(fontsize=8);fig.tight_layout();savefig(fig,"search_improvements")
    fig,ax=plt.subplots(figsize=(3.4,2.6))
    for family in FAMILIES:
        subset=[r for r in noise if r["family"]==family]
        ax.scatter([r["survival_proxy"] for r in subset],[r["simulated_fidelity"] for r in subset],s=12,alpha=.5,label=LABELS[family])
    ax.plot([0,1],[0,1],"k--",linewidth=.8);ax.set(xlim=(0,1),ylim=(0,1),xlabel="Gate-survival proxy",ylabel="Simulated state fidelity");ax.legend(fontsize=7);fig.tight_layout();savefig(fig,"noise_proxy")
    fig,ax=plt.subplots(figsize=(3.4,3.0))
    names=list(summary["pipeline_results"])
    vals=[summary["pipeline_results"][name]["gates"] for name in names]
    readable={"qiskit_l0":"Qiskit L0","qiskit_l1":"Qiskit L1","qiskit_l2":"Qiskit L2","qiskit_l3":"Qiskit L3","tket_common_route":"TKET + route","cirq_common_route":"Cirq + route","hybrid_4seeds":"Hybrid (4)","matched_4seeds":"Matched (4)","ml_seed_predictor":"Learning"}
    ax.barh([readable[n] for n in names],vals,color="#1b4f72");ax.axvline(0,color="black",linewidth=.8);ax.set_xlabel("Gate-count reduction (%)");ax.invert_yaxis();fig.tight_layout();savefig(fig,"baseline_comparison")
    fig,ax=plt.subplots(figsize=(6.5,2.2));ax.axis("off")
    texts=["Logical circuit","Unitary-checked\ncandidates","GP schedule +\nLP/DAG checks","Per-candidate\nnoise simulation","Finite-set\nselection"]
    for i,text in enumerate(texts):
        left=i*1.3
        ax.add_patch(FancyBboxPatch((left,.5),1.1,.8,boxstyle="round,pad=0.05",facecolor="#eaf2f8",edgecolor="#1b4f72"))
        ax.text(left+.55,.9,text,ha="center",va="center",fontsize=8)
        if i<4:ax.annotate("",xy=(left+1.25,.9),xytext=(left+1.1,.9),arrowprops=dict(arrowstyle="->",color="#1b4f72"))
    ax.set(xlim=(-.1,6.5),ylim=(.2,1.6));fig.tight_layout();savefig(fig,"workflow")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
