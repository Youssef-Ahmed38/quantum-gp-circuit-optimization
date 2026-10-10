"""Figures (figures/novelty_*.pdf) and LaTeX tables (paper/novelty_tables.tex) for C1-C8.

Reads only evidence/*.json; any study that has not been run is skipped with a
note, so the script can be rerun as studies finish.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "evidence"
FIG = ROOT / "figures"
TABLES = ROOT / "paper" / "novelty_tables.tex"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def load(path):
    p = EV / path
    return json.loads(p.read_text()) if p.exists() else None


def fmt(x, digits=4):
    return f"{x:.{digits}f}"


WIDE = {"tab:c2", "tab:c5b", "tab:c4a"}


def table(caption, label, header, rows, align=None):
    """Two-column floats span the page; tabulars only ever shrink to fit (adjustbox max width)."""
    align = align or "l" + "r" * (len(header) - 1)
    body = " \\\\\n".join(" & ".join(str(c) for c in r) for r in rows)
    env, width = ("table*", "0.95\\textwidth") if label in WIDE else ("table", "\\columnwidth")
    return (f"\\begin{{{env}}}[t]\n\\centering\n\\caption{{{caption}}}\\label{{{label}}}\n\\footnotesize\n"
            f"\\begin{{adjustbox}}{{max width={width}}}\n"
            f"\\begin{{tabular}}{{{align}}}\n\\toprule\n{' & '.join(header)} \\\\\n\\midrule\n{body} \\\\\n"
            f"\\bottomrule\n\\end{{tabular}}\n\\end{{adjustbox}}\n\\end{{{env}}}\n")


def hardware_controls(out):
    """Pre-registered control arms (H3) on chain A and chain B."""
    runs = [(n, load(f"ibm_dd_protocol/{f}")) for n, f in (("Chain A", "h3_analysis.json"), ("Chain B", "h3-b_analysis.json"))]
    runs = [(n, a) for n, a in runs if a]
    if not runs:
        return "% H3 not yet available\n"
    names = {"gp_dd": "GP-DD", "gp_dd_protect_q54": "GP-DD + protect GP-undecoupled qubit",
             "cpmg8_uniform": "Uniform CPMG-8", "xy4_like": "XY4-like"}
    rows = [[label] + [f"{a['pooled'][k]['p_all_zero']:.3f} $\\pm$ {a['pooled'][k]['shot_se']:.3f}" for _, a in runs]
            for k, label in names.items()]
    for key, label in (("primary_gp_minus_cpmg8", "GP-DD $-$ CPMG-8 (pp, primary)"),
                       ("secondary_gp_minus_protect_q54", "GP-DD $-$ protect (pp)"),
                       ("secondary_gp_minus_xy4", "GP-DD $-$ XY4 (pp)")):
        rows.append([label] + [f"{100 * a[key]['difference']:+.2f} [{100 * a[key]['ci95'][0]:+.2f}, "
                               f"{100 * a[key]['ci95'][1]:+.2f}]" for _, a in runs])
    rows.append(["Predicted order observed"] + ["yes" if a["order_matches"] else "no" for _, a in runs])
    pooled = load("ibm_dd_protocol/h3_pooled.json")
    note = ""
    if pooled:
        note = (f" Pre-registered pooling of the primary over chains: {100 * pooled['pooled_difference']:+.2f} pp, "
                f"95\\% CI [{100 * pooled['pooled_95ci'][0]:+.2f}, {100 * pooled['pooled_95ci'][1]:+.2f}].")
    layouts = "; ".join(f"{n}: qubits {', '.join(str(q) for q in a['layout'])}" for n, a in runs)
    return table("Pre-registered control arms (H3) on \\texttt{ibm\\_marrakesh}, 4 blocks $\\times$ 6000 shots per arm "
                 f"({layouts}): all-qubit return probability with shot-only s.e., and paired differences with "
                 "shot-only 95\\% intervals." + note, "tab:hw3", ["Arm / comparison"] + [n for n, _ in runs], rows,
                 "l" + "c" * len(runs))


def hardware(out):
    """Pre-registered IBM Heron decoupling test (H2) and replication (R1)."""
    runs = [("H2", load("ibm_dd_protocol/h2_analysis.json")), ("R1", load("ibm_dd_protocol/h2-r1_analysis.json"))]
    runs = [(n, a) for n, a in runs if a]
    if not runs:
        return "% Hardware analysis not yet available\n"
    names = {"gp_dd": "GP-DD", "gp_dd_correlated": "GP-DD (cross-window, identical)", "xy4_like": "XY4-like",
             "xx_qiskit_like": "XX (Qiskit-like)", "no_dd": "No DD"}
    rows = []
    for policy in names:
        row = [names[policy]]
        for _, a in runs:
            p = a["pooled"][policy]
            row.append(f"{p['p_all_zero']:.3f} $\\pm$ {p['shot_se']:.3f}")
        rows.append(row)
    rows.append(["GP-DD $-$ best standard (pp)"] + [
        f"{100 * a['primary']['gp_dd_minus_best_standard']:+.2f} [{100 * a['primary']['shot_only_95ci'][0]:+.2f}, "
        f"{100 * a['primary']['shot_only_95ci'][1]:+.2f}]" for _, a in runs])
    pooled = load("ibm_dd_protocol/h2_pooled.json")
    note = ""
    if pooled:
        note = (f" Pre-registered inverse-variance pooling: {100 * pooled['pooled_difference']:+.2f} pp, 95\\% CI "
                f"[{100 * pooled['pooled_95ci'][0]:+.2f}, {100 * pooled['pooled_95ci'][1]:+.2f}].")
    header = ["Policy"] + [f"{n} ({'3$\\times$2000' if n == 'H2' else '4$\\times$6000'} shots)" for n, _ in runs]
    return table("Pre-registered decoupling test on IBM Heron (\\texttt{ibm\\_marrakesh}, qubits 55, 54, 53, 39, 33, "
                 "32; matched 24\\,$\\mu$s makespan): probability that all six qubits return to $|0\\rangle$, with "
                 "shot-only standard errors, and the primary difference with its shot-only 95\\% interval." + note,
                 "tab:hw", header, rows, "l" + "c" * len(runs))


def c2(out):
    s = load("dd_gp_v2/summary.json") or load("dd_gp/summary.json")
    if not s:
        return "% C2 study not yet run\n"
    names = {"alap_xx_qiskit_like": "ALAP+XX", "alap_xy4_qiskit_like": "ALAP+XY4",
             "manuscript_gp_gate_only": "GP (gate only)", "gp_idle_aware_no_dd": "GP idle-aware",
             "gp_dd_fixed_durations": "GP-DD (fixed $\\tau$)", "gp_dd_joint": "GP-DD joint",
             "gp_dd_correlated_fixed_durations": "GP-DD cross-window (fixed $\\tau$)",
             "gp_dd_correlated_joint": "GP-DD cross-window joint"}
    rows = []
    cases = s["by_case"]
    main = "gp_dd_correlated_joint" if "gp_dd_correlated_joint" in next(iter(cases.values())) else "gp_dd_joint"
    fixed = "gp_dd_correlated_fixed_durations" if main == "gp_dd_correlated_joint" else "gp_dd_fixed_durations"

    def cell(e):
        return f"{1e4 * e['diff_vs_best_standard']:+.1f} $\\pm$ {1e4 * e['paired_se']:.1f}"
    for key, entry in cases.items():
        scenario, circuit = key.split("|")
        rows.append([scenario.replace("_", r"\_"), circuit.replace("_", r"\_"),
                     entry["best_standard_policy"].replace("_qiskit_like", "").replace("_", r"\_"),
                     cell(entry["manuscript_gp_gate_only"]), cell(entry[fixed]), cell(entry[main]),
                     str(entry[main]["pulses"])])
    text = table("Joint timing and decoupling allocation at matched makespan: simulated state-fidelity "
                 "change against the best standard policy (ASAP, ALAP, ALAP+XX, ALAP+XY4), in units of $10^{-4}$ "
                 "with paired standard errors over common noise realisations. Columns: the earlier gate-only GP; "
                 "cross-window GP-DD with native (fixed) durations, which is hardware-executable; and cross-window "
                 "GP-DD with duration control. Models are fitted only from simulated single-qubit spectroscopy.",
                 "tab:c2", ["Noise", "Circuit", "Best std.", "Gate-only GP", "GP-DD fixed $\\tau$", "GP-DD joint",
                            "Pulses"], rows, "llcrrrr")
    # Figure: fidelity change per case for the main policies.
    keys = list(cases)
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    width = 0.2
    for i, policy in enumerate(["manuscript_gp_gate_only", "gp_idle_aware_no_dd", fixed, main]):
        values = [1e4 * cases[k][policy]["diff_vs_best_standard"] for k in keys]
        errors = [2e4 * cases[k][policy]["paired_se"] for k in keys]
        ax.bar(np.arange(len(keys)) + (i - 1.5) * width, values, width, yerr=errors, label=names[policy], capsize=1)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels([k.replace("|", "\n") for k in keys], rotation=90, fontsize=5)
    ax.set_ylabel("Fidelity change vs best\nstandard policy ($10^{-4}$)")
    ax.legend(fontsize=6, ncol=4, frameon=False)
    fig.savefig(FIG / "novelty_c2_dd_gp.pdf")
    plt.close(fig)
    o = s["overall"]
    out.append(f"C2 ({main}): {o.get(main + '_significant_wins', o['gp_dd_joint_significant_wins'])} wins / "
               f"{o.get(main + '_significant_losses', o['gp_dd_joint_significant_losses'])} losses of {o['cases']}; "
               f"median model-rank correlation {o['median_model_rank_correlation']:.3f}")
    return text


def c3(out):
    s = load("runtime_optimal/summary.json")
    if not s:
        return "% C3 study not yet run\n"
    labels = {"no_repetition_delay": "0", "fast_reset_5us": "5", "ibm_default_250us": "250"}
    rows = [[labels[k], str(v["cases"]), f"{100 * v['median_reduction_vs_best_other']:.2f}",
             f"{100 * v['max_reduction_vs_best_other']:.2f}", f"{100 * v['median_reduction_vs_nominal']:.2f}",
             "yes" if v["stationarity_all_valid"] else "no"] for k, v in s.items() if k in labels]
    val = s.get("pec_validation", [])
    out.append(f"C3: PEC variance ratio range {min(v['variance_ratio_min'] for v in val):.3f}-"
               f"{max(v['variance_ratio_max'] for v in val):.3f}")
    return table("Runtime-optimal PEC: reduction in modelled QPU time (shots $\\times$ per-shot time) "
                 "to a fixed standard error, over 32 snapshot-anchored Clifford workloads. The best other "
                 "policy is the best of nominal, fastest, error-optimal and optimal-uniform durations.",
                 "tab:c3", ["$t_{\\rm rep}$ ($\\mu$s)", "Cases", "Median vs best (\\%)", "Max vs best (\\%)",
                            "Median vs nominal (\\%)", "KKT check"], rows)


def c4(out):
    s = load("robust_pac/summary.json")
    if not s:
        return "% C4 study not yet run\n"
    a = s["part_a"]
    rows = []
    for method, label in (("plain_gp", "GP (manuscript)"), ("dominance_gp", "GP + dominance"),
                          ("dominance_gp_plus_table7_law", "GP + dominance (+Table VII law)")):
        for side in ("inside", "outside"):
            v = a[f"{method}|{side}_hull"]
            rows.append([label, side, str(v["cases"]), str(v["worse"]), str(v["better"]),
                         f"{100 * v['median_relative_change']:+.2f}"])
    b = s["part_b"]
    out.append(f"C4: pooled coverage robust {b['pooled_robust_coverage']:.3f} vs plug-in {b['pooled_plugin_coverage']:.3f}")
    t1 = table("Baseline dominance (rerun of the misspecification stress test): error laws inside and outside the declared hull "
               "\\{inverse, matched inverse-square\\}; counts of schedules worse/better than the nominal baseline.",
               "tab:c4a", ["Method", "Laws", "Cases", "Worse", "Better", "Median change (\\%)"], rows, "llrrrr")
    rows2 = [[k.replace("_", r"\_"), str(v["trials"]), fmt(v["robust_coverage"], 3), fmt(v["plugin_coverage"], 3),
              fmt(v["median_robust_true_over_plugin_true"], 4)]
             for k, v in b.items() if isinstance(v, dict) and "trials" in v]
    t2 = table("Calibration coverage: fraction of simulated calibrations in which the certified bound "
               "covers the true model error (target 0.95), versus the plug-in prediction.", "tab:c4b",
               ["Circuit", "Trials", "Robust coverage", "Plug-in coverage", "True error ratio"], rows2)
    return t1 + t2


def c5(out):
    s = load("anisotropic/spin_locking.json")
    comp = load("anisotropic/compilation.json")
    if not s:
        return "% C5 study not yet run\n"
    rows = []
    for basis in ("X", "Z"):
        for policy in ("slow", "distributed", "tail", "dd_tail"):
            rows.append([basis, policy.replace("_", r"\_"), fmt(s["simulated_truth"][basis][policy], 3),
                         fmt(s["anisotropic_prediction"][basis][policy], 3),
                         fmt(s["isotropic_prediction"][basis][policy], 3),
                         fmt(s["heron_table_x_retrospective"][basis][policy], 3)])
    if comp:
        violations = sum(r.get("isotropic_plan_true_violations", 0) > 0 for r in comp)
        false_infeasible = sum(not r["isotropic_feasible"] and r["anisotropic_feasible"] for r in comp)
        out.append(f"C5: isotropic plans violating true targets {violations}/{len(comp)}; falsely infeasible {false_infeasible}")
    return table("Native-X policies: simulated truth, anisotropic state-aware prediction calibrated only on "
                 "generic characterisation runs, isotropic prediction, and the measured Heron values "
                 "(retrospective, not used for fitting).", "tab:c5",
                 ["Basis", "Policy", "Simulated", "Anisotropic", "Isotropic", "Heron"], rows, "llrrrr")


def c5b(out):
    s = load("dd_state_aware/summary.json")
    if not s:
        return "% State-aware ablation not yet run\n"
    rows = []
    for key, entry in s.items():
        if not isinstance(entry, dict) or "|" not in key:
            continue
        scenario, circuit = key.split("|")
        def cell(key):
            return f"{1e4 * entry[key]['mean']:+.1f} $\\pm$ {1e4 * entry[key]['paired_se']:.1f}"
        rows.append([scenario.replace("_", r"\_"), circuit.replace("_", r"\_"),
                     cell("aware_minus_gp_dd_state_blind"), cell("aware_minus_alap_xx_qiskit_like"),
                     cell("correlated_minus_alap_xx_qiskit_like"),
                     f"{entry['gp_dd_correlated']['pulses']}/{entry['alap_xx_qiskit_like']['pulses']}"])
    return table("State-aware and cross-window DD allocation under exact amplitude damping (fidelity change, "
                 "$10^{-4}$, paired s.e.): state-aware vs state-blind GP-DD; state-aware vs ALAP+XX; "
                 "cross-window (correlated) GP-DD vs ALAP+XX; pulses used (correlated/XX).", "tab:c5b",
                 ["Noise", "Circuit", "Aware$-$blind", "Aware$-$XX", "Correlated$-$XX", "Pulses"], rows,
                 "llrrrr")


def c6c7(out):
    s = load("theory_checks/summary.json")
    f = load("theory_checks/c7_frontier.json")
    if not s:
        return "% Theory checks not yet run\n"
    out.append(f"C6: {s['c6']['error_bound_violations']} error / {s['c6']['makespan_bound_violations']} makespan bound violations in {s['c6']['trials']} trials")
    if f:
        fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.2))
        for name, v in f.items():
            d = np.array(v["deadlines_us"])
            a.loglog(d / d[0], np.array(v["errors"]) / v["errors"][0], label=name.replace("_", " "))
            b.plot(d / d[0], v["dual_price"], label=name.replace("_", " "))
        a.set_xlabel("Deadline / fastest makespan")
        a.set_ylabel("Optimal error / error at fastest")
        b.set_xlabel("Deadline / fastest makespan")
        b.set_ylabel("Fidelity price $\\nu^*$")
        a.legend(fontsize=6, frameon=False)
        fig.savefig(FIG / "novelty_c7_frontier.pdf")
        plt.close(fig)
    return ""


def c8(out):
    s = load("scale/summary.json")
    if not s:
        return "% Scale study not yet run\n"
    rows = [[str(r["qubits"]), str(r["gates"]), r.get("ip_status", "-"), f"{r['elapsed_s']:.1f}",
             f"{r['relative_gap']:.1e}"] for r in s["rows"] if r["kind"] == "scale"]
    out.append(f"C8: empirical exponent {s['empirical_time_exponent_in_gates']:.2f}")
    return table("Large-scale solves on random circuit DAGs with balanced makespan weight: sparse "
                 "interior point (Clarabel) time, and the independent bound (IP primal $-$ Frank--Wolfe dual) / "
                 "IP primal after 500 path-flow steps.", "tab:c8",
                 ["Qubits", "Gates", "IP status", "IP time (s)", "Indep. gap bound"], rows)


def numbers():
    """\\newcommand macros used in the manuscript text; 'TBD' until a study has run."""
    m = {}
    s = load("dd_gp_v2/summary.json")
    if s:
        o = s["overall"]
        m.update(CTwoWins=o["gp_dd_correlated_joint_significant_wins"],
                 CTwoLosses=o["gp_dd_correlated_joint_significant_losses"], CTwoCases=o["cases"],
                 CTwoMedian=f"{1e4 * o['gp_dd_correlated_joint_median_diff_vs_best_standard']:+.1f}",
                 CTwoRank=f"{o['median_model_rank_correlation']:.2f}",
                 CTwoFixedWins=o["gp_dd_correlated_fixed_durations_significant_wins"],
                 CTwoFixedLosses=o["gp_dd_correlated_fixed_durations_significant_losses"],
                 CTwoGateWins=o["manuscript_gp_gate_only_significant_wins"],
                 CTwoGateLosses=o["manuscript_gp_gate_only_significant_losses"])
    s = load("dd_gp/summary.json")
    if s:
        o = s["overall"]
        m.update(CTwoVoneWins=o["gp_dd_joint_significant_wins"], CTwoVoneLosses=o["gp_dd_joint_significant_losses"],
                 CTwoVoneRank=f"{o['median_model_rank_correlation']:.2f}")
    s = load("runtime_optimal/summary.json")
    if s:
        m.update(CThreeZero=f"{100 * s['no_repetition_delay']['median_reduction_vs_best_other']:.0f}",
                 CThreeZeroMax=f"{100 * s['no_repetition_delay']['max_reduction_vs_best_other']:.0f}",
                 CThreeFive=f"{100 * s['fast_reset_5us']['median_reduction_vs_best_other']:.1f}",
                 CThreeIBM=f"{100 * s['ibm_default_250us']['median_reduction_vs_best_other']:.3f}")
    s = load("robust_pac/summary.json")
    if s:
        a, b = s["part_a"], s["part_b"]
        m.update(CFourRobust=f"{b['pooled_robust_coverage']:.3f}", CFourPlugin=f"{b['pooled_plugin_coverage']:.3f}",
                 CFourInside=a["dominance_gp|inside_hull"]["worse"], CFourInsideCases=a["dominance_gp|inside_hull"]["cases"],
                 CFourPlainInside=a["plain_gp|inside_hull"]["worse"],
                 CFourPlainOutside=a["plain_gp|outside_hull"]["worse"],
                 CFourOutsideCases=a["plain_gp|outside_hull"]["cases"],
                 CFourExtendedOutside=a["dominance_gp_plus_table7_law|outside_hull"]["worse"],
                 CFourTrials=sum(v["trials"] for v in b.values() if isinstance(v, dict) and "trials" in v))
        rows = load("robust_pac/part_b_coverage.json") or []
        if rows:
            ratio = np.median([r["robust_certificate"] / r["robust_true"] for r in rows])
            m.update(CFourSlack=f"{100 * (ratio - 1):.0f}")
    comp = load("anisotropic/compilation.json")
    if comp:
        m.update(CFiveViolations=sum(r.get("isotropic_plan_true_violations", 0) > 0 for r in comp),
                 CFiveFalse=sum(not r["isotropic_feasible"] and r["anisotropic_feasible"] for r in comp),
                 CFiveCases=len(comp))
    s = load("theory_checks/summary.json")
    if s:
        m.update(CSixTrials=s["c6"]["trials"], CSixViolations=s["c6"]["error_bound_violations"] + s["c6"]["makespan_bound_violations"])
    s = load("scale/summary.json")
    if s:
        big = max((r for r in s["rows"] if r["kind"] == "scale"), key=lambda r: r["gates"])
        m.update(CEightGates=f"{big['gates']:,}".replace(",", "{,}"), CEightTime=f"{big['elapsed_s']:.1f}",
                 CEightGap=f"{big['relative_gap']:.0e}")
    names = ["CTwoWins", "CTwoLosses", "CTwoCases", "CTwoRank", "CTwoMedian", "CTwoFixedWins", "CTwoFixedLosses",
             "CTwoGateWins", "CTwoGateLosses", "CTwoVoneWins", "CTwoVoneLosses", "CTwoVoneRank", "CThreeZero", "CThreeZeroMax", "CThreeFive",
             "CThreeIBM", "CFourRobust", "CFourPlugin", "CFourInside", "CFourInsideCases", "CFourPlainInside",
             "CFourPlainOutside", "CFourOutsideCases", "CFourExtendedOutside", "CFourTrials", "CFourSlack",
             "CFiveViolations", "CFiveFalse", "CFiveCases", "CSixTrials", "CSixViolations", "CEightGates",
             "CEightTime", "CEightGap"]
    return "".join(f"\\newcommand{{\\{k}}}{{{m.get(k, 'TBD')}}}\n" for k in names)


def main():
    FIG.mkdir(exist_ok=True)
    notes = []
    tex = "% Auto-generated by benchmarks/write_novelty_outputs.py; do not edit by hand.\n"
    for part in (c2, c3, c4, c5, c5b, c6c7, c8, hardware, hardware_controls):
        tex += part(notes)
    TABLES.write_text(tex, encoding="utf-8")
    # One file per table so each float can sit next to the paragraph that discusses it.
    import re
    for block in re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}\n", tex, flags=re.S):
        label = re.search(r"\\label\{([^}]+)\}", block).group(1)
        (TABLES.parent / f"novelty_{label.replace(':', '_')}.tex").write_text(block, encoding="utf-8")
    (TABLES.parent / "novelty_numbers.tex").write_text(
        "% Auto-generated; do not edit by hand.\n" + numbers(), encoding="utf-8")
    print("\n".join(notes))


if __name__ == "__main__":
    main()
