"""Summarize the model-attainable large-scale study."""
from pathlib import Path
from collections import defaultdict
import json
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "evidence/feasible_scale/results.json"


def main():
    rows = json.loads(DATA.read_text())
    assert len(rows) == 24
    assert len({(r["family"], r["qubits"], r["seed"], r["alpha"]) for r in rows}) == len(rows)
    groups = defaultdict(list)
    for row in rows:
        assert row["lower_duration_expectation"] < row["target"] < row["coordinatewise_min_hazard_expectation"]
        assert row["achieved_min_expectation"] >= row["target"] - 1e-6
        groups[row["qubits"]].append(row)
    summary = []
    for n in sorted(groups):
        subset = groups[n]
        reductions = [100 * (1 - r["gp_makespan_us"] / r["uniform_makespan_us"]) for r in subset]
        summary.append({
            "qubits": n, "n": len(subset),
            "timed_gates_range": [min(r["timed_gates"] for r in subset), max(r["timed_gates"] for r in subset)],
            "target_range": [min(r["target"] for r in subset), max(r["target"] for r in subset)],
            "median_gp_ms": 1000 * statistics.median(r["gp_elapsed_s"] for r in subset),
            "median_reduction_vs_optimal_uniform_percent": statistics.median(reductions),
            "reduction_range_percent": [min(reductions), max(reductions)],
            "max_formulation_relative_difference": max(r["formulation_relative_difference"] for r in subset),
        })
    out = ROOT / "evidence/feasible_scale/summary.json"
    out.write_text(json.dumps({"rows": len(rows), "by_size": summary}, indent=2) + "\n")
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Feasible model-based solves. Targets are chosen separately per instance from its assumed attainable range; times are single-run wall-clock observations, not hardware timings.}",
        r"\label{tab:feasible_scale}",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"Qubits & $n$ & Gates & GP ms & Gain vs. uniform \\",
        r"\midrule",
    ]
    for item in summary:
        lo, hi = item["timed_gates_range"]
        tex.append(f'{item["qubits"]} & {item["n"]} & {lo}--{hi} & {item["median_gp_ms"]:.1f} & {item["median_reduction_vs_optimal_uniform_percent"]:.1f}\\% \\\\')
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (ROOT / "docs/feasible_scale_table.tex").write_text("\n".join(tex) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for family, marker in (("graph", "o"), ("scrambled", "s")):
        subset = [r for r in rows if r["family"] == family]
        axes[0].scatter([r["qubits"] for r in subset], [1000 * r["gp_elapsed_s"] for r in subset],
                        label=family, marker=marker, alpha=.75)
        axes[1].scatter([r["qubits"] for r in subset],
                        [100 * (1 - r["gp_makespan_us"] / r["uniform_makespan_us"]) for r in subset],
                        marker=marker, alpha=.75)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("GP solve wall time (ms)")
    axes[0].legend(frameon=False)
    axes[1].set_ylabel("Makespan reduction vs. optimal uniform (%)")
    for ax in axes:
        ax.set_xticks([32, 64, 128])
        ax.set_xlabel("Qubits")
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(ROOT / "figures/feasible_scale.pdf")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
