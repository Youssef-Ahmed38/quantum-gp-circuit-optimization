"""Generate publication-quality figures for the 128-qubit observable study."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.titlesize': 11,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'figure.autolayout': True
})

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "evidence/observable_study/results.json"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

def main():
    data = json.loads(DATA_PATH.read_text())
    print(f"Loaded {len(data)} configurations.")

    # Figure 1: Feasible Optimization Runtime vs Analytic Screening
    sizes_all = [4, 8, 16, 32, 64, 128]
    sizes_feas = [4, 8, 16, 32]

    feas_medians = []
    screen_medians = []

    for n in sizes_feas:
        feas_cases = [d for d in data if d["qubits"] == n and d["gp"]["feasible"]]
        feas_medians.append(np.median([d["gp"]["elapsed"] * 1000 for d in feas_cases]))

    for n in sizes_all:
        infeas_cases = [d for d in data if d["qubits"] == n and not d["gp"]["feasible"]]
        if infeas_cases:
            screen_medians.append(np.median([d["gp"]["elapsed"] * 1000 for d in infeas_cases]))
        else:
            screen_medians.append(np.median([d["gp"]["elapsed"] * 1000 for d in data if d["qubits"] == n]))

    fig, ax = plt.subplots(figsize=(4.4, 3.0), dpi=300)
    ax.plot(sizes_feas, feas_medians, 'o-', color='#1f77b4', label='Feasible Constrained Solve (Clarabel)')
    ax.plot(sizes_all, screen_medians, 's--', color='#d62728', label=r'$\mathcal{O}(\operatorname{nnz}(H))$ Analytic Screening')

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(sizes_all)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel('Number of Qubits ($N$)')
    ax.set_ylabel('Execution Latency (ms)')
    ax.set_title('Optimization vs. Screening Latency (IBM Heron)')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', frameon=True, fontsize=8)

    f1_pdf = FIG_DIR / "observable_runtime_scaling.pdf"
    f1_png = FIG_DIR / "observable_runtime_scaling.png"
    plt.savefig(f1_pdf)
    plt.savefig(f1_png)
    plt.close()
    print(f"Saved Figure 1 to {f1_pdf}")

    # Figure 2: Makespan Reduction & Operational Expectation Preservation
    feasible = [d for d in data if d["gp"]["feasible"]]
    base_spans = [d["base_makespan"] * 1000 for d in feasible]
    unif_spans = [d["uniform_makespan"] * 1000 for d in feasible]
    opt_spans = [d["gp"]["makespan"] * 1000 for d in feasible]
    rounded_spans = [d["rounded"]["makespan"] * 1000 for d in feasible]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8), dpi=300)

    # Subplot A: Makespan comparison across baselines
    indices = np.arange(len(feasible))
    ax1.plot(indices, base_spans, 'k--', label=r'Nominal Base Durations ($\tau^{\mathrm{base}}_j$)', alpha=0.5)
    ax1.plot(indices, unif_spans, '-.', color='#ff7f0e', label='Optimal Uniform Baseline', alpha=0.8)
    ax1.plot(indices, opt_spans, 'b-', label='Optimal Continuous GP', alpha=0.9)
    ax1.plot(indices, rounded_spans, 'g:', label='Safe Rounded ($q=1.02$)', alpha=0.9)
    ax1.set_xlabel('Configuration Index')
    ax1.set_ylabel('Makespan (ns)')
    ax1.set_title('(a) Latency Compression')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=6.5)

    # Subplot B: Target Expectation Preservation
    targets = [d["target"] for d in feasible]
    achieved = [d["rounded"]["minimum_expectation"] for d in feasible]

    ax2.plot([0.94, 1.0], [0.94, 1.0], 'k--', label='Target Threshold', alpha=0.5)
    ax2.scatter(targets, achieved, color='#2ca02c', edgecolor='black', s=35, zorder=3, label='Achieved Expectation')
    ax2.set_xlabel('Target Expectation ($\tau$)')
    ax2.set_ylabel('Achieved Minimum Expectation $\min_k \langle S_k \\rangle$')
    ax2.set_title('(b) Certified Constraint Preservation')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left', fontsize=7)
    ax2.set_xlim(0.94, 1.0)
    ax2.set_ylim(0.94, 1.0)

    f2_pdf = FIG_DIR / "observable_makespan_fidelity.pdf"
    f2_png = FIG_DIR / "observable_makespan_fidelity.png"
    plt.savefig(f2_pdf)
    plt.savefig(f2_png)
    plt.close()
    print(f"Saved Figure 2 to {f2_pdf}")

    # Figure 3: Monte Carlo Validation against Hoeffding Bound
    sample_cases = [d for d in data if "sampling_validation" in d]
    qubits_mc = [d["qubits"] for d in sample_cases]
    errors_mc = [d["sampling_validation"]["maximum_difference"] for d in sample_cases]
    eps_mc = [d["sampling_validation"]["simultaneous_tolerance"] for d in sample_cases]

    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=300)
    width = 0.35
    x = np.arange(len(sample_cases))

    ax.bar(x - width/2, errors_mc, width, label='Empirical MC Error (16k shots)', color='#1f77b4', edgecolor='black')
    ax.bar(x + width/2, eps_mc, width, label='Hoeffding Bound ($\epsilon$ at 99.9%)', color='#ff7f0e', alpha=0.7, edgecolor='black')

    ax.set_xticks(x)
    labels = [f'{d["family"][:3]}-{d["qubits"]}Q' for d in sample_cases]
    ax.set_xticklabels(labels)
    ax.set_ylabel('Observable Absolute Error')
    ax.set_title('Forward Monte Carlo vs. Hoeffding Bound')
    ax.grid(True, linestyle=':', alpha=0.6, axis='y')
    ax.legend(loc='upper right', frameon=True)

    f3_pdf = FIG_DIR / "observable_monte_carlo_validation.pdf"
    f3_png = FIG_DIR / "observable_monte_carlo_validation.png"
    plt.savefig(f3_pdf)
    plt.savefig(f3_png)
    plt.close()
    print(f"Saved Figure 3 to {f3_pdf}")

if __name__ == "__main__":
    main()
