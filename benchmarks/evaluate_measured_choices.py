"""Retrospective measured-library decision audit; no hardware execution."""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qcgp.measured_choice import evaluate_family


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evidence/measured_duration/measured_points_and_predictions.csv"
OUTPUT = ROOT / "evidence/measured_duration/choice_audit.json"
TABLE = ROOT / "docs/measured_choice_table.tex"


def main():
    with SOURCE.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups = {}
    for row in rows:
        groups.setdefault(row["pulse"], []).append(row)
    results = {name: evaluate_family(group) for name, group in groups.items()}
    payload = dict(
        source_doi="10.5281/zenodo.13808824",
        all_free_local=True, hardware_jobs=0,
        status="Exploratory retrospective comparison; PCHIP considered after earlier law results",
        decision_space="Same withheld measured durations for every method within each pulse family",
        interpretation="Measurement error bars exclude model discrepancy; train-only RMS inflation is a diagnostic, not a confidence interval",
        results=results,
    )
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    lines = [r"\begin{table}[t]", r"\centering\caption{Exploratory measured finite-library duration choices. Entries are measured regret against the lowest-error withheld option, in $10^{-5}$ RB error per gate. The empirical interpolator was examined after earlier law results; these are not prospective gains.}\label{tab:choice_audit}",
             r"\begin{tabular}{lccc}\hline", r"Pulse family & $1/t+t$ & $1/t^2+t$ & PCHIP\\\hline"]
    for name, result in results.items():
        methods = result["methods"]
        vals = [methods[key]["measured_regret_vs_best_test"] * 1e5
                for key in ("inverse_linear", "inverse_square_linear", "pchip")]
        lines.append(f"{name.replace('_', ' ')} & {vals[0]:.3f} & {vals[1]:.3f} & {vals[2]:.3f}\\\\")
    lines.extend([r"\hline\end{tabular}", r"\end{table}"])
    TABLE.write_text("\n".join(lines) + "\n")
    names = list(results)
    policies = [("inverse_linear", "$1/t+t$"),
                ("inverse_square_linear", "$1/t^2+t$"), ("pchip", "PCHIP")]
    fig, ax = plt.subplots(figsize=(7.8, 3.7), constrained_layout=True)
    x = np.arange(len(names))
    for k, (key, label) in enumerate(policies):
        values = [results[name]["methods"][key]["measured_regret_vs_best_test"] * 1e5
                  for name in names]
        ax.bar(x + (k - 1) * .24, values, width=.23, label=label)
    ax.set_xticks(x, [name.replace("_", "\n") for name in names])
    ax.set_ylabel("Measured regret ($10^{-5}$ RB error/gate)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    fig.savefig(ROOT / "figures/measured_choice.pdf")
    fig.savefig(ROOT / "figures/measured_choice.png", dpi=180)
    plt.close(fig)
    print(json.dumps({name: {method: round(record["measured_regret_vs_best_test"], 9)
                             for method, record in result["methods"].items()}
                      for name, result in results.items()}, indent=2))


if __name__ == "__main__":
    main()
