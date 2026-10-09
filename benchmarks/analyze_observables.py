"""Analysis and LaTeX table generation for the 128-qubit observable study."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "evidence/observable_study/results.json"

def main():
    if not DATA_PATH.exists():
        print(f"Data file not found at {DATA_PATH}")
        return

    data = json.loads(DATA_PATH.read_text())
    print(f"Loaded {len(data)} configurations from {DATA_PATH}")

    # Breakdown by qubit count
    sizes = sorted(list(set(d["qubits"] for d in data)))
    families = sorted(list(set(d["family"] for d in data)))

    table_rows = []

    for n in sizes:
        subset = [d for d in data if d["qubits"] == n]
        feasible = [d for d in subset if d["gp"]["feasible"]]
        infeasible = [d for d in subset if not d["gp"]["feasible"]]

        f_times = [d["gp"]["elapsed"] * 1000 for d in feasible]
        inf_times = [d["gp"]["elapsed"] * 1000 for d in infeasible]

        gaps = [d["gp"]["relative_gap"] for d in feasible if d["gp"]["relative_gap"] is not None]
        mean_gap = f"{np.mean(gaps):.2e}" if gaps else "N/A"

        diffs = [d["formulation_relative_difference"] for d in feasible if "formulation_relative_difference" in d]
        max_diff = f"{np.max(diffs):.2e}" if diffs else "N/A"

        med_feas = f"{np.median(f_times):.1f}" if f_times else "N/A"
        med_screen = f"{np.median(inf_times):.2f}" if inf_times else "N/A"

        timed_gates = [d["timed_gates"] for d in subset]
        avg_gates = int(np.mean(timed_gates))

        table_rows.append({
            "qubits": n,
            "total": len(subset),
            "feasible": len(feasible),
            "infeasible": len(infeasible),
            "avg_gates": avg_gates,
            "med_feas_ms": med_feas,
            "med_screen_ms": med_screen,
            "mean_gap": mean_gap,
            "max_diff": max_diff
        })

    # Generate LaTeX table
    tex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Observable-Constrained Scheduling Across Scales (IBM Heron Architecture). Feasible optimization runtime is reported separately from $\mathcal{O}(\operatorname{nnz}(H))$ analytic screening.}",
        r"\label{tab:observables}",
        r"\begin{tabular}{rccccccc}",
        r"\hline",
        r"Qubits & Gates & Feas. & Infeas. & Feas. Solve (ms) & Screen (ms) & Dual Gap & $|\Delta_{\mathrm{form}}|$\\ \hline"
    ]

    for r in table_rows:
        tex_lines.append(
            f"{r['qubits']} & {r['avg_gates']} & {r['feasible']} & {r['infeasible']} & "
            f"{r['med_feas_ms']} & {r['med_screen_ms']} & {r['mean_gap']} & {r['max_diff']} \\\\"
        )

    tex_lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table*}"
    ])

    table_tex = "\n".join(tex_lines) + "\n"
    out_table = ROOT / "docs/observable_tables.tex"
    out_table.write_text(table_tex, encoding="utf-8")
    print(f"Saved LaTeX table to {out_table}")

    # Save summary json
    summary = {
        "configurations": len(data),
        "sizes": sizes,
        "families": families,
        "table_rows": table_rows,
        "overall": {
            "total_feasible": sum(r["feasible"] for r in table_rows),
            "total_infeasible": sum(r["infeasible"] for r in table_rows),
            "median_gp_ms": float(np.median([d["gp"]["elapsed"] * 1000 for d in data])),
            "median_cvx_ms": float(np.median([d["direct"]["elapsed"] * 1000 for d in data])),
            "max_formulation_diff": float(np.max([d["formulation_relative_difference"] for d in data if d["gp"]["feasible"]])),
            "all_safe_rounded_verified": all(
                d["rounded"]["minimum_expectation"] >= d["target"] - 1e-6
                for d in data if d["safe_gp"]["feasible"]
            )
        }
    }

    out_summary = ROOT / "evidence/observable_study/summary.json"
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary to {out_summary}")

if __name__ == "__main__":
    main()
