"""Build a self-contained Overleaf project for the revised master manuscript.

Layout: main.tex (the revised quantum_circuit_optimization_master_paper.tex),
tables/*.tex, figures/*.pdf. The feasible-scale table is regenerated from its
evidence so that it reports per-alpha reductions instead of a pooled median.
"""
import json
import shutil
import statistics
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO.parent
DEST = OUTPUTS / "overleaf_master_paper"
TABLES = ["observable_tables", "physical_tables", "law_comparison_table", "measured_choice_table"]
FIGURES = ["observable_runtime_scaling", "observable_makespan_fidelity", "feasible_scale", "physical_gains",
           "physical_certificates", "measured_duration", "schedule_noise", "direct_matched",
           "ibm_marrakesh_x_duration_holdout_2026-10-10", "observable_monte_carlo_validation",
           "novelty_c2_dd_gp", "novelty_c7_frontier"]


def feasible_scale_table():
    rows = json.loads((REPO / "evidence/feasible_scale/results.json").read_text())
    lines = []
    for n in (32, 64, 128):
        subset = [r for r in rows if r["qubits"] == n]
        gates = [r["timed_gates"] for r in subset]
        ms = statistics.median(r["gp_elapsed_s"] for r in subset) * 1e3
        gain = {a: statistics.median(100 * (1 - r["gp_makespan_us"] / r["uniform_makespan_us"])
                                     for r in subset if r["alpha"] == a) for a in (0.5, 0.8)}
        lines.append(f"{n} & {len(subset)} & {min(gates)}--{max(gates)} & {ms:.1f} & "
                     f"{gain[0.5]:.1f}\\% & {gain[0.8]:.1f}\\% \\\\")
    return ("\\begin{table}[t]\n\\centering\n\\caption{Feasible model-based solves. Targets are chosen separately per "
            "instance from its assumed attainable range; times are single-run wall-clock observations, not hardware "
            "timings. Makespan reduction against the optimal uniform policy is reported separately for each assumed "
            "inverse-duration share $\\alpha$ (median over families and seeds).}\n\\label{tab:feasible_scale}\n"
            "\\footnotesize\n\\begin{adjustbox}{max width=\\columnwidth}\n\\begin{tabular}{rrrrrr}\n\\toprule\n"
            "Qubits & $n$ & Gates & GP ms & Gain, $\\alpha=0.5$ & Gain, $\\alpha=0.8$ \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n")


def main():
    if DEST.exists():
        shutil.rmtree(DEST)
    (DEST / "tables").mkdir(parents=True)
    (DEST / "figures").mkdir()
    shutil.copy(OUTPUTS / "quantum_circuit_optimization_master_paper.tex", DEST / "main.tex")
    for name in TABLES:
        shutil.copy(REPO / "docs" / f"{name}.tex", DEST / "tables" / f"{name}.tex")
    (DEST / "tables" / "feasible_scale_table.tex").write_text(feasible_scale_table(), encoding="utf-8")
    for name in ("novelty_numbers",):
        shutil.copy(REPO / "paper" / f"{name}.tex", DEST / "tables" / f"{name}.tex")
    for path in (REPO / "paper").glob("novelty_tab_*.tex"):
        shutil.copy(path, DEST / "tables" / path.name)
    for name in FIGURES:
        shutil.copy(REPO / "figures" / f"{name}.pdf", DEST / "figures" / f"{name}.pdf")
    archive = OUTPUTS / "quantum_circuit_optimization_master_paper_overleaf.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(DEST.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(DEST).as_posix())
    print(archive)
    print((DEST / "tables" / "feasible_scale_table.tex").read_text())


if __name__ == "__main__":
    main()
