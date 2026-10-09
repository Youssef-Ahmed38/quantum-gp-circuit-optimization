"""Run the complete exploratory benchmark and preserve every candidate/result."""
from pathlib import Path
import argparse
import csv
import importlib.metadata
import json
import platform
import time
import numpy as np
from qiskit import qpy
from qcgp.core import (benchmark, topology, compile_candidate, candidate_metrics,
    critical_path, gp_schedule, lp_schedule, unitary_error, noisy_fidelity,
    normalized_score, certified_selection)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "evidence")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    (out / "circuits").mkdir(parents=True, exist_ok=True)
    families = ["qft", "grover", "vqe_ansatz", "qaoa"]
    sizes = [3] if args.quick else [3, 4, 5, 6]
    instances = [0] if args.quick else [0, 1, 2]
    rows, comparisons, selections = [], [], []
    checkpoint = out / "candidates.partial.json"
    cached = {}
    if checkpoint.exists():
        for row in json.loads(checkpoint.read_text(encoding="utf8")):
            if "gp_solver" not in row:
                schedule_file = out / "circuits" / f"{row['case']}_candidate{row['candidate']}_schedule.json"
                schedule = json.loads(schedule_file.read_text(encoding="utf8"))
                row["gp_solver"] = schedule.get("gp_solver", "CLARABEL")
                row["gp_status"] = schedule["gp_status"]
            cached.setdefault(row["case"], []).append(row)
    resumed = 0
    start = time.perf_counter()
    for family in families:
        for n in sizes:
            for topo in ("line", "ring"):
                for instance in instances:
                    case = f"{family}_{n}_{topo}_{instance}"
                    logical = benchmark(family, n, 101 + instance)
                    coupling = topology(n, topo)
                    candidate_rows, circuits = [], []
                    for k in range(4):
                        if len(cached.get(case, [])) == 4:
                            candidate_rows.append(cached[case][k])
                            if k == 0:
                                resumed += 1
                            continue
                        seed = 1000 * instance + 17 * k + 7
                        physical, compile_time = compile_candidate(logical, coupling, seed)
                        metrics, (durations, edges) = candidate_metrics(physical, coupling)
                        cp = critical_path(durations, edges)
                        gp = gp_schedule(durations, edges)
                        lp = lp_schedule(durations, edges)
                        equivalence = unitary_error(logical, physical)
                        if equivalence > 1e-8:
                            raise RuntimeError(f"Unitary mismatch: {case}, candidate {k}: {equivalence}")
                        gap = abs(gp.makespan-cp.makespan) / max(cp.makespan, 1e-12)
                        if gap > 1e-5 or abs(lp.makespan-cp.makespan) > 1e-7:
                            raise RuntimeError(f"Scheduling optimum mismatch in {case}")
                        row = dict(case=case, family=family, qubits=n, topology=topo,
                            instance=instance, candidate=k, compiler_seed=seed, **metrics,
                            compile_seconds=compile_time, gp_seconds=gp.elapsed,
                            cp_seconds=cp.elapsed, lp_seconds=lp.elapsed,
                            gp_makespan=gp.makespan, lp_makespan=lp.makespan,
                            gp_solver=gp.solver_used, gp_status=gp.status,
                            gp_relative_gap=gap, gp_violation=gp.violation,
                            unitary_max_error=equivalence, simulated_fidelity=None,
                            density_trace_error=None, density_min_eigenvalue=None)
                        if n <= 4:
                            fidelity, trace, minimum = noisy_fidelity(physical, coupling)
                            row.update(simulated_fidelity=fidelity, density_trace_error=trace,
                                       density_min_eigenvalue=minimum)
                        candidate_rows.append(row)
                        circuits.append(physical)
                        # Actual solver output, not a schedule replaced with critical-path values.
                        schedule_path = out / "circuits" / f"{case}_candidate{k}_schedule.json"
                        schedule_path.write_text(json.dumps({
                            "durations": durations.tolist(), "edges": edges,
                            "gp_starts": gp.starts.tolist(), "gp_makespan": gp.makespan,
                            "cp_starts": cp.starts.tolist(), "cp_makespan": cp.makespan,
                            "lp_starts": lp.starts.tolist(), "lp_makespan": lp.makespan,
                            "gp_status": gp.status, "gp_solver": gp.solver_used,
                            "gp_violation": gp.violation,
                        }, indent=2), encoding="utf8")
                    if circuits:
                        with (out / "circuits" / f"{case}.qpy").open("wb") as stream:
                            qpy.dump([logical, *circuits], stream)
                    reference = candidate_rows[0]
                    weights_set = {"balanced": (0.4, 0.3, 0.3),
                                   "time": (0.8, 0.1, 0.1),
                                   "gates": (0.1, 0.8, 0.1),
                                   "error": (0.1, 0.1, 0.8)}
                    for label, weights in weights_set.items():
                        scores = [normalized_score(r, reference, weights) for r in candidate_rows]
                        chosen, matched, raw_gp, gp_scores = certified_selection(
                            candidate_rows, [r["gp_makespan"] for r in candidate_rows], weights)
                        # GP and CP objectives agree within the certified tolerance.
                        # Use exact DAG costs for deterministic ties in BOTH selectors.
                        selections.append(dict(case=case, weight_scheme=label,
                            chosen_candidate=chosen, score=scores[chosen],
                            baseline_score=scores[0], raw_gp_candidate=raw_gp,
                            matched_candidate=matched, gp_scores=gp_scores))
                        if label == "balanced":
                            selected = candidate_rows[chosen]
                            comparison = dict(case=case, family=family, qubits=n,
                                topology=topo, instance=instance, chosen_candidate=chosen,
                                score_improvement=100*(scores[0]-scores[chosen])/scores[0],
                                single_compile_seconds=reference["compile_seconds"],
                                matched_compile_seconds=sum(r["compile_seconds"]+r["cp_seconds"]
                                                            for r in candidate_rows),
                                hybrid_compile_seconds=sum(r["compile_seconds"]+r["gp_seconds"]
                                                           for r in candidate_rows),
                                matched_hybrid_selection_equal=(chosen==matched))
                            for metric in ("depth", "gates", "two_qubit", "makespan", "error_loss"):
                                comparison[f"baseline_{metric}"] = reference[metric]
                                comparison[f"selected_{metric}"] = selected[metric]
                                comparison[f"{metric}_reduction_pct"] = (
                                    100*(reference[metric]-selected[metric])/reference[metric]
                                    if reference[metric] else 0.0)
                            comparison["simulated_fidelity_gain_pp"] = (
                                100*(selected["simulated_fidelity"]-reference["simulated_fidelity"])
                                if n <= 4 else None)
                            comparisons.append(comparison)
                    rows.extend(candidate_rows)
                    print(f"{case}: validated 4 candidates; selected {comparisons[-1]['chosen_candidate']}",
                          flush=True)
                    # Checkpoint raw rows, so a stopped run has explicit partial data.
                    (out / "candidates.partial.json").write_text(json.dumps(rows), encoding="utf8")
    for name, data in (("candidates", rows), ("comparisons", comparisons), ("selections", selections)):
        (out / f"{name}.json").write_text(json.dumps(data, indent=2), encoding="utf8")
        with (out / f"{name}.csv").open("w", newline="", encoding="utf8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    environment = {"python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "packages": {m: importlib.metadata.version(m)
        for m in ("qiskit", "cvxpy", "numpy", "scipy", "clarabel", "matplotlib", "pytest")},
        "wall_seconds": time.perf_counter()-start,
        "resumed_cases": resumed,
        "families": families, "sizes": sizes, "instances": instances,
        "candidate_seeds": "1000*instance + 17*candidate + 7",
        "synthetic_hardware": True, "real_hardware_jobs": 0,
        "noise_scope": "local depolarizing channels after gates, all-zero input, n<=4",
        "duration_unit": "synthetic arbitrary time units"}
    (out / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf8")
    partial = out / "candidates.partial.json"
    partial.unlink()
    print(json.dumps({"cases": len(comparisons), "candidates": len(rows),
        "wall_seconds": environment["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
