import json
import numpy as np

with open("evidence/observable_study/results.json") as f:
    data = json.load(f)

sizes = sorted(list(set(d["qubits"] for d in data)))

print("Total cases:", len(data))
all_feasible = [d for d in data if d["gp"]["feasible"]]
all_infeasible = [d for d in data if not d["gp"]["feasible"]]
print(f"Total Feasible: {len(all_feasible)}, Total Infeasible: {len(all_infeasible)}")

feas_times = [d["gp"]["elapsed"] * 1000 for d in all_feasible]
infeas_times = [d["gp"]["elapsed"] * 1000 for d in all_infeasible]

print(f"All Feasible Solves: median={np.median(feas_times):.2f} ms, mean={np.mean(feas_times):.2f} ms, min={np.min(feas_times):.2f} ms, max={np.max(feas_times):.2f} ms")
print(f"All Infeasible Screening: median={np.median(infeas_times):.2f} ms, mean={np.mean(infeas_times):.2f} ms, min={np.min(infeas_times):.2f} ms, max={np.max(infeas_times):.2f} ms")

print("\n--- Breakdown by Qubit Count ---")
for n in sizes:
    subset = [d for d in data if d["qubits"] == n]
    feasible = [d for d in subset if d["gp"]["feasible"]]
    infeasible = [d for d in subset if not d["gp"]["feasible"]]

    if feasible:
        f_times = [d["gp"]["elapsed"] * 1000 for d in feasible]
        f_med = f"{np.median(f_times):.2f} ms"
        f_mean = f"{np.mean(f_times):.2f} ms"
    else:
        f_med = "N/A"
        f_mean = "N/A"

    if infeasible:
        inf_times = [d["gp"]["elapsed"] * 1000 for d in infeasible]
        inf_med = f"{np.median(inf_times):.2f} ms"
        inf_mean = f"{np.mean(inf_times):.2f} ms"
    else:
        inf_med = "N/A"
        inf_mean = "N/A"

    print(f"Q={n:3d} (gates={subset[0]['timed_gates']:3d}): Feas={len(feasible):2d}/{len(subset):2d} | Feasible Solve: med={f_med:>8s}, mean={f_mean:>8s} | Infeas Screen: med={inf_med:>8s}, mean={inf_mean:>8s}")
