"""C4 study: baseline dominance (Table VII rerun) and calibration coverage.

Part A repeats the misspecification stress test of the manuscript (six
circuits, deadline = nominal makespan) and adds a GP with dominance
constraints over a declared hull of error laws. Part B simulates a noisy
duration-sweep calibration, builds the Scheffe ellipsoid, and checks how often
the certified robust bound actually covers the true model error.

All computation is local; no hardware is used.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from qcgp.core import benchmark, compile_candidate, critical_path, topology
from qcgp.robust import (RobustSpec, coverage_radius, fit_duration_law, law_value,
                         solve_certified_robust)
from qcgp.timing import synthetic_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence" / "robust_pac"
CIRCUITS = [("qft", 3), ("qft", 4), ("grover", 3), ("grover", 4),
            ("vqe_ansatz", 3), ("vqe_ansatz", 4)]


def build(name, n):
    line = topology(n, "line")
    compiled, _ = compile_candidate(benchmark(name, n, 0), line, 0)
    return synthetic_model(compiled, line)


def laws(model):
    a, c, base = model.amplitude_noise, model.gate_noise_rate, model.base
    return {
        "inverse (declared)": [(a, -1), (c, 1)],
        "inverse-square, matched at nominal": [(a * base, -2), (c, 1)],
        "inverse-cube, matched at nominal": [(a * base ** 2, -3), (c, 1)],
        "manuscript Table VII law a/t^2": [(a, -2), (c, 1)],
    }


def mixture(law_a, law_b, weight):
    return [(weight * coefficient, exponent) for coefficient, exponent in law_a] + \
           [((1 - weight) * coefficient, exponent) for coefficient, exponent in law_b]


def compare(value, reference):
    if value < reference * (1 - 1e-9):
        return "better"
    if value > reference * (1 + 1e-9):
        return "worse"
    return "equal"


def part_a():
    rows = []
    for name, n in CIRCUITS:
        model = build(name, n)
        family = laws(model)
        hull = [family["inverse (declared)"], family["inverse-square, matched at nominal"]]
        deadline = critical_path(model.base, model.edges).makespan
        spec = RobustSpec.independent(model.amplitude_noise, model.gate_noise_rate,
                                      np.zeros((len(model.base), 2, 2)))
        kwargs = dict(power=1., deadline=deadline)
        plain = solve_certified_robust(model.lower, model.upper, model.edges, spec, 0., **kwargs)
        dominant = solve_certified_robust(model.lower, model.upper, model.edges, spec, 0.,
                                          dominance_laws=hull, baseline=model.base, **kwargs)
        extended = solve_certified_robust(model.lower, model.upper, model.edges, spec, 0.,
                                          dominance_laws=hull + [family["manuscript Table VII law a/t^2"]],
                                          baseline=model.base, **kwargs)
        tests = dict(family)
        for w in (0.25, 0.5, 0.75):
            tests[f"hull mixture {w:.2f}"] = mixture(hull[0], hull[1], w)
        for label, law in tests.items():
            base_value = law_value(model.base, law)
            record = dict(circuit=f"{name}_{n}", law=label,
                          inside_hull=label.startswith("inverse (") or label.startswith("inverse-square") or label.startswith("hull"),
                          baseline=base_value)
            for method, result in (("plain_gp", plain), ("dominance_gp", dominant),
                                   ("dominance_gp_plus_table7_law", extended)):
                value = law_value(result["durations"], law)
                record[method] = value
                record[method + "_vs_baseline"] = compare(value, base_value)
                record[method + "_relative_change"] = value / base_value - 1
            rows.append(record)
    return rows


def calibrate(model, groups, rng, sigma_fraction, points=6):
    """Simulated duration sweep for one reference gate per group."""
    a, c = model.amplitude_noise, model.gate_noise_rate
    theta, covariance, alpha, beta = [], [], np.zeros(len(a)), np.zeros(len(a))
    for g in range(groups.max() + 1):
        members = groups == g
        alpha[members] = a[members] / np.median(a[members])
        beta[members] = c[members] / np.median(c[members])
        a_ref, c_ref = np.median(a[members]), np.median(c[members])
        base_ref = np.median(model.base[members])
        t = base_ref * np.linspace(0.5, 3.0, points)
        truth = a_ref / t + c_ref * t
        sigma = sigma_fraction * (a_ref / base_ref + c_ref * base_ref)
        observed = truth + rng.normal(0., sigma, size=points)
        estimate, cov = fit_duration_law(t, observed, sigma, 1.)
        theta.append(estimate)
        covariance.append(cov)
    return RobustSpec(groups, alpha, beta, np.array(theta), np.array(covariance))


def coverage_for_circuit(args):
    """One circuit's coverage trials (run in its own process)."""
    (name, n), trials, delta, sigma_fraction, seed = args
    rng = np.random.default_rng([seed, n, len(name)])
    model = build(name, n)
    # synthetic_model assigns rate 6e-5 to 1q gates and 2.2e-4 to 2q gates.
    groups = (model.gate_noise_rate > 1e-4).astype(int)
    truth_law = [(model.amplitude_noise, -1), (model.gate_noise_rate, 1)]
    deadline = critical_path(model.base, model.edges).makespan
    kappa = coverage_radius(delta, 2 * (groups.max() + 1))
    rows, failures, start = [], 0, time.perf_counter()
    for trial in range(trials):
        spec = calibrate(model, groups, rng, sigma_fraction)
        try:
            robust = solve_certified_robust(model.lower, model.upper, model.edges, spec, kappa,
                                            deadline=deadline)
            plugin = solve_certified_robust(model.lower, model.upper, model.edges, spec, 0.,
                                            deadline=deadline)
        except RuntimeError:
            failures += 1
            continue
        rows.append(dict(
            circuit=f"{name}_{n}", trial=trial, kappa=kappa,
            robust_true=law_value(robust["durations"], truth_law),
            robust_certificate=robust["certified_error"],
            robust_covered=law_value(robust["durations"], truth_law) <= robust["certified_error"],
            plugin_true=law_value(plugin["durations"], truth_law),
            plugin_claim=plugin["nominal_error"],
            plugin_covered=law_value(plugin["durations"], truth_law) <= plugin["nominal_error"],
            baseline_true=law_value(model.base, truth_law)))
        if (trial + 1) % 20 == 0:
            print(f"{name}_{n}: {trial + 1}/{trials} trials, {failures} solver failures, "
                  f"{time.perf_counter() - start:.0f}s", flush=True)
    (OUT / f"part_b_{name}_{n}.json").write_text(json.dumps(dict(rows=rows, failures=failures), default=float))
    return rows, failures


def part_b(trials=200, delta=0.05, sigma_fraction=0.10, seed=2026):
    from multiprocessing import Pool
    with Pool(len(CIRCUITS)) as pool:
        results = pool.map(coverage_for_circuit, [(c, trials, delta, sigma_fraction, seed) for c in CIRCUITS])
    rows = [r for chunk, _ in results for r in chunk]
    failures = {f"{name}_{n}": f for (name, n), (_, f) in zip(CIRCUITS, results)}
    return rows, failures


def summarise(a_rows, b_rows):
    summary = {"part_a": {}, "part_b": {}}
    for method in ("plain_gp", "dominance_gp", "dominance_gp_plus_table7_law"):
        for inside in (True, False):
            subset = [r for r in a_rows if r["inside_hull"] == inside]
            summary["part_a"][f"{method}|{'inside' if inside else 'outside'}_hull"] = dict(
                cases=len(subset),
                worse=sum(r[method + "_vs_baseline"] == "worse" for r in subset),
                better=sum(r[method + "_vs_baseline"] == "better" for r in subset),
                median_relative_change=float(np.median([r[method + "_relative_change"] for r in subset])))
    declared = [r for r in a_rows if r["law"] == "inverse (declared)"]
    summary["part_a"]["price_of_dominance_on_declared_law"] = float(np.median(
        [r["dominance_gp"] / r["plain_gp"] - 1 for r in declared]))
    for circuit in sorted({r["circuit"] for r in b_rows}):
        subset = [r for r in b_rows if r["circuit"] == circuit]
        summary["part_b"][circuit] = dict(
            trials=len(subset),
            robust_coverage=float(np.mean([r["robust_covered"] for r in subset])),
            plugin_coverage=float(np.mean([r["plugin_covered"] for r in subset])),
            median_robust_true_over_plugin_true=float(np.median([r["robust_true"] / r["plugin_true"] for r in subset])),
            robust_never_worse_than_baseline_fraction=float(np.mean([r["robust_true"] <= r["baseline_true"] * (1 + 1e-9) for r in subset])))
    pooled = [r["robust_covered"] for r in b_rows]
    summary["part_b"]["pooled_robust_coverage"] = float(np.mean(pooled))
    summary["part_b"]["pooled_plugin_coverage"] = float(np.mean([r["plugin_covered"] for r in b_rows]))
    summary["part_b"]["target_coverage"] = 0.95
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    a_rows = part_a()
    (OUT / "part_a_dominance.json").write_text(json.dumps(a_rows, indent=1, default=float))
    print("part A done", flush=True)
    b_rows, failures = part_b()
    summary = summarise(a_rows, b_rows)
    summary["part_b"]["solver_failures_excluded"] = failures
    summary["elapsed_s"] = time.perf_counter() - start
    summary["hardware_jobs"] = 0
    (OUT / "part_a_dominance.json").write_text(json.dumps(a_rows, indent=1, default=float))
    (OUT / "part_b_coverage.json").write_text(json.dumps(b_rows, indent=1, default=float))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
