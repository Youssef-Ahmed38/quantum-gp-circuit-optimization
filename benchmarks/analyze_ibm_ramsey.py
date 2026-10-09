"""Summarize the exploratory IBM Ramsey/echo pilot without efficacy claims."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest


ORDER = ("ramsey_1us", "ramsey_20us", "ramsey_80us", "echo_80us")
LABELS = ("Ramsey 1 µs", "Ramsey 20 µs", "Ramsey 80 µs", "Echo 80 µs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    names, counts = raw["order"], raw["counts"]
    shots = raw["shots_per_circuit"]
    if len(names) != 8 or len(counts) != 8 or set(names) != set(ORDER):
        raise ValueError("Expected two randomized blocks of four conditions")
    observations = defaultdict(list)
    for index, (name, result) in enumerate(zip(names, counts)):
        if name not in ORDER or sum(result.values()) != shots:
            raise ValueError("Invalid condition or shot total")
        observations[name].append({
            "block": index // 4,
            "zero": int(result.get("0", 0)),
            "one": int(result.get("1", 0)),
            "zero_fraction": int(result.get("0", 0)) / shots,
        })
    if any(sorted(x["block"] for x in observations[name]) != [0, 1]
           for name in ORDER):
        raise ValueError("Missing matched block")
    summary = {
        "job_id": raw["job_id"],
        "backend": raw["backend"],
        "physical_qubit": raw["physical_qubit"],
        "shots_per_circuit": shots,
        "conditions": {},
        "echo_minus_ramsey_80us_by_block": [],
        "interpretation": "Exploratory idle-noise pilot only; two blocks do not establish a stable device-level advantage or validate GP gate-duration optimization.",
    }
    for name in ORDER:
        rows = observations[name]
        success = sum(row["zero"] for row in rows)
        total = len(rows) * shots
        ci = binomtest(success, total).proportion_ci(confidence_level=.95, method="exact")
        summary["conditions"][name] = {
            "runs": rows,
            "zero_total": success,
            "shots_total": total,
            "zero_fraction": success / total,
            "conditional_exact_binomial_95_percent_interval": [ci.low, ci.high],
        }
    for block in (0, 1):
        echo = next(x for x in observations["echo_80us"] if x["block"] == block)
        ramsey = next(x for x in observations["ramsey_80us"] if x["block"] == block)
        summary["echo_minus_ramsey_80us_by_block"].append(
            echo["zero_fraction"] - ramsey["zero_fraction"]
        )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    x = np.arange(len(ORDER))
    y = np.array([summary["conditions"][name]["zero_fraction"] for name in ORDER])
    low = np.array([summary["conditions"][name]
                    ["conditional_exact_binomial_95_percent_interval"][0]
                    for name in ORDER])
    high = np.array([summary["conditions"][name]
                     ["conditional_exact_binomial_95_percent_interval"][1]
                     for name in ORDER])
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.errorbar(x, y, yerr=[y - low, high - y], fmt="o", capsize=4,
                color="#17617a", label="Pooled shots; conditional binomial CI")
    for block, marker in ((0, "x"), (1, "+")):
        ax.scatter(x + (-0.08 if block == 0 else 0.08),
                   [next(row["zero_fraction"] for row in observations[name]
                         if row["block"] == block) for name in ORDER],
                   marker=marker, color="#bb5533", s=45, label=f"Block {block + 1}")
    ax.set_xticks(x, LABELS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Measured Pr(0)")
    ax.set_title("Heron idle-noise pilot, physical qubit 0")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)
    fig.text(0.02, 0.01, "Two blocks, 2,048 shots per circuit. Intervals omit device drift.",
             fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)
    print(f"Wrote {args.summary} and {args.figure}")


if __name__ == "__main__":
    main()
