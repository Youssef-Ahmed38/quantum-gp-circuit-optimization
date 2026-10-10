"""Recompute held-out X-duration contrasts, including blockwise uncertainty."""

import argparse
from collections import defaultdict
import json
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
from scipy import stats


def analyze(payload):
    order, counts = payload["order"], payload["counts"]
    if len(order) != len(counts) or len(order) % payload["blocks"]:
        raise ValueError("Incomplete holdout data")
    block_size = len(order) // payload["blocks"]
    groups = defaultdict(lambda: {"successes": 0, "shots": 0, "blocks": {}})
    for index, (condition, distribution) in enumerate(zip(order, counts)):
        shots = sum(distribution.values())
        if shots != payload["shots_per_circuit"] or set(distribution) - {"0", "1"}:
            raise ValueError("Malformed target counts")
        key = (condition["basis"], condition["length"], condition["mode"])
        block = index // block_size
        if block in groups[key]["blocks"]:
            raise ValueError("Duplicated randomized-block condition")
        successes = distribution.get("0", 0)
        groups[key]["successes"] += successes
        groups[key]["shots"] += shots
        groups[key]["blocks"][block] = successes / shots
    summaries = {}
    for key in sorted(groups):
        basis, length, mode = key
        group = groups[key]
        if len(group["blocks"]) != payload["blocks"]:
            raise ValueError("Missing randomized-block condition")
        name = f"{basis}_m{length}_{mode}"
        summaries[name] = {
            "successes": group["successes"], "shots": group["shots"],
            "return_probability": group["successes"] / group["shots"],
            "block_probabilities": [group["blocks"][i]
                                    for i in range(payload["blocks"])],
        }
    contrasts = {}
    for basis in payload["bases"]:
        for length in payload["lengths"]:
            slow = summaries[f"{basis}_m{length}_slow"]
            for baseline in ("padded", "fast_tail", "dd_tail"):
                other = summaries[f"{basis}_m{length}_{baseline}"]
                p, q = slow["return_probability"], other["return_probability"]
                se = sqrt(p * (1 - p) / slow["shots"]
                          + q * (1 - q) / other["shots"])
                block_diffs = [x - y for x, y in zip(
                    slow["block_probabilities"], other["block_probabilities"]
                )]
                block_interval = stats.t.interval(
                    0.95, df=len(block_diffs) - 1,
                    loc=stats.tmean(block_diffs),
                    scale=stats.sem(block_diffs),
                )
                contrasts[f"{basis}_m{length}_slow_minus_{baseline}"] = {
                    "difference": p - q,
                    "shot_only_normal_95": [p - q - 1.96 * se, p - q + 1.96 * se],
                    "block_differences": block_diffs,
                    "paired_block_t_95": list(block_interval),
                }
    first = contrasts["X_m48_slow_minus_fast_tail"]
    second = contrasts["X_m48_slow_minus_dd_tail"]
    passed = all(
        item["shot_only_normal_95"][0] > 0 and
        all(value > 0 for value in item["block_differences"])
        for item in (first, second)
    )
    return {
        "backend": payload["backend"], "job_id": payload["job_id"],
        "job_status": payload["job_status"],
        "protocol_lock_sha256": payload["protocol_lock_sha256"],
        "nominal_ticks_equal": all(
            len({ticks for row, ticks in zip(payload["order"], payload["nominal_ticks"])
                 if row["basis"] == basis and row["length"] == length}) == 1
            for basis in payload["bases"] for length in payload["lengths"]
        ),
        "conditions": summaries, "contrasts": contrasts,
        "frozen_primary_criterion_met": passed,
        "limits": [
            "A return probability is not average gate fidelity.",
            "The shot-only normal intervals and three-block t intervals do not capture slow device drift, and the latter rely on a tiny number of blocks.",
            "The tested choice was selected after a training job. Only this holdout was independent of selection.",
            "The result tests one supported X gate pair and one device, not the paper's continuous GP duration allocation.",
        ],
    }


def draw(result, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    colors = {"slow": "#1b9e77", "padded": "#7570b3",
              "fast_tail": "#d95f02", "dd_tail": "#4d4d4d"}
    for axis, basis in zip(axes, ("X", "Z")):
        for mode in ("slow", "padded", "fast_tail", "dd_tail"):
            records = [result["conditions"][f"{basis}_m{m}_{mode}"] for m in (16, 48)]
            means = [record["return_probability"] for record in records]
            errors = [1.96 * sqrt(p * (1 - p) / record["shots"])
                      for p, record in zip(means, records)]
            axis.errorbar((16, 48), means, yerr=errors, marker="o", capsize=3,
                          color=colors[mode], label=mode.replace("_", " "))
        axis.set_title(f"{basis}-basis return, physical qubit 1")
        axis.set_xticks((16, 48))
        axis.set_xlabel("Logical X gates")
        axis.set_ylim(0, 1.03)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Observed return probability")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.03), frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = analyze(payload)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.figure:
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        draw(result, args.figure)
    print(json.dumps({"frozen_primary_criterion_met": result["frozen_primary_criterion_met"],
                      "primary_contrasts": {
                          k: result["contrasts"][k]
                          for k in ("X_m48_slow_minus_fast_tail",
                                    "X_m48_slow_minus_dd_tail",
                                    "X_m48_slow_minus_padded")}}, indent=2))


if __name__ == "__main__":
    main()
