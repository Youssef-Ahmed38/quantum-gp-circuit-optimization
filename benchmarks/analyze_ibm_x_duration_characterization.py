"""Analyze every frozen X/XSlow condition without selecting post hoc winners."""

import argparse
from collections import defaultdict
import json
from math import sqrt
from pathlib import Path


def summarize(payload):
    order, counts = payload["order"], payload["counts"]
    if len(order) != len(counts):
        raise ValueError("Order/count length mismatch")
    blocks = payload["blocks"]
    if len(order) % blocks:
        raise ValueError("Incomplete randomized blocks")
    block_size = len(order) // blocks
    totals = defaultdict(lambda: [0, 0])
    by_block = defaultdict(dict)
    for index, (condition, distribution) in enumerate(zip(order, counts)):
        shots = sum(distribution.values())
        if shots != payload["shots_per_circuit"] or set(distribution) - {"0", "1"}:
            raise ValueError("Malformed counts")
        key = (condition["qubit"], condition["basis"],
               condition["length"], condition["mode"])
        success = distribution.get("0", 0)
        totals[key][0] += success
        totals[key][1] += shots
        by_block[key][index // block_size] = success / shots
    results = {}
    for key in sorted(totals):
        successes, shots = totals[key]
        q, basis, length, mode = key
        results[f"q{q}_{basis}_m{length}_{mode}"] = {
            "successes": successes, "shots": shots,
            "return_probability": successes / shots,
            "block_probabilities": [by_block[key][block] for block in range(blocks)],
        }
    contrasts = {}
    for q in payload["qubits"]:
        for basis in payload["bases"]:
            for length in payload["lengths"][1:]:
                slow = results[f"q{q}_{basis}_m{length}_slow"]
                for baseline in ("fast", "padded"):
                    other = results[f"q{q}_{basis}_m{length}_{baseline}"]
                    p, r = slow["return_probability"], other["return_probability"]
                    se = sqrt(p * (1 - p) / slow["shots"]
                              + r * (1 - r) / other["shots"])
                    contrasts[f"q{q}_{basis}_m{length}_slow_minus_{baseline}"] = {
                        "difference": p - r, "shot_only_normal_95": [p - r - 1.96 * se,
                                                                        p - r + 1.96 * se],
                        "block_differences": [a - b for a, b in zip(
                            slow["block_probabilities"], other["block_probabilities"]
                        )],
                    }
    return {
        "backend": payload["backend"], "job_id": payload["job_id"],
        "job_status": payload["job_status"], "conditions": results,
        "contrasts": contrasts,
        "primary_contrasts": {
            f"q{q}_{which}": contrasts[f"q{q}_X_m32_slow_minus_{which}"]
            for q in payload["qubits"] for which in ("fast", "padded")
        },
        "limits": ["Return probability is not gate fidelity.",
                   "Shot-only intervals ignore temporal drift and correlations.",
                   "Only two native X durations exist; the inverse-plus-linear law cannot be independently validated by this job.",
                   "A same-duration gate comparison alone is not a GP scheduler advantage."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis = summarize(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(analysis["primary_contrasts"], indent=2))


if __name__ == "__main__":
    main()
