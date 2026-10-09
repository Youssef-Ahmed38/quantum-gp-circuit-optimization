"""Report prespecified event residuals for a nominal-gate IBM model check."""

import argparse
import json
from pathlib import Path
from math import sqrt


SUCCESS = {
    "single_return": ("0",),
    "bell": ("00", "11"),
    "bell_return": ("00",),
    "ghz": ("000", "111"),
}


def wilson(successes, shots):
    """Shot-only 95% Wilson interval; excludes device drift."""
    z = 1.959963984540054
    p = successes / shots
    denominator = 1 + z * z / shots
    center = (p + z * z / (2 * shots)) / denominator
    half = z * sqrt(p * (1 - p) / shots + z * z / (4 * shots * shots)) / denominator
    return [center - half, center + half]


def analyze(payload):
    if payload.get("dry_run") or "counts" not in payload:
        raise ValueError("Actual IBM counts are required")
    order = payload["order"]
    counts = payload["counts"]
    shots = payload["shots_per_circuit"]
    if len(order) != len(counts) or len(order) != len(SUCCESS) * payload["blocks"]:
        raise ValueError("Protocol order and result count disagree")
    rows = {name: [] for name in SUCCESS}
    for sequence, (name, distribution) in enumerate(zip(order, counts)):
        if name not in SUCCESS or sum(distribution.values()) != shots:
            raise ValueError("Unknown circuit or invalid shot total")
        successes = sum(distribution.get(bitstring, 0) for bitstring in SUCCESS[name])
        prediction = sum(payload["predicted_observed_distributions"][name][bitstring]
                         for bitstring in SUCCESS[name])
        rows[name].append(dict(sequence=sequence, successes=successes, shots=shots,
                               observed=successes / shots, prediction=prediction,
                               residual=successes / shots - prediction,
                               shot_only_wilson_95=wilson(successes, shots)))
    summary = {}
    for name, blocks in rows.items():
        if len(blocks) != payload["blocks"]:
            raise ValueError("Circuit did not occur once per block")
        successes = sum(block["successes"] for block in blocks)
        total = shots * len(blocks)
        observed = successes / total
        prediction = blocks[0]["prediction"]
        summary[name] = dict(successes=successes, shots=total,
                             observed=observed, prediction=prediction,
                             residual=observed - prediction,
                             shot_only_wilson_95=wilson(successes, total),
                             block_observed_range=[min(block["observed"] for block in blocks),
                                                   max(block["observed"] for block in blocks)],
                             within_two_percentage_points=abs(observed - prediction) <= .02)
    return dict(job_id=payload["job_id"], backend=payload["backend"],
                purpose="Nominal fixed-gate noise-model check; not GP hardware improvement",
                practical_tolerance_absolute=.02,
                caveat="Shot-only intervals exclude calibration drift and model uncertainty; reported tolerance is descriptive, not a significance test",
                results=summary, blocks=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["results"], indent=2))


if __name__ == "__main__":
    main()
