"""Analyze the frozen readout-calibration versus held-out QPU predictions."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from analyze_ibm_model_check import wilson
from run_ibm_model_check import readout_probabilities
from run_ibm_readout_holdout import EVENTS, LAYOUT, TARGET_NAMES, fit_operational_readout


TOLERANCE = 0.02


def analyze(payload):
    if payload.get("dry_run") or "target_counts" not in payload:
        raise ValueError("Completed hardware target counts are required")
    shots = payload["shots_per_circuit"]
    blocks = payload["blocks"]
    cal_order, cal_counts = payload["calibration_order"], payload["calibration_counts"]
    target_order, target_counts = payload["target_order"], payload["target_counts"]
    if len(cal_order) != 6 * blocks or len(target_order) != len(TARGET_NAMES) * blocks:
        raise ValueError("Unexpected number of calibration or target circuits")
    empirical = fit_operational_readout(cal_order, cal_counts, shots)
    if empirical != payload["operational_readout"]:
        raise ValueError("Saved readout fit disagrees with raw calibration counts")
    snapshot = readout_probabilities(payload["calibration_snapshot"], len(LAYOUT))
    calibration = {}
    for q in LAYOUT:
        for prepared in (0, 1):
            key = f"readout_q{q}_{prepared}"
            subset = [counts for name, counts in zip(cal_order, cal_counts) if name == key]
            if len(subset) != blocks:
                raise ValueError(f"Missing or duplicate calibrator: {key}")
            errors = sum(counts.get(str(1 - prepared), 0) for counts in subset)
            calibration[key] = {
                "errors": errors,
                "shots": shots * blocks,
                "measured_error": errors / (shots * blocks),
                "snapshot_error": snapshot[q][prepared],
                "shot_only_wilson_95": wilson(errors, shots * blocks),
            }

    rows = {name: [] for name in TARGET_NAMES}
    if len(target_order) != len(target_counts):
        raise ValueError("Target order and result count disagree")
    for sequence, (name, distribution) in enumerate(zip(target_order, target_counts)):
        if name not in rows or sum(distribution.values()) != shots:
            raise ValueError("Unknown target or invalid shot total")
        width = len(EVENTS[name][0])
        if any(len(bitstring) != width or set(bitstring) - {"0", "1"}
               for bitstring in distribution):
            raise ValueError("Malformed target bitstring")
        hits = sum(distribution.get(bitstring, 0) for bitstring in EVENTS[name])
        old = sum(payload["snapshot_predictions"][name][bitstring]
                  for bitstring in EVENTS[name])
        calibrated = sum(payload["calibrated_predictions"][name][bitstring]
                         for bitstring in EVENTS[name])
        rows[name].append({
            "sequence": sequence,
            "successes": hits,
            "shots": shots,
            "observed": hits / shots,
            "snapshot_prediction": old,
            "calibrated_prediction": calibrated,
            "snapshot_residual": hits / shots - old,
            "calibrated_residual": hits / shots - calibrated,
        })
    summary = {}
    for name, entries in rows.items():
        if len(entries) != blocks:
            raise ValueError(f"Missing or duplicated target: {name}")
        hits = sum(entry["successes"] for entry in entries)
        total = shots * blocks
        observed = hits / total
        old = entries[0]["snapshot_prediction"]
        new = entries[0]["calibrated_prediction"]
        summary[name] = {
            "successes": hits,
            "shots": total,
            "observed": observed,
            "snapshot_prediction": old,
            "calibrated_prediction": new,
            "snapshot_residual": observed - old,
            "calibrated_residual": observed - new,
            "shot_only_wilson_95": wilson(hits, total),
            "block_observed_range": [min(row["observed"] for row in entries),
                                     max(row["observed"] for row in entries)],
        }
    snapshot_mae = sum(abs(x["snapshot_residual"]) for x in summary.values()) / len(summary)
    calibrated_mae = sum(abs(x["calibrated_residual"]) for x in summary.values()) / len(summary)
    return {
        "purpose": "Held-out nominal native-gate model check, not GP duration validation",
        "calibration_job_id": payload["calibration_job_id"],
        "target_job_id": payload["target_job_id"],
        "backend": payload["backend"],
        "prespecified_descriptive_tolerance": TOLERANCE,
        "snapshot_event_mae": snapshot_mae,
        "calibrated_event_mae": calibrated_mae,
        "calibrated_within_tolerance_all_families": all(
            abs(x["calibrated_residual"]) <= TOLERANCE for x in summary.values()),
        "calibrated_improves_mae": calibrated_mae < snapshot_mae,
        "calibrator_diagnostic": calibration,
        "target_summary": summary,
        "target_blocks": rows,
        "limitations": [
            "The operational p(0|1) estimate includes error in its X preparation gate.",
            "Shot-only Wilson intervals exclude calibration drift and model uncertainty.",
            "No target circuits were used to fit readout probabilities, but the target device still used fixed native gate durations.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    lock_path = args.input.with_name(args.input.stem + "_prediction_lock.json")
    lock_bytes = lock_path.read_bytes()
    if sha256(lock_bytes).hexdigest() != payload.get("prediction_lock_sha256"):
        raise ValueError("Prediction lock hash mismatch")
    lock = json.loads(lock_bytes)
    for field in ("calibration_counts", "operational_readout", "snapshot_predictions",
                  "calibrated_predictions", "target_order", "target_events"):
        if lock[field] != payload[field]:
            raise ValueError(f"Prediction lock field changed: {field}")
    result = analyze(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"snapshot_event_mae": result["snapshot_event_mae"],
                      "calibrated_event_mae": result["calibrated_event_mae"],
                      "calibrated_within_tolerance_all_families": result["calibrated_within_tolerance_all_families"]}, indent=2))


if __name__ == "__main__":
    main()
