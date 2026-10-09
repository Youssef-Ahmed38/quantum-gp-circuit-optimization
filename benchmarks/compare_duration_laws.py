"""Exploratory leave-one-duration-out comparison of positive error laws.

The inverse-square candidate was suggested after inspecting the earlier
alternating holdout results.  This analysis is exploratory, not confirmatory.
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evidence/measured_duration/measured_points_and_predictions.csv"
OUT = ROOT / "evidence/measured_duration/law_comparison.json"
TABLE = ROOT / "docs/law_comparison_table.tex"
MODELS = {"constant": (0,), "inverse_linear": (-1, 1), "inverse_square_linear": (-2, 1)}


def design(duration_ns, exponents):
    """Scale at 10 ns to keep the small weighted least-squares system stable."""
    scaled = np.asarray(duration_ns, dtype=float) / 10.
    if scaled.ndim != 1 or not np.isfinite(scaled).all() or np.any(scaled <= 0):
        raise ValueError("Durations must be positive finite values")
    return np.column_stack([scaled ** exponent for exponent in exponents])


def leave_one_out(duration_ns, error, sigma, exponents):
    duration_ns = np.asarray(duration_ns, dtype=float)
    error = np.asarray(error, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if error.shape != sigma.shape or error.shape != duration_ns.shape or len(error) < 4:
        raise ValueError("Need at least four matching observations")
    if not np.isfinite(error).all() or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("Errors and uncertainties must be finite; uncertainties positive")
    matrix = design(duration_ns, exponents)
    predicted = np.empty_like(error)
    for held in range(len(error)):
        train = np.arange(len(error)) != held
        coefficients, _ = nnls(matrix[train] / sigma[train, None], error[train] / sigma[train])
        predicted[held] = matrix[held] @ coefficients
    return predicted


def analyze(rows):
    result = []
    for pulse in dict.fromkeys(row["pulse"] for row in rows):
        group = [row for row in rows if row["pulse"] == pulse]
        duration = np.array([float(row["duration_ns"]) for row in group])
        error = np.array([float(row["measured_error"]) for row in group])
        sigma = np.array([float(row["reported_one_sigma"]) for row in group])
        train = np.array([row["split"] == "train" for row in group])
        test = ~train
        if not train.any() or not test.any():
            raise ValueError("Need both alternating-fit and held-out durations")
        metrics = {}
        for name, exponents in MODELS.items():
            prediction = leave_one_out(duration, error, sigma, exponents)
            residual = prediction - error
            metrics[name] = dict(mae=float(np.mean(np.abs(residual))),
                                 rms_standardized_residual=float(np.sqrt(np.mean((residual / sigma) ** 2))),
                                 predictions=prediction.tolist())
            if name != "constant":
                matrix = design(duration, exponents)
                coefficients, _ = nnls(matrix[train] / sigma[train, None], error[train] / sigma[train])
                held_indices = np.flatnonzero(test)
                selected = int(held_indices[np.argmin((matrix @ coefficients)[test])])
                metrics[name]["heldout_decision"] = dict(
                    selected_duration_ns=float(duration[selected]),
                    measured_error=float(error[selected]),
                    regret_vs_best_heldout=float(error[selected] - np.min(error[test])),
                )
        result.append(dict(pulse=pulse, n=len(group), metrics=metrics))
    return result


def main():
    with SOURCE.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = dict(method="Exploratory leave-one-duration-out weighted NNLS; candidate chosen after earlier holdout inspection",
                  units="duration ns; mean RB error per gate; source one-sigma uncertainty",
                  source_doi="10.5281/zenodo.13808824", results=analyze(rows))
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    lines = [r"\begin{table}[t]", r"\centering\caption{Exploratory leave-one-duration-out error-law comparison. MAE columns are in units of $10^{-4}$; RMS is the root mean squared residual divided by source-reported one-sigma uncertainty. The inverse-square candidate was chosen after viewing earlier holdout data.}\label{tab:law_comparison}",
             r"\begin{tabular}{lccc}\hline", r"Pulse family & $1/t+t$ MAE & $1/t^2+t$ MAE & $1/t^2+t$ RMS\\\hline"]
    for group in result["results"]:
        first = group["metrics"]["inverse_linear"]
        second = group["metrics"]["inverse_square_linear"]
        lines.append(f"{group['pulse'].replace('_', ' ')} & {first['mae'] * 1e4:.3f} & {second['mae'] * 1e4:.3f} & {second['rms_standardized_residual']:.2f}\\\\")
    lines.extend([r"\hline\end{tabular}", r"\end{table}"])
    TABLE.write_text("\n".join(lines) + "\n")
    for group in result["results"]:
        print(group["pulse"], group["n"],
              {name: (round(item["mae"], 8), round(item["rms_standardized_residual"], 2))
               for name, item in group["metrics"].items()})


if __name__ == "__main__":
    main()
