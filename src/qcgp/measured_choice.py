"""Retrospective pulse-duration decisions on a measured finite library.

The data-driven interpolator was examined after earlier results and is therefore
exploratory. Source error bars describe RB measurement uncertainty, not model
discrepancy or a guarantee for unseen durations.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import nnls


LAW_EXPONENTS = {"inverse_linear": (-1., 1.),
                 "inverse_square_linear": (-2., 1.)}


def _arrays(rows):
    duration = np.array([float(row["duration_ns"]) for row in rows])
    error = np.array([float(row["measured_error"]) for row in rows])
    sigma = np.array([float(row["reported_one_sigma"]) for row in rows])
    train = np.array([row["split"] == "train" for row in rows])
    if (len(duration) < 7 or not np.all(np.diff(duration) > 0) or
            not np.isfinite(duration).all() or np.any(duration <= 0) or
            not np.isfinite(error).all() or np.any(error < 0) or
            not np.isfinite(sigma).all() or np.any(sigma <= 0) or
            not np.all([row["split"] in ("train", "test") for row in rows]) or
            train.sum() < 4 or (~train).sum() < 2):
        raise ValueError("Ordered positive duration/error observations and train/test split required")
    if np.min(duration[~train]) <= np.min(duration[train]) or np.max(duration[~train]) >= np.max(duration[train]):
        raise ValueError("Test decisions must interpolate within the training range")
    return duration, error, sigma, train


def _law_prediction(duration, error, sigma, train, exponents):
    matrix = np.column_stack([(duration / 10.) ** exponent for exponent in exponents])
    coefficients, _ = nnls(matrix[train] / sigma[train, None], error[train] / sigma[train])
    return matrix @ coefficients


def _interpolation_discrepancy(duration, error, train):
    """Interior training-only leave-one-out RMS; diagnostic, not a CI."""
    indices = np.flatnonzero(train)
    residuals = []
    for index in indices[1:-1]:
        kept = indices[indices != index]
        predicted = PchipInterpolator(duration[kept], error[kept])(duration[index])
        residuals.append(float(predicted - error[index]))
    return float(np.sqrt(np.mean(np.square(residuals))))


def evaluate_family(rows):
    """Fit on marked training points and score only withheld pulse choices."""
    duration, error, sigma, train = _arrays(rows)
    test_indices = np.flatnonzero(~train)
    oracle = float(np.min(error[~train]))
    predictions = {name: _law_prediction(duration, error, sigma, train, power)
                   for name, power in LAW_EXPONENTS.items()}
    predictions["pchip"] = PchipInterpolator(duration[train], error[train])(duration)
    discrepancy = _interpolation_discrepancy(duration, error, train)
    methods = {}
    for name, predicted in predictions.items():
        selected = int(test_indices[np.argmin(predicted[~train])])
        residual = predicted[~train] - error[~train]
        methods[name] = dict(
            selected_duration_ns=float(duration[selected]),
            selected_measured_error=float(error[selected]),
            selected_reported_one_sigma=float(sigma[selected]),
            measured_regret_vs_best_test=float(error[selected] - oracle),
            heldout_mae=float(np.mean(np.abs(residual))),
            heldout_rms_source_sigma=float(np.sqrt(np.mean((residual / sigma[~train]) ** 2))),
        )
        if name == "pchip":
            expanded = np.sqrt(np.square(sigma[~train]) + discrepancy ** 2)
            methods[name]["train_only_discrepancy_rms"] = discrepancy
            methods[name]["heldout_rms_expanded_scale"] = float(
                np.sqrt(np.mean(np.square(residual / expanded))))
            methods[name]["diagnostic_1_96_band_coverage"] = float(
                np.mean(np.abs(residual) <= 1.96 * expanded))
    return dict(n_train=int(train.sum()), n_test=len(test_indices),
                shortest_test_duration_ns=float(duration[test_indices[0]]),
                best_test_duration_ns=float(duration[test_indices[np.argmin(error[~train])]]),
                best_test_measured_error=oracle, methods=methods)
