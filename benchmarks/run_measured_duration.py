"""Held-out test of the inverse-plus-linear law on published experimental RB data."""
import argparse
import csv
from hashlib import md5
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen
import zipfile

import numpy as np
from scipy.optimize import nnls


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence" / "measured_duration"
URL = "https://zenodo.org/api/records/13808824/files/Data_for_Reducing_leakage_errors_v2.zip/content"
EXPECTED_MD5 = "b2d6a335a41bb0f8f9123d5ddfb4247a"
PULSES = ("FAST_DRAG-L", "HD_DRAG-L", "Cosine_DRAG-L", "Cosine_DRAG-P")


def load_archive(path):
    data = path.read_bytes()
    if md5(data).hexdigest() != EXPECTED_MD5:
        raise ValueError("Published archive checksum mismatch")
    return zipfile.ZipFile(io.BytesIO(data))


def extract_rows(archive):
    rows = []
    for pulse in PULSES:
        prefix = f"v2/Fig3/Fig3a/{pulse}/"
        def values(name):
            return np.loadtxt(io.BytesIO(archive.read(prefix + name + ".csv")), delimiter=",")
        duration = values("duration_exp_arr") * 1e9
        error = values("gate_error_exp_arr")
        uncertainty = values("gate_error_exp_uncertainty_arr")
        if not (duration.shape == error.shape == uncertainty.shape):
            raise ValueError("Mismatched measured arrays")
        if not (np.isfinite(duration).all() and np.isfinite(error).all() and np.isfinite(uncertainty).all()):
            raise ValueError("Nonfinite measurements")
        if not ((duration > 0).all() and (error >= 0).all() and (uncertainty > 0).all()):
            raise ValueError("Invalid experimental data")
        order = np.argsort(duration)
        for index, original in enumerate(order):
            rows.append(dict(pulse=pulse, duration_ns=float(duration[original]),
                             measured_error=float(error[original]),
                             reported_one_sigma=float(uncertainty[original]),
                             split="train" if index % 2 == 0 else "test"))
    return rows


def evaluate(rows):
    fits = []
    for pulse in PULSES:
        group = [r for r in rows if r["pulse"] == pulse]
        t = np.array([r["duration_ns"] for r in group])
        e = np.array([r["measured_error"] for r in group])
        sigma = np.array([r["reported_one_sigma"] for r in group])
        train = np.array([r["split"] == "train" for r in group])
        test = ~train
        design = np.column_stack((1 / t, t))
        coefficients, _ = nnls(design[train] / sigma[train, None], e[train] / sigma[train])
        prediction = design @ coefficients
        constant = np.average(e[train], weights=1 / sigma[train] ** 2)
        for row, value in zip(group, prediction):
            row["predicted_error"] = float(value)
            row["constant_prediction"] = float(constant)
        residual = prediction[test] - e[test]
        fits.append(dict(pulse=pulse, train_n=int(train.sum()), test_n=int(test.sum()),
                         inverse_coefficient_error_ns=float(coefficients[0]),
                         linear_coefficient_error_per_ns=float(coefficients[1]),
                         test_mae=float(np.mean(np.abs(residual))),
                         constant_test_mae=float(np.mean(np.abs(constant - e[test]))),
                         test_mean_squared_standardized_residual=float(np.mean((residual / sigma[test]) ** 2)),
                         measured_min_error=float(np.min(e)), measured_min_duration_ns=float(t[np.argmin(e)])))
    return fits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, help="Verified local copy of the Zenodo ZIP")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.archive:
        archive = load_archive(args.archive)
    else:
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "source.zip"
            with urlopen(URL, timeout=120) as response:
                destination.write_bytes(response.read())
            archive = load_archive(destination)
    with archive:
        rows = extract_rows(archive)
    fits = evaluate(rows)
    with (OUT / "measured_points_and_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    payload = dict(source_doi="10.5281/zenodo.13808824", source_archive_url=URL,
                   source_archive_md5=EXPECTED_MD5, figure="Fig. 3a",
                   measurement="Mean RB error per gate and reported one-sigma uncertainty",
                   split="Ascending duration; even-indexed points train, odd-indexed points test; per pulse family",
                   fit="Nonnegative weighted least squares, error=A/t_ns+G*t_ns; weights=1/sigma^2",
                   no_hardware_jobs=True, fits=fits)
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.0), constrained_layout=True)
        for ax, pulse in zip(axes.flat, PULSES):
            group = [r for r in rows if r["pulse"] == pulse]
            train = [r for r in group if r["split"] == "train"]
            test = [r for r in group if r["split"] == "test"]
            ax.errorbar([r["duration_ns"] for r in train], [r["measured_error"] for r in train],
                        yerr=[r["reported_one_sigma"] for r in train], fmt="o", label="fit points")
            ax.errorbar([r["duration_ns"] for r in test], [r["measured_error"] for r in test],
                        yerr=[r["reported_one_sigma"] for r in test], fmt="s", label="held out")
            ax.plot([r["duration_ns"] for r in group], [r["predicted_error"] for r in group], label="inverse+linear fit")
            ax.set(xlabel="Gate duration (ns)", ylabel="RB error per gate", title=pulse)
            ax.legend(fontsize=7)
        fig.savefig(ROOT / "figures/measured_duration.pdf")
        fig.savefig(ROOT / "figures/measured_duration.png", dpi=180)
        plt.close(fig)
    except ImportError:
        pass
    print(json.dumps(fits, indent=2))


if __name__ == "__main__":
    main()
