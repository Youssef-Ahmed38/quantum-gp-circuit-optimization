import csv
from pathlib import Path

import pytest

from qcgp.measured_choice import evaluate_family


SOURCE = (Path(__file__).resolve().parents[1] /
          "evidence/measured_duration/measured_points_and_predictions.csv")


def _groups():
    groups = {}
    with SOURCE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            groups.setdefault(row["pulse"], []).append(row)
    return groups


def test_all_methods_use_the_same_withheld_library():
    for rows in _groups().values():
        result = evaluate_family(rows)
        withheld = {float(row["duration_ns"]) for row in rows if row["split"] == "test"}
        assert result["n_test"] == len(withheld)
        for method in result["methods"].values():
            assert method["selected_duration_ns"] in withheld
            assert method["measured_regret_vs_best_test"] >= -1e-12


def test_withheld_values_do_not_affect_any_selection():
    rows = _groups()["FAST_DRAG-L"]
    before = evaluate_family(rows)
    modified = [dict(row) for row in rows]
    for row in modified:
        if row["split"] == "test":
            row["measured_error"] = str(float(row["measured_error"]) + .01)
    after = evaluate_family(modified)
    for name in before["methods"]:
        assert before["methods"][name]["selected_duration_ns"] == \
               after["methods"][name]["selected_duration_ns"]


def test_invalid_interpolation_range_rejected():
    rows = [dict(row) for row in _groups()["FAST_DRAG-L"]]
    rows[0]["split"] = "test"
    with pytest.raises(ValueError, match="interpolate"):
        evaluate_family(rows)
