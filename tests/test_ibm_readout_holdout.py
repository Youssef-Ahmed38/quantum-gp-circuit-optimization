from pathlib import Path
import sys

import pytest
from qiskit import qasm3
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from run_ibm_model_check import predicted_distribution  # noqa: E402
from run_ibm_readout_holdout import (  # noqa: E402
    BLOCKS, EVENTS, SHOTS, TARGET_NAMES, build_protocol, configured_sampler,
    fit_operational_readout, target_source,
)
from analyze_ibm_readout_holdout import analyze  # noqa: E402


def test_fake_protocol_preserves_mapping_and_ideal_events():
    payload, calibrators, targets, local = build_protocol(FakeMarrakesh())
    assert len(calibrators) == 6 * BLOCKS
    assert len(targets) == len(TARGET_NAMES) * BLOCKS
    assert set(payload["snapshot_predictions"]) == set(TARGET_NAMES)
    for name in TARGET_NAMES:
        circuit = target_source(name).remove_final_measurements(inplace=False)
        ideal = Statevector.from_instruction(circuit).probabilities_dict()
        assert sum(ideal.get(bitstring, 0) for bitstring in EVENTS[name]) == pytest.approx(1.)
        assert sum(payload["snapshot_predictions"][name].values()) == pytest.approx(1.)
        assert qasm3.loads(payload["native_target_qasm3"][name]).num_qubits == local[name].num_qubits
    for index, name in enumerate(payload["calibration_order"]):
        physical = int(name.split("_")[1][1:])
        layout = calibrators[index].layout
        input_qubit = next(iter(layout.input_qubit_mapping))
        assert layout.initial_layout[input_qubit] == physical


def test_operational_readout_uses_only_calibration_counts():
    order = [f"readout_q{q}_{prepared}" for q in range(3) for prepared in (0, 1)]
    counts = [{"0": 90, "1": 10} if name.endswith("_0") else {"0": 20, "1": 80}
              for name in order]
    assert fit_operational_readout(order, counts, 100) == [[.1, .2]] * 3
    with pytest.raises(ValueError, match="Calibration order"):
        fit_operational_readout(order, counts[:-1], 100)


def test_readout_override_changes_predictions_without_changing_gate_model():
    payload, _, _, local = build_protocol(FakeMarrakesh())
    circuit = local["single_return"]
    snapshot = payload["calibration_snapshot"]
    original = predicted_distribution(circuit, snapshot)
    overridden = predicted_distribution(circuit, snapshot, readout_override=[[0., 0.]])
    assert overridden["0"] > original["0"]
    with pytest.raises(ValueError, match="Readout probabilities"):
        predicted_distribution(circuit, snapshot, readout_override=[[1.2, 0.]])


def test_analysis_refuses_dry_run_and_checks_calibration_fit():
    with pytest.raises(ValueError, match="Completed hardware"):
        analyze({"dry_run": True})
    payload, _, _, local = build_protocol(FakeMarrakesh())
    payload["dry_run"] = False
    payload["calibration_job_id"] = "synthetic-cal"
    payload["target_job_id"] = "synthetic-target"
    payload["calibration_counts"] = [
        {"0": SHOTS - 20, "1": 20} if name.endswith("_0")
        else {"0": 30, "1": SHOTS - 30}
        for name in payload["calibration_order"]
    ]
    payload["operational_readout"] = fit_operational_readout(
        payload["calibration_order"], payload["calibration_counts"], SHOTS)
    payload["calibrated_predictions"] = {
        name: predicted_distribution(local[name], payload["calibration_snapshot"],
                                     readout_override=payload["operational_readout"][:local[name].num_qubits])
        for name in TARGET_NAMES
    }
    payload["target_counts"] = [{EVENTS[name][0]: SHOTS} for name in payload["target_order"]]
    result = analyze(payload)
    assert len(result["target_summary"]) == len(TARGET_NAMES)
    assert all(row["shots"] == SHOTS * BLOCKS for row in result["target_summary"].values())
    payload["operational_readout"][0][0] = .7
    with pytest.raises(ValueError, match="readout fit"):
        analyze(payload)


def test_hardware_sampler_options_explicitly_disable_hidden_suppression():
    sampler = configured_sampler(FakeMarrakesh())
    assert sampler.options.dynamical_decoupling.enable is False
    assert sampler.options.twirling.enable_gates is False
    assert sampler.options.twirling.enable_measure is False
    assert sampler.options.max_execution_time == 60
