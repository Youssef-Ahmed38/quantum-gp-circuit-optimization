import importlib.util
from pathlib import Path

import pytest
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh


ROOT = Path(__file__).resolve().parents[1]


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _module("ibm_model_runner", "benchmarks/run_ibm_model_check.py")
analysis = _module("ibm_model_analysis", "benchmarks/analyze_ibm_model_check.py")


def test_fake_backend_compiles_distinct_model_check_circuits():
    payload, isa = runner.build_protocol(FakeMarrakesh(), shots=128, blocks=2)
    assert len(isa) == 8
    assert set(payload["compiled_gate_counts"]) == set(runner.NAMES)
    assert payload["compiled_gate_counts"]["bell"]["cz"] == 1
    assert payload["compiled_gate_counts"]["ghz"]["cz"] == 2
    for probabilities in payload["predicted_observed_distributions"].values():
        assert sum(probabilities.values()) == pytest.approx(1.)


def test_model_analysis_uses_counts_without_claiming_gp_gain():
    payload, _ = runner.build_protocol(FakeMarrakesh(), shots=128, blocks=2)
    payload.update(dry_run=False, job_id="synthetic-test",
                   counts=[{max(payload["predicted_observed_distributions"][name],
                                key=payload["predicted_observed_distributions"][name].get): 128}
                           for name in payload["order"]])
    result = analysis.analyze(payload)
    assert result["purpose"].endswith("not GP hardware improvement")
    assert all(row["shots"] == 256 for row in result["results"].values())


def test_dry_run_cannot_be_analyzed_as_hardware_data():
    with pytest.raises(ValueError, match="Actual IBM counts"):
        analysis.analyze({"dry_run": True})
