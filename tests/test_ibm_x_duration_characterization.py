"""Check that the frozen X-duration protocol encodes the intended controls."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from run_ibm_x_duration_characterization import (
    BASES, BLOCKS, LENGTHS, MODES, QUBITS, build_protocol,
    prepare_fake_backend,
)


def test_same_logical_gate_and_matched_duration_controls():
    backend = prepare_fake_backend()
    payload, circuits = build_protocol(backend)
    assert len(circuits) == BLOCKS * len(QUBITS) * len(BASES) * (
        1 + (len(LENGTHS) - 1) * len(MODES)
    )
    assert all(sum(counts.values()) >= 1 for counts in payload["compiled_counts"])
    for condition, operations, qasm in zip(
        payload["order"], payload["compiled_counts"], payload["qasm3"]
    ):
        length, mode = condition["length"], condition["mode"]
        if mode == "slow":
            assert operations.get("xslow", 0) == length
            assert "xslow" in qasm
        elif mode == "padded":
            assert operations.get("x", 0) == length
            assert operations.get("delay", 0) == length
        else:
            assert operations.get("x", 0) == length
            assert operations.get("delay", 0) == 0
    for info in payload["durations"].values():
        assert abs(info["x_seconds"] + info["pad_dt"] * backend.dt
                   - info["xslow_seconds"]) < 1e-12


def test_frozen_design_has_all_conditions_each_block():
    payload, _ = build_protocol(prepare_fake_backend())
    block_size = len(payload["order"]) // BLOCKS
    for block in range(BLOCKS):
        conditions = payload["order"][block * block_size:(block + 1) * block_size]
        keys = {(row["qubit"], row["basis"], row["mode"], row["length"])
                for row in conditions}
        assert len(keys) == block_size
        for q in QUBITS:
            for basis in BASES:
                assert (q, basis, "fast", 0) in keys
                for length in LENGTHS[1:]:
                    for mode in MODES:
                        assert (q, basis, mode, length) in keys
