"""Check the held-out X-duration comparison before hardware execution."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from run_ibm_x_duration_holdout import (
    BASES, BLOCKS, LENGTHS, MODES, build_protocol, prepare_fake_backend,
)


def test_same_duration_and_intended_native_operations():
    payload, circuits = build_protocol(prepare_fake_backend(), "training-hash")
    assert len(circuits) == BLOCKS * len(BASES) * len(LENGTHS) * len(MODES)
    assert len(payload["order"]) == len(circuits)
    grouped = {}
    for row, ticks, counts in zip(payload["order"], payload["nominal_ticks"],
                                   payload["compiled_counts"]):
        grouped.setdefault((row["basis"], row["length"]), set()).add(ticks)
        length = row["length"]
        if row["mode"] == "slow":
            assert counts.get("xslow", 0) == length
        elif row["mode"] == "dd_tail":
            assert counts.get("x", 0) == length + 2
        else:
            assert counts.get("x", 0) == length
    assert all(len(ticks) == 1 for ticks in grouped.values())


def test_every_randomized_block_is_complete():
    payload, _ = build_protocol(prepare_fake_backend(), "training-hash")
    size = len(payload["order"]) // BLOCKS
    for block in range(BLOCKS):
        rows = payload["order"][block * size:(block + 1) * size]
        assert len({(r["basis"], r["length"], r["mode"]) for r in rows}) == size
