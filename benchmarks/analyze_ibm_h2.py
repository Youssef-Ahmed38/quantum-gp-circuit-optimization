"""Pre-registered analysis of the H2 hardware test (see evidence/ibm_dd_protocol/h2_preregistration.json).

Primary endpoint: P(all six qubits return to 0) per policy, pooled over 3 randomized
blocks x 2000 shots. Primary comparison: gp_dd minus the best of {no_dd,
xx_qiskit_like, xy4_like}, paired by block, with per-block values and a shot-only 95%
interval. Secondary: predicted vs observed ranking (Spearman) and per-qubit marginals.
Written before the H2 results were available.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence" / "ibm_dd_protocol"
STANDARD = ["no_dd", "xx_qiskit_like", "xy4_like"]


def main(tag=""):
    from qiskit_ibm_runtime import QiskitRuntimeService
    logging.disable(logging.WARNING)
    suffix = f"-{tag}" if tag else ""
    record = json.loads((OUT / f"submit-h2{suffix}_job.json").read_text())
    pre = json.loads((OUT / "h2_preregistration.json").read_text())
    job = QiskitRuntimeService().job(record["job_id"])
    result = job.result()
    labels = record["labels"]
    blocks = {}
    marginals = {}
    for position, (block, index) in enumerate(record["order"]):
        counts = result[position].data.c.get_counts()
        shots = sum(counts.values())
        width = len(next(iter(counts)))
        policy = labels[index]
        blocks.setdefault(policy, {})[block] = (counts.get("0" * width, 0), shots)
        m = marginals.setdefault(policy, np.zeros(width))
        for key, value in counts.items():
            for q in range(width):
                if key[::-1][q] == "0":
                    m[q] += value
    pooled = {}
    for policy, per in blocks.items():
        hits = sum(h for h, _ in per.values())
        total = sum(n for _, n in per.values())
        p = hits / total
        pooled[policy] = dict(p_all_zero=p, shot_se=math.sqrt(p * (1 - p) / total),
                              per_block=[per[b][0] / per[b][1] for b in sorted(per)], shots=total)
    best_standard = max(STANDARD, key=lambda k: pooled[k]["p_all_zero"])
    gp = pooled["gp_dd"]
    ref = pooled[best_standard]
    diff = gp["p_all_zero"] - ref["p_all_zero"]
    se = math.sqrt(gp["shot_se"] ** 2 + ref["shot_se"] ** 2)
    per_block_diff = [a - b for a, b in zip(gp["per_block"], ref["per_block"])]
    predicted = pre["predicted_order_best_first"]
    observed = sorted(pooled, key=lambda k: -pooled[k]["p_all_zero"])
    rho = float(spearmanr([predicted.index(k) for k in pooled], [observed.index(k) for k in pooled]).statistic)
    width = len(marginals["gp_dd"])
    qubit_marginals = {policy: (m / pooled[policy]["shots"]).tolist() for policy, m in marginals.items()}
    summary = dict(
        job_id=record["job_id"], backend=record["backend"], layout=record["layout"],
        pooled=pooled, best_standard=best_standard,
        primary=dict(gp_dd_minus_best_standard=diff, shot_only_95ci=[diff - 1.96 * se, diff + 1.96 * se],
                     per_block_differences=per_block_diff,
                     blocks_favouring_gp=sum(d > 0 for d in per_block_diff),
                     interpretation=("GP-DD better than the best standard schedule" if diff - 1.96 * se > 0 else
                                     "GP-DD worse than the best standard schedule" if diff + 1.96 * se < 0 else
                                     "No significant difference from the best standard schedule")),
        secondary=dict(predicted_order=predicted, observed_order=observed, spearman=rho,
                       qubit_marginals_p0=qubit_marginals,
                       qubit54_index=1,
                       qubit54_p0={k: v[1] for k, v in qubit_marginals.items()}),
        caveats=["Shot-only intervals exclude drift; three blocks on one qubit chain is thin evidence",
                 "Readout error is not mitigated and affects all policies equally"])
    (OUT / f"h2{suffix}_analysis.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "")
