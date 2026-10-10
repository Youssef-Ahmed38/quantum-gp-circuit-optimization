"""Pre-registered analysis of H3 (control arms), written before H3 data existed.

Primary: gp_dd minus cpmg8_uniform in P(all six qubits return to 0), pooled over
4 blocks x 6000 shots, shot-only 95% CI, per-block differences.
Secondary: gp_dd minus gp_dd_protect_q54 (value of leaving qubit 54 undecoupled);
gp_dd minus xy4_like (replication of H2/R1); observed vs predicted order
(gp_dd > gp_dd_protect_q54 > cpmg8_uniform > xy4_like).
"""
import json
import logging
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence" / "ibm_dd_protocol"
PREDICTED = ["gp_dd", "gp_dd_protect_q54", "cpmg8_uniform", "xy4_like"]


def main(tag=""):
    from qiskit_ibm_runtime import QiskitRuntimeService
    logging.disable(logging.WARNING)
    suffix = f"-{tag}" if tag else ""
    record = json.loads((OUT / f"submit-h3{suffix}_job.json").read_text())
    result = QiskitRuntimeService().job(record["job_id"]).result()
    per = {}
    marg = {}
    for position, (block, index) in enumerate(record["order"]):
        counts = result[position].data.c.get_counts()
        shots = sum(counts.values())
        width = len(next(iter(counts)))
        policy = record["labels"][index]
        per.setdefault(policy, {})[block] = (counts.get("0" * width, 0), shots)
        m = marg.setdefault(policy, [0] * width)
        for key, value in counts.items():
            for q in range(width):
                m[q] += value if key[::-1][q] == "0" else 0
    pooled = {}
    for policy, blocks in per.items():
        hits = sum(h for h, _ in blocks.values())
        total = sum(n for _, n in blocks.values())
        p = hits / total
        pooled[policy] = dict(p_all_zero=p, shot_se=math.sqrt(p * (1 - p) / total), shots=total,
                              per_block=[blocks[b][0] / blocks[b][1] for b in sorted(blocks)],
                              qubit_p0=[x / total for x in marg[policy]])

    def compare(a, b):
        d = pooled[a]["p_all_zero"] - pooled[b]["p_all_zero"]
        se = math.hypot(pooled[a]["shot_se"], pooled[b]["shot_se"])
        blocks = [x - y for x, y in zip(pooled[a]["per_block"], pooled[b]["per_block"])]
        return dict(difference=d, ci95=[d - 1.96 * se, d + 1.96 * se], per_block=blocks,
                    blocks_favouring=sum(x > 0 for x in blocks),
                    significant=(d - 1.96 * se > 0) or (d + 1.96 * se < 0))

    observed = sorted(pooled, key=lambda k: -pooled[k]["p_all_zero"])
    summary = dict(job_id=record["job_id"], backend=record["backend"], layout=record["layout"], pooled=pooled,
                   primary_gp_minus_cpmg8=compare("gp_dd", "cpmg8_uniform"),
                   secondary_gp_minus_protect_q54=compare("gp_dd", "gp_dd_protect_q54"),
                   secondary_gp_minus_xy4=compare("gp_dd", "xy4_like"),
                   predicted_order=PREDICTED, observed_order=observed,
                   order_matches=observed == PREDICTED)
    (OUT / f"h3{suffix}_analysis.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "")
