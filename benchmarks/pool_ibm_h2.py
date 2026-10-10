"""Pre-registered fixed-effect pooling of H2 and its replication (written before R1 results)."""
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "evidence" / "ibm_dd_protocol"


def primary(analysis):
    d = analysis["primary"]["gp_dd_minus_best_standard"]
    low, high = analysis["primary"]["shot_only_95ci"]
    return d, (high - low) / (2 * 1.96)


runs = {"H2": json.loads((OUT / "h2_analysis.json").read_text()),
        "R1": json.loads((OUT / "h2-r1_analysis.json").read_text())}
weights, total = 0., 0.
rows = {}
for name, analysis in runs.items():
    d, se = primary(analysis)
    rows[name] = dict(difference=d, se=se, best_standard=analysis["best_standard"])
    weights += 1 / se ** 2
    total += d / se ** 2
pooled = total / weights
se = math.sqrt(1 / weights)
summary = dict(runs=rows, pooled_difference=pooled, pooled_se=se, pooled_95ci=[pooled - 1.96 * se, pooled + 1.96 * se],
               significant=pooled - 1.96 * se > 0 or pooled + 1.96 * se < 0)
(OUT / "h2_pooled.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
