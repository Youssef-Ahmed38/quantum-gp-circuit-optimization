"""Pre-registered fixed-effect pooling of the H3 primary (gp_dd - cpmg8_uniform) over chains A and B.
Written before chain-B results existed."""
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "evidence" / "ibm_dd_protocol"
rows = {}
num = den = 0.
for chain, name in (("A", "h3_analysis.json"), ("B", "h3-b_analysis.json")):
    primary = json.loads((OUT / name).read_text())["primary_gp_minus_cpmg8"]
    d = primary["difference"]
    se = (primary["ci95"][1] - primary["ci95"][0]) / (2 * 1.96)
    rows[chain] = dict(difference=d, se=se)
    num += d / se ** 2
    den += 1 / se ** 2
pooled, se = num / den, math.sqrt(1 / den)
summary = dict(chains=rows, pooled_difference=pooled, pooled_se=se, pooled_95ci=[pooled - 1.96 * se, pooled + 1.96 * se],
               significant=pooled - 1.96 * se > 0 or pooled + 1.96 * se < 0)
(OUT / "h3_pooled.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
