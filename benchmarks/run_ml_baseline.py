"""Exploratory free seed-selection baseline; not a published RL reproduction.

Leave-one-qubit-size-out evaluation prevents identical size strata from appearing
in training and testing. Training uses only logical structural input features.
"""
from pathlib import Path
import csv
import json
import time
import importlib.metadata
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from qiskit import qpy
from qcgp.core import normalized_score


def main():
    out = Path(__file__).resolve().parents[1]/"evidence"
    cases=json.loads((out/"comparisons.json").read_text(encoding="utf8"))
    raw=json.loads((out/"candidates.json").read_text(encoding="utf8"))
    lookup={(r["case"],r["candidate"]):r for r in raw}
    families=["qft","grover","vqe_ansatz","qaoa"]
    features=[]
    for case in cases:
        with (out/"circuits"/f"{case['case']}.qpy").open("rb") as f:
            logical=qpy.load(f)[0]
        two=sum(len(ins.qubits)==2 for ins in logical.data)
        features.append([case["qubits"],logical.depth(),len(logical.data),two,
                         int(case["topology"]=="ring"),
                         *[int(case["family"]==name) for name in families]])
    x=np.asarray(features,float)
    y=np.asarray([c["chosen_candidate"] for c in cases])
    rows=[]; folds=[]
    for heldout in sorted({c["qubits"] for c in cases}):
        train=np.asarray([c["qubits"]!=heldout for c in cases])
        test=~train
        assert train.any() and test.any()
        model=RandomForestClassifier(n_estimators=200,max_depth=6,
                                     random_state=20261009,n_jobs=1)
        start=time.perf_counter(); model.fit(x[train],y[train]); fit=time.perf_counter()-start
        start=time.perf_counter(); predictions=model.predict(x[test]); inference=time.perf_counter()-start
        folds.append(dict(heldout_size=heldout,train_cases=int(train.sum()),
                          test_cases=int(test.sum()),fit_seconds=fit,inference_seconds=inference))
        for index,prediction in zip(np.flatnonzero(test),predictions):
            case=cases[index]; baseline=lookup[(case["case"],0)]
            selected=lookup[(case["case"],int(prediction))]
            row={k:case[k] for k in ("case","family","qubits","topology","instance")}
            row.update(predicted_candidate=int(prediction),oracle_candidate=int(y[index]),
                       normalized_score=normalized_score(selected,baseline),
                       oracle_score=1-case["score_improvement"]/100,
                       **{k:selected[k] for k in ("depth","gates","two_qubit","makespan",
                                                 "error_loss","simulated_fidelity")})
            rows.append(row)
    (out/"ml_baseline.json").write_text(json.dumps(rows,indent=2),encoding="utf8")
    with (out/"ml_baseline.csv").open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/"ml_environment.json").write_text(json.dumps({"scikit_learn":importlib.metadata.version("scikit-learn"),
        "features":["qubits","logical_depth","logical_gates","logical_2q","ring","family_one_hot"],
        "split":"leave-one-qubit-size-out","folds":folds,
        "published_ML_method_reproduction":False},indent=2),encoding="utf8")
    print(f"Out-of-size predictions evaluated for {len(rows)} configurations")


if __name__=="__main__":main()
