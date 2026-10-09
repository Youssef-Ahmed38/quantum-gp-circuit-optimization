"""Re-open saved circuits and independently verify serialized scheduling evidence."""
from pathlib import Path
import json
from qiskit import qpy
from qcgp.core import extract, topology, critical_path, constraint_violation, unitary_error
import numpy as np


def main():
    folder=Path(__file__).resolve().parents[1]/"evidence"
    raw=json.loads((folder/"candidates.json").read_text(encoding="utf8"))
    baselines=json.loads((folder/"baselines.json").read_text(encoding="utf8"))
    loaded={}; spot_checks=set()
    for row in raw:
        case=row["case"]
        if case not in loaded:
            with (folder/"circuits"/f"{case}.qpy").open("rb") as f: loaded[case]=qpy.load(f)
        assert len(loaded[case])==5
        circuit=loaded[case][row["candidate"]+1]
        durations,edges,_,_=extract(circuit,topology(row["qubits"],row["topology"]))
        assert len(durations)==row["gates"] and circuit.depth()==row["depth"]
        path=folder/"circuits"/f"{case}_candidate{row['candidate']}_schedule.json"
        schedule=json.loads(path.read_text(encoding="utf8"))
        assert np.max(abs(durations-np.array(schedule["durations"])))<1e-12
        assert edges==[tuple(e) for e in schedule["edges"]]
        independent=critical_path(durations,edges).makespan
        actual_violation=constraint_violation(np.array(schedule["gp_starts"]),schedule["gp_makespan"],durations,edges)
        assert abs(independent-schedule["gp_makespan"])<=1e-5*max(1,independent)
        assert actual_violation<=1e-5*max(1,independent)
        assert abs(actual_violation-row["gp_violation"])<1e-9
        key=row["family"],row["qubits"],row["topology"]
        if key not in spot_checks:
            assert unitary_error(loaded[case][0],circuit)<1e-8
            spot_checks.add(key)
    for row in baselines:
        path=folder/f"baseline_{row['case']}_{row['method']}.qpy"
        with path.open("rb") as f: circuits=qpy.load(f)
        assert len(circuits)==1
        circuit=circuits[0]
        durations,_,_,_=extract(circuit,topology(row["qubits"],row["topology"]))
        assert len(durations)==row["gates"] and circuit.depth()==row["depth"]
    status={"saved_candidate_circuits_reopened":len(raw),"saved_baseline_circuits_reopened":len(baselines),
            "schedules_independently_rechecked":len(raw),"serialized_unitary_spot_checks":len(spot_checks),
            "all_checks_passed":True}
    (folder/"serialization_validation.json").write_text(json.dumps(status,indent=2),encoding="utf8")
    print(json.dumps(status,indent=2))


if __name__=="__main__":main()
