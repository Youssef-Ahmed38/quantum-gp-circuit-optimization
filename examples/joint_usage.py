"""Free, local duration allocation and conditional certificates."""
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qcgp import JointGPTimingPass
from qcgp.timing import synthetic_model, optimize_timing, simulate_timing
from qcgp.pathflow import optimize_pathflow
import numpy as np


def main():
    circuit=QuantumCircuit(2)
    circuit.sx(0);circuit.cx(0,1);circuit.rz(.4,1);circuit.x(0)
    model=synthetic_model(circuit)  # Explicitly constructed; no calibration.
    flow=optimize_pathflow(model,weights=(.6,.3,.1))
    direct=optimize_timing(model,weights=(.6,.3,.1),method='convex')
    if not flow.certified:raise RuntimeError('Optimization did not meet requested gap')
    if abs(flow.timing.objective-direct.objective)>2e-5:raise RuntimeError('Independent solver disagreement')
    actual,ideal=simulate_timing(circuit,model,flow.timing)
    fidelity=float(np.real(np.trace(actual.data@ideal.data)))
    assert fidelity>=flow.timing.fidelity_lower_bound-1e-9
    manager=PassManager([JointGPTimingPass(synthetic_model)])
    annotated=manager.run(circuit)
    assert annotated==circuit
    print({'durations':flow.timing.durations.tolist(),'latency':flow.timing.makespan,
        'risk':flow.timing.risk,'effort':flow.timing.effort,'numerical_relative_gap':flow.relative_gap,
        'conditional_fidelity_floor':flow.timing.fidelity_lower_bound,'simulated_fidelity':fidelity,
        'direct_objective':direct.objective,'free_local_only':True})


if __name__=='__main__':main()
