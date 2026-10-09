"""Free local, finite-pulse timing and conditional channel certificate example."""
from qiskit import QuantumCircuit
from qcgp.physical import PhysicalParameters,physical_model,physical_channel
from qcgp.pathflow import optimize_pathflow
from qcgp.library import duration_library,optimize_library
from qcgp.characterization import characterize_library,library_certificate,transfer_gp_lower_bound
from qcgp.timing import reference_metrics
import json

def main():
    circuit=QuantumCircuit(2);circuit.rx(.8,0);circuit.rxx(.6,0,1);circuit.rz(-.4,1)
    parameters=PhysicalParameters((110.,130.),(180.,190.))
    model=physical_model(circuit,parameters);flow=optimize_pathflow(model);library=duration_library(model)
    bounds,rates,_=characterize_library(circuit,parameters,library)
    finite=optimize_library(model,library,pulse_bounds=bounds,support_rates=rates)
    _,metadata=physical_channel(circuit,parameters,finite.durations)
    report=dict(continuous_gap=flow.relative_gap,finite_gap=finite.relative_gap,
        general_certificate=metadata['certificate'],library_certificate=library_certificate(model,finite.durations,library,bounds,rates),
        transferred_bound=transfer_gp_lower_bound(model,library,bounds,rates,flow.dual_bound,reference_metrics(model)),
        hardware_calibrated=False,all_free_local=True)
    print(json.dumps(report,indent=2))
    return report
if __name__=='__main__':main()
