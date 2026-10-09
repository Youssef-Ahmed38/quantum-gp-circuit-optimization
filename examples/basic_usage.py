from qiskit.transpiler import PassManager
from qcgp import HybridCompiler, GPSchedulingPass
from qcgp.core import benchmark, topology

logical = benchmark("qaoa", 3, 101)
coupling = topology(3, "line")
result = HybridCompiler(coupling).compile(logical)
manager = PassManager([GPSchedulingPass(coupling)])
scheduled_circuit = manager.run(result.circuit)
print("Selected candidate:", result.selected_candidate)
print("Makespan:", manager.property_set["gp_schedule"].makespan)
print("Guarantee:", result.guarantee)
