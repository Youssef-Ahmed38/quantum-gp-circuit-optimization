"""Create and execute a small reproducible notebook without external services."""
from pathlib import Path
import contextlib
import io
import json

ROOT=Path(__file__).resolve().parents[1]
code="""from qiskit import QuantumCircuit
from qcgp.timing import synthetic_model, optimize_timing, simulate_timing
from qcgp.pathflow import optimize_pathflow
import numpy as np
circuit=QuantumCircuit(2)
circuit.sx(0); circuit.cx(0,1); circuit.rz(.4,1); circuit.x(0)
model=synthetic_model(circuit)
flow=optimize_pathflow(model,weights=(.6,.3,.1))
direct=optimize_timing(model,weights=(.6,.3,.1),method='convex')
assert flow.certified
assert abs(flow.timing.objective-direct.objective)<2e-5
actual,ideal=simulate_timing(circuit,model,flow.timing)
fidelity=float(np.real(np.trace(actual.data@ideal.data)))
assert fidelity>=flow.timing.fidelity_lower_bound-1e-9
print('Durations:',flow.timing.durations)
print('Numerical relative optimum gap:',flow.relative_gap)
print('Conditional fidelity floor:',flow.timing.fidelity_lower_bound)
print('Simulated fidelity:',fidelity)
"""


def main():
    buffer=io.StringIO()
    with contextlib.redirect_stdout(buffer):exec(compile(code,'joint_framework.ipynb','exec'),{})
    book=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.12.14')),cells=[
        dict(cell_type='markdown',id='model-scope',metadata={},source=['# Certified joint timing\n','Free local example with constructed coefficients. The certificate is conditional on the noise model, and this notebook does not synthesize hardware pulses.\n']),
        dict(cell_type='code',id='joint-example',metadata={},source=code.splitlines(keepends=True),execution_count=1,outputs=[dict(output_type='stream',name='stdout',text=buffer.getvalue().splitlines(keepends=True))])])
    path=ROOT/'examples/joint_framework.ipynb';path.write_text(json.dumps(book,indent=2)+'\n')
    reread=json.loads(path.read_text());assert reread['cells'][1]['execution_count']==1
    print(buffer.getvalue())


if __name__=='__main__':main()
