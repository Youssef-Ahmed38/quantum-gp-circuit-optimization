# Quantum circuit timing with geometric programming

Research code, recorded simulation data, and an editable manuscript for convex quantum-gate duration allocation. The main optimizer acts on a **fixed circuit dependency graph**. It allocates positive gate durations subject to timing and model-based error constraints; it does not prove a globally optimal gate decomposition, mapping, or routing.

## What is in this repository

- `src/qcgp/`: geometric-programming and convex solvers, scheduling utilities, observable and physical-noise models, and Qiskit analysis passes.
- `benchmarks/`: scripts for QFT, Grover, VQE, QAOA, timing, observable, physical-channel, and baseline studies.
- `evidence/`: recorded inputs and outputs, including a saved `ibm_torino` calibration snapshot, serialized circuits, benchmark tables, protocols, and a SHA-256 file manifest.
- `tests/`: unit and integration tests.
- `examples/`: runnable demonstrations and notebooks.
- `figures/`: plots generated for the research manuscript.
- `paper/`: editable LaTeX manuscript and a rendered PDF.
- `docs/REPRODUCIBILITY.md`: a guide to reproducing and interpreting the evidence.

## Evidence boundary

**All experiments in this repository use free local simulators or calculations. No IBM Quantum job was run for these results.** The `ibm_torino` file is a saved calibration snapshot used as model input; calibration-anchored simulation is not hardware validation. The guarantees proved in the manuscript hold within the stated mathematical and noise-model assumptions. In particular, the continuous timing solver does not imply that an IBM transmon backend exposes arbitrary gate-duration control.

This is a research artifact accompanying a manuscript under consideration. The manuscript and results have not been endorsed or certified by GitHub, IBM, Qiskit, or the journal. No DOI has been assigned to this repository.

## Install

Python 3.12 or newer is required. In a fresh virtual environment, run:

```bash
python -m pip install -e ".[test,plots,ecos,baselines]"
```

The dependency versions are specified in `pyproject.toml`. The `baselines` extra is needed for the TKET and Cirq adapter tests; `ml` adds the machine-learning comparison packages. A full optional dependency list is in `requirements-full.txt`.

## Run a small example

```bash
python examples/quick_reproduce_pass.py
```

The example loads the saved calibration snapshot and applies `ConvexDurationAllocationPass` inside a Qiskit `PassManager`. The pass records timing results in the pass manager's `property_set`; it is an analysis pass and does not change the circuit's gates.

## Check and reproduce

```bash
python -m pytest -q tests
python benchmarks/verify_evidence.py
python reproduce_all.py
```

`reproduce_all.py` runs the test suite, the short pass example, and the published analyses and figures using local computation. Full benchmark drivers are available in `benchmarks/`; they can take substantially longer and may overwrite recorded evidence, so use a fresh copy of the repository for a complete rerun. See [reproducibility notes](docs/REPRODUCIBILITY.md) for protocols, data provenance, and limitations.

To compile the manuscript, run a LaTeX engine with the packages declared in the source from the `paper/` directory:

```bash
cd paper
tectonic quantum_gp_manuscript.tex
```

The LaTeX file refers to tables in `../docs/` and figures in `../figures/`.

## Citation and license

Citation metadata is in [CITATION.cff](CITATION.cff). Please cite a published version if one becomes available; there is currently no verified journal DOI. The source code and documentation are released under the [MIT License](LICENSE). The saved calibration snapshot and third-party dependencies remain subject to their original provenance and terms.

## Contact

Youssef Ahmed Ramadan — `p-youssef.ramdan@zewailcity.edu.eg`
