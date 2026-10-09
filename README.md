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
- `docs/REVIEW_RESPONSE.md`: a point-by-point account of the latest scientific critique, revisions, and unresolved evidence gaps.
- `docs/TARGET_CALIBRATION_PROTOCOL.md`: the prospective experiment needed to test physical benefit on a target device.
- `docs/HARDWARE_STATUS.md`: free Heron QPU access, idle-noise, and prespecified nominal-gate model checks, with raw counts and explicit limits.
- `docs/ENHANCEMENT_ROADMAP.md`: decision gates and a matched experimental protocol for an executable optimizer claim.
- `docs/GP_VALIDATION_2026-10-10.md`: independent global-optimum bounds, measured-duration decisions, matched simulator results, and the remaining physical-validation boundary.

## Evidence boundary

**The GP performance comparisons use local simulators or calculations.** Three free Open-instance jobs on `ibm_marrakesh` provide an access control, an exploratory idle-noise pilot, and a prespecified fixed-native-gate model check. The last test found that all four predicted success probabilities missed a two-percentage-point descriptive tolerance; none of the jobs executed GP-selected gate durations or established a GP hardware benefit. See [hardware status](docs/HARDWARE_STATUS.md) and the [frozen model-check protocol](docs/IBM_MODEL_CHECK_PROTOCOL.md). The `ibm_torino` file is a saved calibration snapshot used as model input; calibration-anchored simulation is not hardware validation. The guarantees proved in the manuscript hold within the stated mathematical and noise-model assumptions. In particular, the continuous timing solver does not imply that an IBM transmon backend exposes arbitrary gate-duration control.

This is a research artifact accompanying a manuscript under consideration. The manuscript and results have not been endorsed or certified by GitHub, IBM, Qiskit, or the journal. No DOI has been assigned to this repository.

The added `evidence/feasible_scale/` study exercises feasible continuous solves at 32, 64, and 128 qubits. Its targets are selected from each instance's *assumed-model* attainable range. They are not measured hardware targets, and the study does not validate the inverse-duration noise law. The primary matched-deadline fidelity comparison shows little modeled benefit.

The `evidence/schedule_noise/` study adds matched 168-ns Qiskit ASAP/ALAP and X-X dynamical-decoupling schedules, with independent local density-matrix evaluation of idle and anisotropic noise. It also exports reported error/duration calibration pairs. Those pairs are cross-sectional cached calibration entries, not a controlled measured duration sweep, so no duration-response curve is fitted. Run `python benchmarks/run_schedule_noise.py` to reproduce the study; see [reproducibility notes](docs/REPRODUCIBILITY.md) for the exact noise assumptions and data-access status.

The `evidence/measured_duration/` study uses published experimental single-qubit randomized-benchmarking error-versus-duration measurements with uncertainties. A held-out test of the paper's inverse-plus-linear error law shows substantial residuals relative to the reported uncertainty. Run `python benchmarks/run_measured_duration.py` to fetch and verify the public source archive and regenerate the evidence. The measurements are from a different platform and cannot calibrate the IBM-snapshot circuit study.

An exploratory `A/t**2 + G*t` model lowers leave-one-duration-out mean absolute error across the four measured pulse families while retaining the continuous geometric-program structure. It still misses the measurements by more than their reported uncertainty and was proposed after inspecting earlier holdout data. Reproduce with `python benchmarks/compare_duration_laws.py`; see `src/qcgp/generalized.py` for the declared-exponent solver.

`benchmarks/evaluate_measured_choices.py` adds a retrospective decision audit on that measured dataset. Both inverse-power laws and a shape-preserving empirical interpolator choose from the same withheld duration options. The interpolator selects the lowest reported-error duration in three of four pulse families, but it was examined after earlier results and its residuals still exceed source measurement uncertainties. The script records a separate train-only model-discrepancy diagnostic; it does not claim a calibrated confidence interval or a prospective device gain. Outputs are in `evidence/measured_duration/choice_audit.json`, `docs/measured_choice_table.tex`, and `figures/measured_choice.pdf`.

`solve_robust_generalized` in `src/qcgp/generalized.py` accepts a declared finite set of coefficient scenarios and minimizes the worst scenario error under the same DAG and deadline. Its epigraph remains a geometric program. The supplied scenario set is an assumption; it gains a physical coverage interpretation only after a target-device calibration study supports it.

The `evidence/direct_matched/` study directly compares idealized GP duration choices with Qiskit ASAP/ALAP scheduling, with and without two-X dynamical decoupling, on the same three-qubit circuit and 168-ns deadline. Its local density-matrix evaluator tests aligned and misspecified duration-error laws, anisotropic Pauli noise, and shot-static detuning. The GP variants use hypothetical continuous alternate-duration pulses; the Qiskit variants use fixed nominal durations. A small simulated GP gain under Markovian assumptions reverses strongly in favor of DD under static detuning. This is a sensitivity study, not a calibrated device result. Run `python benchmarks/run_direct_matched.py` to regenerate the JSON and plots.

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
For a nontrivial physical layout, pass the circuit-to-snapshot index mapping with `physical_qubits`; the identity mapping is only a default for small examples. Snapshot-based observable optimization currently uses the free Clarabel solver.

## Check and reproduce

```bash
python -m pytest -q tests
python benchmarks/verify_evidence.py
python reproduce_all.py
```

`reproduce_all.py` runs the test suite, the short pass example, and the published analyses and figures using local computation. Full benchmark drivers are available in `benchmarks/`; they can take substantially longer and may overwrite recorded evidence, so use a fresh copy of the repository for a complete rerun. See [reproducibility notes](docs/REPRODUCIBILITY.md) for protocols, data provenance, and limitations.

To regenerate the added feasible large-circuit study in a fresh copy, run `python benchmarks/run_feasible_scale.py` followed by `python benchmarks/analyze_feasible_scale.py`. The saved protocol and raw records are in `evidence/feasible_scale/`.

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
