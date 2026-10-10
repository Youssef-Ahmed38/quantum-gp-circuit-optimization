# Quantum circuit timing with geometric programming

Research code, recorded simulation data, and an editable manuscript for convex quantum-gate duration allocation. The main optimizer acts on a **fixed circuit dependency graph**. It allocates positive gate durations subject to timing and model-based error constraints; it does not prove a globally optimal gate decomposition, mapping, or routing.

## What is in this repository

- `src/qcgp/`: geometric-programming and convex solvers, scheduling utilities, observable and physical-noise models, and Qiskit analysis passes.
- `benchmarks/`: scripts for QFT, Grover, VQE, QAOA, timing, observable, physical-channel, and baseline studies.
- `evidence/`: recorded inputs and outputs, including a saved `ibm_torino` calibration snapshot, serialized circuits, benchmark tables, protocols, and a SHA-256 file manifest.
- `tests/`: unit and integration tests.
- `examples/`: runnable demonstrations and notebooks.
- Figures and LaTeX tables are not stored in the repository; the benchmark scripts regenerate them locally (PDF figures under `figures/`, LaTeX tables under `docs/` and `paper/`).
- `docs/REPRODUCIBILITY.md`: a guide to reproducing and interpreting the evidence.
- `docs/REVIEW_RESPONSE.md`: a point-by-point account of the latest scientific critique, revisions, and unresolved evidence gaps.
- `docs/TARGET_CALIBRATION_PROTOCOL.md`: the prospective experiment needed to test physical benefit on a target device.
- `docs/HARDWARE_STATUS.md`: free Heron QPU access, model checks, and a training-selected held-out native-X comparison, with raw counts and explicit limits.
- `docs/ENHANCEMENT_ROADMAP.md`: decision gates and a matched experimental protocol for an executable optimizer claim.
- `docs/GP_VALIDATION_2026-10-10.md`: independent global-optimum bounds, measured-duration decisions, matched simulator results, and the remaining physical-validation boundary.

## Evidence boundary

**The GP performance comparisons use local simulators or calculations.** Seven free Open-instance jobs on `ibm_marrakesh` provide an access control, an idle-noise pilot, two nominal-model checks, and a native-X training/held-out comparison. The first fixed-gate model check missed its descriptive tolerance; the second passed after calibration changed, without support for empirical readout correction. In the new native-X holdout, a longer physical X implementation improved X-basis return by 4.25 percentage points over an equal-duration Qiskit two-X DD control at 48 gates on one qubit, but ranked worse on a Z-basis input. **This is a measured, observable-specific gate-choice result, not a GP-selected hardware improvement.** No job has executed the paper's continuous GP duration policy. See [hardware status](docs/HARDWARE_STATUS.md), the [X characterization protocol](docs/IBM_X_DURATION_CHARACTERIZATION_PROTOCOL.md), and the [X holdout protocol](docs/IBM_X_DURATION_HOLDOUT_PROTOCOL.md). The `ibm_torino` file is a saved calibration snapshot used as model input; calibration-anchored simulation is not hardware validation. The guarantees proved in the manuscript hold within the stated mathematical and noise-model assumptions.

This is a research artifact accompanying a manuscript under consideration. The manuscript and results have not been endorsed or certified by GitHub, IBM, Qiskit, or the journal. No DOI has been assigned to this repository.

The added `evidence/feasible_scale/` study exercises feasible continuous solves at 32, 64, and 128 qubits. Its targets are selected from each instance's *assumed-model* attainable range. They are not measured hardware targets, and the study does not validate the inverse-duration noise law. The primary matched-deadline fidelity comparison shows little modeled benefit.

The `evidence/schedule_noise/` study adds matched 168-ns Qiskit ASAP/ALAP and X-X dynamical-decoupling schedules, with independent local density-matrix evaluation of idle and anisotropic noise. It also exports reported error/duration calibration pairs. Those pairs are cross-sectional cached calibration entries, not a controlled measured duration sweep, so no duration-response curve is fitted. Run `python benchmarks/run_schedule_noise.py` to reproduce the study; see [reproducibility notes](docs/REPRODUCIBILITY.md) for the exact noise assumptions and data-access status.

The `evidence/measured_duration/` study uses published experimental single-qubit randomized-benchmarking error-versus-duration measurements with uncertainties. A held-out test of the paper's inverse-plus-linear error law shows substantial residuals relative to the reported uncertainty. Run `python benchmarks/run_measured_duration.py` to fetch and verify the public source archive and regenerate the evidence. The measurements are from a different platform and cannot calibrate the IBM-snapshot circuit study.

An exploratory `A/t**2 + G*t` model lowers leave-one-duration-out mean absolute error across the four measured pulse families while retaining the continuous geometric-program structure. It still misses the measurements by more than their reported uncertainty and was proposed after inspecting earlier holdout data. Reproduce with `python benchmarks/compare_duration_laws.py`; see `src/qcgp/generalized.py` for the declared-exponent solver.

`benchmarks/evaluate_measured_choices.py` adds a retrospective decision audit on that measured dataset. Both inverse-power laws and a shape-preserving empirical interpolator choose from the same withheld duration options. The interpolator selects the lowest reported-error duration in three of four pulse families, but it was examined after earlier results and its residuals still exceed source measurement uncertainties. The script records a separate train-only model-discrepancy diagnostic; it does not claim a calibrated confidence interval or a prospective device gain. Outputs are in `evidence/measured_duration/choice_audit.json`; the script also regenerates a LaTeX table and a PDF figure locally.

`solve_robust_generalized` in `src/qcgp/generalized.py` accepts a declared finite set of coefficient scenarios and minimizes the worst scenario error under the same DAG and deadline. Its epigraph remains a geometric program. The supplied scenario set is an assumption; it gains a physical coverage interpretation only after a target-device calibration study supports it.

The `evidence/direct_matched/` study directly compares idealized GP duration choices with Qiskit ASAP/ALAP scheduling, with and without two-X dynamical decoupling, on the same three-qubit circuit and 168-ns deadline. Its local density-matrix evaluator tests aligned and misspecified duration-error laws, anisotropic Pauli noise, and shot-static detuning. The GP variants use hypothetical continuous alternate-duration pulses; the Qiskit variants use fixed nominal durations. A small simulated GP gain under Markovian assumptions reverses strongly in favor of DD under static detuning. This is a sensitivity study, not a calibrated device result. Run `python benchmarks/run_direct_matched.py` to regenerate the JSON and plots.

## GP resource-allocation extensions (C1–C8)

Statements and proofs are in [docs/THEORY.md](docs/THEORY.md); results are in [docs/NOVELTY_RESULTS.md](docs/NOVELTY_RESULTS.md). All of these studies run on local simulators. The hardware protocol is dry-run only until you submit it.

| | Module | Study (`benchmarks/`) | Evidence |
|---|---|---|---|
| C2 joint timing + DD (convex, perspective) | `ddgp.py`, `noisesim.py` | `run_dd_gp_sim.py` | `evidence/dd_gp/` |
| C3 runtime-optimal PEC | `mitigation.py` | `run_runtime_optimal.py` | `evidence/runtime_optimal/` |
| C4 certified robust GP, baseline dominance | `robust.py` | `run_robust_pac.py` | `evidence/robust_pac/` |
| C1/C5 anisotropic, state-aware constraints | `anisotropic.py` | `run_anisotropic_prediction.py` | `evidence/anisotropic/` |
| C6 bicriteria rounding, C7 frontier | `approximation.py`, `frontier.py` | `run_theory_checks.py` | `evidence/theory_checks/` |
| C8 level-vectorised path-flow | `scale.py` | `run_scale.py` | `evidence/scale/` |
| Hardware H1/H2 (minimal free tier) | `hardware.py` | `ibm_dd_protocol.py dry-run` | `evidence/ibm_dd_protocol/` |

Unit tests: `python -m pytest tests/test_novelty.py`.

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

The manuscript source and rendered PDFs are not part of this repository. All numbers, tables and figures in the manuscript are generated from the evidence here by `benchmarks/write_novelty_outputs.py` and the other benchmark scripts.

## Citation and license

Citation metadata is in [CITATION.cff](CITATION.cff). Please cite a published version if one becomes available; there is currently no verified journal DOI. The source code and documentation are released under the [MIT License](LICENSE). The saved calibration snapshot and third-party dependencies remain subject to their original provenance and terms.

## Contact

Youssef Ahmed Ramadan — `p-youssef.ramdan@zewailcity.edu.eg`
