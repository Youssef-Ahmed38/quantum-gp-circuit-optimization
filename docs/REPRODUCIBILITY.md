# Reproducibility and interpretation

The latest independent solver and model-validation audit is in [GP_VALIDATION_2026-10-10.md](GP_VALIDATION_2026-10-10.md). Run `python benchmarks/validate_generalized_certificate.py` to regenerate its twelve supporting-plane lower-bound checks from a fixed seed. This uses free local CVXPY/Clarabel and SciPy/HiGHS solvers; no IBM credential or QPU job is required.

## What was run

The study combines Qiskit circuit construction, convex optimization, saved calibration parameters, local noise models, and locally generated benchmark results. It uses no paid hardware execution. The protocol file at `evidence/observable_study/protocol.json` records `all_free_local: true` and `hardware_jobs: 0` for that original observable study, not for the later repository as a whole. Two separate free Open-instance QPU jobs are documented in [HARDWARE_STATUS.md](HARDWARE_STATUS.md); neither implements the GP duration policy. Other evidence subdirectories contain their own result and environment files.

The saved `evidence/observable_study/props_torino.json` is a Qiskit fake-backend calibration snapshot. It anchors nominal errors and durations in a model; it is not a record of experiments run on `ibm_torino`.

## Measured-data boundary and scheduling stress test

Run `python benchmarks/run_measured_duration.py` to download the publicly available [experimental single-qubit RB duration sweep](https://doi.org/10.5281/zenodo.13808824), check the archive MD5, extract Figure 3a's four pulse families, and reproduce the held-out fits, CSV, and plot. To avoid another download, use `--archive PATH` with a verified local ZIP. The archived measurements include one-sigma uncertainties and document how RB error per gate was derived. We fit each pulse family independently to `A/t + G*t` with nonnegative weighted least squares on alternating durations; the other durations are held out. The measured error is gate-specific and pulse-specific. These measurements are external to the target IBM snapshot and show large uncertainty-normalized residuals; they do not calibrate the proposed circuit optimizer.

Run `python benchmarks/compare_duration_laws.py` after the extraction to compare the original `A/t + G*t` law with `A/t**2 + G*t` using leave-one-duration-out refits. The inverse-square candidate was proposed **after inspecting** the earlier alternating-holdout results; this second analysis is exploratory and must not be described as independent confirmation. Its mean absolute error is lower in all four pulse families, but its uncertainty-normalized residual remains large. Both laws choose the same held-out duration in each pulse family when fitted on alternating training points, so the new law has no demonstrated decision benefit here. `src/qcgp/generalized.py` implements the corresponding declared-exponent continuous GP; it has not been calibrated for IBM native gates or compared directly with DD on hardware.

Run `python benchmarks/evaluate_measured_choices.py` to regenerate `evidence/measured_duration/choice_audit.json`, `docs/measured_choice_table.tex`, and `figures/measured_choice.pdf`. It fits only the marked training durations and restricts each method to the same withheld measured pulse options. Shape-preserving cubic interpolation (PCHIP) is a retrospective empirical comparator examined after earlier law results; it is not a GP law and is not prospectively validated. The source-reported one-sigma RB errors are retained. A separate discrepancy scale is estimated from training-only leave-one-out predictions at interior training durations, then inspected on withheld residuals. The `1.96`-scale band coverage is a diagnostic, not a confidence interval: it has neither an exchangeability argument nor a calibrated target-device guarantee. Selection is verified not to depend on withheld errors by `tests/test_measured_choice.py`.

`solve_robust_generalized` implements the finite-scenario extension proved in the manuscript. Supply one row per complete coefficient scenario; this preserves correlations within that row. It solves a minimax GP with a checked deadline and independently recomputes the worst scenario objective. The checked-in public measurements do not justify any target-device scenario distribution or coverage statement. See `tests/test_generalized.py` for single-scenario equivalence and multi-scenario checks.

Run `python benchmarks/run_schedule_noise.py` to regenerate `evidence/schedule_noise/results.json`, the 566-row `reported_calibration_pairs.csv`, and `figures/schedule_noise.pdf`. The CSV contains reported gate errors and durations from the saved Qiskit fake-backend calibration file, with parameter timestamps. It is **cross-sectional calibration metadata**, not a repeated error-versus-duration sweep for one gate. We do not fit the GP duration law from those points. A separate [Bristol/4TU two-qubit duration-sweep archive](https://doi.org/10.4121/cec2d36b-c31b-4c70-a66f-65d3ef7bd514.v1) describes randomized benchmarking at varied gate durations, but its file download returned HTTP 503 (`StorageMaintenance`) on 2026-10-09. No values from that archive enter this study.

The scheduling benchmark invokes Qiskit's ASAP and ALAP analysis passes, each with delay padding or two-X dynamical-decoupling padding, on the same three-qubit circuit and native durations. All four schedules have 168-ns makespan. An independent local density-matrix calculation evaluates ideal-state fidelity under three **assumed** regimes: isotropic Markovian gate noise, Z-biased Markovian gate noise, and Z-biased noise plus shot-static Gaussian detuning during idles. Idle amplitude damping and Markovian dephasing are included in every regime. The simulator uses seven-node Gauss-Hermite quadrature per qubit, no hardware calls, and no paid service. Its detuning acts only during explicit delays, and it does not represent drive-calibrated pulses or hardware execution. This is a sensitivity test of scheduling and omitted idle noise, not a GP-versus-DD performance claim.

Run `python benchmarks/run_direct_matched.py` to regenerate `evidence/direct_matched/results.json` and `figures/direct_matched.pdf`. It compares four GP policies (inverse powers one or two, each placed ASAP or ALAP) against the four Qiskit schedules on the same five-gate, three-qubit circuit at a common 168-ns deadline. The cached Torino snapshot supplies one nominal duration and error per gate; the GP duration curves, 0.8 drive-error share, half-to-one-and-a-half nominal duration bounds, idle coherence times, and static-detuning scale are declared scenario assumptions. They are not estimated from the snapshot. The independent density-matrix evaluator applies duration-dependent or deliberately flat per-gate Pauli errors and idle-only T1/Tphi and shot-static Z noise. The GP continuous-duration choices are hypothetical alternate pulses, while Qiskit uses fixed nominal gates; equal circuit and deadline do not imply equal hardware action sets. The static-noise scenario favors ASAP+DD decisively, so the results do not support general GP superiority. The JSON records every policy and scenario, the assumed parameters, and the local Qiskit version.

Run `python benchmarks/check_direct_quadrature.py` to regenerate `evidence/direct_matched/quadrature_check.json`. For the leading GP-only and DD policies in the static-detuning scenario, seven- and nine-node quadrature differ by at most $1.3\times10^{-12}$ in ideal-state fidelity. This checks numerical integration convergence, not whether the noise assumptions describe a device.

The `feasible_scale` experiment uses the same snapshot and an instance-specific target chosen as the geometric mean of the worst-generator expectation at lower durations and at the coordinatewise minimum-hazard vector. This guarantees a nontrivial target attainable within the assumed post-gate isotropic Pauli model. It is a solver scalability study, not an independent noise-model test. Its 24 observations use two circuit families, two seeds, two assumed drive-error shares, and three sizes. The reported wall times are single runs; they are descriptive rather than confidence intervals. The comparison is against the earliest uniform duration scaling that meets the same target. Run `python benchmarks/run_feasible_scale.py` and `python benchmarks/analyze_feasible_scale.py` in a fresh copy to regenerate it.

## Fast verification

From the repository root in the environment described by `pyproject.toml`:

```bash
python -m pytest -q tests
python examples/quick_reproduce_pass.py
python benchmarks/verify_evidence.py
```

The first command checks core behavior. The second exercises the Qiskit pass with a small circuit. The third reopens serialized circuits and rechecks saved scheduling records.

`evidence/SHA256SUMS.txt` lists SHA-256 hashes of the checked-in evidence files. It detects accidental data changes; it is not a substitute for independent replication or physical-device validation.

## Recompute analyses

```bash
python reproduce_all.py
```

This runs analyses over saved evidence and regenerates manuscript tables and figures. Full data-generation drivers are in `benchmarks/run_*.py` and should be run in a fresh checkout because they may rewrite evidence. Their runtimes depend on the selected workload and solver.

## Scope of the mathematics

Global optimality applies to the stated continuous duration-allocation model on a fixed dependency graph. Finite circuit candidate search and hardware routing are separate discrete problems. Observable bounds rely on the specified Pauli/Markov assumptions; physical-channel bounds rely on their stated noise model. Results based on a calibration snapshot should not be described as hardware execution or device validation.

## Public data access

The `evidence/` tree holds the data supplied with this repository. No Zenodo archive or DOI exists for this release. If a DOI is assigned later, add the verified identifier to the manuscript and citation metadata.
