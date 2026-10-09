# Independent GP validation and evidence boundary (2026-10-10)

The continuous fixed-DAG duration allocator was checked at three distinct levels. The conclusion is **numerical optimization within the declared model is supported; the assumed gate-error law and physical improvement are not validated for the IBM device**.

## Independent optimum bound

Run `python benchmarks/validate_generalized_certificate.py`. Twelve deterministic instances (3, 5, and 8 gates; inverse powers 1 and 2; with and without deadlines) were solved by the CVXPY/Clarabel GP. A separate SciPy/HiGHS linear program minimized a supporting hyperplane of the convex objective over the full timing polytope. Convexity makes this an objective lower bound, while the GP duration vector evaluated on an independently calculated DAG makespan gives an upper bound. The largest recorded absolute and relative gaps are `0.00010059603688006291` and `1.1012729434349249e-05`. The script fails if a gap exceeds its declared tolerance. Inputs, durations, and bounds are in [`../evidence/generalized_certificate/results.json`](../evidence/generalized_certificate/results.json).

This certificate checks small-instance numerical optimality **for the specified coefficients and error law**. It neither establishes those coefficients experimentally nor certifies full quantum compilation or the separate 128-qubit observable study.

## External duration measurements

Run `python benchmarks/evaluate_measured_choices.py`. The source is a published single-qubit randomized-benchmarking duration sweep, with a training/test split recorded per pulse family. Both tested inverse-power GP laws select the same withheld duration in each family. They select the lowest measured-error withheld option in only one of four families. The recorded regret in the other families is positive, including `3.0756007098786675e-05` and `3.451917433427987e-05` RB error per gate in FAST_DRAG-L and Cosine_DRAG-P. The measured data are from a different platform, and the empirical PCHIP comparator was examined after earlier results; this is a retrospective model-adequacy check, not a prospective IBM trial. Raw decisions and reported measurement errors are in [`../evidence/measured_duration/choice_audit.json`](../evidence/measured_duration/choice_audit.json).

## Matched-deadline simulation

Run `python benchmarks/run_direct_matched.py`. A three-qubit circuit and 168-ns deadline are shared by all local policies. Under an aligned Markovian model, GP `p=1` with ALAP placement has simulated ideal-state fidelity `0.993483307253756`, versus `0.9931617411069962` for Qiskit ALAP. Under shot-static detuning, GP `p=1` ALAP has `0.9326712275071676`, versus `0.9934484024879879` for ASAP with two-X dynamical decoupling. These are deterministic model calculations, so no sampling confidence interval or physical-device benefit follows. The GP policies have hypothetical variable-duration gates; the Qiskit controls use fixed nominal gates. Complete results are in [`../evidence/direct_matched/results.json`](../evidence/direct_matched/results.json).

## Reproducibility check

The full repository test suite passed before the later IBM model-check additions: **158 tests**, including a check that the Qiskit duration pass leaves the emitted circuit unchanged while recording its analysis result. Two pre-existing CVXPY warnings occurred in the large-Grover regression test; they did not change the passing result. The local experiments use free solvers and simulators. A subsequent free IBM native-gate model check is documented in [HARDWARE_STATUS.md](HARDWARE_STATUS.md). Its four prespecified outcome predictions all missed a descriptive two-percentage-point tolerance; neither that job nor the earlier idle-noise pilot implements GP-chosen gate durations.

The remaining physical test requires multiple characterized, selectable implementations of the same target gate (or a newly formulated IBM-native scheduling action set), a training-only decision rule, and randomized matched hardware comparisons. IBM's current QPUs do not expose arbitrary pulse-level gate-duration control, so repeating an unchanged circuit on the Open instance would not validate the continuous GP allocation.
