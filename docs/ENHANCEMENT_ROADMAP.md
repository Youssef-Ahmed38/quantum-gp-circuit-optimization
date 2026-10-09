# What would demonstrate the contribution?

## Current evidence boundary

The mathematical contribution is an exact convex formulation for **continuous gate-duration allocation on a fixed circuit DAG under an assumed noise law**. The implementation is an analysis pass: it records a duration vector but does not emit an executable IBM circuit with those durations. The current matched GP-versus-Qiskit results are local simulations with hypothetical alternate-duration pulses. The free Heron jobs in `evidence/hardware/` measure access and idle-time behavior, not GP-selected gate durations or a matched scheduler comparison. The measured external duration sweep exposes substantial model discrepancy.

The next claim to establish is therefore specific: **a calibrated, executable policy chosen by the proposed optimizer improves a prespecified measured task metric over strong controls on the same device**. A real-QPU job alone cannot establish that claim.

## Decision gate: choose an executable action set

1. **Preserve the original gate-duration claim only where multiple calibrated implementations of the same logical gate can actually be executed.** Identify a device and gate family with an accessible finite pulse library. Record every pulse, duration, physical location, calibration epoch, and error estimate. If no such device is accessible, keep the original GP theorem as a modeling result and do not claim hardware gain from it.
2. **For the current IBM Open instance, study an executable scheduling variant.** IBM documents that current QPUs do not expose pulse-level control; the available testable controls include circuit timing, delays, and dynamical decoupling. Define a *new* optimization problem for those controls, prove exactly what remains convex, and generate ISA circuits whose timing survives transpilation. A scheduler variant must be identified as an extension, not presented as physical validation of the original variable-gate-duration GP.

Official IBM references: [Qiskit 2.0 pulse migration](https://quantum.cloud.ibm.com/docs/en/guides/qiskit-2.0), [transpiler scheduling stages](https://quantum.cloud.ibm.com/docs/en/guides/transpiler-stages), and [runtime DD options](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/options-models-dynamical-decoupling-options).

## Minimum convincing experiment

1. **Freeze the hypothesis before more QPU jobs.** Declare one primary metric (for example, success probability for an ideal-output circuit), the target circuits and physical qubits, the decision rule, the baseline list, the number of blocks and shots, and the analysis method. Keep calibration and test blocks separate. Report all failed submissions and regressions.
2. **Calibrate the chosen action set.** On the same qubit or edge and within a documented calibration epoch, measure error versus each *available* control option, with repeated randomized sequences and uncertainty. Split calibration data from held-out durations, circuits, and later epochs. Reject or enlarge the model if held-out residuals exceed measurement uncertainty. A single nominal gate point cannot determine an inverse-duration coefficient.
3. **Produce distinct executable circuits.** For every policy, export the transpiled ISA circuit, physical layout, gate counts, exact delays, and makespan. Assert that the backend accepts the instructions and that policy-specific differences survive compilation. If GP and a baseline compile to the same circuit, no performance comparison is possible.
4. **Run matched controls.** Use the same logical circuits, layout, native operations, total deadline, shots, calibration window, and measurement procedure for GP, Qiskit optimization level 3, ASAP, ALAP, DD, and a simple calibrated discrete-choice heuristic. Randomize policy order within repeated blocks. Include small QFT, Grover, VQE, and QAOA instances only where the output metric can be interpreted honestly; use additional structured idle-sensitive circuits to stress the timing mechanism.
5. **Report effect sizes with uncertainty.** Show paired block-level differences, uncertainty that includes between-block drift, failures, runtime, QPU usage, gate count, depth, physical makespan, and idle distribution. Do not convert the observed frequency in the one-qubit pilot into an algorithmic fidelity or GP speedup.
6. **Release the full audit trail.** Publish source circuits, exact compiled circuits, backend-calibration metadata, job IDs, raw counts, analysis code, seeds, environment versions, negative results, and manuscript figure-generation scripts. Keep account credentials and the IBM account CRN out of the public repository.

## Go/no-go rules for the manuscript

- **No physical performance claim** until a distinct, executable optimizer choice has beaten matched baselines on held-out hardware blocks with uncertainty and regressions reported.
- **No validated inverse-plus-linear gate law** until measured *gate*-duration sweeps on the target device support both fit and held-out prediction; idle-delay Ramsey data are a different quantity.
- **No general circuit-optimality claim** from the fixed-DAG continuous theorem. Gate synthesis, mapping, routing, and discrete pulse selection remain separate problems.
- If the hardware test is neutral or negative, publish the convex method and the counterexample honestly as a methods paper, with claims scaled to the evidence.
