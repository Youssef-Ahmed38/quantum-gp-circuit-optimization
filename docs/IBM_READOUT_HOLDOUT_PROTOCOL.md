# Frozen free-QPU readout-model diagnostic

This protocol was prepared before any new IBM submission. It follows the failed nominal-gate model check in [IBM_MODEL_CHECK_PROTOCOL.md](IBM_MODEL_CHECK_PROTOCOL.md), whose four event predictions missed a prespecified two-percentage-point tolerance. The diagnostic tests whether the discrepancy is principally associated with the backend-reported readout entries. It does **not** select GP gate durations or demonstrate a GP hardware advantage.

## Fixed design

- Backend and plan: `ibm_marrakesh` on the user's free IBM Open instance only. The instance displayed 18 seconds of total use and 9 minutes 42 seconds remaining on 2026-10-10 before this protocol. Two jobs are capped at 60 seconds of QPU execution each.
- Physical qubits: 0, 1, and 2. The same native CZ edges 0–1 and 1–2 are required. Qiskit optimization level 0, seed `20261011`, and explicit disabling of Sampler dynamical decoupling, gate twirling, and measurement twirling are fixed in the script.
- Calibration job first: three randomized blocks of six circuits, preparing logical `0` and `1` separately on each physical qubit, with 2,048 shots per circuit. Counts from these circuits **alone** estimate operational `P(1|0)` and `P(0|1)` per qubit. The latter includes error in the X preparation gate and is not a pure assignment-error estimate.
- Target job second: three randomized blocks of six circuits, also 2,048 shots each. Families are one-qubit return to `0`, one-qubit return to `1`, Bell preparation, Bell return, GHZ preparation, and GHZ return. The first four families include repeats or close variants of the previous check; the latter two add an independent return and a preparation-direction control. Exact native ISA circuits and block order are saved to QPY and JSON.
- Two prespecified predictions for each target family: (A) the original snapshot-anchored independent post-gate isotropic Pauli plus reported readout model; (B) the **same gate model** with only the six operational readout probabilities replaced by calibration-job estimates. The second prediction is serialized with a SHA-256 lock **before** the target job is submitted. Target counts are never used for fitting.
- Prespecified events are `0`, `1`, Bell parity `00`/`11`, Bell return `00`, GHZ parity `000`/`111`, and GHZ return `000`, respectively. Report pooled observed-minus-predicted event probabilities, all block frequencies, and shot-only Wilson intervals. The primary descriptive comparison is mean absolute event error across the six target families. A model is called within the preregistered practical tolerance only if **all six** pooled event residuals are at most two percentage points in absolute value; this is not a hypothesis test.
- Save the pre-job backend snapshot, QPY, all counts, both job IDs and statuses, both predictions, timing of the prediction lock, and all failures. Preserve negative outcomes. No paid instance is authorized.

## Commands

```text
python benchmarks/run_ibm_readout_holdout.py --fake --output WORKDIR/readout_holdout_dry_run.json
python benchmarks/run_ibm_readout_holdout.py --token-file PRIVATE_KEY_FILE --instance OPEN_INSTANCE_CRN --output evidence/hardware/ibm_marrakesh_readout_holdout_2026-10-10.json
python benchmarks/analyze_ibm_readout_holdout.py evidence/hardware/ibm_marrakesh_readout_holdout_2026-10-10.json --output evidence/hardware/ibm_marrakesh_readout_holdout_analysis_2026-10-10.json
```

The key file and CRN are outside public artifacts. The user requested retention of a new local key file after testing. Retaining it does not make a one-day IBM key valid after its configured expiration.

Even if calibrated fixed-gate predictions pass, that supports only this nominal-gate readout-adjusted model on small circuits at this time. It cannot identify the inverse-duration versus exposure coefficients, validate a gate-duration optimum, or establish that GP outperforms established schedulers on physical hardware. A negative result will be reported as a model failure, not retuned against the target counts and renamed a confirmation.
