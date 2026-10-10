# IBM Quantum hardware access check (2026-10-10)

The Open instance in the user's IBM Quantum account showed 10 minutes of free QPU time before this check. A 1,024-shot control circuit completed on `ibm_marrakesh` (Heron r2); workload `db4li2amb58s7389cha0` reported 1,018 zeros and 6 ones. The full submitted QASM, raw counts, and the exported outcome plot are in [`../evidence/hardware/ibm_marrakesh_control_2026-10-10.json`](../evidence/hardware/ibm_marrakesh_control_2026-10-10.json). IBM showed 2 seconds of QPU usage for the completed job.

This is an access and measurement control, **not validation of the GP optimizer**. It does not compare matched schedules, and the compiler may simplify the submitted consecutive Hadamard gates. A Composer submission containing a 1,000 ns `delay` returned HTTP 500 before a job was created. That response does not establish whether the instruction is unsupported or whether the failure was transient.

Qiskit Runtime did accept explicit delays. An exploratory randomized-order, two-block Ramsey/echo pilot then completed on the same QPU as job `db4ll9o4qg6s73c2buj0`. The Open instance showed 7 seconds of QPU usage for this job. Each of the four conditions had two circuits of 2,048 shots on physical qubit 0:

| Circuit | Zero counts / shots | Measured Pr(0) |
| --- | ---: | ---: |
| Ramsey, 1 µs delay | 4,025 / 4,096 | 0.9827 |
| Ramsey, 20 µs delay | 2,655 / 4,096 | 0.6482 |
| Ramsey, 80 µs delay | 1,947 / 4,096 | 0.4753 |
| Echo, two 40 µs delays with two X gates | 2,402 / 4,096 | 0.5864 |

The per-block echo minus plain-Ramsey difference at 80 µs was 0.1021 and 0.1201. These are **exploratory observations**, not an established scheduling improvement: the echo circuit adds X gates, only two blocks were run, calibration can drift, and shot-level binomial intervals do not cover device variation. The qubit-0 calibration snapshot reported T1 = 298.5 µs, T2 = 40.4 µs, and readout error = 0.00671; the snapshot time and exact values are in the [raw result](../evidence/hardware/ibm_marrakesh_ramsey_2026-10-10.json). The [analysis](../evidence/hardware/ibm_marrakesh_ramsey_analysis.json), [plot](../evidence/hardware/ibm_marrakesh_ramsey.png), and [submission script](../benchmarks/run_ibm_ramsey.py) make the pilot inspectable. A temporary API key was deleted after result retrieval.

The paper's continuous gate-duration variables require characterized, implementable gate calibrations. Neither job supplies an error-versus-**gate**-duration curve or a deployable GP duration choice. A GP hardware-performance result can be added to the manuscript only after obtaining those controls and submitting matched randomized repetitions of the proposed policy and established schedulers, with effect sizes and uncertainty. Any later QPU experiment should remain within the Open instance's free allowance.

## Prespecified nominal-gate model check

A subsequent free Open-instance Sampler job, `db4nov04qg6s73c2e8d0`, completed on `ibm_marrakesh` with 12 circuits (four Clifford circuit families in three randomized blocks, 2,048 shots each). The design and event definitions were frozen in [IBM_MODEL_CHECK_PROTOCOL.md](IBM_MODEL_CHECK_PROTOCOL.md) before submission. An IBM backend calibration snapshot was saved before submission, and the paper's independent post-gate isotropic Pauli model plus reported readout assignment probabilities generated predictions before any QPU counts were retrieved. The circuits used fixed native gates on physical qubits 0–2. They **did not implement GP-selected gate durations**.

| Prespecified success event | Observed / shots | Predicted probability | Observed minus predicted |
| --- | ---: | ---: | ---: |
| Single-qubit return, `0` | 6,119 / 6,144 | 0.9685 | +0.0274 |
| Bell, `00` or `11` | 6,065 / 6,144 | 0.9619 | +0.0253 |
| Bell return, `00` | 6,000 / 6,144 | 0.9512 | +0.0254 |
| GHZ, `000` or `111` | 5,984 / 6,144 | 0.9201 | +0.0538 |

All four absolute residuals exceed the prespecified descriptive tolerance of two percentage points. The model was pessimistic for these circuits, and this experiment does **not** validate its quantitative predictions on the QPU. In particular, the pre-job snapshot reported `prob_meas1_prep0 = 0.03027` for qubit 0, whereas the measured single-qubit return error was only 25/6,144 = 0.00407. Calibration mismatch, readout treatment, and the simplistic independent Pauli-channel assumption are possible explanations; these data do not identify a cause. The shot-only Wilson intervals and three block ranges are in the [analysis](../evidence/hardware/ibm_marrakesh_model_check_analysis_2026-10-10.json). They do not cover device drift or calibration uncertainty. The [raw evidence](../evidence/hardware/ibm_marrakesh_model_check_2026-10-10.json) contains the predeclared probability distributions, calibration snapshot, circuit order and counts; the companion QPY file preserves exact ISA circuits. The [submission](../benchmarks/run_ibm_model_check.py) and [analysis](../benchmarks/analyze_ibm_model_check.py) scripts reproduce the protocol and calculations.

## Separate readout-calibration and held-out target check

The [frozen protocol](IBM_READOUT_HOLDOUT_PROTOCOL.md) followed the failure above. Two additional free Open-instance jobs completed on `ibm_marrakesh`: readout controls `db4o3784qg6s73c2ekqg` and held-out targets `db4o3bo4qg6s73c2ekvg`. Each job contained 18 circuits at 2,048 shots. Calibration results yielded six operational readout estimates; both the original snapshot and readout-adjusted target predictions were saved in a SHA-256 [prediction lock](../evidence/hardware/ibm_marrakesh_readout_holdout_2026-10-10_prediction_lock.json) before the target job was submitted. The lock hash in the [raw result](../evidence/hardware/ibm_marrakesh_readout_holdout_2026-10-10.json) matches the lock file. Neither target results nor outcomes were used to fit predictions. The calibrator's `|1>` preparation includes X-gate error, so it does not isolate readout error perfectly.

| Target family | Observed event probability | Snapshot prediction | Calibrated prediction |
| --- | ---: | ---: | ---: |
| Single return | 0.99365 | 0.99482 | 0.99514 |
| Single-one return | 0.99186 | 0.99174 | 0.99239 |
| Bell parity | 0.98486 | 0.98708 | 0.98717 |
| Bell return | 0.97396 | 0.98251 | 0.98508 |
| GHZ parity | 0.97233 | 0.97283 | 0.97547 |
| GHZ return | 0.95866 | 0.96324 | 0.96863 |

All six calibrated residuals were within the prespecified descriptive 0.02 tolerance. The snapshot prediction was **better**, however: event mean absolute error 0.00286 versus 0.00476 for readout adjustment. Thus the proposed explanation that separate readout calibration would improve prediction was **not supported** on this holdout. The new device snapshot reported much lower readout errors than the earlier one; for qubit 0, `prob_meas1_prep0` changed from 0.03027 to 0.00391, and for qubit 2 from 0.04565 to 0.01270. This is consistent with calibration sensitivity but cannot establish the cause of the first mismatch. Raw counts, exact ISA circuits, block ranges, and shot-only Wilson intervals are available in the [analysis](../evidence/hardware/ibm_marrakesh_readout_holdout_analysis_2026-10-10.json), [raw file](../evidence/hardware/ibm_marrakesh_readout_holdout_2026-10-10.json), and companion QPY file. Both jobs report `DONE` and were capped at 60 seconds of execution. No gate duration was selected by GP in either job.

## Executable native X-duration experiment

The device exposes two physical implementations of the same logical X operation: `x` (36 ns) and `xslow` (1,000 ns). We froze a [characterization protocol](IBM_X_DURATION_CHARACTERIZATION_PROTOCOL.md) before running job `db4ob04lf4us73c31490` on the free Open instance. It compared fast X, slow X, and fast X followed by matched delays on physical qubits 0 and 1, in X- and Z-basis return experiments. On qubit 1, the 32-gate X-basis slow-minus-padded difference was +0.0120 across three consistently positive randomized blocks, but its shot-only 95% interval included zero. The full [raw training data](../evidence/hardware/ibm_marrakesh_x_duration_characterization_2026-10-10.json), [analysis](../evidence/hardware/ibm_marrakesh_x_duration_analysis_2026-10-10.json), and QPY are archived.

We selected qubit 1 on those training data and froze a [separate holdout protocol](IBM_X_DURATION_HOLDOUT_PROTOCOL.md) before job `db4ocm2mb58s7389fnag`. New sequence lengths 16 and 48, both bases, four equal-duration policies, and three randomized blocks were submitted together at 2,048 shots each. The standard comparator used Qiskit's ALAP schedule analysis and two-X dynamical-decoupling pass; the other controls used distributed or tail delays. The protocol lock records the training-data SHA-256 hash and shuffled target order before submission. The [raw held-out file](../evidence/hardware/ibm_marrakesh_x_duration_holdout_2026-10-10.json), [lock](../evidence/hardware/ibm_marrakesh_x_duration_holdout_2026-10-10_protocol_lock.json), [analysis](../evidence/hardware/ibm_marrakesh_x_duration_holdout_analysis_2026-10-10.json), QPY, and [figure](../figures/ibm_marrakesh_x_duration_holdout_2026-10-10.pdf) preserve all results.

| Equal-duration policy, 48 logical X gates | X-basis return | Z-basis return |
| --- | ---: | ---: |
| Native slow X | 0.9010 | 0.8296 |
| Fast X, distributed padding | 0.8826 | 0.8013 |
| Fast X, tail padding | 0.3052 | 0.9849 |
| Fast X, two-X DD tail | 0.8586 | 0.8753 |

Each cell is a return-to-zero frequency over 6,144 shots, **not average gate fidelity**. The frozen X-basis criterion was met: slow X beat fast-tail by +0.5959 and DD-tail by +0.0425, both positive in all three blocks with shot-only 95% intervals above zero. It beat distributed padding by +0.0184 (shot-only interval +0.0074 to +0.0294). In the Z basis, fast-tail and DD-tail beat slow X. Three blocks on one qubit do not establish long-term or device-wide performance. This is a **physical improvement in one native gate choice for one observable and schedule**, not a GP-selected circuit or validation of the manuscript's continuous inverse-duration model. The basis reversal is evidence that the current isotropic post-gate model is inadequate for ranking these physical policies.

Both additional jobs completed on the free Open instance. Afterward, the instance page reported 1m50s used in the current cycle and 8m10s remaining. No paid service was used.

The temporary one-day API key used for this run was not stored in this repository. Its downloaded and temporary local copies were removed after result retrieval. At the user's request, the IBM Cloud key remains active until its configured one-day expiration rather than being revoked immediately.
