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

The temporary one-day API key used for this run was not stored in this repository. Its downloaded and temporary local copies were removed after result retrieval. At the user's request, the IBM Cloud key remains active until its configured one-day expiration rather than being revoked immediately.
