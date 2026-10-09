# Prespecified free IBM native-gate model check

This experiment tests the paper's **nominal, fixed-gate Pauli noise model** on `ibm_marrakesh`; it does **not** execute GP-chosen alternate gate durations or establish GP hardware improvement. The experiment must be described that way whether its result is positive, neutral, or negative.

The account's Open instance displayed 9 min 51 s of free QPU allocation before this run. No paid plan or simulator is needed. The protocol compiles with a fake backend first and refuses unexpected native operations, unavailable CZ edges, non-equivalent unitaries, or invalid predicted distributions. Actual job submission requires a credential supplied outside the public repository.

## Frozen design

- QPU: `ibm_marrakesh`, physical qubits 0, 1, 2; native CZ edges 0–1 and 1–2 must be available.
- Circuits: a one-qubit return-to-zero control, a Bell preparation, its inverse return-to-zero, and a three-qubit GHZ preparation. The transpiled native gates and exact layout are saved in QPY. The source and native QASM are recorded.
- Three randomized blocks, with each of the four circuit types appearing once per block; 2,048 shots per circuit, in one Sampler job. The random seed is `20261010`.
- Before submission, take a current backend calibration snapshot. Predict each measured bitstring distribution using independent post-gate isotropic Pauli channels anchored to the reported native-gate error rates, followed by independent per-qubit readout assignment errors. The nominal prediction does not depend on the paper's unidentifiable inverse-duration split.
- Prespecified reported events: `0` for the one-qubit control, `00` or `11` for Bell, `00` for Bell return, and `000` or `111` for GHZ. The primary reported quantity for each is observed minus predicted event probability. Report each block, pooled counts, shot-only Wilson intervals, and block range. An absolute discrepancy of two percentage points is a descriptive practical threshold, not a significance test.
- Save raw counts, calibration fields, predictions fixed before submission, transpiled QPY, job ID, compiler gate counts, and all results. Include failed submissions and calibration limitations. Do not report hardware advantage for GP from this test.

Dry-run command:

```text
python benchmarks/run_ibm_model_check.py --fake --output WORKDIR/ibm_model_check_dry_run.json
```

Actual submission, after obtaining a temporary API key through the account's normal access flow:

```text
python benchmarks/run_ibm_model_check.py --token-file PRIVATE_TOKEN_FILE --instance OPEN_INSTANCE_CRN --output evidence/hardware/ibm_marrakesh_model_check.json
python benchmarks/analyze_ibm_model_check.py evidence/hardware/ibm_marrakesh_model_check.json --output evidence/hardware/ibm_marrakesh_model_check_analysis.json
```

The token and CRN must remain outside repository artifacts and logs. The temporary key should be revoked after result retrieval. A favorable fixed-gate model check would support only one modeling assumption at nominal duration. It cannot identify error versus **gate duration**, transfer the inverse-duration law, or validate a GP duration decision. Those claims still require a target device exposing at least two calibrated implementations of the same unitary and matched policy comparisons on held-out circuits.
