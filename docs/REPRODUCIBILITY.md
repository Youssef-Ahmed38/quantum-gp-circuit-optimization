# Reproducibility and interpretation

## What was run

The study combines Qiskit circuit construction, convex optimization, saved calibration parameters, local noise models, and locally generated benchmark results. It uses no paid hardware execution. The protocol file at `evidence/observable_study/protocol.json` explicitly records `all_free_local: true`, `hardware_jobs: 0`, model assumptions, and the calibration-snapshot source. Other evidence subdirectories contain their own result and environment files.

The saved `evidence/observable_study/props_torino.json` is a Qiskit fake-backend calibration snapshot. It anchors nominal errors and durations in a model; it is not a record of experiments run on `ibm_torino`.

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
