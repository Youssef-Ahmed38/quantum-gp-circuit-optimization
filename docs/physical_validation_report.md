# Physical and finite-library validation

Declared spin-control simulation, not hardware calibration. All negative results retained.

| Regime | Mean process fidelity gain vs slack (pp) | Regressions | Mean latency change (%) |
|---|---:|---:|---:|
| low_drive | -0.008 | 16 / 48 | 0.05 |
| balanced | 3.966 | 0 / 48 | 32.55 |
| high_drive | 15.341 | 0 / 48 | 216.97 |

{
  "configurations": 144,
  "native_circuits": 48,
  "policy_channels": 1296,
  "state_checks": 2592,
  "pulse_entries": 25056,
  "independent_ode_diamond_checks": 6
}

The characterized MILP optimizes an operational error upper bound, not actual state/process fidelity. The GP transfer bound can be loose; LP/MILP bounds remain separate. General-noise floors can be vacuous. These findings do not establish a field-changing result or Q1 readiness.

See raw cases/results, transfer_validation, stronger_certificate_validation and summary for all checks and denominators.
