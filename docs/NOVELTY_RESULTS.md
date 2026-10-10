# Results for contributions C1–C8 (simulation only)

Date: 2026-10-10. All numbers come from `evidence/*/summary.json` and are regenerated into LaTeX by
`benchmarks/write_novelty_outputs.py`. **No hardware job has been run for these contributions.** Favourable
and unfavourable outcomes are both reported.

## Verdict at a glance

| | Claim | Status | Headline |
|---|---|---|---|
| C1 | Log-posynomial catalogue | Proved; checked | PEC is unbiased and its variance matches γ²−1 to within 1% (Pauli-frame sampling) |
| C2 | Joint timing + DD is convex | Proved; mixed in simulation; **positive on hardware** | Sim: 11 wins / 8 losses of 20. IBM Heron, pre-registered: +2.18 pp over XY4 (95% CI 1.43–2.92), predicted ranking held |
| C3 | Runtime-optimal PEC closed form | Proved; checked | −21% QPU time (median) at t_rep = 0; ≈0% at IBM's 250 µs, as predicted |
| C4 | Certified robust GP; dominance | Proved; checked | Coverage 1.000 vs plug-in 0.502; Table VII regressions 6/12 → 0/12 at ≤0.3 pp cost |
| C5 | Anisotropic / state-aware constraints | Proved; checked | Predicts the X/Z reversal that isotropic models cannot represent |
| C6 | NP-hardness; bicriteria rounding | Proved; checked | 0 bound violations in 300 random instances |
| C7 | Log-log convex frontier, dual price | Proved; checked | Convex on all 4 circuits; slope = −dual within finite-difference error |
| C8 | Scale | Measured | 133,160 gates solved to optimality in 18 min (sparse interior point) |

## C2 — joint timing and dynamical decoupling (`evidence/dd_gp_v2/`, `evidence/dd_gp/`, `evidence/dd_state_aware/`)

Setup: 4 circuits × 5 noise spectra; deadline = nominal ASAP makespan. The GP model is fitted only from
simulated single-qubit spectroscopy. Evaluation uses an independent time-correlated density-matrix simulator
with common random numbers; 300 trajectories per case.

**Protocol v2 (primary).** Count of significant wins and losses against the best of ASAP / ALAP / ALAP+XX /
ALAP+XY4 (|Δ| > max(2 s.e., 10⁻⁶)):

| Policy | Wins | Losses | Median Δ (10⁻⁴) |
|---|---|---|---|
| GP-DD joint, cross-window | 11 | 8 | +4.1 |
| GP-DD joint, per-window | 11 | 9 | +4.1 |
| Idle-aware GP, no DD | 11 | 9 | +2.3 |
| Gate-only GP (earlier manuscript) | 14 | 6 | +0.9 |
| GP-DD cross-window, fixed native durations | 3 | 9 | −0.1 |
| GP-DD per-window, fixed native durations | 3 | 10 | −0.1 |

Model-to-simulator policy-rank correlation: median Spearman 0.74.

Interpretation:
- The largest gains are on the mirror circuit (+19 to +147×10⁻⁴) and on QFT (+7 to +30×10⁻⁴). They come
  mainly from stretching gates into slack, which IBM Heron does not expose.
- **The hardware-executable variant (fixed durations) mostly does not beat the best standard schedule.**
- The worst case is the repetition code under quasi-static noise. Cross-window allocation shrinks the loss from
  −236 to −57×10⁻⁴ but does not remove it.

**Protocol v1 (superseded, kept for transparency).** v1 charged decoherence during gates twice: in the gate law
`g τ` and again in the simulator. Under v1 the joint GP won 8 and lost 12 cases (rank correlation 0.26). v2 makes
the simulator the only source of decoherence: the truth gate law keeps the drive term only, and the model's
`g_j` is the fitted idle rate of the gate's qubits. The change fixes an inconsistency diagnosed from the per-case
pattern. It was not chosen to improve the result, and both runs are in the repository.

**State-aware and cross-window ablation (exact amplitude damping, fixed durations).**
- On the excited-idle circuit the state-aware GP places DD only on qubits idling in |1⟩: +8.6×10⁻⁴ over both
  ALAP and the state-blind GP, and +70×10⁻⁴ over blanket XX, in all three noise scenarios.
- On GHZ with an ancilla wait it beats XX by 20–24×10⁻⁴.
- On the repetition code, per-window GP loses to XX (−93 and −59×10⁻⁴). Cross-window protection turns this into
  −11.8 ± 5.7 (1/f) and +1.5 ± 3.5 (quasi-static), using 6 pulses where XX uses 16.

## C3 — runtime-optimal PEC (`evidence/runtime_optimal/`)

32 snapshot-anchored Clifford workloads (8–64 qubits). The table gives the median (max) reduction in QPU time
to a fixed standard error against the best of nominal, fastest, error-optimal and best-uniform durations.

| t_rep | vs best heuristic | vs nominal | Runtime-optimal wins |
|---|---|---|---|
| 0 | 21% (43%) | 44% | 32/32 |
| 5 µs | 2.3% (8.9%) | 3.7% | 32/32 |
| 250 µs | 0.004% | 1.6% | 32/32 |

The closed form's implied flows lie in [0, 1] and vanish off critical paths in all 96 cases (tolerance scaled by
1 + t_rep/M because recovering the flows amplifies solver error). PEC sampling (200k shots): bias within 2.1 s.e.;
the variance-to-(γ²−1) ratio lies in 0.990–1.013.

## C4 — certified robustness (`evidence/robust_pac/`)

**Part A — dominance (rerun of the manuscript's Table VII; deadline = nominal makespan).**

| Laws | Pairs | Plain GP worse | Dominance GP worse | Dominance + Table VII law worse |
|---|---|---|---|---|
| Inside hull (inverse; inverse-square matched at nominal; mixtures) | 30 | 0 | 0 | 0 |
| Outside hull (inverse-cube matched; manuscript `a/τ²`) | 12 | 6 | 6 | **0** |

- The manuscript's 100% regression (Table VII) comes entirely from its `a/τ²` law. That law isn't unit-matched
  to the declared law, so it lies outside any hull built from matched laws. The matched inverse-cube law is also
  outside the hull, yet causes no regression.
- Declaring the `a/τ²` law removes all regressions. The cost is at most 0.3 percentage points of improvement on
  the declared law; for example, QFT-4 goes from −6.25% to −5.96%.

**Part B — calibration coverage.**
- Setup: 6 circuits × 200 simulated noisy duration sweeps (σ = 10%); Scheffé radius at δ = 0.05; 0 solver
  failures.
- Pooled coverage of the certified bound: **1.000** (1,200/1,200; target 0.95).
- Plug-in prediction coverage: **0.502**.
- The bound is conservative: certificate / true error has median 1.23 (minimum 1.11). Causes: simultaneous
  validity over all schedules, and the diagonal covariance bound.
- The robust schedule's true error is within 0.01% of the plug-in schedule's, and is never worse than the
  nominal baseline.
- So the contribution is a guarantee that actually holds, not a better schedule.

## C5 — anisotropic and state-aware constraints (`evidence/anisotropic/`)

Native-X policies, 48 gates, single-qubit Lindblad + classical-noise simulation:

| Basis | Policy | Simulated | Anisotropic prediction | Isotropic prediction | Heron (measured, retrospective) |
|---|---|---|---|---|---|
| X | slow | 0.918 | 0.921 | 0.917 | 0.901 |
| X | distributed | 0.919 | 0.919 | 0.922 | 0.883 |
| X | tail | 0.559 | 0.540 | 0.641 | 0.305 |
| X | dd_tail | 0.873 | 0.817 | 0.866 | 0.859 |
| Z | slow | 0.902 | 0.912 | 0.917 | 0.830 |
| Z | distributed | 0.907 | 0.925 | 0.922 | 0.801 |
| Z | tail | 0.978 | 0.997 | 0.641 | 0.985 |
| Z | dd_tail | 0.905 | 0.922 | 0.866 | 0.875 |

- The anisotropic, state-aware model gets the reversal right: tail is worst in X and best in Z.
- The isotropic model cannot (Proposition C5.1): it predicts 0.64 for the Z-basis tail where the truth is 0.98.
- The anisotropic model misorders slow vs distributed in X, a near-tie (0.0013).
- The simulator was not tuned to the Heron data.
- Compilation with Z-biased noise: isotropic plans violate true targets in 3 of 6 workloads and wrongly declare
  2 attainable targets infeasible.

## C6 / C7 (`evidence/theory_checks/`)

- **C6:** 300 random DAGs with random finite libraries; exhaustive library optimum versus (q,q)-bicriteria
  rounding gives 0 error-bound and 0 makespan-bound violations. Worst observed error ratio 1.037; median 0.944.
- **C7:** on QFT, Grover, VQE and QAOA (4 qubits) the frontier is log-log convex (minimum second difference
  0.41–0.57), and the finite-difference slope equals −dual within 0.01. Moving from the fastest to the free
  optimum removes 51–56% of model error.

## C8 — scale (`evidence/scale/`)

Random circuits with a balanced makespan weight:

| Qubits | Gates | Sparse IP | Independent FW bound after 500 steps |
|---|---|---|---|
| 100 | 3,361 | optimal, 1.5 s | 0.17% |
| 500 | 19,996 | optimal, 56 s | 2.1% |
| 1,000 | 53,242 | optimal, 184 s | 3.4% |
| 2,000 | 133,160 | optimal, 1,088 s | 6.4% |

- Empirical time exponent ≈ 1.75 in gates. Model construction takes under 0.1 s at every size.
- Frank–Wolfe alone converges slowly under realistic makespan pressure (gaps 0.08–3% after 4,000 steps), so it
  serves as an independent but loose certificate.
- An earlier run with a negligible makespan weight converged in 0–2 iterations and is not reported as evidence.

## What this means for the paper

- The theory contributions (C1, C2 convexity, C3–C7) are solid and checked.
- The empirical case for joint timing + DD is positive on average but not uniform.
  - It depends on duration control that current IBM hardware does not offer.
  - Cross-window low-frequency noise is a real failure mode, only partly fixed.
- C3 has a clear, honest regime statement.
- The manuscript should present C2 as "exact convex allocation with characterised regimes of benefit", not as a
  uniform improvement.
- H2 is a genuine falsification test.

## Hardware: IBM ibm_marrakesh, free Open plan (2026-10-10)

**H1 calibration** (job `db54a1o4qg6s73c2v460`; qubits 55, 54, 53; CPMG n ∈ {0,2,4,8}, w ∈ {4,16,40,80} µs; T1;
3 blocks × 1,000 shots).
- Qubits 55 and 53 show strong low-frequency, partly coherent detuning: a single echo raises 16 µs survival from
  0.42 to about 0.90, and free precession revives after its first zero crossing.
- T1 is 125–160 µs, with a readout floor of about 3%.
- Qubit 54 is not described well by any model in the family (log-residual cost 0.44).
- **Changes before freezing**, all documented in `h2_preregistration.json`:
  - static term, SPAM offset, and Ramsey points restricted to before the first zero crossing;
  - pulse counts capped at the H1-calibrated maximum of 8 per window. The uncapped fit would have inserted 2,358
    pulses because pulse error is unidentifiable from n ≤ 8;
  - pulse error floored at the backend-reported X error.

**H2 pre-registered test** (job `db54ep4vf2bc73cvcvbg`; qubits 55, 54, 53, 39, 33, 32; GHZ-5 + ancilla with a 20 µs
wait; matched 24 µs makespan; 3 blocks × 2,000 shots).
- Circuit hash `a98edb43…` was verified identical at submission. The analysis script's hash is recorded in
  `h2_analysis_script.sha256`.

| Policy | Pulses | P(000000) | Per block |
|---|---|---|---|
| GP-DD | 34 | 0.348 ± 0.006 | 0.356, 0.345, 0.344 |
| GP-DD cross-window (identical schedule) | 34 | 0.345 ± 0.006 | 0.346, 0.344, 0.346 |
| XY4-like | 36 | 0.338 ± 0.006 | 0.336, 0.340, 0.339 |
| XX (Qiskit-like) | 18 | 0.321 ± 0.006 | 0.335, 0.318, 0.310 |
| No DD | 0 | 0.291 ± 0.006 | 0.287, 0.292, 0.294 |

- **Primary (pre-registered): GP-DD − best standard (XY4) = +0.98 pp, shot-only 95% CI [−0.72, +2.68] pp. Not
  significant.** All 3 blocks favour GP-DD (+1.95, +0.50, +0.50 pp).
- **Secondary:** the predicted ranking of all five policies matched the observed ranking exactly (Spearman 1.0).
- The model's riskiest decision was to leave qubit 54 without DD. Its marginal P(0) was 0.871, against 0.846 under
  XY4 and 0.862 under XX.
- GP-DD beats Qiskit-like XX by +2.7 pp (about 3σ) and no DD by +5.7 pp.
- **Exploratory (not pre-registered):** pooling the two identical GP schedules gives +0.86 pp over XY4, CI about
  [−0.6, +2.3] pp. Still not significant.
- **Caveats:** one run, one qubit chain, shot-only intervals that exclude drift, and no readout mitigation (which
  affects all policies equally).
- **Implication:** the model's predictions transfer to hardware. Establishing a significant advantage over XY4 needs
  more blocks or replications: about 4× the shots for a 2σ effect of this size, still within the free tier.

**R1 pre-registered replication** (job `db54jfo4qg6s73c2ve5g`; identical circuits, hash `a98edb43…` verified at
submission; same qubits; 4 blocks × 6,000 shots).
- The pre-registration (`h2_r1_preregistration.json`) and the analysis and pooling scripts were hashed before
  submission. All three verified unchanged after the run.

| Policy | P(000000) | Per block |
|---|---|---|
| GP-DD cross-window (identical schedule) | 0.327 ± 0.003 | 0.330, 0.318, 0.325, 0.334 |
| GP-DD | 0.326 ± 0.003 | 0.318, 0.331, 0.330, 0.323 |
| XY4-like | 0.301 ± 0.003 | 0.304, 0.299, 0.306, 0.295 |
| XX (Qiskit-like) | 0.296 ± 0.003 | 0.293, 0.292, 0.300, 0.300 |
| No DD | 0.290 ± 0.003 | 0.289, 0.299, 0.284, 0.288 |

- **R1 primary: GP-DD − XY4 = +2.46 pp, 95% CI [+1.63, +3.29]; 4/4 blocks favour GP-DD. Significant.**
- **Pre-registered fixed-effect pooling (H2 + R1): +2.18 pp, 95% CI [+1.43, +2.92]. Significant.**
- Observed ranking vs the H2 prediction: Spearman 0.9. The only swap is between the two GP variants, which are
  identical schedules.
- Qubit 54, left without DD by the model: marginal P(0) 0.878 under GP-DD, 0.858 under XY4 and 0.869 under XX.
- **Caveats:**
  - Absolute fidelities drifted down between runs (XY4 0.338 → 0.301) while the GP advantage grew. The two runs'
    differences are consistent (z ≈ 1.5).
  - All runs are on one qubit chain of one device on one day. Intervals are shot-only.
  - Generality across chains, devices and days is not established.

**Total QPU use for this study:** 3 jobs (H1, H2, R1), about 1 minute of shot time on the free Open plan.

**H3 pre-registered control arms, two qubit chains** (4 blocks × 6,000 shots per arm).
- Chain A: job `db5549o4qg6s73c303kg`, qubits 55, 54, 53, 39, 33, 32.
- Chain B: calibration `db554vclf4us73c3i8ag` and test `db557cclf4us73c3ibeg`, qubits 9, 8, 7, 6, 5, 4, disjoint from chain A and its neighbours, with its own frozen model.
- Pre-registrations and analysis/pooling scripts were hashed before each run and verified afterwards. The analysis script gained an optional tag argument after the chain-A pre-registration. The byte-identical pre-registered copy (`analyze_ibm_h3.prereg.py`, hash verified) produces identical output.

| Comparison | Chain A | Chain B | Pooled |
|---|---|---|---|
| **Primary: GP-DD − uniform CPMG-8** | +1.69 pp [+1.05, +2.32], 4/4 blocks | +1.32 pp [+0.46, +2.18], 4/4 blocks | **+1.56 pp [+1.05, +2.07]** |
| GP-DD − protect the GP-undecoupled qubit | +0.71 [+0.07, +1.35] | +0.23 [−0.63, +1.09] | — |
| GP-DD − XY4 | +2.31 [+1.68, +2.94] | +0.20 [−0.67, +1.06] | — |
| Predicted order of the four arms | matched | not matched (XY4 2nd) | — |

- **Conclusion:** the GP's allocation beats a pulse-matched uniform schedule that has nearly twice as many pulses, on two disjoint chains. The gain is therefore not explained by pulse count.
- **Against XY4 the result is chain-dependent:** a large win on chain A, a tie on chain B. The model over-predicted XY4's deficit on chain B. A plausible cause is XY4's robustness to pulse errors from alternating X/Y, against the GP's X-only CPMG blocks; a natural extension is GP selection over robust sequence families.
- **Drift:** absolute fidelities on chain A fell from about 0.33 (H2/R1) to about 0.15 (H3). The paired within-run comparisons remain valid.
- **Total hardware use:** 7 free-tier jobs, about 2 minutes of shot time.
