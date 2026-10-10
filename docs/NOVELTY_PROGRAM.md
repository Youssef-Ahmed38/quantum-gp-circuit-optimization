# Novelty Program: Geometric Programming for Quantum Resource Allocation

Status (2026-10-10): **implemented**. Full statements and proofs are in
[THEORY.md](THEORY.md); simulator results are in [NOVELTY_RESULTS.md](NOVELTY_RESULTS.md).
The IBM jobs H1/H2 are built and dry-run only; nothing has been submitted.

Constraint honoured throughout: all evaluation runs on free local simulators;
IBM Quantum Open-plan time (10 min/month) is used for exactly two small,
pre-registered jobs (Section 5).

---

## 0. Why the current paper scores low on novelty

The current results (convexity of `a/τ + γτ` with a longest-path makespan,
Kelley–Fulkerson duality, log-grid rounding, no-jump channel bounds) are each
textbook. The paper also optimises a decision (continuous gate duration) that
cannot be executed on IBM hardware, and its own data show ≈0 pp median gain.

A 10/10-novelty paper needs three things the current one lacks:

1. **A unifying thesis** that is new and broader than "durations are convex".
2. **At least one non-obvious theorem** with a closed form a referee cannot find elsewhere.
3. **A GP-chosen decision that is executable on real hardware**, tested prospectively.

---

## 1. Unifying thesis (new framing — keeps GP as the main idea)

> **Log-posynomial fidelity principle.** Under Markovian Pauli–Lindblad noise
> whose rates depend on physical resources by power laws (gate durations,
> idle windows, dynamical-decoupling pulse counts, shot counts), the
> quantities a quantum compiler cares about — log-fidelity of tracked
> observables, PEC sampling overhead, estimator variance, and total QPU
> wall-clock time — are all generalized posynomials or exponentials of
> posynomials. Their joint allocation is therefore a geometric program, or
> convex in log coordinates, and is solved to global optimality with a dual certificate.

New title proposal:
**"Geometric Programming for Quantum Resource Allocation: Globally Optimal
Co-Design of Gate Timing, Dynamical Decoupling, and Error-Mitigation Budgets"**

The existing timing results become Section 2 "Preliminaries" (credited to
Boyd et al., Kelley, Fulkerson), *not* headline theorems.

---

## 2. Contributions

Each item lists: statement, proof sketch, why it is new, how it is tested.

### C1. Catalogue theorem: which quantum allocation problems are GPs

**Statement.** For a fixed circuit DAG with post-gate Pauli–Lindblad channels
of intensity `λ_jP(x)` that are posynomials in the resource vector `x`:
1. Tracked Clifford expectations satisfy `⟨S_k⟩ = exp(-2 Σ_j Σ_{P∈A_kj} λ_jP(x))`,
   so `⟨S_k⟩ ≥ t_k` is a posynomial constraint.
2. PEC sampling overhead `γ_PEC = exp(2 Σ λ(x))` is log-posynomial.
3. Mitigated-estimator variance `σ² γ_PEC² / s` is convex in log coordinates.
4. A posynomial ≤ 1 constraint stays a GP after any of 1–3 are combined.

**Proof sketch.** (1) Adjoint Pauli-channel generator gives `-2λ` decay on each
anticommuting component (already Eq. 20, generalised to per-Pauli rates).
(2) Standard PEC overhead for sparse Pauli–Lindblad models (van den Berg 2023).
(3) In `y = log x`, `Σλ(e^y)` is a sum of exponentials of affine functions,
hence convex; `-log s` is linear in `log s`.

**Novelty.** First statement that timing, decoupling and mitigation budgets live
in one convex class. **Test:** unit tests + dense Lindblad agreement (exists).

### C2. Joint gate-duration + dynamical-decoupling allocation is exactly convex (headline theorem)

Model: on every idle window `w` of qubit `q`, a CPMG/XX block with `n` π-pulses
under a power-law dephasing spectrum `S(ω) = A_q/ω^β` gives the filter-function
attenuation (Cywiński et al. 2008 scaling)

    χ_q(w, n) = c_β A_q w^{1+β} n^{-β}   (+ n ε_π for pulse errors).

**Theorem C2.** With gate hazards `a_j/τ_j + γ_j τ_j`, idle windows defined by
the schedule (`w = s_next - f_prev`, affine in start/finish times), and the DD
hazard above, the joint problem

    min over (τ, f, w, n):  Σ_j (a_j/τ_j + γ_j τ_j) + Σ_windows (c_β A w^{1+β} n^{-β} + ε_π n)
    s.t. precedence, deadline M, n τ_π ≤ w, ℓ ≤ τ ≤ u, n ≥ 0

is **jointly convex** in the original variables, so its relaxation is
solved to global optimality.

**Proof sketch.** `w^{1+β} n^{-β} = n·(w/n)^{1+β}` is the perspective of the
convex map `x ↦ x^{1+β}`, hence jointly convex on `n > 0`; composition with the
affine map `(f, τ) ↦ w` preserves convexity. Gate terms are convex. Constraints
are linear. For fixed windows the problem is an exact GP (monomial + posynomial).

**Corollaries.**
- *Closed-form DD count:* for fixed `w`, `n* + 1 = (β c_β A w^{1+β}/ε_π)^{1/(1+β)}` (the model uses `(n+1)^{-β}` so that `n = 0` is free induction).
- *Drive-vs-idle threshold (n = 0, β = 0):* gate `j` should be stretched into
  its slack **iff** `γ_j < η_j^idle` (idle rate of its qubits); otherwise
  `τ_j* = clip(√(a_j/(γ_j - η_j)), ℓ_j, u_j)`. This removes the non-uniqueness of Remark 1.
- *Signomial warning:* idle-aware timing is convex but **not** a GP in the
  `(τ, M)` variables (idle = `M - Στ` carries a negative coefficient). We state
  this explicitly — a structural result in its own right.
- *Integer rounding:* `n ↦ χ(w,n)` is convex, so the best of `⌊n*⌋, ⌈n*⌉`
  (restricted to even `n` for XX) has a bounded, computable loss.

**Why it matters.** It turns the paper's biggest loss (Qiskit DD beats GP under
quasi-static noise, Table IX) into the method's main feature: DD is a
decision variable *inside* the GP. Delays and X-pulses are executable on IBM
Heron today, so this is the hardware-testable contribution.

**Novelty check needed:** ADAPT (Das et al., MICRO 2021), VAQEM (Ravi et al.,
HPCA 2022), Ezzell et al. DD survey (2023), Uhrig/UDD, Biercuk filter
functions. None, to our knowledge, proves joint convexity of timing + DD.

### C3. Runtime-optimal mitigated estimation (closed-form path-flow stationarity)

The cost users actually pay is **QPU time = shots × makespan**. With PEC to
precision `ε`, the required shots are `s = σ² exp(4Λ(τ))/ε²`, where `Λ = Σ_j λ_j(τ)`.

**Theorem C3.** Minimising total QPU time `T(τ) = (σ²/ε²) exp(4Λ(τ)) M(τ)` is
convex in `y = log τ`. At an interior optimum,

    τ_j* = sqrt( a_j / (γ_j + p_j / (4 M*)) ),

where `p_j ∈ [0,1]` is the optimal path-flow (dual) weight through gate `j`.
Off-critical gates (`p_j = 0`) take their error-optimal duration `√(a_j/γ_j)`;
critical gates are compressed by a factor set by the makespan.

**Proof sketch.** `log T = 4Λ(e^y) + log max_P Σ_{j∈P} e^{y_j} + const`; the first
term is a sum of exponentials, the second a max of log-sum-exps — both convex.
Differentiate; the subgradient of `log M` is `p_j τ_j / M` with `p` in the
convex hull of critical paths (Theorem 2 machinery).

**Novelty.** Prior work optimises fidelity *or* latency; this gives the first
closed-form optimum of the *product* that governs mitigated-experiment cost,
and it reuses the path-flow dual. Extends verbatim to idle/DD variables (C2).

**Test (simulator):** Aer + Pauli–Lindblad noise; implement PEC by
quasi-probability sampling; measure shots-to-precision × makespan vs ASAP,
uniform, and fidelity-optimal schedules on MQT Bench circuits (≤12 qubits).

### C4. Learn-then-optimize with a statistical guarantee (ellipsoidal robust GP)

Fitted coefficients `θ_j = (a_j, γ_j)` come with a confidence ellipsoid
`{θ : (θ-θ̂)ᵀ Σ⁻¹ (θ-θ̂) ≤ κ²}` at level `1-δ`.

**Theorem C4.** The worst-case gate hazard over the ellipsoid is
`θ̂ᵀv(τ) + κ ‖Σ^{1/2} v(τ)‖` with `v = (1/τ, τ)`. Using `Σ ≼ 2 diag(Σ)`,

    worst-case ≤ â/τ + γ̂τ + κ √(2 Σ_aa/τ² + 2 Σ_γγ τ²),

a **generalized posynomial**, so the robust problem is an exact GP. With
probability `≥ 1-δ` (over calibration data, within the model class) the true
model error of the returned schedule is below its certificate.

**Baseline-dominance add-on.** Add `E_s(τ) ≤ E_s(τ_base)` for every scenario
`s` (inverse powers `p ∈ {1,2}`, share range of α, ellipsoid corners). The
problem is always feasible (baseline satisfies it) and the result is never
worse than the baseline for any law in the **convex hull** of the scenarios,
because each `E_s` is linear in its coefficients.

**Why it matters.** Directly fixes Table VII (6/6 regressions under a
mis-specified law) by construction, and gives the first PAC-style guarantee for
a quantum scheduling decision.

**Test:** re-run Table VII and the matched-deadline study with the dominance
constraint; regressions must be 0 inside the hull. Report behaviour outside
the hull honestly (e.g. an inverse-cube truth).

### C5. Anisotropic, observable-aware allocation (explains the X/Z reversal)

Give each local Pauli its own law `a_jP/τ + γ_jP τ`. The constraint in C1 stays
posynomial, so anisotropy is free. **Physical prediction:** a continuous X
drive spin-locks X-basis states (decay set by `T_1ρ`, not `T_2*`), so
`γ_X ≪ γ_Z`. The model then *predicts* the measured native-X result (X basis:
slow X wins; Z basis: slow X loses). This turns the paper's falsification of
the isotropic model into a confirmation of the anisotropic one — **only if** the
prediction is pre-registered before new data (Section 5, job H1 includes it).

### C6. Complexity and approximation of discrete pulse libraries

**Theorem C6a (hardness).** Choosing one pulse per gate from finite libraries to
minimise error under a deadline is strongly NP-hard, by reduction from the
deadline discrete time–cost trade-off problem (De, Dunne, Ghosh, Wells, Oper.
Res. 1997). Bounded-degree circuit DAGs suffice by inserting zero-cost
fixed-duration identity nodes — *this reduction must be written carefully*.

**Theorem C6b (bicriteria approximation).** Solving the continuous GP with
`γ̃ = qγ` and rounding upward on a ratio-`q` log grid gives a schedule with
makespan `≤ q·M*` and error `≤ OPT_cont(qγ)`, in polynomial time (cf. Skutella
1998 for the classical analogue).

**Why it matters.** It answers why a continuous relaxation is the right tool,
not just a convenient one.

### C7. Latency–fidelity Pareto frontier is log-log convex; dual = marginal price

**Statement.** Let `F(D)` be the optimal log-error at deadline `D`. Then
`F(e^d)` is convex in `d`, and `-dF/d log D` equals the optimal multiplier of
the deadline constraint ("fidelity per log-nanosecond").

**Proof.** Standard GP sensitivity (the optimal value of a GP is convex in the
log of the constraint right-hand sides). Cheap to prove, but it gives compiler
writers a **usable knob**: report the price, and stop compressing when the price
falls below a threshold. Figure: frontiers for QFT, QAOA, GHZ, mirror circuits.

### C8. Scalability via path-flow Frank–Wolfe

Show the dual Frank–Wolfe in `pathflow.py` solves 10⁴–10⁵-gate DAGs (1,000+
qubits, Stim-generated Clifford circuits) in seconds with a certified gap,
against Clarabel/MOSEK-free interior point. Pure optimisation — no quantum
simulation needed. Report empirical scaling exponents, not "linear time".

---

## 3. Contribution → module map

| Contribution | New / changed code | New benchmark script |
|---|---|---|
| C1 | `observables.py` (per-Pauli rates) | `tests/test_catalogue.py` |
| C2 | new `src/qcgp/ddgp.py` (CVXPY, perspective via `power`/`quad_over_lin`-style atoms) | `benchmarks/run_dd_gp_sim.py` |
| C3 | new `src/qcgp/mitigation.py` (PEC sampler, runtime objective) | `benchmarks/run_runtime_optimal.py` |
| C4 | `generalized.py` (ellipsoid term, dominance constraints) | `benchmarks/run_robust_pac.py`, rerun Table VII |
| C5 | `observables.py` anisotropic sensitivity | `benchmarks/run_anisotropic_prediction.py` |
| C6 | paper only + `library.py` rounding check | `tests/test_bicriteria.py` |
| C7 | new `src/qcgp/frontier.py` | `benchmarks/run_frontier.py` |
| C8 | `pathflow.py` | `benchmarks/run_scale.py` |

---

## 4. Simulator evaluation (all free)

| Tool | Use | Size |
|---|---|---|
| Stim | Clifford circuits, observable tracking, scale study | 100–1,000+ qubits |
| Qiskit Aer (density matrix / trajectories) | Markovian Pauli–Lindblad + PEC | ≤ 12 qubits |
| NumPy/QuTiP time-dependent trajectories | 1/f^β and quasi-static dephasing for DD (C2) | ≤ 6 qubits exact |
| MQT Bench / QASMBench | benchmark circuits (QFT, QAOA, GHZ, VQE, Grover, mirror) | as above |

**Baselines (all at matched makespan):** ASAP, ALAP, optimal uniform, Qiskit
DD (XX, XY4, UR), adapted Lussi et al., fidelity-only GP, latency-only GP.

**Protocol rules (to avoid the current paper's statistical problems):**
- Report per-instance distributions and per-group results, never a lone median.
- Report the fraction of instances where GP returns the baseline schedule.
- Mis-specification sweep: true law ∈ {1/τ, 1/τ², 1/τ³, mixtures}, β ∈ {0, 0.5, 1, 2, quasi-static}.
- Seeds and configs fixed in a manifest before running; SHA-256 locked.

**Success criteria (pre-declared):**
- C2: GP-DD beats best Qiskit DD at matched makespan under ≥ 3 of 5 noise
  spectra, and never loses by more than its certificate under the others.
- C3: ≥ 20% median QPU-time reduction vs ASAP + PEC at equal precision.
- C4: 0 regressions inside the scenario hull; regressions outside are reported.

---

## 5. Minimal IBM free-tier experiments (≈ 3–4 min total)

Only DD and idle decisions are executable on Heron (no pulse-duration control),
so hardware tests target C2 and C5. Gate-duration claims remain simulator-only
and the paper must say so.

**H1 — spectroscopy + anisotropy calibration (~60–90 s QPU).**
3 qubits measured in parallel; CPMG with `n ∈ {0, 2, 4, 8}` over
`w ∈ {4, 16, 40, 80} µs`; X- and Z-basis inputs; 1,000 shots; 3 randomized
blocks. Fit `(A_q, β_q, ε_π)` per qubit. Freeze the fitted model and
**pre-register** the C2 schedules and the C5 X/Z prediction (hash the file).

**H2 — prospective held-out test (~90–120 s QPU).**
2 idle-heavy circuits (GHZ-5 with mid-circuit idle layers; 5-qubit mirror
circuit) × 4 policies {no DD, Qiskit XX, Qiskit XY4, GP-DD} × 3 randomized
blocks × 2,000 shots, on qubits **not** used in H1 where possible. Primary
endpoint: GHZ parity / mirror return probability, paired block differences.

Leave ≥ 5 min of the monthly allowance as a reserve for one re-run if a
calibration update lands mid-job.

---

## 6. Execution order

1. **Literature gate (before any code):** confirm novelty of C2, C3, C4
   (search terms above). If C2 joint convexity is known, C3 becomes the headline.
2. **Theory:** write full proofs for C2, C3, C4 (highest value), then C6, C7.
3. **Code + simulator studies:** C4 (cheapest, fixes Table VII) → C2 → C3 → C5 → C7 → C8.
4. **Hardware:** H1, freeze, pre-register, H2.
5. **Rewrite:** new title/abstract around Section 1; move old theorems to
   preliminaries; delete the 24.5% plateau claim; cut appendices to supplement.

## 7. Honest expectation

Novelty is judged by referees, so no plan guarantees 10/10. With C1–C4
proved and C2 confirmed prospectively on hardware, the paper becomes a
framework contribution with a closed-form, hardware-validated result
(≈ 8–9/10 at Quantum / PRX Quantum level). If H2 is null, C1–C4 still stand as
theory + simulator contributions (≈ 7/10, strong for IEEE TQE / QST).
