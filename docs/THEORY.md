# Theory for contributions C1–C8

Status: proofs written for the manuscript. Each result carries its assumptions; none of them asserts
anything about hardware beyond the declared model class. Code references point to the
implementation and to the numerical checks that exercise each statement.

## 0. Setting and notation

A fixed circuit has timed gates `j = 1..m` with durations `τ_j ∈ [ℓ_j, u_j]`, `0 < ℓ_j ≤ u_j`, and a
dependency DAG `D` (consecutive gates on a shared qubit). `P` is the set of source–sink paths and
`M(τ) = max_{P∈P} Σ_{j∈P} τ_j`. A gate's Pauli–Lindblad channel has total intensity
`λ_j(τ_j) = a_j τ_j^{-p} + g_j τ_j` (`a_j, g_j ≥ 0`, declared `p > 0`), or per local Pauli `P`,
`λ_{jP}(τ_j) = a_{jP} τ_j^{-p} + g_{jP} τ_j`.

**Preliminaries (known results, credited, not claimed as new).** Convexity and GP form of
`Σ_j λ_j(τ_j) + b M(τ)` (Boyd et al. 2005, 2007); the path-flow minimax dual and its
Frank–Wolfe solution (Kelley 1961; Fulkerson 1961; Jaggi 2013); no-jump channel bounds
(Dalibard–Castin–Mølmer 1992); exact Pauli decay of stabilizer expectations
(van den Berg et al. 2023). These are Section 2 of the manuscript.

---

## C1. Log-posynomial catalogue

**Theorem C1.** Let `x > 0` collect controllable resources (durations, idle windows, pulse counts,
shots) and let every Pauli intensity `λ_{jP}(x)` be a posynomial in `x`. For a Clifford circuit
with ideal stabilizer generators `S_k`:

1. `⟨S_k⟩ = exp(−2 Σ_j Σ_{P∈A_{kj}} λ_{jP}(x))`, where `A_{kj}` is the set of local Paulis at gate
   `j` that anticommute with `S_k` propagated backward to `j`. Hence `⟨S_k⟩ ≥ t_k` is the
   posynomial constraint `Σ_j Σ_{P∈A_{kj}} λ_{jP}(x) ≤ −½ ln t_k`.
2. The PEC sampling overhead is `γ(x) = exp(2 Σ_{j,P} λ_{jP}(x))`, and the variance of the PEC
   estimator of any ±1-valued observable is `γ² − ⟨O⟩² ≤ γ²`.
3. With `y = log x`, both `log γ(e^y)` and `log(γ(e^y)²/s)` are convex.

*Proof.* (1) The adjoint of `L_P(ρ) = λ(PρP − ρ)` maps a Pauli `O` to `−2λ O` if `{O,P}=0` and to
0 otherwise. Independent commuting generators compose, and ideal Clifford layers map Paulis to
Paulis, so each anticommuting `(j,P)` contributes a factor `e^{−2λ_{jP}}`. (2) The single-Pauli
channel `ρ ↦ (1−q)ρ + qPρP`, `q = (1−e^{−2λ})/2`, has inverse `αρ − βPρP` with
`α = (e^{2λ}+1)/2`, `β = (e^{2λ}−1)/2`; its quasi-probability norm is `α + β = e^{2λ}`, and norms
multiply. Each shot returns `±γ`, so `E[X²] = γ²`. (3) A posynomial in `e^y` is a sum of
exponentials of affine functions of `y`, hence convex; `−log s` is linear in `log s`. ∎

*Checks:* `tests/test_observables.py` (dense Lindblad agreement); PEC unbiasedness and the
variance law are verified by Pauli-frame sampling in `evidence/runtime_optimal/summary.json`.

---

## C2. Joint gate-duration and dynamical-decoupling allocation (headline)

**Model.** Let `W` be the idle windows: between consecutive timed gates on a qubit, and from a
qubit's last gate to the common makespan `M` (time before a qubit's first gate is excluded:
the qubit is still in its initial computational state). With start times `s`, a window `k` on qubit
`q` between gates `a` and `b` has width `w_k = s_b − s_a − τ_a` (or `M − s_a − τ_a`), which is
affine in `(s, τ, M)`. A window carrying `n_k ≥ 0` refocusing π-pulses under a power-law
dephasing spectrum `S(ω) ∝ ω^{−β_q}` has hazard

`h_q(w, n) = c_q w^{1+β_q} (n+1)^{−β_q} + η_q w + ε_q n`,   subject to `n τ_π ≤ w`,

which interpolates free induction (`n = 0`, `χ ∝ w^{1+β}`) and the CPMG scaling
`χ ∝ w^{1+β} n^{−β}` (Cywiński et al. 2008) and adds Markovian decay `η_q` and per-pulse error `ε_q`.

**Theorem C2 (joint convexity).** The problem

`min Σ_j (a_j/τ_j + g_j τ_j) + Σ_{k∈W} h_{q(k)}(w_k, n_k)` over `(τ, s, M, n)`

subject to `ℓ ≤ τ ≤ u`, `s ≥ 0`, `w_k ≥ n_k τ_π ≥ 0`, `M ≤ D`, is a convex program. Its optimal
value lower-bounds the value of every schedule with integer pulse counts.

*Proof.* For `β ≥ 0`, `φ(x) = x^{1+β}` is convex on `x ≥ 0`, and its perspective
`(w, t) ↦ t φ(w/t) = w^{1+β} t^{−β}` is jointly convex on `t > 0`. With `t = n + 1` (affine)
and `w` affine in the decision variables, each `c_q w^{1+β}(n+1)^{−β}` is convex; the remaining
terms are linear or convex, and the constraints are linear. Integer `n` is a subset of the
real relaxation. ∎

**Corollary C2.1 (optimal pulse count).** For a fixed window `w` with `β > 0`, the relaxed optimum is
`n* + 1 = max{1, (β c w^{1+β} / ε)^{1/(1+β)}}`, capped by `w/τ_π`. Because `n ↦ h(w,n)` is convex,
the best even integer is one of the two even neighbours of `n*`.

**Corollary C2.2 (drive-versus-idle threshold).** Consider white idle noise (`β = 0`, total rate
`η'_q = c_q + η_q`) and a gate `j` on qubit set `Q_j` whose neighbours are fixed, so that lengthening
`τ_j` by δ shortens the adjacent idle windows on each `q ∈ Q_j` by δ. Then
`τ_j* = clip(√(a_j/(g_j − η̄_j)), ℓ_j, τ_j^max)` if `g_j > η̄_j := Σ_{q∈Q_j} η'_q`, and
`τ_j* = τ_j^max` (fill all available slack) otherwise. *Proof:* the objective in `τ_j` is
`a_j/τ_j + (g_j − η̄_j) τ_j + const`. ∎ A gate should therefore be stretched exactly when the
driven qubit loses coherence more slowly than the idle one. This also removes the
non-uniqueness of off-critical durations in the manuscript's Remark 1.

**Remark C2.3 (not a GP).** Idle time `M − Σ τ` enters with a negative coefficient on `τ`, so the
idle-aware problem is a convex program but not a geometric program in `(τ, M)`. For fixed
windows it is an exact GP. The manuscript states this explicitly.

*Implementation:* `src/qcgp/ddgp.py` (geometric-mean cone for the perspective, even-integer
rounding, re-solve, certified relaxation gap). *Checks:* `tests/test_novelty.py`;
`evidence/dd_gp/` (independent time-correlated simulator, model fitted only from spectroscopy).

---

## C3. Runtime-optimal error-mitigated estimation

**Theorem C3.** Let `Λ(τ) = Σ_j λ_j(τ_j)` with `p = 1`, let `t_rep ≥ 0` be the per-shot repetition
overhead and `ε` the target standard error. The PEC QPU time
`T(τ) = ε^{−2} exp(4Λ(τ)) (M(τ) + t_rep)` is log-convex in `y = log τ`. If `τ*` is optimal and
`τ_j*` is interior, then

`τ_j* = √( a_j / (g_j + p_j / (4(M* + t_rep))) )`,

where `p ∈ ∂M(τ*)` lies in the convex hull of incidence vectors of critical paths: `p_j ∈ [0,1]`, and
`p_j = 0` for every gate on no critical path.

*Proof.* `log T = 4Λ(e^y) + log(M(e^y) + t_rep) − 2 log ε`. The first term is a sum of exponentials
of affine functions. `M(e^y) + t_rep = max_P (Σ_{j∈P} e^{y_j} + t_rep)`, and the log of each
argument is a log-sum-exp of affine functions; a maximum of convex functions is convex. For
interior `τ_j` the optimality condition in `τ` coordinates is
`0 = 4(−a_j/τ_j² + g_j) + p_j/(M + t_rep)` with `p ∈ ∂M(τ*) = conv{1_P : P critical}` (Danskin).
Solve for `τ_j`. ∎

**Corollary C3.1.** As `t_rep → ∞`, `τ_j* → √(a_j/g_j)` (the error-optimal duration): when
repetition overhead dominates, minimising QPU time reduces to minimising error, and the
latency-optimal schedule is never optimal. On IBM hardware with the default ~250 µs repetition
delay, the gain over error-optimal durations is therefore small. The manuscript reports this.

*Implementation:* `src/qcgp/mitigation.py`. *Checks:* `tests/test_novelty.py` (Nelder–Mead
reproduces the closed form on a two-path DAG), `evidence/runtime_optimal/` (implied flows lie
in `[0,1]` and vanish off critical paths; PEC sampling confirms unbiasedness and variance `γ²−1`).

---

## C4. Certified robust allocation and baseline dominance

**Setting.** Gates are grouped (e.g. by type). Gate `j` in group `g` has error
`A_g α_j τ_j^{−p} + G_g β_j τ_j` with known multipliers `α, β ≥ 0`. A calibration produces a Gaussian
estimate `θ̂_g` of `θ_g = (A_g, G_g)` with known covariance `Σ_g`, independent across groups.

**Lemma C4.1.** For 2×2 positive semidefinite `Σ`, `Σ ≼ 2 diag(Σ)`; if `Σ` is diagonal, `Σ = diag(Σ)`.
*Proof:* `2 diag(Σ) − Σ = [[σ₁₁, −σ₁₂], [−σ₁₂, σ₂₂]]` has nonnegative diagonal and determinant
`σ₁₁σ₂₂ − σ₁₂² ≥ 0`. ∎

**Theorem C4 (calibration-coverage certificate).** Let `v_g(τ) = (Σ_{j∈g} α_j τ_j^{−p}, Σ_{j∈g} β_j τ_j)`
and `E = {θ : Σ_g (θ_g − θ̂_g)ᵀ Σ_g^{−1} (θ_g − θ̂_g) ≤ κ²}`. Then for every `τ`,

`sup_{θ∈E} E_θ(τ) = E_{θ̂}(τ) + κ √(Σ_g v_gᵀ Σ_g v_g) ≤ E_{θ̂⁺}(τ) + κ √(c Σ_g (σ_{g,11} v_{g,1}² + σ_{g,22} v_{g,2}²)) =: C(τ)`

with `c ∈ {1, 2}` from Lemma C4.1 and `θ̂⁺ = max(θ̂, 0)`. `C(τ)` is a generalized posynomial, so
`min_τ C(τ)` under DAG and deadline constraints is an exact GP. If `κ² = χ²_{2G}(1 − δ)`, then with
probability at least `1 − δ` over the calibration data, `E_θ(τ) ≤ C(τ)` holds **simultaneously
for all τ**, in particular for the data-dependent optimiser `τ̂`.

*Proof.* The supremum of a linear function `θ ↦ Σ_g θ_gᵀ v_g` over the ellipsoid is
`θ̂ᵀv + κ‖Σ^{1/2}v‖` (Cauchy–Schwarz). Lemma C4.1 bounds `v_gᵀΣ_g v_g`; replacing `θ̂` by `θ̂⁺`
can only increase the nominal term because `v ≥ 0`. A posynomial raised to the power ½ is a
generalized posynomial (Boyd et al. 2007, §5), and the epigraph `r ≥ √(·)` becomes the posynomial
constraint `(·)/r² ≤ 1`. Under the Gaussian model the Mahalanobis statistic is `χ²_{2G}`, so
`P(θ ∈ E) = 1 − δ`, and on that event the bound holds for every `τ`. ∎

**Theorem C4' (baseline dominance on a law hull).** Let `E_1, …, E_S` be declared error laws, each
of the form `E_s(τ) = Σ_r Σ_j c^{(s)}_{rj} τ_j^{e_r}` with `c ≥ 0`, and let `τ⁰` be a baseline
schedule that meets the deadline. Adding the constraints `E_s(τ) ≤ E_s(τ⁰)` for all `s` keeps the
GP feasible, and the optimiser `τ̂` satisfies `E(τ̂) ≤ E(τ⁰)` for every law `E` in the convex hull
of `{E_s}`. *Proof.* `τ⁰` satisfies every added constraint. For `E = Σ_s μ_s E_s` with `μ` in the
simplex, `E(τ̂) = Σ_s μ_s E_s(τ̂) ≤ Σ_s μ_s E_s(τ⁰) = E(τ⁰)`. ∎ No statement is made for laws outside
the hull; the manuscript tests such laws explicitly.

*Implementation:* `src/qcgp/robust.py`. *Checks:* `tests/test_novelty.py` (sampled ellipsoid worst
case equals the formula; hull dominance), `evidence/robust_pac/` (Table VII rerun; coverage).

---

## C5. Anisotropic and state-aware constraints

**Proposition C5.1 (support-only decay under isotropy).** If gate `j`'s channel is isotropic on its
`k_j` qubits, then for every observable whose back-propagated restriction to `supp(j)` is not the
identity, exactly `2·4^{k_j−1}` of the `4^{k_j} − 1` local non-identity Paulis anticommute with it.
Hence the decay of `⟨O⟩` depends only on the set of gates its back-propagated support touches,
not on its Pauli type. In particular, single-qubit X, Y and Z observables of an idle or
single-qubit-Clifford wire decay identically, so every schedule ranking is basis-independent.

*Proof.* Fix a non-identity Pauli `R` on `k` qubits. The map `P ↦ [P anticommutes with R]` is a
nonzero linear functional on `F₂^{2k}` (symplectic product), so exactly half of the `4^k` Paulis
anticommute with `R`, and the identity does not. ∎

**Consequence.** A basis-dependent ranking measured on hardware (Table X of the manuscript)
falsifies isotropy at the level of the model, not merely a coefficient value. Per-Pauli
intensities are the minimal extension: by Theorem C1(1) the constraints stay posynomial.

**Proposition C5.2 (state-aware T1 weights).** For Clifford circuits, the ideal single-qubit
Bloch vector of each qubit in each idle window is computable by stabilizer tracking, with
`⟨Z_q⟩ ∈ {−1, 0, +1}`. Weighting the amplitude-damping exposure of a window by the excited
population `(1 − ⟨Z_q⟩)/2 ≥ 0` multiplies a posynomial term by a nonnegative constant and
preserves the GP. This captures the non-unital fact that a qubit idling in `|0⟩` does not decay,
which no Pauli-twirled model can represent.

*Implementation:* `src/qcgp/anisotropic.py`. *Checks:* `evidence/anisotropic/` (a model calibrated
only from generic characterisation reproduces the X/Z ranking reversal of the four native-X
policies; an isotropic model cannot).

---

## C6. Discrete pulse libraries: hardness and bicriteria approximation

**Theorem C6.1 (NP-hardness).** Given a fixed DAG, finite libraries `L_j ⊂ [ℓ_j, u_j]` of
characterized `(duration, error)` options, and a deadline `D`, minimising total error subject to
`M ≤ D` is strongly NP-hard.

*Proof.* Reduction from the deadline version of the discrete time–cost trade-off problem
(DTCTP), which is strongly NP-hard on general activity-on-node DAGs (De, Dunne, Ghosh and Wells,
*Operations Research* 45(2), 1997). Map each activity to a timed gate and each mode
`(t, cost)` to a library option `(t, error = cost)`; the deadline is unchanged. The instance is a
circuit dependency DAG: assign one qubit to each DAG edge `(i, i')` and let gate `j` act on the
qubits of all its incident edges, so the only precedences between gates are the given edges and
their transitive closure. Optimal values coincide. ∎

*Scope.* The reduction uses gates of unbounded arity. Hardness for circuits restricted to
two-qubit gates is not claimed here; it is stated as an open question.

**Theorem C6.2 (bicriteria approximation).** Let `q > 1`. Solve the continuous problem with
exposure coefficients `q g_j` under deadline `D`, obtaining `τ^c`, and round each duration up on
the grid `ℓ_j q^k`, clamped at `u_j`, obtaining `τ^r`. Then `M(τ^r) ≤ q D` and, for every
library `L ⊂ [ℓ, u]`, `E(τ^r) ≤ q · OPT_L(D)`.

*Proof.* Rounding gives `τ^c ≤ τ^r ≤ q τ^c` componentwise (clamping keeps `τ^r ≤ u` and `τ^r ≥ τ^c`
because `τ^c ≤ u`). Path sums scale by at most `q`, so `M(τ^r) ≤ q M(τ^c) ≤ q D`. Next,
`E(τ^r) = Σ a/τ^r + g τ^r ≤ Σ a/τ^c + q g τ^c = OPT_cont(qg; D)`. For any feasible `τ`,
`Σ a/τ + q g τ ≤ q (Σ a/τ + g τ)`, so `OPT_cont(qg; D) ≤ q OPT_cont(g; D) ≤ q OPT_L(D)`, the last
step because every library schedule is feasible for the continuous problem. ∎

*Implementation:* `src/qcgp/approximation.py`. *Checks:* `evidence/theory_checks/` (300 random
instances against exhaustive library optima: no violations).

---

## C7. Latency–fidelity frontier

**Proposition C7.** Let `E*(D)` be the optimal error under deadline `D`. Then `d ↦ log E*(e^d)` is
convex on its domain, and wherever it is differentiable,
`d log E*/d log D = −ν*(D)`, where `ν*` is the optimal multiplier of the log-form deadline
constraint `log M ≤ log D`.

*Proof.* In log coordinates the problem is `min f₀(y)` subject to convex constraints and
`g(y) ≤ u` with `u = log D`, where `f₀ = log E` and `g = log M` are convex. The optimal value is
convex in the right-hand side perturbation `u`, and `−ν*` is its (super)gradient (Boyd and
Vandenberghe, §5.6). ∎

**Use.** `ν*` is a dimensionless *fidelity price*: the relative error reduction bought by a
1 % relative latency increase. A compiler can stop compressing once `ν*` crosses a threshold.

*Implementation:* `src/qcgp/frontier.py`. *Checks:* positive second differences and
`slope ≈ −ν*` on four benchmark circuits (`evidence/theory_checks/`).

---

## C8. Scalability

Two solvers are used. (1) A sparse, fully vectorised interior-point formulation (`sparse_interior_point`
in `src/qcgp/scale.py`): precedence constraints are a single sparse matrix, so model construction is linear
in circuit size, and Clarabel returns a primal–dual certificate. (2) The path-flow Frank–Wolfe method with a
level-vectorised longest-path oracle (`O(m + r)` work per step). Its dual value is a valid lower bound at
every iterate, so it independently bounds the suboptimality of the interior-point solution. Under realistic
makespan pressure Frank–Wolfe converges slowly (relative gaps of 0.1–3 % after 4,000 steps on 3k–133k
gates), so it is used as a certificate, not as the primary solver. Running times are reported empirically
(`evidence/scale/`), not as complexity bounds.

---

## What remains open

1. Hardness of C6.1 for two-qubit-gate circuits.
2. A coverage guarantee for C4 under non-Gaussian or misspecified calibration models.
3. Joint convexity in C2 when the dephasing spectrum is not a single power law (e.g. the sum of a
   quasi-static term and 1/f noise); the fitted single-power model is then an approximation, and
   the manuscript quantifies its accuracy in simulation.
4. Prospective hardware confirmation of C2 (protocol in `benchmarks/ibm_dd_protocol.py`).
5. **Cross-window correlation (found in simulation).** The window hazard of C2 is additive over windows,
   which is exact only if dephasing noise decorrelates between windows. Quasi-static and low-frequency 1/f
   noise do not: unrefocused phases from successive windows on one qubit add coherently, so the true hazard
   scales like `(Σ_k w_k)^{1+β}` rather than `Σ_k w_k^{1+β}`. In the repetition-code benchmark with
   quasi-static or 1/f noise, blanket XX decoupling therefore beats the per-window GP
   (`evidence/dd_state_aware/`), even though the fitted single-window model is accurate to a few
   percent. **Fixed schedules (the hardware case) are solved:** with protection indicators `z_k ∈ {0,1}`
   (balanced CPMG block in window `k`), the low-frequency hazard of qubit `q` is
   `c_q (Σ_k w_k (1 − z_k))^{1+β_q}`, convex in `z`, plus per-window residual, Markov and state-aware terms.
   The objective separates by qubit, so it is solved exactly by enumeration for up to 16 windows per qubit
   and by the rounded convex relaxation otherwise (`correlated_protection` in `src/qcgp/ddgp.py`).
   Jointly optimising placement and durations with cross-window coupling remains open: natural coupled
   terms such as `c (Σ_k w_k (n_k+1)^{-β/(1+β)})^{1+β}` are not jointly convex.
