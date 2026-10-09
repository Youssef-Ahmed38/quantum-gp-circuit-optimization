"""Build complete IEEE research draft from verified raw-data summaries."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]

def main():
    summary=json.loads((ROOT/'evidence/physical_study/summary.json').read_text())
    previous=(ROOT/'docs/joint_manuscript.tex').read_text()
    preamble=previous.split(r'\begin{abstract}')[0].replace(r'\input{joint_results.tex}','')
    abstract=r'''
\begin{abstract}
Quantum gate scheduling involves a tradeoff between control strength, exposure to noise, and latency. We formulate bounded duration allocation on a fixed circuit dependency graph as an exact geometric program (GP), while showing that an ordinary convex formulation solves the same problem. A path-flow dual provides an inspectable optimum bracket. We extend conditional operational guarantees to nonunitary Markovian jumps and bounded coherent mismatch, and specify RX/RZ/XX control Hamiltonians with relaxation, dephasing, and duration-dependent drive noise. For available pulse libraries, a separate mixed-integer linear model optimizes compositional channel-error bounds; a residual inequality transfers a continuous GP dual bound to these nonposynomial costs. Free local experiments cover @CONFIG@ configurations, @CHANNEL@ simulated channels, and @STATE@ input-state checks across six circuit families. Matched comparisons include uniform scaling, a reimplementation of published slack scheduling, continuous allocation, and finite-library optimization. At the same fastest-latency deadline, mean process-fidelity changes range from $-0.009$ to $0.026$ percentage points, with regressions in every regime; larger unconstrained gains require longer schedules. The results support a reproducible integration of established optimization and quantum-error tools, while leaving hardware calibration, broader scalability, and independent novelty assessment unresolved.
\end{abstract}
\begin{IEEEkeywords}
Quantum compilation, geometric programming, convex optimization, gate scheduling, Markovian noise, pulse libraries.
\end{IEEEkeywords}
'''
    start=previous.index(r'\section{Introduction}')
    end=previous.index(r'\section{Implementation and Reproducibility}')
    foundation=previous[start:end]
    foundation=foundation.replace('Our fastest-deadline comparator is a continuous surrogate inspired by that scheduling principle; it is not a reproduction of their pulse library or hardware experiments.','The earlier study used a continuous fastest-deadline surrogate. The present extension reimplements their library-selection decisions on a common simulated library, with the adaptations stated in Section VIII. It does not reproduce their hardware calibration.')
    foundation=foundation.replace('The main simulator uses local isotropic replacement noise and globally covariant background replacement noise. The independent pulse simulator uses local $Z$ dephasing for background noise.','The earlier unitary-jump study uses local isotropic replacement and globally covariant background replacement noise; an independent pulse simulator uses local $Z$ dephasing. The new general-jump study is specified below.')
    foundation=foundation.replace('In the numerical model, a shortest enclosing eigenphase arc selects a minimal-norm generator modulo global phase; matrix exponentiation verifies its unitary.','The earlier model uses a shortest enclosing eigenphase arc; the new study instead uses explicit Pauli generators, with matrix exponentiation checking their unitaries.')
    foundation=foundation.replace(r'\subsection{Noise Certificates and Optimization Certificates}',r'\subsection{Noise Certificates and Optimization Certificates}')
    foundation=foundation.replace('We specialize these ideas to one explicit timing model and verify their implementation independently.','Compositional robustness for quantum programs has also been formalized by Hung et al. \\cite{hung}. Robust quantum control via sequential convex programming is established \\cite{kosut}. We specialize related tools to explicit duration and library models and verify their implementation independently.')
    extension=r'''
\section{General Noise and Explicit Controls}
The exponential certificate in Theorem~\ref{th:noise} relies on unitary jumps. Relaxation has nonunitary jump operators and requires a different bound. Let $\Delta H(t)$ be the coherent Hamiltonian mismatch and define
\begin{equation}
N=\int\sum_\mu\|L_\mu(t)\|_\infty^2dt,\quad C=\int\|\Delta H(t)\|_\infty dt.
\end{equation}
\begin{proposition}[General-jump certificates]
For finite-dimensional Markovian dynamics with these integrable bounds,
\begin{align}
\tfrac12\|\mathcal N-\mathcal U\|_\diamond&\le\min(1,N+C),\\
F_{\mathrm{pur}}&\ge\max\{(1-N-C)_+,(1-N/2-C)_+^2\}.
\end{align}
Observable error for $\|O\|_\infty\le1$ is at most $\min(2,2N+2C)$; $(z)_+=\max(0,z)$.
\end{proposition}
\begin{proof}
The variation-of-constants identity writes the channel difference as an integral of the generator difference between CPTP propagators, each of diamond norm one. Multiplication norms give $\|\mathcal D[L]\|_\diamond\le2\|L\|^2$ and $\|[\Delta H,\cdot]\|_\diamond\le2\|\Delta H\|$. Integration gives the channel bound and the first fidelity floor by measuring the ideal purified-state projector. In the ideal interaction picture, the no-jump propagator obeys $\dot K_0=(-\mathrm i\widetilde{\Delta H}-\widetilde\Gamma/2)K_0$, with $\Gamma=\sum L_\mu^\dagger L_\mu\succeq0$. Its operator norm is at most one. Thus $\|K_0-I\|\le C+N/2$. Its overlap amplitude with any normalized purification is at least $(1-C-N/2)_+$, and other trajectories contribute nonnegative overlap. Squaring gives the second floor. Fidelity monotonicity and trace-norm duality give reduced-state and observable claims.
\end{proof}
These are elementary perturbation and trajectory arguments, rather than a claim of new general open-system theory. The stronger fidelity floor does not replace the channel-distance bound.

Our declared spin-control family uses $K_i=\theta_iP_i/2$ for $P_i=X,Z,X\otimes X$, realizing RX, RZ and RXX respectively. Always-on jumps are $\sqrt{1/T_{1q}}\sigma^-_q$ and $\sqrt{1/(2T_{\phi q})}Z_q$. During gate $i$, additional jumps are $\sqrt{\eta_i}K_i/\tau_i$ and $\sqrt{\gamma_i}P_i$. Consequently $N=R$ in \eqref{eq:risk}, with
\begin{equation}
b=\sum_q(1/T_{1q}+1/(2T_{\phi q})),\qquad a_i=\eta_i\kappa_i^2.
\end{equation}
A relative coherent strength mismatch $\epsilon$ adds $C=|\epsilon|\sum_i\kappa_i$. For duration-independent $\epsilon$, this changes the certificate without changing the allocation optimum. The model assumes local independent memory noise, available X/Z/XX controls and all-to-all pair availability. It does not include leakage, finite bandwidth, crosstalk or transmon calibration.

\section{Finite Pulse Libraries and Certificate Transfer}
\subsection{Exact Finite Selection and Relaxation}
For library durations $d_{ik}\in[\ell_i,u_i]$, choose binary $x_{ik}$ with $\sum_kx_{ik}=1$ and $\tau_i=\sum_kd_{ik}x_{ik}$. Substituting these affine durations into the completion constraints gives a mixed-integer linear program (MILP), whose continuous-envelope objective is
\begin{equation}
J_{\mathrm{lib}}=\sum_{ik}(A_i/d_{ik}+C_id_{ik})x_{ik}+BM.
\end{equation}
Relaxing $x_{ik}\in[0,1]$ gives an LP lower bound. A feasible discrete policy supplies an upper bound. The continuous GP dual bound is also a lower bound for this same cost. Fractional solutions are not emitted as pulses, and a time-limited MILP incumbent retains its recorded dual gap. No polynomial-time discrete-selection claim is made.

\subsection{Simulated-Channel Costs}
For each local pulse channel $\mathcal E_{ik}$ and intended channel $\mathcal U_i$, let $J_{ik}$ be the unnormalized Hermitian Choi matrix of their difference. Watrous's diamond-norm SDP \cite{watrous} gives the feasible dual upper bound
\begin{equation}
\delta_{ik}=\min\{1,\tfrac12\|\operatorname{Tr}_{\rm out}|J_{ik}|\|_\infty\}.
\end{equation}
Indeed, the block matrix with diagonal blocks $|J_{ik}|$ and off-diagonal blocks $-J_{ik}$ is positive semidefinite. The normalized Choi trace distance supplies a lower bound, independently checked against small diamond-norm SDPs.

Let $b_i$ be the memory rate summed over gate $i$'s support. Its characterized pulse channel already includes that memory noise. For independent local noise and disjoint concurrent controls, scheduled evolution factors into local pulse channels and interleaved idle channels. Telescoping the channel difference and using CPTP contractivity gives
\begin{align}
Q(x,M)&=bM+\sum_{ik}(\delta_{ik}-b_id_{ik})x_{ik},\\
\tfrac12\|\mathcal N-\mathcal U\|_\diamond&\le\min(1,Q),\quad F_{\rm pur}\ge(1-Q)_+.
\end{align}
The idle exposure $bM-\sum_i b_i\tau_i$ is nonnegative because a qubit executes at most one gate at a time. This subtraction prevents double-counting active memory noise. Correlated noise generally invalidates this factorization. Some local coefficients are negative, so these characterized costs are not asserted to form a GP; their finite MILP and LP remain valid.

\begin{proposition}[GP-to-library transfer]
For the risk-only objective normalized by $R_0$, let $D_{\rm GP}$ be a feasible continuous dual bound and
\begin{equation}
e_{ik}=(\delta_{ik}-b_id_{ik}-a_i/d_{ik}-\gamma_id_{ik})/R_0.
\end{equation}
Then the characterized finite optimum satisfies
\begin{equation}
J_{\rm char}^*\ge\max\{0,D_{\rm GP}+\sum_i\min_ke_{ik}\}.
\end{equation}
\end{proposition}
\begin{proof}
Every library policy lies in the GP duration box. At such a policy the characterized objective equals the GP objective plus $\sum_i e_{i,k(i)}$. Weak duality lower-bounds the former by $D_{\rm GP}$; each residual is at least its gatewise minimum. Minimizing over policies preserves the inequality. Nonnegativity of the complete error radius allows clipping at zero.
\end{proof}
This is a conservative model-perturbation argument. It requires a valid dual bound, rather than an approximate GP primal value, and may be vacuous. Independent LP/MILP bounds can strengthen the bracket. Its integration with channel-characterized policies is the candidate application contribution; it is not a new general perturbation principle.

\section{Implementation and Experimental Protocol}
The Python implementation provides modular timing models, GP/ordinary-convex interfaces, a path-flow solver, finite LP/MILP selection, and independent channel simulation. Qiskit analysis passes annotate supplied circuit models rather than emitting device pulses. CVXPY with open-source CLARABEL/ECOS/SCS handles convex problems; SciPy/HiGHS handles finite choices. No cloud account or commercial solver is used. The following executable pattern is supplied in an example and notebook:
\begin{quote}\small\ttfamily
model = physical\_model(circuit, parameters)\\
flow = optimize\_pathflow(model)\\
library = duration\_library(model)\\
bounds, rates, records = characterize\_library(\\
\hspace*{1em}circuit, parameters, library)\\
policy = optimize\_library(model, library,\\
\hspace*{1em}pulse\_bounds=bounds, support\_rates=rates)
\end{quote}

\subsection{Declared Design}
The extended study has six families: QFT, Grover, VQE ansatz, QAOA, seeded Pauli circuits and mirror circuits. Each uses two or three qubits and seeds 101--104. Seeded RX/RZ preparation precedes QFT/Grover; these are distinct input-preparation circuits, not distinct algorithm definitions. Mirror inverse pairs are intentionally retained rather than cancelled. Across three drive-noise regimes $\eta=0.002,0.02,0.2$, the study contains @CONFIG@ configurations and @CIRCUIT@ native circuits. Each has six available duration ratios: $0.5,0.75,1,1.5,2,3$ times its base duration.

Time units are microseconds and $\hbar=1$. Base duration is $|\theta|/\omega$, with $\omega=1.5$ for one-body controls and $0.35$ for pair controls. $T_1$ and $T_\phi$ are independently seeded uniform draws over $[80,160]$ and $[120,240]$ respectively; the same size/seed values are shared across families. Active rate is $0.0005$ per microsecond; pair drive-noise coefficient is $4\eta$. These constructed coefficients and the design were saved before measurement. They are not estimated device parameters.

\subsection{Matched Policies}
Nine policies are evaluated: fixed, fastest, optimized uniform envelope, published slack, nearest-library GP, envelope MILP, continuous GP, characterized MILP, and optimized characterized uniform. Uniform policies enumerate all six common duration ratios. The published-slack comparator follows Lussi et al.'s shortest-pulse initialization, early/late times, priority by rotation-to-duration ratio, and extension only when the next library choice fits. We recompute critical-path times globally, use absolute signed angles, and apply a common Pauli library to all controls. Their calibrated single-qubit IBM pulses and fixed entangling gates are not reproduced. Comparison therefore isolates scheduling decisions within our model.

The separate original compilation study compares Qiskit levels 0--3, TKET/Cirq logical adapters with common routing, and an exploratory held-out-size learning selector. It found no GP advantage for fixed durations. Those baselines do not provide matched pulse-selection implementations. We do not claim superiority over native TKET/Cirq physical pipelines or published ML compilers.

\subsection{Validation and Analysis}
Full interval-by-interval Lindblad superoperators are computed with dense matrix exponentials. Each policy receives process fidelity, zero-input and seeded Haar-input fidelity, trace distance, and a bounded Pauli observable error. Six cases additionally use an adaptive differential-equation integrator and a diamond-norm SDP. All circuit translations, trace preservation and Choi positivity are checked. General and finite-library certificate inequalities are tested on saved channels. @TESTS@ unit/integration tests cover analytic relaxation, coherent mismatch, independent solvers, exhaustive small pulse policies, and infeasible deadlines.

All @CHANNEL@ policy channels and @STATE@ state checks remain in the analysis, including regressions and solver iteration limits. Paired descriptive comparisons are stratified by noise regime. Exploratory bootstrap intervals resample size/seed clusters because physical parameters are shared across families; they are not population or hardware confidence claims. The stronger no-jump floor and transfer bound were derived during the study and audited afterward against every saved configuration. They were not preregistered outcomes.

\section{Results}
\subsection{Matched Fidelity and Latency}
\input{physical_tables.tex}
Table~\ref{tab:physical} reports characterized finite selection against published slack. Positive mean changes are conditional on the declared model; the optimizer minimizes a conservative channel-error bound, not fidelity. Regressions demonstrate that a better certificate objective need not improve each actual circuit output. Figure~\ref{fig:gains} retains the entire paired distribution.
\begin{figure}[t]
\centering\includegraphics[width=\columnwidth]{../figures/physical_gains.pdf}
\caption{Paired process-fidelity changes against adapted published slack on the same simulated library. All configurations, including negative changes, are retained.}\label{fig:gains}
\end{figure}

\subsection{Certificates and Numerical Gaps}
The general trace-distance floor is nonvacuous for @LINEAR@ of @CHANNEL@ policy channels; the no-jump floor raises this count to @STRONG@. The minimum checked fidelity slack is @SLACK@. These are deterministic bounds conditional on the modeled generators, not statements about model calibration. Figure~\ref{fig:cert} compares certificate floors with the minimum of process and two input-state fidelities.
\begin{figure}[t]
\centering\includegraphics[width=\columnwidth]{../figures/physical_certificates.pdf}
\caption{General-jump certificate coverage, including vacuous floors. The diagonal marks equality; simulated fidelities must lie above it.}\label{fig:cert}
\end{figure}
The largest independent exponential/ODE channel difference is @ODE@. Translation unitary error is at most @UNITARY@; trace-preservation error is at most @TP@. Envelope and characterized finite selection have @FINITENON@ and @CHARNON@ uncertified-optimal incumbents respectively, with maximum recorded relative gaps @FINITEGAP@ and @CHARGAP@. Their feasible incumbents and lower bounds are retained. @FLOWNON@ continuous solves reach their iteration limit; the largest gap is @FLOWGAP@. A valid lower bound does not require meeting the requested tolerance.

Transferred GP lower bounds are positive in @TRANSFER@ of @CONFIG@ cases, but their mean gap to the nearest-library GP policy's characterized score is @TRANSFERGAP@\%. This comparison concerns the emitted GP policy, not the optimal MILP policy. Figure~\ref{fig:transfer} also shows independent finite LP lower bounds. The transfer argument remains valid even when loose; the evidence does not support substituting it for the stronger finite optimization bracket.
\begin{figure}[t]
\centering\includegraphics[width=\columnwidth]{../figures/physical_transfer.pdf}
\caption{Transferred GP and finite LP lower scores versus the emitted characterized MILP score. The diagonal represents a tight optimum bracket.}\label{fig:transfer}
\end{figure}

\subsection{Previous Study and Runtime Scope}
The preserved unitary-jump study has 96 configurations and 1152 state checks. Mean risk reductions against optimized uniform scaling range from 1.65\% to 2.73\%, with larger latency relative to base duration. These results use a different noise class and must not be pooled with the general-jump study. Its equal-model ordinary convex comparator confirms that GP does not uniquely cause quality improvements.
@RUNTIME@
Full physical channel simulation is limited to three qubits. Superoperator dimension is $4^n\times4^n$, so dense simulation is not a scalable evidence strategy. Earlier larger timing-only cases do not establish larger-system physical fidelity.

\section{Discussion and Submission Boundaries}
The framework separates continuous allocation optimality, finite-library selection gaps, conditional channel error, and calibration uncertainty. This separation corrects claims that a continuous GP globally optimizes arbitrary quantum synthesis or automatically raises fidelity. General relaxation is now included, and controls have explicit Pauli Hamiltonians. Nevertheless, finite bandwidth, leakage, correlated errors and restricted interaction availability can invalidate these models.

The principal candidate contribution is a reproducible GP-to-library integration with operational certificates and an explicit residual transfer bound. GP, convex project scheduling, trajectory bounds, channel-norm SDPs and MILP are established mathematics. Duration-aware robust scheduling has close prior work. Independent expert assessment is required to determine whether the integration supplies sufficient novelty and usefulness for a selective journal.

A stronger submission requires a convincing application-scale benchmark, empirically justified pulse/noise libraries available without paid execution, and a comparison against methods optimized for the same physical task. Calibrated public datasets or externally reproduced model parameters can support this work without hardware expenditure. Model uncertainty must be assessed before treating a simulator certificate as a device guarantee. No experimental Eagle/Heron result, universal adoption claim, quantum advantage, or journal acceptance is asserted.

The package includes raw circuits/channels, parameter provenance, tests, scripts, notebook outputs and hash manifests. A GitHub repository and archival DOI have not been created. The author must review ownership and licensing, source citations, manuscript accuracy and journal-specific declarations before deposition and submission. Quartiles must be checked for the selected year, category and index; IEEE formatting alone does not establish Q1 readiness.

\section{Conclusion}
An exact continuous GP and its ordinary convex equivalent provide bounded duration allocation on fixed circuit DAGs. Explicit controls, general-jump error bounds, finite pulse selection and certificate transfer make the framework more operationally interpretable. Free local tests and comparative simulations expose both improvements and limitations. The resulting research package supports independent review and further study; it does not establish a field-changing breakthrough.
'''
    values={'CONFIG':summary['counts']['configurations'],'CHANNEL':summary['counts']['policy_channels'],'STATE':summary['counts']['state_checks'],
        'CIRCUIT':summary['counts']['native_circuits'],'TESTS':119,'LINEAR':summary['certificate_coverage']['linear_nonvacuous'],
        'STRONG':summary['certificate_coverage']['stronger_nonvacuous'],'TRANSFER':summary['certificate_coverage']['transfer_nonvacuous']}
    n=summary['numerics'];cases=json.loads((ROOT/'evidence/physical_study/cases.json').read_text())
    values.update(SLACK=f"{n['minimum_general_fidelity_slack']:.2e}",ODE=f"{n['ode_difference']['maximum']:.2e}",UNITARY=f"{n['unitary_error_max']:.2e}",TP=f"{n['tp_error_max']:.2e}",FINITENON=n['finite_nonoptimal_count'],CHARNON=n['characterized_nonoptimal_count'],FINITEGAP=f"{n['finite_mip_gap']['maximum']:.2e}",CHARGAP=f"{n['characterized_mip_gap']['maximum']:.2e}",FLOWNON=sum(not v['continuous']['certified'] for v in cases.values()),FLOWGAP=f"{max(v['continuous']['relative_gap'] for v in cases.values()):.2e}",TRANSFERGAP=f"{100*summary['certificate_coverage']['transfer_gap']['mean']:.2f}")
    runtime=json.loads((ROOT/'evidence/physical_study/runtime.json').read_text())
    description='Repeated solver timings use six balanced two-qubit cases, one warmup and three recorded repetitions per method, with one BLAS/OpenMP thread and cached characterization. Median solve times are '
    description+=', '.join(f"{v['median']:.3f} s ({k.replace('_',' ')})" for k,v in runtime['summary'].items())+'. Characterization and simulation are excluded; these measurements establish neither asymptotic superiority nor a general runtime advantage.'
    values['RUNTIME']=description
    deadline=json.loads((ROOT/'evidence/physical_study/deadline_summary.json').read_text())
    matched='An exploratory follow-up evaluates characterized MILP selection under the same fastest-latency deadline as the slack comparator. Mean process-fidelity changes are '
    matched+=', '.join(f"{v['mean_gain_pp']:.3f} pp ({k.replace('_',' ')}; {v['regressions']}/{v['n']} regressions)" for k,v in deadline['regimes'].items())
    matched+='. These results isolate allocation from additional latency; the large unconstrained gains cannot be interpreted as equal-latency improvements. The follow-up adds '+str(deadline['channels'])+' channel simulations and '+str(deadline['state_checks'])+' state checks and was designed after inspecting the first study.'
    extension=extension.replace('Figure~\\ref{fig:gains} retains the entire paired distribution.','Figure~\\ref{fig:gains} retains the entire paired distribution.\n\n'+matched)
    tables=r'''\begin{table}[t]
\caption{Characterized MILP versus adapted published slack (48 cases per regime). Fidelity changes are percentage points (pp); intervals are exploratory cluster bootstraps.}\label{tab:physical}
\centering\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrr}\toprule
Regime & Mean pp & Interval & Regressions & Latency \%\\\midrule
'''
    for regime,label in [('low_drive','Low'),('balanced','Balanced'),('high_drive','High')]:
        r=summary['regimes'][regime]['published_slack'];tables+=f"{label} & {r['process_gain_pp']['mean']:.3f} & [{r['interval']['low']:.2f},{r['interval']['high']:.2f}] & {r['process_regressions']} & {r['latency_change_percent']['mean']:.2f}\\\\\n"
    tables+=r'\bottomrule\end{tabular}\end{table}'
    (ROOT/'docs/physical_tables.tex').write_text(tables+'\n')
    refs=previous[previous.index(r'\begin{thebibliography}'):previous.index(r'\end{thebibliography}')].replace('{9}','{10}',1)
    refs+=r'''
\bibitem{watrous} J. Watrous, ``Simpler semidefinite programs for completely bounded norms,'' \emph{Chicago Journal of Theoretical Computer Science}, Art. no. 8, 2013. [Online]. Available: \url{https://cs.uwaterloo.ca/~watrous/Papers/SimplerSDPforCBnorm.pdf}
\bibitem{hung} S.-H. Hung, K. Hietala, S. Zhu, M. Ying, M. Hicks, and X. Wu, ``Quantitative robustness analysis of quantum programs,'' \emph{Proceedings of the ACM on Programming Languages}, vol. 3, POPL, Art. no. 31, 2019, doi: 10.1145/3290344. Extended version: \url{https://www.cs.umd.edu/~mwh/papers/qcerrs-full.pdf}
\bibitem{kosut} R. L. Kosut, M. D. Grace, and C. Brif, ``Robust control of quantum gates via sequential convex programming,'' \emph{Physical Review A}, vol. 88, Art. no. 052326, 2013, doi: 10.1103/PhysRevA.88.052326.
\end{thebibliography}
\end{document}
'''
    text=preamble+abstract+foundation+extension+refs
    for key,value in values.items():text=text.replace('@'+key+'@',str(value))
    if any('@'+key+'@' in text for key in values):raise ValueError('Unresolved manuscript placeholder')
    (ROOT/'docs/physical_manuscript.tex').write_text(text,encoding='utf8')
    print('Generated complete IEEE source and measured table.')
if __name__=='__main__':main()
