"""Tests for contributions C1-C8 (see docs/THEORY.md)."""
import math

import numpy as np
import pytest

from qcgp.anisotropic import AnisotropicModel, isotropic_split
from qcgp.approximation import bicriteria, library_optimum
from qcgp.ddgp import DDProblem, IdleModel, asap, model_objective, solve_dd_allocation
from qcgp.frontier import pareto_frontier
from qcgp.noisesim import GateLaw, QubitPhysics, ScheduledCircuit, gate, simulate, spectroscopy
from qcgp.robust import RobustSpec, coverage_radius, law_value, solve_certified_robust
from qcgp.scale import fast_pathflow, random_circuit_dag


# C4 ---------------------------------------------------------------------------

def test_robust_worst_case_matches_ellipsoid_sampling():
    lo, up, edges = np.array([1., 1.]), np.array([4., 4.]), ((0, 1),)
    cov = np.array([[[.04, 0], [0, .0025]], [[.01, 0], [0, .0025]]])
    spec = RobustSpec.independent(np.array([2., 1.]), np.array([.5, .5]), cov)
    r = solve_certified_robust(lo, up, edges, spec, 1., deadline=4.)
    t = r["durations"]
    rng = np.random.default_rng(0)
    worst = 0.
    for _ in range(50000):
        u = rng.normal(size=4)
        u /= np.linalg.norm(u)
        th = np.array([[2, .5], [1, .5]]) + np.stack([np.linalg.cholesky(c) @ u[2 * i:2 * i + 2]
                                                      for i, c in enumerate(cov)])
        worst = max(worst, sum(th[i, 0] / t[i] + th[i, 1] * t[i] for i in range(2)))
    assert worst <= r["certified_error"] + 1e-9
    assert worst == pytest.approx(r["exact_worst_case"], rel=2e-3)


def test_dominance_never_worse_inside_hull():
    lo, up, edges = np.array([1., 1.]), np.array([4., 4.]), ((0, 1),)
    spec = RobustSpec.independent(np.array([2., 1.]), np.array([.5, .5]), np.zeros((2, 2, 2)))
    base = np.array([2., 2.])
    laws = [[(np.array([2., 1.]), -1), (np.array([.5, .5]), 1)],
            [(np.array([4., 2.]), -2), (np.array([.5, .5]), 1)]]
    r = solve_certified_robust(lo, up, edges, spec, 0., deadline=4., dominance_laws=laws, baseline=base)
    for w in np.linspace(0, 1, 11):
        mix = [(w * c, e) for c, e in laws[0]] + [((1 - w) * c, e) for c, e in laws[1]]
        assert law_value(r["durations"], mix) <= law_value(base, mix) * (1 + 1e-6)


def test_coverage_radius_matches_chi_square():
    assert coverage_radius(0.05, 2) == pytest.approx(math.sqrt(-2 * math.log(0.05)), rel=1e-9)


# C2 ---------------------------------------------------------------------------

def forced_window_problem(model):
    """Qubit 0 is entangled, then waits 10 us for a block on qubit 1."""
    gates = (gate("cx", 0, 1), gate("block", 1), gate("cx", 0, 1))
    base = np.array([0.08, 10.0, 0.08])
    quiet = IdleModel(0., 0., 0., 0.)
    return DDProblem(2, gates, base, base.copy(), base.copy(), np.zeros(3), np.zeros(3), (model, quiet), 0.001)


def test_dd_relaxation_matches_closed_form_pulse_count():
    model = IdleModel(amplitude=1e-3, beta=1.0, markov=0., pulse_error=1e-4)
    problem = forced_window_problem(model)
    _, _, deadline = asap(problem)
    result = solve_dd_allocation(problem, deadline, optimize_durations=False)
    w = 10.
    # minimise c w^2 / (n+1) + eps n  ->  n + 1 = sqrt(c w^2 / eps)
    expected = math.sqrt(model.amplitude * w ** 2 / model.pulse_error) - 1
    window = [k for k, (q, a, b) in enumerate(problem.windows()) if q == 0 and b is not None][0]
    assert result["relaxation_value"] <= result["integer_value"] + 1e-12
    assert abs(result["pulses"][window] - expected) <= 2.0  # even rounding
    assert result["relative_gap"] < 0.01


def test_dd_zero_pulses_for_white_noise():
    model = IdleModel(amplitude=0., beta=0., markov=1e-3, pulse_error=1e-4)
    problem = forced_window_problem(model)
    _, _, deadline = asap(problem)
    result = solve_dd_allocation(problem, deadline, optimize_durations=False)
    assert result["pulses"].sum() == 0


def test_state_aware_relaxation_places_dd_only_on_excited_idle():
    """With pure T1, DD halves the loss of a qubit idling in |1> and doubles nothing in |0>."""
    t1 = IdleModel(amplitude=0., beta=0., markov=0., pulse_error=1e-6, relaxation=1e-2)
    quiet = IdleModel(0., 0., 0., 0.)
    # Qubit 0 waits in |1>, qubit 2 waits in |0>, both for a 10 us block on qubit 1.
    gates = (gate("x", 0), gate("cz", 0, 1), gate("cz", 2, 1), gate("block", 1), gate("cz", 0, 1), gate("cz", 2, 1))
    base = np.array([.036, .084, .084, 10., .084, .084])
    problem = DDProblem(3, gates, base, base.copy(), base.copy(), np.zeros(6), np.zeros(6), (t1, quiet, t1), .036)
    populations = dict(zip(problem.windows(), problem.excited_populations()))
    assert populations[(0, 1, 4)] == pytest.approx(1.)
    assert populations[(2, 2, 5)] == pytest.approx(0.)
    _, _, deadline = asap(problem)
    result = solve_dd_allocation(problem, deadline, optimize_durations=False)
    counts = dict(zip(problem.windows(), result["pulses"]))
    assert counts[(0, 1, 4)] >= 2
    assert counts[(2, 2, 5)] == 0


def test_simulator_matches_quasi_static_ramsey():
    physics = QubitPhysics(t1=1e12, white_dephasing=0., quasi_static=0.5, spectral_s0=0., pulse_error=0.)
    rows = spectroscopy(physics, [2.0], [0, 2], trajectories=20000, seed=3)
    assert rows[0]["contrast"] == pytest.approx(math.exp(-0.5 * 0.25 * 2.05 ** 2), abs=0.01)
    assert rows[1]["contrast"] > 0.99


def test_simulator_noiseless_identity():
    quiet = QubitPhysics(t1=1e12, white_dephasing=0., quasi_static=0., spectral_s0=0., pulse_error=0.)
    gates = [gate("h", 0), gate("cx", 0, 1), gate("rz", 1, theta=0.7), gate("sx", 1)]
    sched = ScheduledCircuit(2, gates, {0: 0., 1: .1, 3: 1.}, {0: .05, 1: .08, 3: .05}, 1.2,
                             {0: [(.5, "X"), (.9, "X")]})
    f = simulate(sched, [quiet, quiet], GateLaw(np.zeros(3), np.zeros(3)), trajectories=3)
    assert np.allclose(f, 1.)


# C3 ---------------------------------------------------------------------------

def test_runtime_closed_form_on_two_path_dag():
    from scipy.optimize import minimize
    a = np.array([2.0, 0.5, 1.0])
    g = np.array([0.02, 0.02, 0.03])
    paths = [[0, 2], [1, 2]]

    def objective(y):
        t = np.exp(y)
        return 4 * np.sum(a / t + g * t) + np.log(max(t[p].sum() for p in paths))

    r = minimize(objective, np.ones(3), method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-12, maxiter=20000))
    t = np.exp(r.x)
    M = max(t[p].sum() for p in paths)
    p = np.array([1., 0., 1.])
    assert np.allclose(np.sqrt(a / (g + p / (4 * M))), t, rtol=1e-4)


# C5 ---------------------------------------------------------------------------

def test_isotropic_noise_gives_basis_independent_single_qubit_decay():
    ia, ig = isotropic_split(3e-3, 1e-3, 1)
    # Each single-qubit Pauli anticommutes with exactly two of {X, Y, Z}.
    for observable in range(3):
        anti = np.array([i != observable for i in range(3)], dtype=float)
        assert anti @ ia == pytest.approx(2 * 1e-3)
        assert anti @ ig == pytest.approx(2 / 3 * 1e-3)


# C6 / C7 / C8 -----------------------------------------------------------------

def test_bicriteria_bounds_on_random_instances():
    rng = np.random.default_rng(1)
    for _ in range(20):
        m = 4
        edges = tuple((i, j) for j in range(m) for i in range(j) if rng.random() < .4)
        lower = rng.uniform(.5, 1.5, m)
        upper = 4 * lower
        inv, lin = rng.uniform(.1, 2., m), rng.uniform(.01, .5, m)
        library = [np.concatenate([[lower[j]], rng.uniform(lower[j], upper[j], 3)]) for j in range(m)]
        from qcgp.core import critical_path
        deadline = 1.5 * critical_path(lower, edges).makespan
        q = 1.1
        opt, _ = library_optimum(library, inv, lin, edges, deadline)
        r = bicriteria(lower, upper, inv, lin, edges, deadline, q)
        assert r["makespan"] <= q * deadline * (1 + 1e-7)
        assert r["error"] <= q * opt * (1 + 1e-7)


def test_frontier_is_log_log_convex_with_dual_slope():
    lower = np.array([1., 1., 1.])
    upper = np.array([5., 5., 5.])
    inv, lin = np.array([2., 1., 3.]), np.array([.1, .2, .1])
    edges = ((0, 2), (1, 2))
    res = pareto_frontier(lower, upper, inv, lin, edges, np.geomspace(2.05, 9., 25))
    assert res["min_second_difference"] > -1e-6
    slope, price = np.array(res["finite_difference_slope"]), np.array(res["dual_price"])
    assert np.allclose(slope[2:-2], -price[2:-2], atol=0.02)


def test_fast_pathflow_matches_interior_point():
    from qcgp.timing import TimingModel, optimize_timing
    arity, edges = random_circuit_dag(8, 6, seed=2)
    base = np.where(arity == 1, .036, .084)
    a, c = 1e-4 * base * 3, 1e-3 / base
    model = TimingModel(base, .5 * base, 3 * base, a, c, np.ones(len(base)), 1e-3, tuple((int(i), int(j)) for i, j in edges))
    gp = optimize_timing(model, (1, 0, 0), references=np.ones(3))
    fast = fast_pathflow(.5 * base, 3 * base, a, c, 1e-3, edges, tolerance=1e-8)
    assert fast["primal"] == pytest.approx(gp.objective, rel=1e-5)


def test_hardware_circuit_reproduces_schedule_timeline():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qcgp.hardware import schedule_to_circuit
    model = IdleModel(2e-4, 1.0, 1e-3, 3e-4)
    gates = (gate("h", 0), gate("cx", 0, 1), gate("cx", 1, 2), gate("block", 2), gate("cx", 1, 2),
             gate("cx", 0, 1), gate("h", 0))
    base = np.array([.036, .084, .084, 20., .084, .084, .036])
    problem = DDProblem(3, gates, base, base.copy(), base.copy(), np.zeros(7), np.zeros(7), (model,) * 3, .036)
    _, _, deadline = asap(problem)
    r = solve_dd_allocation(problem, deadline, optimize_durations=False)
    assert r["pulses"].sum() > 0
    c = schedule_to_circuit(problem, r["durations"], r["starts"], r["makespan"], r["pulses"], measure=False)
    duration = {"h": .036, "x": .036, "cx": .084}
    for q in range(3):
        total = 0.
        for ins in c.data:
            if q in [c.find_bit(b).index for b in ins.qubits]:
                op = ins.operation
                total += float(op.duration) if op.name == "delay" else duration[op.name]
        # Leading delays pad every qubit from t = 0, so each timeline spans the makespan.
        assert total == pytest.approx(r["makespan"], abs=1e-6)
    ideal = QuantumCircuit.from_instructions([i for i in c.data if i.operation.name != "delay"], qubits=c.qubits)
    assert Operator(ideal).equiv(Operator(QuantumCircuit(3)))


def test_correlated_protection_counts_cross_window_accumulation():
    from qcgp.ddgp import correlated_protection
    # Qubit 0 has two 5 us windows (waiting on two blocks on qubit 1).
    gates = (gate("h", 0), gate("cz", 0, 1), gate("block", 1), gate("cz", 0, 1), gate("block", 1),
             gate("cz", 0, 1), gate("h", 0))
    base = np.array([.036, .084, 5., .084, 5., .084, .036])
    quiet = IdleModel(0., 0., 0., 0.)

    def solve(eps):
        model = IdleModel(amplitude=1e-3, beta=1.0, markov=0., pulse_error=eps)
        problem = DDProblem(2, gates, base, base.copy(), base.copy(), np.zeros(7), np.zeros(7), (model, quiet), .036)
        d, s, M = asap(problem)
        counts = correlated_protection(problem, d, s, M)
        return {k: c for k, c in zip(problem.windows(), counts) if k[0] == 0}

    cheap = solve(1e-5)
    assert all(cheap[k] >= 2 for k in [(0, 1, 3), (0, 3, 5)])
    # Per window, protecting saves c w^2 (1 - 1/(n+1)) = 0.025 * 2/3 at n = 2; joint coupling saves more.
    expensive = solve(0.05)
    assert sum(expensive.values()) == 0
