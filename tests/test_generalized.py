import numpy as np
import pytest

from qcgp.generalized import (GeneralizedProblem, dual_duration_response,
                              solve_generalized, solve_robust_generalized)


@pytest.mark.parametrize("power,inverse,expected", [(1., 4., np.sqrt(2.)), (2., 8., 2.)])
def test_single_gate_matches_analytic_stationary_point(power, inverse, expected):
    problem = GeneralizedProblem(np.array([1.]), np.array([4.]),
                                 np.array([inverse]), np.array([1.]), (), 1., power)
    result = solve_generalized(problem)
    assert result["durations"][0] == pytest.approx(expected, rel=1e-4)
    response = dual_duration_response(np.array([inverse]), np.array([1.]),
                                      np.array([1.]), np.array([1.]), np.array([4.]), power)
    assert response[0] == pytest.approx(expected)


def test_generalized_precedence_and_bounds():
    problem = GeneralizedProblem(np.array([1., 2.]), np.array([4., 4.]),
                                 np.array([8., 8.]), np.array([1., 1.]),
                                 ((0, 1),), 1., 2.)
    result = solve_generalized(problem)
    assert result["makespan"] == pytest.approx(sum(result["durations"]), rel=1e-8)
    assert np.all(np.array(result["durations"]) >= problem.lower)


def test_generalized_rejects_invalid_power():
    problem = GeneralizedProblem(np.array([1.]), np.array([4.]),
                                 np.array([8.]), np.array([1.]), (), 1., 0.)
    with pytest.raises(ValueError):
        solve_generalized(problem)


def test_zero_noise_coefficients_reduce_to_earliest_schedule():
    problem = GeneralizedProblem(np.array([1., 1.]), np.array([4., 4.]),
                                 np.zeros(2), np.zeros(2), ((0, 1),), 1., 2.)
    result = solve_generalized(problem)
    assert result["makespan"] == pytest.approx(2., abs=1e-5)


def test_deadline_is_enforced():
    problem = GeneralizedProblem(np.array([1., 1.]), np.array([4., 4.]),
                                 np.array([8., 8.]), np.zeros(2), ((0, 1),),
                                 1e-8, 2., deadline=3.)
    result = solve_generalized(problem)
    assert result["makespan"] <= 3. + 1e-5
    with pytest.raises(ValueError):
        solve_generalized(GeneralizedProblem(np.array([1., 1.]), np.array([4., 4.]),
                                             np.array([8., 8.]), np.zeros(2),
                                             ((0, 1),), 1., 2., deadline=1.))


def test_one_scenario_robust_gp_matches_ordinary_gp():
    problem = GeneralizedProblem(np.array([1., 1.]), np.array([4., 4.]),
                                 np.array([8., 2.]), np.array([1., 3.]),
                                 ((0, 1),), .2, 2., deadline=5.)
    ordinary = solve_generalized(problem)
    robust = solve_robust_generalized(problem, problem.inverse[None, :],
                                      problem.linear[None, :])
    assert robust["durations"] == pytest.approx(ordinary["durations"], rel=2e-4)
    assert robust["objective"] == pytest.approx(ordinary["objective"], rel=1e-5)


def test_robust_gp_reports_actual_worst_scenario_and_deadline():
    problem = GeneralizedProblem(np.array([1., 1.]), np.array([4., 4.]),
                                 np.array([1., 1.]), np.array([1., 1.]),
                                 ((0, 1),), .01, 2., deadline=5.)
    inverse = np.array([[8., 1.], [1., 8.]])
    linear = np.array([[.5, 2.], [2., .5]])
    result = solve_robust_generalized(problem, inverse, linear)
    duration = np.asarray(result["durations"])
    independent = np.sum(inverse / duration[None, :] ** 2
                         + linear * duration[None, :], axis=1)
    assert result["makespan"] <= 5. + 1e-5
    assert result["scenario_errors"] == pytest.approx(independent)
    assert result["worst_error"] == pytest.approx(max(independent))


def test_robust_gp_rejects_invalid_uncertainty_scenarios():
    problem = GeneralizedProblem(np.array([1.]), np.array([4.]),
                                 np.array([1.]), np.array([1.]), (), .01, 2.)
    with pytest.raises(ValueError, match="scenario"):
        solve_robust_generalized(problem, np.array([[-1.]]), np.array([[1.]]))
