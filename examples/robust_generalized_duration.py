"""Free local example: declared-scenario robust duration allocation.

The coefficients are synthetic and carry no statistical or device calibration.
"""
import numpy as np

from qcgp.generalized import (GeneralizedProblem, solve_generalized,
                              solve_robust_generalized)


def main():
    inverse = np.array([[8., 1.], [1., 8.]])
    linear = np.array([[.5, 2.], [2., .5]])
    problem = GeneralizedProblem(lower=np.array([1., 1.]),
                                 upper=np.array([4., 4.]),
                                 inverse=inverse[0], linear=linear[0],
                                 edges=((0, 1),), makespan_weight=.01,
                                 inverse_power=2., deadline=5.)
    nominal = solve_generalized(problem)
    robust = solve_robust_generalized(problem, inverse, linear)
    nominal_durations = np.asarray(nominal["durations"])
    nominal_scenario_errors = np.sum(
        inverse / nominal_durations[None, :] ** 2
        + linear * nominal_durations[None, :], axis=1)
    print("Nominal durations:", nominal["durations"])
    print("Nominal-policy errors by scenario:", nominal_scenario_errors.tolist())
    print("Robust durations:", robust["durations"])
    print("Robust-policy errors by scenario:", robust["scenario_errors"])
    print("All quantities are synthetic model outputs.")


if __name__ == "__main__":
    main()
