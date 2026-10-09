"""Solve a declared inverse-square duration-allocation example locally."""
import numpy as np

from qcgp.generalized import GeneralizedProblem, solve_generalized


problem = GeneralizedProblem(
    lower=np.array([1., 1., 2.]),
    upper=np.array([4., 4., 5.]),
    inverse=np.array([8., 6., 12.]),
    linear=np.array([1., 1., 1.]),
    edges=((0, 2), (1, 2)),
    makespan_weight=1.,
    inverse_power=2.,
)
print(solve_generalized(problem))
