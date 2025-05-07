import numpy as np
from DePrinceLab.linear_system.utils import LinearSystem
from DePrinceLab.linear_system.solvers import brute_force, recommended


def build_atomic_energy() -> LinearSystem:
    # Solution of this linear systems gives the expression for the atomic
    # energy
    matrix = np.array([
        [1, 0, 1, 1],
        [0, 0, 2, 3],
        [0, 1, -1, -4],
        [0, 1, 0, -2],
    ])
    rhs = np.array([1, 2, -2, 0])
    solution = np.array([1, 4, -2, 2])
    return LinearSystem(matrix=matrix, rhs=rhs, solution=solution)


def test_atomic_energy():
    problem = build_atomic_energy()

    _ = brute_force(problem)
    _ = recommended(problem)

