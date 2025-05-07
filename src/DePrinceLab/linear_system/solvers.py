from DePrinceLab.linear_system.utils import LinearSystem
import numpy as np
from numpy.typing import NDArray
from scipy import linalg


def brute_force(ls: LinearSystem) -> NDArray:
    solution = linalg.inv(ls.matrix).dot(ls.rhs)

    if ls.solution is None:
        ls.solution = solution
    else:
        assert np.allclose(ls.solution, solution), "brute force must match"

    return solution


def recommended(ls: LinearSystem) -> NDArray:
    solution = linalg.solve(ls.matrix, ls.rhs)

    if ls.solution is None:
        ls.solution = solution
    else:
        assert np.allclose(ls.solution, solution), "solve must match"

    return solution


def jacobi(ls: LinearSystem) -> NDArray:
    MAXITER = 100
    THRESHOLD = 1e-6
    matrix = ls.matrix
    rhs = ls.rhs

    diagonal_elements = matrix.diagonal()
    diagonal = np.diag(diagonal_elements)
    remider = matrix - diagonal

    inv_diagonal = np.diag(1.0/diagonal_elements)
    jacobi = inv_diagonal @ (-remider)

    solution = rhs.copy()
    for _ in range(MAXITER):
        old_solution = solution.copy()
        solution = jacobi @ solution + inv_diagonal @ rhs

        curr_norm = np.linalg.norm(solution, ord=np.inf)
        diff_norm = np.linalg.norm(solution - old_solution, ord=np.inf)
        if diff_norm / curr_norm < THRESHOLD:
            break
    else:
        raise RuntimeError("Jacobi method didn't converge")


    if ls.solution is None:
        ls.solution = solution
    else:
        assert np.allclose(ls.solution, solution), "Jacobi must match"

    print(f'Jacobi {solution=}')
    return solution
