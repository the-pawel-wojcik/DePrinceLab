from typing import Any
from DePrinceLab.linear_system.utils import LinearSystem
import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from scipy.sparse.linalg import LinearOperator, gmres


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

    return solution


def gmres_unclever(ls: LinearSystem) -> NDArray:
    matrix = ls.matrix
    rhs = ls.rhs
    # generalized minimal residual
    solution, exit_code = gmres(
        matrix,
        rhs,
        rtol=1e-12,
    )
    if exit_code != 0:
        raise RuntimeError("GMRES didn't converge")

    if ls.solution is None:
        ls.solution = solution
    else:
        assert np.allclose(ls.solution, solution), "gmres unclever must match"

    return solution


class RawMatTimesVec(LinearOperator):
    """ `gmres_the_way` helper. """

    def __init__(self: Any, matrix: NDArray) -> None:
        """ The point is to NOT store the whole matrix.
        But here for testing, I will still keep it. """
        self.matrix = matrix
        self.shape = matrix.shape
        self.dtype = matrix.dtype

    def _matvec(self, x: NDArray) :
        """ This is where the implementation of mat@vec should go. """
        return self.matrix @ x.reshape(-1, 1)


def gmres_the_way(ls: LinearSystem) -> NDArray:
    matrix = ls.matrix
    rhs = ls.rhs

    solution, exit_code = gmres(
        RawMatTimesVec(matrix),
        rhs,
        rtol=1e-12,
    )
    if exit_code != 0:
        raise RuntimeError("GMRES didn't converge")

    if ls.solution is None:
        ls.solution = solution
    else:
        assert np.allclose(ls.solution, solution), "`gmres_the_way` must match"

    return solution
