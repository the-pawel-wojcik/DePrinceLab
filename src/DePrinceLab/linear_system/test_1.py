import numpy as np
from DePrinceLab.linear_system.utils import LinearSystem
from DePrinceLab.linear_system.solvers import (
    brute_force, recommended, jacobi
)
from numpy.typing import NDArray


def add_random_noise(
    matrix: NDArray,
    scale: float,
    distribution: str ='uniform',
):
    """
    Adds random noise to a matrix.

    Parameters:
    - matrix: The input matrix.
    - scale: The scale of the random noise.
    - distribution: The distribution of the random noise. Can be 'uniform' or
      'normal'

    Returns:
    - The matrix with added random noise.
    """
    if distribution == 'uniform':
        random_values = np.random.uniform(-scale, scale, matrix.shape)
    elif distribution == 'normal':
        random_values = np.random.normal(0, scale, matrix.shape)
    else:
        raise ValueError("Distribution must be 'uniform' or 'normal'.")

    return matrix + random_values


def build_diagonal_dominant() -> LinearSystem:
    matrix = np.diag([5, 11, 13, 9])
    matrix = add_random_noise(matrix, 1e-3)
    rhs = np.array([1, 2, -2, 0])
    return LinearSystem(matrix=matrix, rhs=rhs)


def test_diagonal_dominant():
    problem = build_diagonal_dominant()

    _ = brute_force(problem)
    _ = recommended(problem)
    _ = jacobi(problem)


def main():
    test_diagonal_dominant()


if __name__ == "__main__":
    main()
