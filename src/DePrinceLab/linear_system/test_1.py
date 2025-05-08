import numpy as np
from DePrinceLab.linear_system.utils import LinearSystem
from DePrinceLab.linear_system.solvers import (
    brute_force, recommended, jacobi, gmres_unclever, gmres_the_way
)
from numpy.typing import NDArray


def add_random_noise(
    matrix: NDArray,
    scale: float,
    distribution: str ='uniform',
    seed: int | None = None,
):
    """
    Adds random noise to a matrix.

    Parameters:
    - matrix: The input matrix.
    - scale: The scale of the random noise.
    - distribution: The distribution of the random noise. Can be 'uniform' or
      'normal'
    - seed (int): Optional random seed for reproducibility.

    Returns:
    - The matrix with added random noise.
    """
    if seed is not None:
        rng = np.random.default_rng(seed=seed)
    else:
        rng =  np.random.default_rng()

    if distribution == 'uniform':
        random_values = rng.uniform(low=-scale, high=scale, size=matrix.shape)
    elif distribution == 'normal':
        random_values = rng.normal(loc=0, scale=scale, size=matrix.shape)
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
    _ = gmres_unclever(problem)
    _ = gmres_the_way(problem)


if __name__ == "__main__":
    test_diagonal_dominant()
