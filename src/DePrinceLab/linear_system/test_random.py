import numpy as np
from numpy.random import Generator
from DePrinceLab.linear_system.utils import LinearSystem
from DePrinceLab.linear_system.solvers import (
    brute_force, recommended, gmres_unclever, gmres_the_way
)


def build_random_linear_system(
    n: int,
    seed: int = 20250508,
) -> LinearSystem:
    """
    Generates a linear system problem `matrix @ solution = rhs`.

    Args:
    - n (int): Size of the square matrix A.
    - seed (int): Optional random seed.

    Returns:
    - ls (LinearSystem): a dataclass with `matrix`, `rhs`, and `solution`
    """
    rng: Generator = np.random.default_rng(seed=seed)

    matrix = rng.random(size=(n,n))
    solution = rng.random(size=(n))
    rhs = np.dot(matrix, solution)

    return LinearSystem(matrix=matrix, solution=solution, rhs=rhs)


def test_random():
    problem = build_random_linear_system(15, seed=20250508)

    _ = brute_force(problem)
    _ = recommended(problem)
    _ = gmres_unclever(problem)
    _ = gmres_the_way(problem)
    # print("The problem:")
    # with np.printoptions(precision=2):
    #     print(problem)
    # print("Solved by all tested solvers!")


if __name__ == "__main__":
    test_random()
