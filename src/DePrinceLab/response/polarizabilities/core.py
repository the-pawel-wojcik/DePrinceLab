from DePrinceLab.response.operators import (
    MatmulLike,
)
from numpy.typing import NDArray
from scipy.sparse.linalg import gmres


def gmres_solve(matrix: NDArray | MatmulLike, rhs: NDArray) -> NDArray:
    """
    solves `matrix @ solution = rhs`
    """
    solution, exit_code = gmres(matrix, rhs, rtol=1e-12)
    if exit_code != 0:
        raise RuntimeError("GMRES didn't converge")
    return solution


def pretty_print(pol):
    header = '==> Static Dipole Polarizability <=='
    width = len(header)
    print()
    print(header.center(width))
    print()

    pad = ' ' * 2
    fmt = '9.4f'
    for key, val in pol.items():
        print(f'{pad}{key} = {val:{fmt}}'.center(width))
    print()
