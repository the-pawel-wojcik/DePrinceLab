from scipy.sparse.linalg import gmres
from DePrinceLab.linear_system.test_0 import build_atomic_energy

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html

def main():
    problem = build_atomic_energy()
    matrix = problem.matrix
    rhs = problem.rhs

    # generalized minimal residual
    solution, exit_code = gmres(
        matrix,
        rhs,
        maxiter=100,
    )
    if exit_code != 0:
        raise RuntimeError("GMRES didn't converge")

    print(f'{solution=}')


if __name__ == "__main__":
    main()
