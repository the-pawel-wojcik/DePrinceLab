from DePrinceLab.response.operators import (
    MatmulLike,
    MockOrbitalHessianAction,
    OrbitalHessianAction,
)
from DePrinceLab.response.electronic_structure import scf
from DePrinceLab.response.hf_orbital_hessian_builders import (
    buld_h_aa, build_h_ab, build_h_ba, build_h_bb, build_complete_matrix,
)
from DePrinceLab.response.intermediates_builders import (
    extract_intermediates,
)
import numpy as np
from numpy import einsum
from numpy.typing import NDArray
from scipy.sparse.linalg import gmres


def find_polarizabilities_directly(
    h_aa, h_ab, h_ba, h_bb,
    nmo, noa, nob,
    mua_x, mua_y, mua_z,
    mub_x, mub_y, mub_z,
    oa, va, ob, vb,
    **_,
):
    h_complete = build_complete_matrix(h_aa, h_ab, h_ba, h_bb, nmo, noa, nob)
    hinv = np.linalg.inv(h_complete)
    # combine spin dipole integrals into a single vector the length of hinv
    mu_x_vec = np.hstack((mua_x[oa, va].flatten(), mub_x[ob, vb].flatten()))
    mu_y_vec = np.hstack((mua_y[oa, va].flatten(), mub_y[ob, vb].flatten()))
    mu_z_vec = np.hstack((mua_z[oa, va].flatten(), mub_z[ob, vb].flatten()))

    # response vectors
    kappa_x = 2 * einsum('pq,q->p', hinv, mu_x_vec)
    kappa_y = 2 * einsum('pq,q->p', hinv, mu_y_vec)
    kappa_z = 2 * einsum('pq,q->p', hinv, mu_z_vec)

    polarizabilities = {
        "xx":  2 * einsum('p,p->', mu_x_vec, kappa_x),
        "xy":  2 * einsum('p,p->', mu_x_vec, kappa_y),
        "xz":  2 * einsum('p,p->', mu_x_vec, kappa_z),
        "yy":  2 * einsum('p,p->', mu_y_vec, kappa_y),
        "yz":  2 * einsum('p,p->', mu_y_vec, kappa_z),
        "zz":  2 * einsum('p,p->', mu_z_vec, kappa_z),
    }

    return polarizabilities


def gmres_solve(matrix: NDArray | MatmulLike, rhs: NDArray) -> NDArray:
    """
    solves `matrix @ solution = rhs`
    """
    solution, exit_code = gmres(matrix, rhs, rtol=1e-12)
    if exit_code != 0:
        raise RuntimeError("GMRES didn't converge")
    return solution


def find_polarizabilities_iteratively_no_storage(
    orbital_hessian_action: MatmulLike,
    mua_x: NDArray, mua_y: NDArray, mua_z: NDArray,
    mub_x: NDArray, mub_y: NDArray, mub_z: NDArray,
    oa, va, ob, vb,
    **_,
):
    """
    Solve the equation
    `orbital_hessian @ response = dipole_moment`
    for the `response` using the GMRES iterative algorithm.

    Use the GMRES version which does not store the orbital_hessian matrix,
    but instead uses the operator that calculates the `orbital_hessian@vector`
    value.
    """
    # combine spin dipole integrals into a single vector the length of hinv
    dipole_moment = {
        "x": np.hstack((mua_x[oa, va].flatten(), mub_x[ob, vb].flatten())),
        "y": np.hstack((mua_y[oa, va].flatten(), mub_y[ob, vb].flatten())),
        "z": np.hstack((mua_z[oa, va].flatten(), mub_z[ob, vb].flatten())),
    }

    response = {
        xyz: 2 * gmres_solve(orbital_hessian_action, dipole_moment[xyz])
        for xyz in ["x", "y", "z"]
    }

    polarizabilities = {
        "xx":  2 * dipole_moment['x'] @ response['x'],
        "xy":  2 * dipole_moment['x'] @ response['y'],
        "xz":  2 * dipole_moment['x'] @ response['z'],
        "yy":  2 * dipole_moment['y'] @ response['y'],
        "yz":  2 * dipole_moment['y'] @ response['z'],
        "zz":  2 * dipole_moment['z'] @ response['z'],
    }

    return polarizabilities


def find_polarizabilities_iteratively(
    h_aa, h_ab, h_ba, h_bb,
    nmo, noa, nob,
    mua_x, mua_y, mua_z,
    mub_x, mub_y, mub_z,
    oa, va, ob, vb,
    **_,
):
    """
    Solve the equation
    `orbital_hessian @ response = dipole_moment`
    for the `response` using the GMRES iterative algorithm.
    """
    orbital_hessian = build_complete_matrix(
        h_aa, h_ab, h_ba, h_bb, nmo, noa, nob
    )

    orbital_hessian_action = MockOrbitalHessianAction(orbital_hessian)

    # combine spin dipole integrals into a single vector the length of hinv
    dipole_moment = {
        "x": np.hstack((mua_x[oa, va].flatten(), mub_x[ob, vb].flatten())),
        "y": np.hstack((mua_y[oa, va].flatten(), mub_y[ob, vb].flatten())),
        "z": np.hstack((mua_z[oa, va].flatten(), mub_z[ob, vb].flatten())),
    }

    response = {
        xyz: 2 * gmres_solve(orbital_hessian_action, dipole_moment[xyz])
        for xyz in ["x", "y", "z"]
    }

    polarizabilities = {
        "xx":  2 * dipole_moment['x'] @ response['x'],
        "xy":  2 * dipole_moment['x'] @ response['y'],
        "xz":  2 * dipole_moment['x'] @ response['z'],
        "yy":  2 * dipole_moment['y'] @ response['y'],
        "yz":  2 * dipole_moment['y'] @ response['z'],
        "zz":  2 * dipole_moment['z'] @ response['z'],
    }

    return polarizabilities


def print_polarizabilities(pol):
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


def main():
    _, wfn = scf()
    intermediates = extract_intermediates(wfn)
    h_aa = buld_h_aa(**intermediates)
    h_ab = build_h_ab(**intermediates)
    h_ba = build_h_ba(**intermediates)
    h_bb = build_h_bb(**intermediates)

    print("Polarizabilities:")

    do_direct = True
    if do_direct:
        pol_direct = find_polarizabilities_directly(
            h_aa, h_ab, h_ba, h_bb, **intermediates
        )

        print("1) from inverted HF orbitals Hessian:")
        print_polarizabilities(pol_direct)

    do_iterative = True
    if do_iterative:
        pol_iterative = find_polarizabilities_iteratively(
            h_aa, h_ab, h_ba, h_bb, **intermediates
        )

        print("2) from GMRES iterative solution:")
        print_polarizabilities(pol_iterative)

    do_action = True
    if do_action:
        orbital_hessian_action = OrbitalHessianAction(
            fock_aa=intermediates['f_aa'],
            fock_bb=intermediates['f_bb'],
            **intermediates,
        )
        pol_iterative = find_polarizabilities_iteratively_no_storage(
            orbital_hessian_action, **intermediates
        )
        print("3) from GMRES iterative solution (no hessian storage):")
        print_polarizabilities(pol_iterative)


if __name__ == "__main__":
    main()
