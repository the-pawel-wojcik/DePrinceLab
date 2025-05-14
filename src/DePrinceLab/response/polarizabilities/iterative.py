from DePrinceLab.response.intermediates_builders import (
    Intermediates,
)
from DePrinceLab.response.hf_orbital_hessian_builders import (
    build_complete_matrix,
)
from DePrinceLab.response.operators import MockOrbitalHessianAction
from DePrinceLab.response.polarizabilities.core import gmres_solve
import numpy as np


def calculate(intermediates: Intermediates):
    """
    Solve the equation
    `orbital_hessian @ response = dipole_moment`
    for the `response` using the GMRES iterative algorithm.
    """

    orbital_hessian = build_complete_matrix(intermediates)
    orbital_hessian_action = MockOrbitalHessianAction(orbital_hessian)

    # combine spin dipole integrals into a single vector the length of hinv
    mua_x = intermediates.mua_x
    mua_y = intermediates.mua_y
    mua_z = intermediates.mua_z
    mub_x = intermediates.mub_x
    mub_y = intermediates.mub_y
    mub_z = intermediates.mub_z
    oa = intermediates.oa
    va = intermediates.va
    ob = intermediates.ob
    vb = intermediates.vb
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
