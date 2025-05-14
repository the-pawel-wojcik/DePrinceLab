from DePrinceLab.response.intermediates_builders import Intermediates
from DePrinceLab.response.hf_orbital_hessian_builders\
    import build_complete_matrix
import numpy as np
from numpy import einsum


def calculate(intermediates: Intermediates):
    """ Find polarizabilities by inverting the HF orbital Hessian. """

    # combine spin dipole integrals into a single vector the length of hinv
    mub_x = intermediates.mub_x
    mua_x = intermediates.mua_x
    mua_y = intermediates.mua_y
    mub_y = intermediates.mub_y
    mua_z = intermediates.mua_z
    mub_z = intermediates.mub_z
    oa = intermediates.oa
    va = intermediates. va
    ob = intermediates.ob
    vb = intermediates.vb
    mu_x_vec = np.hstack((mua_x[oa, va].flatten(), mub_x[ob, vb].flatten()))
    mu_y_vec = np.hstack((mua_y[oa, va].flatten(), mub_y[ob, vb].flatten()))
    mu_z_vec = np.hstack((mua_z[oa, va].flatten(), mub_z[ob, vb].flatten()))

    h_complete = build_complete_matrix(intermediates)
    hinv = np.linalg.inv(h_complete)
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
