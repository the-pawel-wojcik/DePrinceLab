from typing import Any
import psi4
import numpy as np
from numpy import einsum
from numpy.typing import NDArray

def extract_intermediates(wfn):
    Ca = wfn.Ca()
    Cb = wfn.Cb()

    # use Psi4's MintsHelper to generate ERIs and dipole integrals
    mints = psi4.core.MintsHelper(wfn.basisset())

    # build the integrals in chemists' notation
    g_aaaa = np.asarray(mints.mo_eri(Ca, Ca, Ca, Ca))
    g_aabb = np.asarray(mints.mo_eri(Ca, Ca, Cb, Cb))
    g_bbbb = np.asarray(mints.mo_eri(Cb, Cb, Cb, Cb))

    # antisymmetrized integrals in physicists' notation
    g_aaaa = g_aaaa.transpose(0, 2, 1, 3) - g_aaaa.transpose(0, 2, 3, 1)
    g_bbbb = g_bbbb.transpose(0, 2, 1, 3) - g_bbbb.transpose(0, 2, 3, 1)
    g_abab = g_aabb.transpose(0, 2, 1, 3)

    noa = wfn.nalpha()
    nob = wfn.nbeta()

    oa = slice(None, noa)
    ob = slice(None, nob)
    va = slice(noa, None)
    vb = slice(nob, None)

    nmo = wfn.nmo()
    kd_aa = np.eye(nmo)
    kd_bb = np.eye(nmo)

    # orbital energies
    f_aa = np.diag(wfn.epsilon_a())
    f_bb = np.diag(wfn.epsilon_b())

    # dipole integrals
    mu = mints.ao_dipole()

    mu_x = np.asarray(mu[0])
    mu_y = np.asarray(mu[1])
    mu_z = np.asarray(mu[2])

    # transform the dipole integrals to the MO basis
    Ca = np.asarray(Ca)
    Cb = np.asarray(Cb)

    mua_x = Ca.T @ mu_x @ Ca
    mua_y = Ca.T @ mu_y @ Ca
    mua_z = Ca.T @ mu_z @ Ca

    mub_x = Cb.T @ mu_x @ Cb
    mub_y = Cb.T @ mu_y @ Cb
    mub_z = Cb.T @ mu_z @ Cb

    return {
        'mua_x': mua_x,
        'mua_y': mua_y,
        'mua_z': mua_z,
        'mub_x': mub_x,
        'mub_y': mub_y,
        'mub_z': mub_z,
        'kd_aa': kd_aa,
        'kd_bb': kd_bb,
        'f_aa': f_aa,
        'f_bb': f_bb,
        'va': va,
        'vb': vb,
        'oa': oa,
        'ob': ob,
        'nmo': nmo,
        'noa': noa,
        'nob': nob,
        'g_aaaa': g_aaaa,
        'g_abab': g_abab,
        'g_bbbb': g_bbbb,
    }


def buld_h_aa(kd_aa, f_aa, va, oa, g_aaaa, **_) -> NDArray:
    h_aa = -1.00 * einsum('ab,ji->jbia', kd_aa[va, va], f_aa[oa, oa])
    h_aa +=  1.00 * einsum('ij,ab->jbia', kd_aa[oa, oa], f_aa[va, va])
    h_aa += -1.00 * einsum('ba,ij->jbia', kd_aa[va, va], f_aa[oa, oa])
    h_aa +=  1.00 * einsum('ji,ba->jbia', kd_aa[oa, oa], f_aa[va, va])
    h_aa += -1.00 * einsum('jiab->jbia', g_aaaa[oa, oa, va, va])
    h_aa +=  1.00 * einsum('jabi->jbia', g_aaaa[oa, va, va, oa])
    h_aa +=  1.00 * einsum('ibaj->jbia', g_aaaa[oa, va, va, oa])
    h_aa += -1.00 * einsum('abji->jbia', g_aaaa[va, va, oa, oa])

    return h_aa


def build_h_ab(g_abab, oa, ob, va, vb, **_) -> NDArray:
    h_ab =  1.00 * einsum('jiba->jbia', g_abab[oa, ob, va, vb])
    h_ab +=  1.00 * einsum('jabi->jbia', g_abab[oa, vb, va, ob])
    h_ab +=  1.00 * einsum('bija->jbia', g_abab[va, ob, oa, vb])
    h_ab +=  1.00 * einsum('baji->jbia', g_abab[va, vb, oa, ob])
    return h_ab


def build_h_ba(g_abab, oa, ob, va, vb, **_) -> NDArray:
    h_ba =  1.00 * einsum('ijab->jbia', g_abab[oa, ob, va, vb])
    h_ba +=  1.00 * einsum('ajib->jbia', g_abab[va, ob, oa, vb])
    h_ba +=  1.00 * einsum('ibaj->jbia', g_abab[oa, vb, va, ob])
    h_ba +=  1.00 * einsum('abij->jbia', g_abab[va, vb, oa, ob])
    return h_ba


def build_h_bb(g_bbbb, kd_bb, f_bb, vb, ob, **_) -> NDArray:
    h_bb = -1.00 * einsum('ab,ji->jbia', kd_bb[vb, vb], f_bb[ob, ob])
    h_bb +=  1.00 * einsum('ij,ab->jbia', kd_bb[ob, ob], f_bb[vb, vb])
    h_bb += -1.00 * einsum('ba,ij->jbia', kd_bb[vb, vb], f_bb[ob, ob])
    h_bb +=  1.00 * einsum('ji,ba->jbia', kd_bb[ob, ob], f_bb[vb, vb])
    h_bb += -1.00 * einsum('jiab->jbia', g_bbbb[ob, ob, vb, vb])
    h_bb +=  1.00 * einsum('jabi->jbia', g_bbbb[ob, vb, vb, ob])
    h_bb +=  1.00 * einsum('ibaj->jbia', g_bbbb[ob, vb, vb, ob])
    h_bb += -1.00 * einsum('abji->jbia', g_bbbb[vb, vb, ob, ob])
    return h_bb


def scf() -> tuple[float, Any]:

    mol = psi4.geometry("""
    0 1
    O1	0.00000   0.00000   0.11572
    H2	0.00000   0.74879  -0.46288
    H3	0.00000  -0.74879  -0.46288
    symmetry c1
    """)   

    psi4.set_options({'basis': 'cc-pvdz',
                      'scf_type': 'pk',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-12})

    psi4.core.be_quiet()

    # compute the Hartree-Fock energy and wavefunction
    energy, wfn = psi4.energy('SCF', return_wfn=True)
    return energy, wfn


def invert(h_aa, h_ab, h_ba, h_bb, nmo, noa, nob, **_):
    # number of alpha- and beta-spin virtual orbitals
    nva = nmo - noa
    nvb = nmo - nob

    # reshape tensors
    h_aa = h_aa.reshape(noa*nva, noa*nva)
    h_ab = h_ab.reshape(noa*nva, nob*nvb)
    h_ba = h_ba.reshape(nob*nvb, noa*nva)
    h_bb = h_bb.reshape(nob*nvb, nob*nvb)

    # pack into super matrix
    h = np.block([[h_aa, h_ab], [h_ba, h_bb]])

    # invert super matrix
    hinv = np.linalg.inv(h)
    return hinv

def find_polarizabilities(
    hinv,
    mua_x, mua_y, mua_z,
    mub_x, mub_y, mub_z,
    oa, va, ob, vb,
    **_,
):
    # combine spin dipole integrals into a single vector the length of hinv
    mu_x_vec = np.hstack((mua_x[oa, va].flatten(), mub_x[ob, vb].flatten()))
    mu_y_vec = np.hstack((mua_y[oa, va].flatten(), mub_y[ob, vb].flatten()))
    mu_z_vec = np.hstack((mua_z[oa, va].flatten(), mub_z[ob, vb].flatten()))

    # response vectors
    kappa_x = 2 * einsum('pq,q->p', hinv, mu_x_vec) 
    kappa_y = 2 * einsum('pq,q->p', hinv, mu_y_vec)
    kappa_z = 2 * einsum('pq,q->p', hinv, mu_z_vec) 

    print('')
    print('    ==> Static Dipole Polarizability <==')
    print('')

    alpha_xx = 2 * einsum('p,p->', mu_x_vec, kappa_x)
    alpha_xy = 2 * einsum('p,p->', mu_x_vec, kappa_y)
    alpha_xz = 2 * einsum('p,p->', mu_x_vec, kappa_z)
    alpha_yy = 2 * einsum('p,p->', mu_y_vec, kappa_y)
    alpha_yz = 2 * einsum('p,p->', mu_y_vec, kappa_z)
    alpha_zz = 2 * einsum('p,p->', mu_z_vec, kappa_z)

    pad = ' ' * 2
    fmt = '9.4f'
    print(f'{pad}{alpha_xx=:{fmt}}')
    print(f'{pad}{alpha_xy=:{fmt}}')
    print(f'{pad}{alpha_xz=:{fmt}}')
    print(f'{pad}{alpha_yy=:{fmt}}')
    print(f'{pad}{alpha_yz=:{fmt}}')
    print(f'{pad}{alpha_zz=:{fmt}}')
    print('')


def main():
    _, wfn = scf()
    intermediates = extract_intermediates(wfn)
    h_aa = buld_h_aa(**intermediates)
    h_ab = build_h_ab(**intermediates)
    h_ba = build_h_ba(**intermediates)
    h_bb = build_h_bb(**intermediates)

    h_inv = invert(h_aa, h_ab, h_ba, h_bb, **intermediates)
    find_polarizabilities(h_inv, **intermediates)


if __name__ == "__main__":
    main()
