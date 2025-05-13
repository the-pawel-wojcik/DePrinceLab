from typing import Any, TypeVar
import psi4
from psi4.core import Wavefunction, Matrix
import numpy as np
from numpy import einsum
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, gmres


def extract_intermediates(wfn: Wavefunction):
    orbitals_up: Matrix = wfn.Ca()
    orbitals_down: Matrix = wfn.Cb()

    # use Psi4's MintsHelper to generate ERIs and dipole integrals
    mints = psi4.core.MintsHelper(wfn.basisset())

    # build the integrals in chemists' notation
    g_aaaa = np.asarray(mints.mo_eri(
        orbitals_up, orbitals_up, orbitals_up, orbitals_up)
    )
    g_aabb = np.asarray(
        mints.mo_eri(orbitals_up, orbitals_up, orbitals_down, orbitals_down)
    )
    g_bbbb = np.asarray(
        mints.mo_eri(
            orbitals_down, orbitals_down, orbitals_down, orbitals_down
        )
    )

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
    identity_aa = np.eye(nmo)
    identity_bb = np.eye(nmo)

    # orbital energies
    f_aa = np.diag(wfn.epsilon_a())
    f_bb = np.diag(wfn.epsilon_b())

    # dipole integrals
    mu = mints.ao_dipole()

    mu_x = np.asarray(mu[0])
    mu_y = np.asarray(mu[1])
    mu_z = np.asarray(mu[2])

    # transform the dipole integrals to the MO basis
    orbitals_up = np.asarray(orbitals_up)
    orbitals_down = np.asarray(orbitals_down)

    mua_x = orbitals_up.T @ mu_x @ orbitals_up
    mua_y = orbitals_up.T @ mu_y @ orbitals_up
    mua_z = orbitals_up.T @ mu_z @ orbitals_up

    mub_x = orbitals_down.T @ mu_x @ orbitals_down
    mub_y = orbitals_down.T @ mu_y @ orbitals_down
    mub_z = orbitals_down.T @ mu_z @ orbitals_down

    return {
        'mua_x': mua_x,
        'mua_y': mua_y,
        'mua_z': mua_z,
        'mub_x': mub_x,
        'mub_y': mub_y,
        'mub_z': mub_z,
        'identity_aa': identity_aa,
        'identity_bb': identity_bb,
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


def buld_h_aa(identity_aa, f_aa, va, oa, g_aaaa, **_) -> NDArray:
    h_aa = -1.00 * einsum('ab,ji->jbia', identity_aa[va, va], f_aa[oa, oa])
    h_aa += 1.00 * einsum('ij,ab->jbia', identity_aa[oa, oa], f_aa[va, va])
    h_aa += -1.00 * einsum('ba,ij->jbia', identity_aa[va, va], f_aa[oa, oa])
    h_aa += 1.00 * einsum('ji,ba->jbia', identity_aa[oa, oa], f_aa[va, va])
    h_aa += -1.00 * einsum('jiab->jbia', g_aaaa[oa, oa, va, va])
    h_aa += 1.00 * einsum('jabi->jbia', g_aaaa[oa, va, va, oa])
    h_aa += 1.00 * einsum('ibaj->jbia', g_aaaa[oa, va, va, oa])
    h_aa += -1.00 * einsum('abji->jbia', g_aaaa[va, va, oa, oa])

    return h_aa


def build_h_ab(g_abab, oa, ob, va, vb, **_) -> NDArray:
    h_ab = 1.00 * einsum('jiba->jbia', g_abab[oa, ob, va, vb])
    h_ab += 1.00 * einsum('jabi->jbia', g_abab[oa, vb, va, ob])
    h_ab += 1.00 * einsum('bija->jbia', g_abab[va, ob, oa, vb])
    h_ab += 1.00 * einsum('baji->jbia', g_abab[va, vb, oa, ob])
    return h_ab


def build_h_ba(g_abab, oa, ob, va, vb, **_) -> NDArray:
    h_ba = 1.00 * einsum('ijab->jbia', g_abab[oa, ob, va, vb])
    h_ba += 1.00 * einsum('ajib->jbia', g_abab[va, ob, oa, vb])
    h_ba += 1.00 * einsum('ibaj->jbia', g_abab[oa, vb, va, ob])
    h_ba += 1.00 * einsum('abij->jbia', g_abab[va, vb, oa, ob])
    return h_ba


def build_h_bb(g_bbbb, identity_bb, f_bb, vb, ob, **_) -> NDArray:
    h_bb = -1.00 * einsum('ab,ji->jbia', identity_bb[vb, vb], f_bb[ob, ob])
    h_bb += 1.00 * einsum('ij,ab->jbia', identity_bb[ob, ob], f_bb[vb, vb])
    h_bb += -1.00 * einsum('ba,ij->jbia', identity_bb[vb, vb], f_bb[ob, ob])
    h_bb += 1.00 * einsum('ji,ba->jbia', identity_bb[ob, ob], f_bb[vb, vb])
    h_bb += -1.00 * einsum('jiab->jbia', g_bbbb[ob, ob, vb, vb])
    h_bb += 1.00 * einsum('jabi->jbia', g_bbbb[ob, vb, vb, ob])
    h_bb += 1.00 * einsum('ibaj->jbia', g_bbbb[ob, vb, vb, ob])
    h_bb += -1.00 * einsum('abji->jbia', g_bbbb[vb, vb, ob, ob])
    return h_bb


def scf() -> tuple[float, Wavefunction]:

    _ = psi4.geometry("""
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


def build_complete_matrix(h_aa, h_ab, h_ba, h_bb, nmo, noa, nob, **_):
    # number of virtual orbital with spin up (ap) or down (bown)
    nva = nmo - noa
    nvb = nmo - nob

    # reshape matrices
    h_aa = h_aa.reshape(noa*nva, noa*nva)
    h_ab = h_ab.reshape(noa*nva, nob*nvb)
    h_ba = h_ba.reshape(nob*nvb, noa*nva)
    h_bb = h_bb.reshape(nob*nvb, nob*nvb)

    # pack into super matrix
    h = np.block([[h_aa, h_ab], [h_ba, h_bb]])
    return h


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


MatmulLike = TypeVar('MatmulLike', bound=LinearOperator, contravariant=True)


def gmres_solve(matrix: NDArray | MatmulLike, rhs: NDArray) -> NDArray:
    """
    solves `matrix @ solution = rhs`
    """
    solution, exit_code = gmres(matrix, rhs, rtol=1e-12)
    if exit_code != 0:
        raise RuntimeError("GMRES didn't converge")
    return solution


class OrbitalHessianAction(LinearOperator):
    """ GMRES helper. Calculates the result of `orbital_hessian @ vector` """

    def __init__(
        self: Any,
        nmo: int,
        noa: int,
        nob: int,
        identity_aa: NDArray,
        identity_bb: NDArray,
        va: slice,
        oa: slice,
        vb: slice,
        ob: slice,
        fock_aa: NDArray,
        fock_bb: NDArray,
        g_aaaa: NDArray,
        g_abab: NDArray,
        g_bbbb: NDArray,
        **_,
    ) -> None:
        # These are needed for the action
        self.n_occuped_up = noa
        self.n_valence_up = nmo - noa
        self.n_occuped_down = nob
        self.n_valence_down = nmo - nob
        self.identity_aa = identity_aa
        self.identity_bb = identity_bb
        self.valence_slice_up = va
        self.occupied_slice_up = oa
        self.valence_slice_down = vb
        self.occupied_slice_down = ob
        self.fock_aa = fock_aa
        self.fock_bb = fock_bb
        self.g_aaaa = g_aaaa
        self.g_abab = g_abab
        self.g_bbbb = g_bbbb
        """ Scipy needs these. """
        self.dtype = fock_aa.dtype
        dim = noa * (nmo-noa) + nob * (nmo-nob)
        self.shape = (dim, dim)

    def _matuu_times_up(self, up: NDArray):
        id_aa = self.identity_aa
        f_aa = self.fock_aa
        va = self.valence_slice_up
        oa = self.occupied_slice_up
        g_aaaa = self.g_aaaa

        out = -1.00 * einsum(
            'ab,ji,ia->jb', id_aa[va, va], f_aa[oa, oa], up
        )
        out += 1.00 * einsum(
            'ij,ab,ia->jb', id_aa[oa, oa], f_aa[va, va], up
        )
        out += -1.00 * einsum(
            'ba,ij,ia->jb', id_aa[va, va], f_aa[oa, oa], up
        )
        out += 1.00 * einsum(
            'ji,ba,ia->jb', id_aa[oa, oa], f_aa[va, va], up
        )
        out += -1.00 * einsum(
            'jiab,ia->jb', g_aaaa[oa, oa, va, va], up
        )
        out += 1.00 * einsum(
            'jabi,ia->jb', g_aaaa[oa, va, va, oa], up
        )
        out += 1.00 * einsum(
            'ibaj,ia->jb', g_aaaa[oa, va, va, oa], up
        )
        out += -1.00 * einsum(
            'abji,ia->jb', g_aaaa[va, va, oa, oa], up
        )
        return out

    def _matud_times_down(self, down: NDArray) -> NDArray:
        g_abab = self.g_abab
        oa = self.occupied_slice_up
        ob = self.occupied_slice_down
        va = self.valence_slice_up
        vb = self.valence_slice_down
        out = 1.00 * einsum('jiba,ia->jb', g_abab[oa, ob, va, vb], down)
        out += 1.00 * einsum('jabi,ia->jb', g_abab[oa, vb, va, ob], down)
        out += 1.00 * einsum('bija,ia->jb', g_abab[va, ob, oa, vb], down)
        out += 1.00 * einsum('baji,ia->jb', g_abab[va, vb, oa, ob], down)
        return out

    def _matdu_times_up(self, up: NDArray) -> NDArray:
        g_abab = self.g_abab
        oa = self.occupied_slice_up
        va = self.valence_slice_up
        ob = self.occupied_slice_down
        vb = self.valence_slice_down

        out = 1.00 * einsum('ijab,ia->jb', g_abab[oa, ob, va, vb], up)
        out += 1.00 * einsum('ajib,ia->jb', g_abab[va, ob, oa, vb], up)
        out += 1.00 * einsum('ibaj,ia->jb', g_abab[oa, vb, va, ob], up)
        out += 1.00 * einsum('abij,ia->jb', g_abab[va, vb, oa, ob], up)

        return out

    def _matdd_times_down(self, down: NDArray) -> NDArray:
        id_bb = self.identity_bb
        f_bb = self.fock_bb
        g_bbbb = self.g_bbbb

        vb = self.valence_slice_down
        ob = self.occupied_slice_down

        out = -1.00 * einsum(
            'ab,ji,ia->jb', id_bb[vb, vb], f_bb[ob, ob], down
        )
        out += 1.00 * einsum(
            'ij,ab,ia->jb', id_bb[ob, ob], f_bb[vb, vb], down
        )
        out += -1.00 * einsum(
            'ba,ij,ia->jb', id_bb[vb, vb], f_bb[ob, ob], down
        )
        out += 1.00 * einsum(
            'ji,ba,ia->jb', id_bb[ob, ob], f_bb[vb, vb], down
        )
        out += -1.00 * einsum(
            'jiab,ia->jb', g_bbbb[ob, ob, vb, vb], down
        )
        out += 1.00 * einsum(
            'jabi,ia->jb', g_bbbb[ob, vb, vb, ob], down
        )
        out += 1.00 * einsum(
            'ibaj,ia->jb', g_bbbb[ob, vb, vb, ob], down
        )
        out += -1.00 * einsum(
            'abji,ia->jb', g_bbbb[vb, vb, ob, ob], down
        )

        return out

    def _matvec(self, x: NDArray):
        """ This is where the implementation of mat@vec should go. """

        dim_up = self.n_occuped_up * self.n_valence_up
        up = x[:dim_up].reshape(self.n_occuped_up, self.n_valence_up)
        dim_down = self.n_occuped_down * self.n_valence_down
        down = x[-dim_down:].reshape(self.n_occuped_down, self.n_valence_down)

        out_up = self._matuu_times_up(up) + self._matud_times_down(down)
        out_down = self._matdu_times_up(up) + self._matdd_times_down(down)

        out = np.hstack((out_up.flatten(), out_down.flatten()))
        return out


class MockOrbitalHessianAction(LinearOperator):
    """ GMRES helper. Calculates the result of `orbital_hessian @ vector` """

    def __init__(self: Any, matrix: NDArray) -> None:
        """ The point is to NOT store the whole matrix.
        But here for testing, I will still keep it. """
        self.matrix = matrix
        self.shape = matrix.shape
        self.dtype = matrix.dtype

    def _matvec(self, x: NDArray):
        """ This is where the implementation of mat@vec should go. """
        return self.matrix @ x.reshape(-1, 1)


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

    # pol_direct = find_polarizabilities_directly(
    #     h_aa, h_ab, h_ba, h_bb, **intermediates
    # )

    # print("Polarizabilities:")
    # print("1) from inverted HF orbitals Hessian:")
    # print_polarizabilities(pol_direct)

    pol_iterative = find_polarizabilities_iteratively(
        h_aa, h_ab, h_ba, h_bb, **intermediates
    )

    print("2) from GMRES iterative solution:")
    print_polarizabilities(pol_iterative)

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
