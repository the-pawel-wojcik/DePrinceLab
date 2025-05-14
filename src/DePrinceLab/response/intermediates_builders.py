from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from psi4.core import Wavefunction, Matrix, MintsHelper


@dataclass
class Intermediates:
    mua_x: NDArray
    mua_y: NDArray
    mua_z: NDArray
    mub_x: NDArray
    mub_y: NDArray
    mub_z: NDArray
    identity_aa: NDArray
    identity_bb: NDArray
    f_aa: NDArray
    f_bb: NDArray
    va: slice
    vb: slice
    oa: slice
    ob: slice
    nmo: int
    noa: int
    nob: int
    g_aaaa: NDArray
    g_abab: NDArray
    g_bbbb: NDArray


def extract_intermediates(wfn: Wavefunction) -> Intermediates:
    orbitals_up: Matrix = wfn.Ca()
    orbitals_down: Matrix = wfn.Cb()

    # use Psi4's MintsHelper to generate ERIs and dipole integrals
    mints = MintsHelper(wfn.basisset())

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

    return Intermediates(
        mua_x=mua_x,
        mua_y=mua_y,
        mua_z=mua_z,
        mub_x=mub_x,
        mub_y=mub_y,
        mub_z=mub_z,
        identity_aa=identity_aa,
        identity_bb=identity_bb,
        f_aa=f_aa,
        f_bb=f_bb,
        va=va,
        vb=vb,
        oa=oa,
        ob=ob,
        nmo=nmo,
        noa=noa,
        nob=nob,
        g_aaaa=g_aaaa,
        g_abab=g_abab,
        g_bbbb=g_bbbb,
    )
