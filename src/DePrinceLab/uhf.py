from __future__ import annotations
import numpy as np
import psi4
from psi4.core import Molecule
from dataclasses import dataclass
from numpy.typing import NDArray


@dataclass
class Integrals:
    """
    The electron repulsion integrals (eri) are stored in the chemists'
    notation, i.e., (11|22)
    """
    n_up: int
    n_down: int
    core_Hamiltonian: NDArray
    electron_repulsion: NDArray
    overlap: NDArray



def get_integrals(mol: Molecule, options: dict) -> Integrals:
    """
    :param mol: a psi4 molecule object
    :param options: an options dictionary for psi4
    """
    psi4.activate(mol)
    psi4.set_options(options)
    # suppress psi4 printing 
    psi4.core.be_quiet()
    # we need a wave function object to evaluate the integrals
    wfn = psi4.core.Wavefunction.build(
        mol,
        psi4.core.get_global_option('BASIS'),
    )
    # get integrals from MintsHelper
    mints = psi4.core.MintsHelper(wfn.basisset())
    kinetic_energy = np.asarray(mints.ao_kinetic())
    potential_energy = np.asarray(mints.ao_potential())
    integrals = Integrals(
        n_up=wfn.nalpha(),
        n_down=wfn.nbeta(),
        core_Hamiltonian=kinetic_energy + potential_energy,
        electron_repulsion=np.asarray(mints.ao_eri()),
        overlap=np.asarray(mints.ao_overlap()),
    )
    return integrals


@dataclass
class Solution:
    fock: NDArray
    density: NDArray
    orbitals: NDArray
    occ_slice: slice


    def copy(self) -> Solution:
        return Solution(
            fock=self.fock.copy(),
            density=self.density.copy(),
            orbitals=self.orbitals.copy(),
            occ_slice=self.occ_slice,
        )


def build_guess(
    integrals: Integrals,
    lowdin: NDArray
) -> tuple[Solution, Solution]:
    """ Build the "core" guess. 

    Return a tuple of guess: one for spin up and one for spin down. """

    # core guess is formed by diagonalizing the electronic Hamiltonian that is
    # missing the electron-electron interactions
    core_Fock_up = integrals.core_Hamiltonian

    occupied_up = slice(0, integrals.n_up)
    orbitals_up = diagonalize_fock(core_Fock_up, lowdin)
    density_up = build_density(orbitals_up, occupied_up)

    up = Solution(
        density=density_up,
        orbitals=orbitals_up,
        fock=core_Fock_up,
        occ_slice=occupied_up
    )
    down = up.copy()

    return up, down


def scf_energy(
    up: Solution,
    down: Solution,
    core_Hamiltonian: NDArray
) -> float:
    energy = 0.5 * np.dot(
        up.density.flatten(),
        (core_Hamiltonian + up.fock).flatten()
    ) 
    energy += 0.5 * np.dot(
        down.density.flatten(),
        (core_Hamiltonian + down.fock).flatten()
    )
    return energy


def print_SCF_header():
    print('==> Begin SCF Iterations <==')
    fields = ('Iter', 'energy', 'dE', 'dD')
    field_width = 80 // len(fields)
    header = ''.join(field.center(field_width) for field in fields)
    print(header)


def build_focks(up: Solution, down: Solution, integrals: Integrals):
    coulomb_up = np.einsum(
        'rs, pqrs -> pq', up.density, integrals.electron_repulsion,
    )
    coulomb_down = np.einsum(
        'rs, pqrs -> pq', down.density, integrals.electron_repulsion,
    )

    exchange_up = np.einsum(
        'rs, psrq -> pq', up.density, integrals.electron_repulsion,
    )
    exchange_down = np.einsum(
        'rs, psrq -> pq', down.density, integrals.electron_repulsion,
    )

    base = integrals.core_Hamiltonian + coulomb_up + coulomb_down 
    fock_up = base - exchange_up
    fock_down = base - exchange_down

    return fock_up, fock_down


def build_density(orbitals: NDArray, occupied: slice) -> NDArray:
    density = np.einsum(
        'pi,qi->pq', orbitals[:, occupied], orbitals[:, occupied]
    )
    return density


def diagonalize_fock(
    fock: NDArray,
    lowdin: NDArray
) -> NDArray:
    """
    lowdin: square root of the inverse of the diagonalized overlap matrix, S.
    It's needed in the symmetric Löwdin's diagonalization.
    """
    # transform Fock matrices to the orthogonal basis
    transed_Fock = lowdin.transpose() @ fock @ lowdin
    fock_eigensystem = np.linalg.eigh(transed_Fock)
    # back transform the MO coefficient matrices to the non-orthogonal basis
    orbitals = lowdin @ fock_eigensystem.eigenvectors
    return orbitals


def iterative_solution(
    up: Solution,
    down: Solution,
    integrals: Integrals,
    overlap_inv_root: NDArray,
    mol: Molecule,
):
    e_convergence = 1e-8
    d_convergence = 1e-6
    maxiter = 100

    print_SCF_header()

    old_energy = scf_energy(up, down, integrals.core_Hamiltonian)
    old_up = up.copy()
    old_down = down.copy()

    for iter in range (0, maxiter):
        # build Fock matrix using current density
        fock_up, fock_down = build_focks(up, down, integrals)

        mo_coeffcients_up = diagonalize_fock(fock_up, overlap_inv_root)
        density_up = build_density(mo_coeffcients_up, up.occ_slice)
        up = Solution(
            fock=fock_up,
            orbitals=mo_coeffcients_up,
            density=density_up,
            occ_slice=up.occ_slice,
        )
        mo_coeffcients_down = diagonalize_fock(fock_down, overlap_inv_root)
        density_down = build_density(mo_coeffcients_down, down.occ_slice)
        down = Solution(
            fock=fock_down,
            orbitals=mo_coeffcients_down,
            density=density_down,
            occ_slice=down.occ_slice,
        )

        # current energy
        energy = scf_energy(up, down, integrals.core_Hamiltonian)
        
        dE = np.abs(energy - old_energy)
        dD = np.linalg.norm(up.density - old_up.density) 
        dD += np.linalg.norm(down.density - old_down.density)

        # energy is reported with the nuclear repulsion contribution
        fmt='.12f'
        fields = (
            f'{iter}',
            f'{energy + mol.nuclear_repulsion_energy():{fmt}}',
            f'{dE:{fmt}}',
            f'{dD:{fmt}}',
        )
        field_width = 80 // 4
        report_line = ''.join(field.center(field_width) for field in fields)
        print(report_line)

        # save energy and density
        old_energy = energy
        old_up = up.copy()
        old_down = down.copy()

        # convergence check
        if dE < e_convergence and dD < d_convergence:
            break

        iter += 1
    else:
        print('SCF did not converge...')
        return

    print('SCF converged!')
    print(f'SCF total energy: {energy + mol.nuclear_repulsion_energy():.12f}')


def build_Lowdin_transformation(integrals: Integrals) -> NDArray:
    """ In Lowdin's symmetric orthogonalization one needs to transfrom back and
    forward with the inverse of the square root of the overlap matrix
    eigenvalues. This function builds the transformation matrix. """

    overlap_esystem = np.linalg.eigh(integrals.overlap)
    overlap_inv_root = np.diagflat(overlap_esystem.eigenvalues ** (-0.5))
    transformation = overlap_esystem.eigenvectors
    lowdin = transformation @ overlap_inv_root @ transformation.T

    return lowdin


def main():
    mol = psi4.geometry("""
        0 1
        O
        H 1 1.0
        H 1 1.0 2 104.5
        symmetry c1
    """)
    options = {'basis': 'cc-pvdz'}
    integrals = get_integrals(mol, options)
    lowdin = build_Lowdin_transformation(integrals)
    up, down = build_guess(integrals, lowdin)
    iterative_solution(up, down, integrals, lowdin, mol)


if __name__ == "__main__":
    main()
