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
class Guess:
    density: NDArray
    fock: NDArray
    occ_slice: slice


def build_guess(
    integrals: Integrals,
    overlap_inv_root: NDArray
) -> tuple[Guess, Guess]:
    """ Build the "core" guess. 

    Return a tuple of guess: one for spin up and one for spin down. """
    # core guess
    core_Fock_up = integrals.core_Hamiltonian

    core_Fock_up = overlap_inv_root.T @ core_Fock_up @ overlap_inv_root
    core_Fock_eigensys = np.linalg.eigh(core_Fock_up)
    guess_up = overlap_inv_root @ core_Fock_eigensys.eigenvectors

    guess_down = guess_up.copy()
    core_Fock_down = core_Fock_up.copy()
    # Density matrices
    occupied_up = slice(0, integrals.n_up)
    occupied_down = slice(0, integrals.n_down)
    density_up = np.einsum(
        'pi,qi->pq',
        guess_up[:, occupied_up],
        guess_up[:, occupied_up]
    )
    density_down = np.einsum(
        'pi,qi->pq',
        guess_down[:, occupied_down],
        guess_down[:, occupied_down]
    )

    up = Guess(
        density=density_up,
        fock=core_Fock_up,
        occ_slice=occupied_up
    )
    down = Guess(
        density=density_down,
        fock=core_Fock_down,
        occ_slice=occupied_down,
    )
    return up, down


def scf_energy(
    up: Guess,
    down: Guess,
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


def build_focks(up: Guess, down: Guess, integrals: Integrals):
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


def diagonalize_fock(
    fock: NDArray,
    transform: NDArray
) -> NDArray:
    """
    transform: square root of the inverse of the overlap matrix, S.
    """
    # transform Fock matrices to the orthogonal basis
    transed_Fock = transform.transpose() @ fock @ transform
    fock_eigensystem = np.linalg.eigh(transed_Fock)
    # back transform the MO coefficient matrices to the non-orthogonal basis
    mo_coefficients = transform @ fock_eigensystem.eigenvectors
    return mo_coefficients


def iterative_solution(
    up: Guess,
    down: Guess,
    integrals: Integrals,
    overlap_inv_root: NDArray,
    mol: Molecule,
):
    e_convergence = 1e-8
    d_convergence = 1e-6
    maxiter = 100

    print_SCF_header()

    old_energy = scf_energy(up, down, integrals.core_Hamiltonian)
    old_up = Guess(up.density.copy(), up.fock.copy(), up.occ_slice)
    old_down = Guess(down.density.copy(), down.fock.copy(), down.occ_slice)

    for iter in range (0, maxiter):
        # build Fock matrix using current density
        fock_up, fock_down = build_focks(up, down, integrals)
        mo_coeffcients_up = diagonalize_fock(fock_up, overlap_inv_root)
        mo_coeffcients_down = diagonalize_fock(fock_down, overlap_inv_root)

        density_up = np.einsum(
            'pi,qi->pq',
            mo_coeffcients_up[:, up.occ_slice],
            mo_coeffcients_up[:, up.occ_slice]
        )
        density_down = np.einsum(
            'pi,qi->pq',
            mo_coeffcients_down[:, down.occ_slice],
            mo_coeffcients_down[:, down.occ_slice]
        )
        
        up = Guess(density=density_up, fock=fock_up, occ_slice=up.occ_slice)
        down = Guess(
                density=density_down, fock=fock_down, occ_slice=down.occ_slice
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
        old_up = Guess(up.density.copy(), up.fock.copy(), up.occ_slice)
        old_down = Guess(down.density.copy(), down.fock.copy(), down.occ_slice)

        # convergence check
        if dE < e_convergence and dD < d_convergence:
            break

        iter += 1
    else:
        print('SCF did not converge...')
        return

    print('SCF converged!')
    print(f'SCF total energy: {energy + mol.nuclear_repulsion_energy():.12f}')


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

    overlap_esystem = np.linalg.eigh(integrals.overlap)
    overlap_inv_root = np.diagflat(overlap_esystem.eigenvalues ** (-0.5))
    transformation = overlap_esystem.eigenvectors
    overlap_inv_root = transformation @ overlap_inv_root @ transformation.T

    up, down = build_guess(integrals, overlap_inv_root)
    iterative_solution(up, down, integrals, overlap_inv_root, mol)


if __name__ == "__main__":
    main()
