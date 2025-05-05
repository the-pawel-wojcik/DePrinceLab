from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import psi4
from psi4.core import Molecule

from diis import Solver_DIIS

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


    def __post_init__(self):
        """ TODO: """
        self.lowdin = self.build_Lowdin_transformation()


    def build_Lowdin_transformation(self) -> NDArray:
        """ In Lowdin's symmetric orthogonalization one needs to transfrom back
        and forth with the inverse of the square root of the overlap matrix
        eigenvalues. This function builds the transformation matrix. """

        overlap_esystem = np.linalg.eigh(self.overlap)
        overlap_inv_root = np.diagflat(overlap_esystem.eigenvalues ** (-0.5))
        transformation = overlap_esystem.eigenvectors
        lowdin = transformation @ overlap_inv_root @ transformation.T

        return lowdin



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


def build_core_guess(
    integrals: Integrals,
    lowdin: NDArray
) -> tuple[Solution, Solution]:
    """ 
    The "core" guess is formed by diagonalizing the electronic Hamiltonian that
    is missing the electron-electron interactions.

    Return tuple: [spin up, spin down] """

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
    core_Hamiltonian: NDArray,
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


def print_SCF_header(use_diis: bool = False):
    print('==> Begin SCF Iterations <==')
    if use_diis is False:
        fields = ('Iter', 'energy', 'dE', 'dD')
    else:
        fields = ('Iter', 'energy', 'dE', 'g_norm')
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
    fock_in_lowdin = lowdin.T @ fock @ lowdin
    fock_eigensystem = np.linalg.eigh(fock_in_lowdin)
    # back transform the MO coefficient matrices to the non-orthogonal basis
    orbitals = lowdin @ fock_eigensystem.eigenvectors
    return orbitals


def default_solver(
    fock: NDArray,
    lowdin: NDArray,
    occ_slice: slice,
) -> Solution:
    orbitals = diagonalize_fock(fock, lowdin)
    density = build_density(orbitals, occ_slice)
    solution = Solution(
        fock=fock,
        orbitals=orbitals,
        density=density,
        occ_slice=occ_slice
    )
    return solution


""" part of DIIS """
def get_orbital_gradient(fock_lowdin, density, lowdin, overlap):
    commutator = fock_lowdin @ density @ overlap
    commutator -= overlap @ density @ fock_lowdin
    orbital_gradient = lowdin.T @ commutator @ lowdin
    return orbital_gradient

""" part of DIIS """
def find_diis_error_vec(up, down, lowdin, integrals) -> NDArray:
    error_up = get_orbital_gradient(
        up.fock, up.density,
        lowdin, integrals.overlap,
    )
    error_down = get_orbital_gradient(
        down.fock, down.density,
        lowdin, integrals.overlap,
    )
    error_vector = np.hstack(
        (error_up.flatten(), error_down.flatten())
    )
    return error_vector


""" part of DIIS """
def diis_solver(
    up: Solution,
    down: Solution,
    integrals: Integrals,
    lowdin: NDArray,
    diis: Solver_DIIS,
) -> tuple[Solution, Solution, float]:

    error_vector = find_diis_error_vec(up, down, lowdin, integrals)
    # DIIS extrapolation
    fock_lowdin_up = lowdin.T @ up.fock @ lowdin
    fock_lowdin_down = lowdin.T @ down.fock @ lowdin
    soln_vector = np.hstack(
        (fock_lowdin_up.flatten(), fock_lowdin_down.flatten())
    )
    soln_vector = diis.extrapolate(soln_vector, error_vector)

    # reshape flattened extrapolated solution vector
    F_a = up.fock
    F_b = down.fock
    F_up = soln_vector[:int(len(F_a)**2)].reshape(len(F_a), len(F_a))
    F_down = soln_vector[int(len(F_a)**2):].reshape(len(F_b), len(F_b))

    up_eigensystem = np.linalg.eigh(F_up)
    down_eigensystem = np.linalg.eigh(F_down)

    orbitals_up = lowdin @ up_eigensystem.eigenvectors
    density_up = build_density(orbitals_up, up.occ_slice)

    orbitals_down = lowdin @ down_eigensystem.eigenvectors
    density_down = build_density(orbitals_down, down.occ_slice)

    solution_up = Solution(
        fock=up.fock,
        density=density_up,
        orbitals=orbitals_up,
        occ_slice=up.occ_slice,
    )
    solution_down = Solution(
        fock=down.fock,
        density=density_down,
        orbitals=orbitals_down,
        occ_slice=down.occ_slice,
    )

    # orbital gradient norm (for convergence)
    g_norm = float(np.linalg.norm(error_vector))
    return solution_up, solution_down, g_norm


def print_iter_result(
    iter: int, energy: float, mol: Molecule, dE: float, last: float,
):
    """
    last is dD for regular SCF and g_norm for DIIS
    """
    # energy is reported with the nuclear repulsion contribution
    fmt='.12f'
    fields = (
        f'{iter}',
        f'{energy + mol.nuclear_repulsion_energy():{fmt}}',
        f'{dE:{fmt}}',
        f'{last:{fmt}}',
    )
    field_width = 80 // 4
    report_line = ''.join(field.center(field_width) for field in fields)
    print(report_line)


def iterative_solution(
    up: Solution,
    down: Solution,
    integrals: Integrals,
    lowdin: NDArray,
    mol: Molecule,
    use_diis: bool = False
) -> tuple[Solution, Solution]:
    e_convergence = 1e-8
    d_convergence = 1e-6
    maxiter = 100

    print_SCF_header(use_diis)

    old_energy = scf_energy(up, down, integrals.core_Hamiltonian)
    old_up = up.copy()
    old_down = down.copy()

    if use_diis is True:
        diis = Solver_DIIS(max_stored_vecs=8)

    for iter in range(maxiter):
        # build Fock matrices using current density
        up.fock, down.fock = build_focks(up, down, integrals)

        if use_diis is False:
            up = default_solver(up.fock, lowdin, up.occ_slice)
            down = default_solver(down.fock, lowdin, down.occ_slice)
        else:
            up, down, g_norm = diis_solver(up, down, integrals, lowdin, diis)

        # current energy
        energy = scf_energy(up, down, integrals.core_Hamiltonian)
        dE = np.abs(energy - old_energy)

        if use_diis is False:
            dD = np.linalg.norm(up.density - old_up.density) 
            dD += np.linalg.norm(down.density - old_down.density)
            print_iter_result(iter, energy, mol, dE, float(dD))
        else:
            print_iter_result(iter, energy, mol, dE, g_norm)

        # save energy and density
        old_energy = energy
        old_up = up.copy()
        old_down = down.copy()

        # convergence check
        converged = False
        if use_diis is False:
            converged = dE < e_convergence and dD < d_convergence
        else:
            converged = dE < e_convergence and g_norm < d_convergence

        if converged:
            break

        iter += 1
    else:
        print('SCF did not converge...')
        return up, down

    print('SCF converged!')
    print(f'SCF total energy: {energy + mol.nuclear_repulsion_energy():.12f}')
    return up, down


class SCF:
    def __init__(
        self,
        integrals: Integrals,
        mol: Molecule,
        use_diis: bool = False,
    ):
        self.integrals = integrals
        self.mol = mol
        self.use_diis = use_diis


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
    scf = SCF(integrals, mol)
    lowdin = integrals.lowdin
    up, down = build_core_guess(integrals, lowdin)
    up, down = iterative_solution(
        up, down, integrals, lowdin, mol, use_diis=True
    )


if __name__ == "__main__":
    main()
