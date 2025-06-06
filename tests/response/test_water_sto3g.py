import math

import psi4
from psi4.core import Molecule, Wavefunction
from DePrinceLab.response.intermediates_builders import extract_intermediates
from DePrinceLab.response.polarizabilities.core import pretty_print
from DePrinceLab.response.polarizabilities.operators import (
    FasterOrbitalHessianAction
)
from DePrinceLab.response.polarizabilities.solve_gmres import\
    calculate as pol_with_gmres


def scf_water_sto_3G() -> tuple[Molecule, float, Wavefunction]:
    """ Geometry from CCCBDB: HF/STO-3G """
    mol: Molecule = psi4.geometry("""
    0 1
    O1	0.0000   0.0000   0.1272
    H2	0.0000   0.7581  -0.5086
    H3	0.0000  -0.7581  -0.5086
    symmetry c1
    """)

    psi4.set_options({'basis': 'sto-3g',
                      'scf_type': 'pk',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-12})

    psi4.core.be_quiet()

    # compute the Hartree-Fock energy and wavefunction
    energy, wfn = psi4.energy('SCF', molecule=mol, return_wfn=True)

    return mol, energy, wfn


def test_water_sto3g_polarizability():
    _, _, wfn = scf_water_sto_3G()
    intermediates = extract_intermediates(wfn)
    orbital_hessian_action = FasterOrbitalHessianAction(intermediates)
    pol = pol_with_gmres(orbital_hessian_action, intermediates)
    print("Polarizabilities of H₂O HF/STO-3G.")
    pretty_print(pol)

    # Assures go against the values from CCCBDB
    assert math.isclose(pol['xx'], 0.04, abs_tol=0.006)
    assert math.isclose(pol['xy'], 0.00, abs_tol=0.006)
    assert math.isclose(pol['xz'], 0.00, abs_tol=0.006)
    assert math.isclose(pol['yy'], 5.51, abs_tol=0.006)
    assert math.isclose(pol['yz'], 0.00, abs_tol=0.006)
    assert math.isclose(pol['zz'], 2.57, abs_tol=0.006)


if __name__ == "__main__":
    test_water_sto3g_polarizability()
