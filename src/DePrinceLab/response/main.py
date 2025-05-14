from DePrinceLab.response.electronic_structure import scf
from DePrinceLab.response.intermediates_builders import extract_intermediates
from DePrinceLab.response.hf_orbital_hessian_builders import (
    build_complete_matrix,
)
from DePrinceLab.response.polarizabilities.core import pretty_print
from DePrinceLab.response.polarizabilities.operators import (
    MockOrbitalHessianAction,
    OrbitalHessianAction,
    FasterOrbitalHessianAction,
)
from DePrinceLab.response.polarizabilities.invert_hessian import\
    calculate as pol_by_invert
from DePrinceLab.response.polarizabilities.solve_gmres import\
    calculate as pol_with_gmres


def main():
    _, wfn = scf()
    intermediates = extract_intermediates(wfn)

    print("Polarizabilities:")

    do_direct = True
    if do_direct:
        pol_direct = pol_by_invert(intermediates)
        print("1) from inverted HF orbitals Hessian:")
        pretty_print(pol_direct)

    do_iterative = True
    if do_iterative:
        orbital_hessian = build_complete_matrix(intermediates)
        orbital_hessian_action = MockOrbitalHessianAction(orbital_hessian)
        pol_iterative = pol_with_gmres(orbital_hessian_action, intermediates)
        print("2) from GMRES iterative solution (with Hessian storage):")
        pretty_print(pol_iterative)

    do_action = True
    if do_action:
        orbital_hessian_action = OrbitalHessianAction(intermediates)
        pol = pol_with_gmres(orbital_hessian_action, intermediates)
        print("3) from GMRES iterative solution:")
        pretty_print(pol)

    do_fast_action = True
    if do_fast_action:
        orbital_hessian_action = FasterOrbitalHessianAction(intermediates)
        pol = pol_with_gmres(orbital_hessian_action, intermediates)
        print("4) from GMRES iterative solution (optimized build):")
        pretty_print(pol)


if __name__ == "__main__":
    main()
