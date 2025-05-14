from DePrinceLab.response.electronic_structure import scf
from DePrinceLab.response.intermediates_builders import extract_intermediates
from DePrinceLab.response.polarizabilities.invert_hessian import\
    calculate as pol_by_invert
from DePrinceLab.response.polarizabilities.iterative import\
    calculate as pol_by_iterate
from DePrinceLab.response.polarizabilities.no_hessian_storage import\
    calculate as pol_no_storage
from DePrinceLab.response.polarizabilities.core import pretty_print


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
        pol_iterative = pol_by_iterate(intermediates)
        print("2) from GMRES iterative solution:")
        pretty_print(pol_iterative)

    do_action = True
    if do_action:
        pol = pol_no_storage(intermediates)
        print("3) from GMRES iterative solution (no hessian storage):")
        pretty_print(pol)


if __name__ == "__main__":
    main()
