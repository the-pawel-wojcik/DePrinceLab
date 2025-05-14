import pdaggerq as rome
from pdaggerq.parser import contracted_strings_to_tensor_terms


def build_expression():
    pq = rome.pq_helper('fermi')
    # The HACK
    # The UCC cluster operator looks a lot like the commutator terms
    pq.set_unitary_cc(True)
    for ham in ['f', 'v']:
        pq.add_double_commutator(1.0, [ham], ['a*(a)', 'a(i)'], ['t1'])
        pq.add_double_commutator(-1.0, [ham], ['a*(i)', 'a(a)'], ['t1'])
    pq.simplify()
    return pq


def print_restricted(pq):
    print()
    terms = pq.strings()
    for term in terms:
        print(term)


def print_unrestricted(pq):
    print()
    for s1 in ['a', 'b']:
        spin_labels = {
            'i': s1,
            'a': s1,
        }
        terms = pq.strings(spin_labels=spin_labels)
        terms = contracted_strings_to_tensor_terms(terms)

        print(f'# The {s1} spin-block:')
        for term in terms:
            # print(f'# {term}')
            pyterm = term.einsum_string(
                update_val='h_'+s1,
                output_variables=('i', 'a'),
            )
            print(f'{pyterm}')
        print()


def main():
    print('<HF| [[H, (aₐ†aᵢ-aᵢ†aₐ)], (aₐ†aᵢ-aᵢ†aₐ)] |HF> v _{ia}')
    pq = build_expression()
    use_spinorbitals = True
    if use_spinorbitals is False:
        print_restricted(pq)
    else:
        print_unrestricted(pq)


if __name__ == "__main__":
    main()
