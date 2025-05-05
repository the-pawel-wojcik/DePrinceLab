import pdaggerq
from pdaggerq.parser import contracted_strings_to_tensor_terms

pq = pdaggerq.pq_helper('fermi')

pq.add_double_commutator( 1.0, ['f'], ['a*(a)', 'a(i)'], ['a*(b)', 'a(j)'])
pq.add_double_commutator(-1.0, ['f'], ['a*(i)', 'a(a)'], ['a*(b)', 'a(j)'])
pq.add_double_commutator(-1.0, ['f'], ['a*(a)', 'a(i)'], ['a*(j)', 'a(b)'])
pq.add_double_commutator( 1.0, ['f'], ['a*(i)', 'a(a)'], ['a*(j)', 'a(b)'])

pq.add_double_commutator( 1.0, ['v'], ['a*(a)', 'a(i)'], ['a*(b)', 'a(j)'])
pq.add_double_commutator(-1.0, ['v'], ['a*(i)', 'a(a)'], ['a*(b)', 'a(j)'])
pq.add_double_commutator(-1.0, ['v'], ['a*(a)', 'a(i)'], ['a*(j)', 'a(b)'])
pq.add_double_commutator( 1.0, ['v'], ['a*(i)', 'a(a)'], ['a*(j)', 'a(b)'])

pq.simplify()

for s1 in ['a', 'b']:
    for s2 in ['a', 'b']:

        spin_labels = {
                'j': s1,
                'b': s1,
                'i': s2,
                'a': s2,
                }
        terms = pq.strings(spin_labels = spin_labels)
        terms = contracted_strings_to_tensor_terms(terms)

        print('')
        print('# %s-%s spin block' % (s1, s2))
        print('')
        for term in terms:
            string = term.einsum_string(
                update_val='h_' + s1 + s2,
                output_variables=('j', 'b', 'i', 'a')
            )
            print('#', term)
            print(f'{string}')
