<HF| [[H, (aₐ†aᵢ-aᵢ†aₐ)], (aₐ†aᵢ-aᵢ†aₐ)] |HF>

# The a-a spin-block:
# -1.00 d(b,a)*f_aa(i,j)
h_aa += -1.00 * einsum('ba,ij->jbia', kd_aa[va, va], f_aa[oa, oa])
#  1.00 d(j,i)*f_aa(b,a)
h_aa +=  1.00 * einsum('ji,ba->jbia', kd_aa[oa, oa], f_aa[va, va])
# -1.00 d(a,b)*f_aa(j,i)
h_aa += -1.00 * einsum('ab,ji->jbia', kd_aa[va, va], f_aa[oa, oa])
#  1.00 d(i,j)*f_aa(a,b)
h_aa +=  1.00 * einsum('ij,ab->jbia', kd_aa[oa, oa], f_aa[va, va])
# -1.00 <j,i||a,b>_aaaa
h_aa += -1.00 * einsum('jiab->jbia', g_aaaa[oa, oa, va, va])
#  1.00 <i,b||a,j>_aaaa
h_aa +=  1.00 * einsum('ibaj->jbia', g_aaaa[oa, va, va, oa])
#  1.00 <j,a||b,i>_aaaa
h_aa +=  1.00 * einsum('jabi->jbia', g_aaaa[oa, va, va, oa])
# -1.00 <a,b||j,i>_aaaa
h_aa += -1.00 * einsum('abji->jbia', g_aaaa[va, va, oa, oa])

# The a-b spin-block:
#  1.00 <j,i||b,a>_abab
h_ab +=  1.00 * einsum('jiba->jbia', g_abab[oa, ob, va, vb])
#  1.00 <b,i||j,a>_abab
h_ab +=  1.00 * einsum('bija->jbia', g_abab[va, ob, oa, vb])
#  1.00 <j,a||b,i>_abab
h_ab +=  1.00 * einsum('jabi->jbia', g_abab[oa, vb, va, ob])
#  1.00 <b,a||j,i>_abab
h_ab +=  1.00 * einsum('baji->jbia', g_abab[va, vb, oa, ob])

# The b-a spin-block:
#  1.00 <i,j||a,b>_abab
h_ba +=  1.00 * einsum('ijab->jbia', g_abab[oa, ob, va, vb])
#  1.00 <i,b||a,j>_abab
h_ba +=  1.00 * einsum('ibaj->jbia', g_abab[oa, vb, va, ob])
#  1.00 <a,j||i,b>_abab
h_ba +=  1.00 * einsum('ajib->jbia', g_abab[va, ob, oa, vb])
#  1.00 <a,b||i,j>_abab
h_ba +=  1.00 * einsum('abij->jbia', g_abab[va, vb, oa, ob])

# The b-b spin-block:
# -1.00 d(b,a)*f_bb(i,j)
h_bb += -1.00 * einsum('ba,ij->jbia', kd_bb[vb, vb], f_bb[ob, ob])
#  1.00 d(j,i)*f_bb(b,a)
h_bb +=  1.00 * einsum('ji,ba->jbia', kd_bb[ob, ob], f_bb[vb, vb])
# -1.00 d(a,b)*f_bb(j,i)
h_bb += -1.00 * einsum('ab,ji->jbia', kd_bb[vb, vb], f_bb[ob, ob])
#  1.00 d(i,j)*f_bb(a,b)
h_bb +=  1.00 * einsum('ij,ab->jbia', kd_bb[ob, ob], f_bb[vb, vb])
# -1.00 <j,i||a,b>_bbbb
h_bb += -1.00 * einsum('jiab->jbia', g_bbbb[ob, ob, vb, vb])
#  1.00 <i,b||a,j>_bbbb
h_bb +=  1.00 * einsum('ibaj->jbia', g_bbbb[ob, vb, vb, ob])
#  1.00 <j,a||b,i>_bbbb
h_bb +=  1.00 * einsum('jabi->jbia', g_bbbb[ob, vb, vb, ob])
# -1.00 <a,b||j,i>_bbbb
h_bb += -1.00 * einsum('abji->jbia', g_bbbb[vb, vb, ob, ob])

