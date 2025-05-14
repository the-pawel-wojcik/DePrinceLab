from DePrinceLab.response.intermediates_builders import Intermediates
import numpy as np
from numpy import einsum
from numpy.typing import NDArray


def build_h_aa(intermediates: Intermediates) -> NDArray:
    identity_aa = intermediates.identity_aa
    f_aa = intermediates.f_aa
    va = intermediates.va
    oa = intermediates.oa
    g_aaaa = intermediates.g_aaaa

    h_aa = -1.00 * einsum('ab,ji->jbia', identity_aa[va, va], f_aa[oa, oa])
    h_aa += 1.00 * einsum('ij,ab->jbia', identity_aa[oa, oa], f_aa[va, va])
    h_aa += -1.00 * einsum('ba,ij->jbia', identity_aa[va, va], f_aa[oa, oa])
    h_aa += 1.00 * einsum('ji,ba->jbia', identity_aa[oa, oa], f_aa[va, va])
    h_aa += -1.00 * einsum('jiab->jbia', g_aaaa[oa, oa, va, va])
    h_aa += 1.00 * einsum('jabi->jbia', g_aaaa[oa, va, va, oa])
    h_aa += 1.00 * einsum('ibaj->jbia', g_aaaa[oa, va, va, oa])
    h_aa += -1.00 * einsum('abji->jbia', g_aaaa[va, va, oa, oa])

    return h_aa


def build_h_ab(intermediates: Intermediates) -> NDArray:
    g_abab = intermediates.g_abab
    va = intermediates.va
    oa = intermediates.oa
    vb = intermediates.vb
    ob = intermediates.ob

    h_ab = 1.00 * einsum('jiba->jbia', g_abab[oa, ob, va, vb])
    h_ab += 1.00 * einsum('jabi->jbia', g_abab[oa, vb, va, ob])
    h_ab += 1.00 * einsum('bija->jbia', g_abab[va, ob, oa, vb])
    h_ab += 1.00 * einsum('baji->jbia', g_abab[va, vb, oa, ob])
    return h_ab


def build_h_ba(intermediates: Intermediates) -> NDArray:
    g_abab = intermediates.g_abab
    va = intermediates.va
    oa = intermediates.oa
    vb = intermediates.vb
    ob = intermediates.ob
    h_ba = 1.00 * einsum('ijab->jbia', g_abab[oa, ob, va, vb])
    h_ba += 1.00 * einsum('ajib->jbia', g_abab[va, ob, oa, vb])
    h_ba += 1.00 * einsum('ibaj->jbia', g_abab[oa, vb, va, ob])
    h_ba += 1.00 * einsum('abij->jbia', g_abab[va, vb, oa, ob])
    return h_ba


def build_h_bb(intermediates: Intermediates) -> NDArray:
    g_bbbb = intermediates.g_bbbb
    identity_bb = intermediates.identity_bb
    f_bb = intermediates.f_bb
    vb = intermediates.vb
    ob = intermediates.ob

    h_bb = -1.00 * einsum('ab,ji->jbia', identity_bb[vb, vb], f_bb[ob, ob])
    h_bb += 1.00 * einsum('ij,ab->jbia', identity_bb[ob, ob], f_bb[vb, vb])
    h_bb += -1.00 * einsum('ba,ij->jbia', identity_bb[vb, vb], f_bb[ob, ob])
    h_bb += 1.00 * einsum('ji,ba->jbia', identity_bb[ob, ob], f_bb[vb, vb])
    h_bb += -1.00 * einsum('jiab->jbia', g_bbbb[ob, ob, vb, vb])
    h_bb += 1.00 * einsum('jabi->jbia', g_bbbb[ob, vb, vb, ob])
    h_bb += 1.00 * einsum('ibaj->jbia', g_bbbb[ob, vb, vb, ob])
    h_bb += -1.00 * einsum('abji->jbia', g_bbbb[vb, vb, ob, ob])
    return h_bb


def build_complete_matrix(intermediates: Intermediates):
    nmo = intermediates.nmo
    noa = intermediates.noa
    nob = intermediates.nob

    h_aa = build_h_aa(intermediates)
    h_ab = build_h_ab(intermediates)
    h_ba = build_h_ba(intermediates)
    h_bb = build_h_bb(intermediates)

    # number of virtual orbital with spin up (ap) or down (bown)
    nva = nmo - noa
    nvb = nmo - nob

    # reshape matrices
    h_aa = h_aa.reshape(noa*nva, noa*nva)
    h_ab = h_ab.reshape(noa*nva, nob*nvb)
    h_ba = h_ba.reshape(nob*nvb, noa*nva)
    h_bb = h_bb.reshape(nob*nvb, nob*nvb)

    # pack into super matrix
    h = np.block([[h_aa, h_ab], [h_ba, h_bb]])
    return h
