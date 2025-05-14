import numpy as np
from numpy import einsum
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator
from typing import Any, TypeVar

MatmulLike = TypeVar('MatmulLike', bound=LinearOperator, contravariant=True)


class MockOrbitalHessianAction(LinearOperator):
    """ GMRES helper. Calculates the result of `orbital_hessian @ vector`
    A test class. Under the hood it stores the matrix and returns the regular
    `matrix@vector`.
    """

    def __init__(self: Any, matrix: NDArray) -> None:
        """ The point is to NOT store the whole matrix.
        But here for testing, I will still keep it. """
        self.matrix = matrix
        self.shape = matrix.shape
        self.dtype = matrix.dtype

    def _matvec(self, x: NDArray):
        """ This is where the implementation of mat@vec should go. """
        return self.matrix @ x.reshape(-1, 1)


class OrbitalHessianAction(LinearOperator):
    """ GMRES helper. Calculates the result of `orbital_hessian @ vector` """

    def __init__(
        self: Any,
        nmo: int,
        noa: int,
        nob: int,
        identity_aa: NDArray,
        identity_bb: NDArray,
        va: slice,
        oa: slice,
        vb: slice,
        ob: slice,
        fock_aa: NDArray,
        fock_bb: NDArray,
        g_aaaa: NDArray,
        g_abab: NDArray,
        g_bbbb: NDArray,
        **_,
    ) -> None:
        # These are needed for the action
        self.n_occuped_up = noa
        self.n_valence_up = nmo - noa
        self.n_occuped_down = nob
        self.n_valence_down = nmo - nob
        self.identity_aa = identity_aa
        self.identity_bb = identity_bb
        self.valence_slice_up = va
        self.occupied_slice_up = oa
        self.valence_slice_down = vb
        self.occupied_slice_down = ob
        self.fock_aa = fock_aa
        self.fock_bb = fock_bb
        self.g_aaaa = g_aaaa
        self.g_abab = g_abab
        self.g_bbbb = g_bbbb
        """ Scipy needs these. """
        self.dtype = fock_aa.dtype
        dim = noa * (nmo-noa) + nob * (nmo-nob)
        self.shape = (dim, dim)

    def _matuu_times_up(self, up: NDArray):
        id_aa = self.identity_aa
        f_aa = self.fock_aa
        va = self.valence_slice_up
        oa = self.occupied_slice_up
        g_aaaa = self.g_aaaa

        out = -1.00 * einsum(
            'ab,ji,ia->jb', id_aa[va, va], f_aa[oa, oa], up
        )
        out += 1.00 * einsum(
            'ij,ab,ia->jb', id_aa[oa, oa], f_aa[va, va], up
        )
        out += -1.00 * einsum(
            'ba,ij,ia->jb', id_aa[va, va], f_aa[oa, oa], up
        )
        out += 1.00 * einsum(
            'ji,ba,ia->jb', id_aa[oa, oa], f_aa[va, va], up
        )
        out += -1.00 * einsum(
            'jiab,ia->jb', g_aaaa[oa, oa, va, va], up
        )
        out += 1.00 * einsum(
            'jabi,ia->jb', g_aaaa[oa, va, va, oa], up
        )
        out += 1.00 * einsum(
            'ibaj,ia->jb', g_aaaa[oa, va, va, oa], up
        )
        out += -1.00 * einsum(
            'abji,ia->jb', g_aaaa[va, va, oa, oa], up
        )
        return out

    def _matud_times_down(self, down: NDArray) -> NDArray:
        g_abab = self.g_abab
        oa = self.occupied_slice_up
        ob = self.occupied_slice_down
        va = self.valence_slice_up
        vb = self.valence_slice_down
        out = 1.00 * einsum('jiba,ia->jb', g_abab[oa, ob, va, vb], down)
        out += 1.00 * einsum('jabi,ia->jb', g_abab[oa, vb, va, ob], down)
        out += 1.00 * einsum('bija,ia->jb', g_abab[va, ob, oa, vb], down)
        out += 1.00 * einsum('baji,ia->jb', g_abab[va, vb, oa, ob], down)
        return out

    def _matdu_times_up(self, up: NDArray) -> NDArray:
        g_abab = self.g_abab
        oa = self.occupied_slice_up
        va = self.valence_slice_up
        ob = self.occupied_slice_down
        vb = self.valence_slice_down

        out = 1.00 * einsum('ijab,ia->jb', g_abab[oa, ob, va, vb], up)
        out += 1.00 * einsum('ajib,ia->jb', g_abab[va, ob, oa, vb], up)
        out += 1.00 * einsum('ibaj,ia->jb', g_abab[oa, vb, va, ob], up)
        out += 1.00 * einsum('abij,ia->jb', g_abab[va, vb, oa, ob], up)

        return out

    def _matdd_times_down(self, down: NDArray) -> NDArray:
        id_bb = self.identity_bb
        f_bb = self.fock_bb
        g_bbbb = self.g_bbbb

        vb = self.valence_slice_down
        ob = self.occupied_slice_down

        out = -1.00 * einsum(
            'ab,ji,ia->jb', id_bb[vb, vb], f_bb[ob, ob], down
        )
        out += 1.00 * einsum(
            'ij,ab,ia->jb', id_bb[ob, ob], f_bb[vb, vb], down
        )
        out += -1.00 * einsum(
            'ba,ij,ia->jb', id_bb[vb, vb], f_bb[ob, ob], down
        )
        out += 1.00 * einsum(
            'ji,ba,ia->jb', id_bb[ob, ob], f_bb[vb, vb], down
        )
        out += -1.00 * einsum(
            'jiab,ia->jb', g_bbbb[ob, ob, vb, vb], down
        )
        out += 1.00 * einsum(
            'jabi,ia->jb', g_bbbb[ob, vb, vb, ob], down
        )
        out += 1.00 * einsum(
            'ibaj,ia->jb', g_bbbb[ob, vb, vb, ob], down
        )
        out += -1.00 * einsum(
            'abji,ia->jb', g_bbbb[vb, vb, ob, ob], down
        )

        return out

    def _matvec(self, x: NDArray):
        """ This is where the implementation of mat@vec should go. """

        dim_up = self.n_occuped_up * self.n_valence_up
        up = x[:dim_up].reshape(self.n_occuped_up, self.n_valence_up)
        dim_down = self.n_occuped_down * self.n_valence_down
        down = x[-dim_down:].reshape(self.n_occuped_down, self.n_valence_down)

        out_up = self._matuu_times_up(up) + self._matud_times_down(down)
        out_down = self._matdu_times_up(up) + self._matdd_times_down(down)

        out = np.hstack((out_up.flatten(), out_down.flatten()))
        return out
