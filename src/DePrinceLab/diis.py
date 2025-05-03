""" Direct Inversion of the Iterative Subspace (DIIS) with the Pulay's error
https://doi.org/10.1002/jcc.540030413 """

import numpy as np
from numpy.typing import NDArray

class Solver_DIIS:

    def __init__(self, max_stored_vecs: int, start_iter: int = 2):
        """
        DIIS extrapolation class

        :param max_n_vec: the maximum number of vectors to store
        :param start_iter: when to start the extrapolation
        """
        
        self.max_stored = max_stored_vecs
        self.start_iter = start_iter
        self.iteration = 0
        self.soln_vector = []
        self.error_vector = []

    def extrapolate(self, soln_vector, error_vector) -> NDArray:
        """
        perform DIIS extrapolation

        :param soln_vector: a flattened solution vector for the current DIIS
        iteration
        :param error_vector: a flattened error vector for the current DIIS
        iteration
        """

        # do not extrapolate until we reach start_iter
        if self.iteration < self.start_iter :
            self.iteration += 1
            return soln_vector
        self.iteration += 1
            
        # add current solution/error vectors to lists of solution/error
        # vectors
        self.soln_vector.append(soln_vector)
        self.error_vector.append(error_vector)

        # check if we need to remove old vectors
        if len(self.soln_vector) > self.max_stored:
            self.soln_vector.pop(0)
            self.error_vector.pop(0)

        # build B matrix
        dim = len(self.soln_vector) + 1
        B = np.zeros([dim, dim])
        for row in range(dim-1):
            for col in range(row, dim-1):
                dot = np.dot(self.error_vector[row], self.error_vector[col])
                B[row, col] = dot
                B[col, row] = dot
            B[row, dim-1] = -1
            B[dim-1, row] = -1

        # right-hand side of DIIS equation [0, 0, ..., -1]
        rhs = np.zeros([dim], dtype=np.float64)
        rhs[-1] = -1.0

        # solve the DIIS equation
        c = np.linalg.solve(B, rhs)

        # extrapolate solution
        new_soln_vector = np.zeros(len(self.soln_vector[0]), dtype = np.float64)
        for row in range (0, dim-1):
            new_soln_vector += c[row] * self.soln_vector[row]

        # return extrapolated solution
        return new_soln_vector
