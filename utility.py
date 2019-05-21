import numpy as np


def check_matrix_invertible(matrix):
    """Check if a matrix is invertible, returns True or False."""
    return matrix.shape[0] == matrix.shape[1] and np.linalg.matrix_rank(matrix) == matrix.shape[0]
