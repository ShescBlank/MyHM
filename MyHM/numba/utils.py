from numba import njit
import numpy as _np

from numba import prange

# @njit(cache=True)
@njit(cache=True, parallel=True)
def numba_dot(adm_leaves, nadm_leaves, b, length, dtype, n_threads):
    # Single thread version:
    # result_vector = _np.zeros(length, dtype=dtype)
    # for i in range(-len(nadm_leaves), len(adm_leaves)):
    #     if i < 0:
    #         leaf = nadm_leaves[i]
    #     else:
    #         leaf = adm_leaves[i]
    #     if leaf.v_vectors is None:
    #         result_vector[leaf.rows] += leaf.matrix_block @ b[leaf.cols]
    #     else:
    #         result_vector[leaf.rows] += leaf.u_vectors.T @ (leaf.v_vectors @ b[leaf.cols])
    # return result_vector

    # Parallel version:
    n_leaves = len(nadm_leaves) + len(adm_leaves)
    result_vector = _np.zeros((n_threads, length), dtype=dtype)
    for t in prange(n_threads):
        # Chunks:
        if t < n_leaves % n_threads:
            size = n_leaves // n_threads + 1
            first = t * size
        else:
            size = n_leaves // n_threads
            first = t * size + n_leaves % n_threads
        
        # Dot:
        for i in range(first, first + size):
            if i < len(nadm_leaves):
                leaf = nadm_leaves[i]
            else:
                leaf = adm_leaves[i - len(nadm_leaves)]
            if leaf.v_vectors is None:
                result_vector[t, leaf.rows] += leaf.matrix_block @ b[leaf.cols]
            else:
                result_vector[t, leaf.rows] += leaf.u_vectors.T @ (leaf.v_vectors @ b[leaf.cols])
    return result_vector.sum(axis=0)
