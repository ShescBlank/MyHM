from numba import njit
import numpy as _np

from numba import prange, get_thread_id

# TODO: fastmath=True in numba_dot and compression?

# @njit(cache=True)
@njit(cache=True, parallel=True)
def numba_dot(adm_leaves, nadm_leaves, b, length, dtype, n_threads):
    # =============================================================================================================
    # # Single thread version:
    # result_vector = _np.zeros(length, dtype=dtype)
    # for leaf_id in range(-len(nadm_leaves), len(adm_leaves)):
    #     if leaf_id < 0:
    #         leaf = nadm_leaves[leaf_id]
    #     else:
    #         leaf = adm_leaves[leaf_id]
    #     if leaf.v_vectors is None:
    #         result_vector[leaf.rows] += leaf.matrix_block @ b[leaf.cols]
    #     else:
    #         result_vector[leaf.rows] += leaf.u_vectors.T @ (leaf.v_vectors @ b[leaf.cols])
    # return result_vector

    # =============================================================================================================
    # # Parallel version 1:
    # n_leaves = len(nadm_leaves) + len(adm_leaves)
    # result_vector = _np.zeros((n_threads, length), dtype=dtype)
    # for t in prange(n_threads):
    #     # Chunks:
    #     if t < n_leaves % n_threads:
    #         size = n_leaves // n_threads + 1
    #         first = t * size
    #     else:
    #         size = n_leaves // n_threads
    #         first = t * size + n_leaves % n_threads
        
    #     # Dot:
    #     for leaf_id in range(first, first + size):
    #         if leaf_id < len(nadm_leaves):
    #             leaf = nadm_leaves[leaf_id]
    #         else:
    #             leaf = adm_leaves[leaf_id - len(nadm_leaves)]
    #         if leaf.v_vectors is None:
    #             result_vector[t, leaf.rows] += leaf.matrix_block @ b[leaf.cols]
    #         else:
    #             result_vector[t, leaf.rows] += leaf.u_vectors.T @ (leaf.v_vectors @ b[leaf.cols])
    # return result_vector.sum(axis=0)

    # =============================================================================================================
    # # Parallel version 2:
    # result_vector = _np.zeros((n_threads, length), dtype=dtype)
    # for leaf_id in prange(-len(nadm_leaves), len(adm_leaves)):
    #     thread_id = get_thread_id()
    #     if leaf_id < 0:
    #         leaf = nadm_leaves[leaf_id]
    #     else:
    #         leaf = adm_leaves[leaf_id]
    #     if leaf.v_vectors is None:
    #         result_vector[thread_id, leaf.rows] += leaf.matrix_block @ b[leaf.cols]
    #     else:
    #         result_vector[thread_id, leaf.rows] += leaf.u_vectors.T @ (leaf.v_vectors @ b[leaf.cols])
    # return result_vector.sum(axis=0)

    # =============================================================================================================
    # Parallel version 3:
    # The previous version had problems with oversubscription and the creation of too many intermediate results.
    # This version uses for loops to calculate matvecs and the results are calculated in-place.
    result_vector = _np.zeros((n_threads, length), dtype=dtype)
    for leaf_id in prange(-len(nadm_leaves), len(adm_leaves)):
        thread_id = get_thread_id()
        if leaf_id < 0:
            leaf = nadm_leaves[leaf_id]
        else:
            leaf = adm_leaves[leaf_id]
        if leaf.v_vectors is None:
            # Matrix_block x vector:
            # result_vector[thread_id, leaf.rows] += leaf.matrix_block @ b[leaf.cols]
            m, n = leaf.matrix_block.shape
            for i in range(m):
                for j in range(n):
                    result_vector[thread_id, leaf.rows[i]] += leaf.matrix_block[i, j] * b[leaf.cols[j]]
        else:
            # Low-rank_approximation x vector:
            # result_vector[thread_id, leaf.rows] += leaf.u_vectors.T @ (leaf.v_vectors @ b[leaf.cols])
            k, m = leaf.u_vectors.shape
            n = leaf.v_vectors.shape[1]
            for l in range(k):
                aux = 0
                for j in range(n):
                    aux += leaf.v_vectors[l, j] * b[leaf.cols[j]]
                for i in range(m):
                    result_vector[thread_id, leaf.rows[i]] += leaf.u_vectors[l, i] * aux

    # return result_vector.sum(axis=0)
    for i in prange(length):
        for t in range(1, n_threads):
            result_vector[0, i] += result_vector[t, i]
    return result_vector[0]

    # =============================================================================================================