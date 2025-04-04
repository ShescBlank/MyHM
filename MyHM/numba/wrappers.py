import numpy as np
from numba import njit, int64, float64, prange, typeof, int32

from collections import OrderedDict
from numba.experimental import jitclass
from numba import optional, from_dtype

def wrapper_numba_node3d(np_int_rows_dtype, np_int_cols_dtype, np_values_dtype):
    spec = OrderedDict()
    spec['rows'] = from_dtype(np_int_rows_dtype)[:]
    spec['cols'] = from_dtype(np_int_cols_dtype)[:]
    spec['matrix_block'] = optional(from_dtype(np_values_dtype)[:,::1])
    spec['u_vectors'] = optional(from_dtype(np_values_dtype)[:,::1])
    spec['v_vectors'] = optional(from_dtype(np_values_dtype)[:,::1])

    @jitclass(spec)
    class NumbaNode3D:
        def __init__(self, rows, cols):
            self.rows = rows
            self.cols = cols
            self.matrix_block = None
            self.u_vectors = None
            self.v_vectors = None
    
    return NumbaNode3D

def wrapper_compression_numba_debug(nodes_rows, nodes_cols, info, numba_assembler, numba_compressor, dtype):

    @njit((typeof(nodes_rows), typeof(nodes_cols), typeof(info), typeof(numba_assembler), typeof(numba_compressor), int64, float64, typeof(dtype)), parallel=True, cache=True)
    def parallel_compression_numba(nodes_rows, nodes_cols, info, numba_assembler, numba_compressor, n_nadm, epsilon, dtype):
        # ''' Version without NumbaNodes '''
        
        # n_adm = len(nodes_rows) - n_nadm
        # results_nadm = [np.empty((0,0), dtype=dtype)] * (n_nadm)
        # results_adm = [(np.empty((0,0), dtype=dtype), np.empty((0,0), dtype=dtype))] * (n_adm)

        # # for index in prange(len(nodes_rows)):
        # for index in prange(-n_nadm, n_adm): # Uso esto para evitar un warning
        #     rows = nodes_rows[index + n_nadm]
        #     cols = nodes_cols[index + n_nadm]
        #     # if index < n_nadm:
        #     if index < 0: # Uso esto para evitar un warning
        #         results_nadm[index] = numba_assembler(rows, cols, info, dtype)
        #     else:
        #         # results_adm[index - n_nadm] = numba_compressor(rows, cols, info, numba_assembler, epsilon, dtype)
        #         results_adm[index] = numba_compressor(rows, cols, info, numba_assembler, epsilon, dtype)
        # return results_nadm, results_adm

        """
        DEBUG VERSION: Calculates the compression and then forgets it, it only saves the space used.
        """
        # TODO: Modificado para calcular fácilemente la complejidad de almacenamiento
        n_adm = len(nodes_rows) - n_nadm
        results_nadm = np.zeros(1, dtype=np.int64)
        results_adm = np.zeros(n_adm, dtype=np.int64)

        # for index in prange(len(nodes_rows)):
        for index in prange(-n_nadm, n_adm): # Uso esto para evitar un warning
            rows = nodes_rows[index + n_nadm]
            cols = nodes_cols[index + n_nadm]
            if index >= 0:
                u, v = numba_compressor(rows, cols, info, numba_assembler, epsilon, dtype)
                if v.shape[1] == 0:
                    results_adm[index] = u.shape[0] * u.shape[1]
                else:
                    results_adm[index] = (u.shape[0] * u.shape[1]) + (v.shape[0] * v.shape[1])
        return results_nadm, results_adm

    return parallel_compression_numba

def wrapper_compression_numba(leaves, info, numba_assembler, numba_compressor, dtype):
    """ 
    Wrapper to compile and return the compression function.
    Parallelised with Numba.
    """

    @njit((typeof(leaves), typeof(leaves), typeof(info), typeof(numba_assembler), typeof(numba_compressor), float64, typeof(dtype)), parallel=True, cache=True)
    def parallel_compression_numba(adm_leaves, nadm_leaves, info, numba_assembler, numba_compressor, epsilon, dtype):
        n_adm, n_nadm = len(adm_leaves), len(nadm_leaves)

        for index in prange(-n_nadm, n_adm):
            if index < 0:
                nadm_leaves[index].matrix_block = numba_assembler(nadm_leaves[index].rows, nadm_leaves[index].cols, info, dtype)
            else:
                adm_leaves[index].u_vectors, adm_leaves[index].v_vectors = numba_compressor(
                    adm_leaves[index].rows, adm_leaves[index].cols, info, numba_assembler, epsilon, dtype
                )
                if adm_leaves[index].v_vectors.shape[1] == 0:
                    adm_leaves[index].matrix_block = adm_leaves[index].u_vectors
                    adm_leaves[index].u_vectors = None
                    adm_leaves[index].v_vectors = None

    return parallel_compression_numba
