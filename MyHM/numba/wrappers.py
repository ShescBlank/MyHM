import numpy as np
from numba import njit, int64, float64, prange, typeof, int32
from MyHM.numba.aca import ACAPP_numba

# # Versión separada en dos funciones:
# def wrapper_compression_numba(list_dofs, info_class, numba_assembler, dtype):
    
#     @njit((typeof(list_dofs), typeof(list_dofs), info_class.class_type.instance_type, typeof(numba_assembler), typeof(dtype)), parallel=True, cache=True)
#     def parallel_compression_nadm_numba(nodes_rows, nodes_cols, info, numba_assembler, dtype):
#         # Las matrices singulares se suman afuera
#         results = [np.empty((0,0), dtype=dtype)] * len(nodes_rows)
#         # for index in prange(len(nodes_rows)):
#         for index in prange(-len(nodes_rows), 0): # TODO: Usar esto para evitar el warning?
#             rows = nodes_rows[index]
#             cols = nodes_cols[index]
#             results[index] = numba_assembler(rows, cols, info, dtype)
#         return results

#     @njit((typeof(list_dofs), typeof(list_dofs), info_class.class_type.instance_type, typeof(numba_assembler), float64, typeof(dtype)), parallel=True, cache=True)
#     def parallel_compression_adm_numba(nodes_rows, nodes_cols, info, numba_assembler, epsilon, dtype):
#         results = [(np.empty((0,0), dtype=dtype), np.empty((0,0), dtype=dtype))] * len(nodes_rows)
#         # for index in prange(len(nodes_rows)):
#         for index in prange(-len(nodes_rows), 0): # TODO: Usar esto para evitar el warning?
#             rows = nodes_rows[index]
#             cols = nodes_cols[index]
#             results[index] = ACAPP_numba(rows, cols, info, numba_assembler, epsilon, dtype)
#         return results

#     return parallel_compression_nadm_numba, parallel_compression_adm_numba

# Versión todo junto:
def wrapper_compression_numba(list_dofs, info_class, numba_assembler, dtype):

    @njit((typeof(list_dofs), typeof(list_dofs), info_class.class_type.instance_type, typeof(numba_assembler), int64, float64, typeof(dtype)), parallel=True, cache=True)
    def parallel_compression_numba(nodes_rows, nodes_cols, info, numba_assembler, n_nadm, epsilon, dtype):
        n_adm = len(nodes_rows) - n_nadm
        results_nadm = [np.empty((0,0), dtype=dtype)] * (n_nadm)
        results_adm = [(np.empty((0,0), dtype=dtype), np.empty((0,0), dtype=dtype))] * (n_adm)

        # for index in prange(len(nodes_rows)):
        for index in prange(-n_nadm, n_adm): # Uso esto para evitar un warning
            rows = nodes_rows[index + n_nadm]
            cols = nodes_cols[index + n_nadm]
            # if index < n_nadm:
            if index < 0: # Uso esto para evitar un warning
                results_nadm[index] = numba_assembler(rows, cols, info, dtype)
            else:
                # results_adm[index - n_nadm] = ACAPP_numba(rows, cols, info, numba_assembler, epsilon, dtype)
                results_adm[index] = ACAPP_numba(rows, cols, info, numba_assembler, epsilon, dtype)
        return results_nadm, results_adm

    return parallel_compression_numba