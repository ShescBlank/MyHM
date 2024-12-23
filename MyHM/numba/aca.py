import numpy as np
from numba import njit

@njit()
def ACAPP_numba(rows, cols, info, numba_assembler, epsilon=0.1, dtype=np.complex128):
    m, n = len(rows), len(cols)
    mask_row = np.zeros(n, dtype=np.bool_) # OJO
    mask_col = np.zeros(m, dtype=np.bool_) # OJO
    I = []
    J = []
    k = 0
    i_star = 0

    u_vectors = []
    v_vectors = []

    # Stopping criterion:
    sum_uv_norm_square = 0
    
    while True:
        # Row of original matrix:
        if mask_col[i_star] != True and np.sum(~mask_row) > 0:
            mask_col[i_star] = True
        R_row = numba_assembler(rows[i_star:i_star+1], cols, info, dtype).flatten()

        aux = np.zeros_like(R_row) # OJO: no deja =0 directamente
        # if k > 0:
        #     aux = np.array(u_vectors)[:, i_star].T @ np.array(v_vectors)
        for l in range(k): # OJO: lo de arriba no le gusta
            aux += u_vectors[l][i_star] * v_vectors[l]

        R_row -= aux
        R_row_copy = np.copy(R_row)
        # R_row_copy[np.array(J)] = 0 # OJO: no me deja sin np.array
        R_row_copy[mask_row] = 0 # OJO: alternativa al anterior
        j_star = np.argmax(np.abs(R_row_copy))
        delta = R_row_copy[j_star]

        # if delta == 0 or abs(delta)/first_abs_delta <= 1e-15: # Agregamos un pequeño margen (puede ser complejo)
        if delta == 0:
            # if np.sum(mask_col) >= min(m, n):
            if np.sum(mask_col) == m: # # OJO: acá tenía min(m,n), pero eso no es correcto
                if len(u_vectors) == 0:
                    if m + n <= m * n:
                        u_vectors.append(np.zeros(m, dtype=R_row.dtype)) # 1xm
                        v_vectors.append(np.zeros(n, dtype=R_row.dtype)) # 1xn
                    else: # En caso de que la compresión use más espacio que el bloque completo
                        # # Versión retornando listas:
                        # v_vectors.clear()
                        # matrix = numba_assembler(rows, cols, info, dtype)
                        # u_vectors = [matrix[i,:].flatten() for i in range(len(matrix))]
                        # return u_vectors, v_vectors

                        # Versión retornando arrays:
                        array_u = numba_assembler(rows, cols, info, dtype)
                        # array_v = np.array([[0]], dtype=R_row.dtype)
                        array_v = np.array([()], dtype=R_row.dtype)
                        return array_u, array_v
                break
            i_star = np.random.choice(np.arange(m)[~mask_col])
            continue
        else:
            v = R_row / delta

            # Column of original matrix:
            if mask_row[j_star] != True and np.sum(~mask_col) > 0:
                mask_row[j_star] = True
            u = numba_assembler(rows, cols[j_star:j_star+1], info, dtype).flatten()

            aux = np.zeros_like(u) # OJO: no deja =0 directamente
            # if k>0:
            #     aux = np.asarray(v_vectors)[:, j_star].T @ np.asarray(u_vectors)
            for l in range(k): # OJO: lo de arriba no le gusta
                aux += v_vectors[l][j_star] * u_vectors[l]

            u -= aux
            k += 1
            u_vectors.append(u)
            v_vectors.append(v)
        I.append(i_star)
        J.append(j_star)
        u_copy = np.copy(u_vectors[-1])
        # u_copy[np.array(I)] = 0 # OJO: no me deja sin np.array
        u_copy[mask_col] = 0 # OJO: alternativa al anterior
        i_star = np.argmax(np.abs(u_copy))

        # # Check if compression is still viable:
        if len(u_vectors) * len(u_vectors[0]) + len(v_vectors) * len(v_vectors[0]) > m * n:
            # # Versión retornando listas:
            # v_vectors.clear()
            # matrix = numba_assembler(rows, cols, info, dtype)
            # u_vectors = [matrix[i,:].flatten() for i in range(len(matrix))]
            # return u_vectors, v_vectors

            # Versión retornando arrays:
            array_u = numba_assembler(rows, cols, info, dtype)
            # array_v = np.array([[0]], dtype=R_row.dtype)
            array_v = np.array([()], dtype=R_row.dtype)
            return array_u, array_v

        # Stopping criterion:
        u = u_vectors[-1]
        v = v_vectors[-1]
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        sum_uv_norm_square += norm_u**2 * norm_v**2
        error_rel = norm_u * norm_v / np.sqrt(sum_uv_norm_square)
        if error_rel <= epsilon or len(u_vectors) == min(m, n): # Epsilon reached or rank completed
            # print("Reached Epsilon")
            break

    # # Versión retornando listas:
    # return u_vectors, v_vectors # Así que me tengo que acordar de transformarlos a np.arrays cuando los reciba en el árbol!

    # Versión retornando arrays:
    array_u = np.empty((len(u_vectors), m), dtype=R_row.dtype)
    array_v = np.empty((len(v_vectors), n), dtype=R_row.dtype)
    for i in range(len(u_vectors)):
        array_u[i] = u_vectors[i]
        array_v[i] = v_vectors[i]
    return array_u, array_v