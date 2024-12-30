import numpy as _np
from numba import njit

@njit()
def ACAPP_numba(rows, cols, info, numba_assembler, epsilon=0.1, dtype=_np.complex128):
    '''Standard ACA with partial pivoting'''
    m, n = len(rows), len(cols)
    mask_row = _np.zeros(n, dtype=_np.bool_) # OJO
    mask_col = _np.zeros(m, dtype=_np.bool_) # OJO
    k = 0
    i_star = 0

    u_vectors = []
    v_vectors = []

    # Stopping criterion:
    sum_uv_norm_square = 0
    
    while True:
        # Row of original matrix:
        R_row = numba_assembler(rows[i_star:i_star+1], cols, info, dtype).flatten()
        mask_col[i_star] = True

        aux = _np.zeros_like(R_row) # OJO: no deja =0 directamente
        # if k > 0:
        #     aux = _np.array(u_vectors)[:, i_star].T @ _np.array(v_vectors)
        for l in range(k): # OJO: lo de arriba no le gusta
            aux += u_vectors[l][i_star] * v_vectors[l]
        R_row -= aux

        R_row_abs = _np.abs(R_row)
        R_row_abs[mask_row] = -1 
        j_star = _np.argmax(R_row_abs)
        assert R_row_abs[j_star] != -1, "-1 found in R_row_abs"
        if R_row_abs[j_star] == -1: # En teoría esto nunca debería ocurrir (por la condición de salida más abajo)
            delta = 0
        else:
            delta = R_row[j_star]

        if delta == 0:
            if _np.sum(mask_col) == m: # OJO: acá tenía min(m,n), pero eso no es correcto
                if len(u_vectors) == 0:
                    if m + n <= m * n:
                        u_vectors.append(_np.zeros(m, dtype=R_row.dtype)) # 1xm
                        v_vectors.append(_np.zeros(n, dtype=R_row.dtype)) # 1xn
                    else: # En caso de que la compresión use más espacio que el bloque completo
                        # # Versión retornando listas:
                        # v_vectors.clear()
                        # matrix = numba_assembler(rows, cols, info, dtype)
                        # u_vectors = [matrix[i,:].flatten() for i in range(len(matrix))]
                        # return u_vectors, v_vectors

                        # Versión retornando arrays:
                        array_u = numba_assembler(rows, cols, info, dtype)
                        array_v = _np.array([()], dtype=R_row.dtype)
                        return array_u, array_v
                break
            i_star = _np.random.choice(_np.arange(m)[~mask_col]) # Quizás se puede hacer un arange afuera en vez de crear uno cada vez acá?
            continue
        else:
            v = R_row / delta

            # Column of original matrix:
            u = numba_assembler(rows, cols[j_star:j_star+1], info, dtype).flatten()
            mask_row[j_star] = True

            aux = _np.zeros_like(u) # OJO: no deja =0 directamente
            # if k>0:
            #     aux = _np.asarray(v_vectors)[:, j_star].T @ _np.asarray(u_vectors)
            for l in range(k): # OJO: lo de arriba no le gusta
                aux += v_vectors[l][j_star] * u_vectors[l]
            u -= aux

            k += 1
            u_vectors.append(u)
            v_vectors.append(v)
        u_abs = _np.abs(u_vectors[-1])
        u_abs[mask_col] = -1 # OJO: alternativa al anterior
        i_star = _np.argmax(u_abs)

        # Check if compression is still viable:
        if len(u_vectors) * m + len(v_vectors) * n > m * n:
            # # Versión retornando listas:
            # v_vectors.clear()
            # matrix = numba_assembler(rows, cols, info, dtype)
            # u_vectors = [matrix[i,:].flatten() for i in range(len(matrix))]
            # return u_vectors, v_vectors

            # Versión retornando arrays:
            array_u = numba_assembler(rows, cols, info, dtype)
            array_v = _np.array([()], dtype=R_row.dtype)
            return array_u, array_v

        # Stopping criterion:
        u = u_vectors[-1]
        v = v_vectors[-1]
        norm_u = _np.linalg.norm(u)
        norm_v = _np.linalg.norm(v)
        sum_uv_norm_square += norm_u**2 * norm_v**2
        # NEW PAPER ============
        aux = 0
        for i in range(k - 1):
            aux += _np.vdot(u_vectors[i], u) * _np.vdot(v_vectors[i], v)
        sum_uv_norm_square += 2 * _np.real(aux)
        # ============ NEW PAPER
        error_rel = norm_u * norm_v / _np.sqrt(sum_uv_norm_square)
        if error_rel <= epsilon or _np.sum(mask_col) == m or _np.sum(mask_row) == n: # Epsilon reached or rank completed (checked entire matrix)
            # print("Reached Epsilon")
            break

    # # Versión retornando listas:
    # return u_vectors, v_vectors # Con este me tengo que acordar de transformarlos a _np.arrays cuando los reciba en el árbol!

    # Versión retornando arrays:
    array_u = _np.empty((len(u_vectors), m), dtype=R_row.dtype)
    array_v = _np.empty((len(v_vectors), n), dtype=R_row.dtype)
    for i in range(len(u_vectors)):
        array_u[i] = u_vectors[i]
        array_v[i] = v_vectors[i]
    return array_u, array_v