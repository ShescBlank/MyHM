import numpy as _np
from scipy.sparse import lil_matrix

# https://www.epfl.ch/labs/anchp/wp-content/uploads/2018/10/lecture4-slides.pdf
# https://web.archive.org/web/20060903235945id_/http://esl.eng.ohio-state.edu/~csg/papers/79.pdf
def ACAPP(A, epsilon = 0.1, exact_error=False):
    # Saqué variables extras, saqué I,J, puse -1 en los abs, quité los copy y cambié una de las condiciones de salida
    # R = _np.copy(A)
    R = A
    m, n = R.shape
    mask_row = _np.zeros(n, dtype=bool)
    mask_col = _np.zeros(m, dtype=bool)
    k = 0
    i_star = 0

    u_vectors = []
    v_vectors = []

    # Stopping criterion:
    Fnorm_square = 0
    norm_A = _np.linalg.norm(A)
    
    while True:
        aux = 0
        # for l in range(k):
        #     aux += u_vectors[l][i_star] * v_vectors[l]
        if k > 0:
            aux = _np.asarray(u_vectors)[:, i_star].T @ _np.asarray(v_vectors)

        # Row of original matrix:
        R_row = R[i_star, :] - aux # Resto la aproximación que llevamos de la fila original
        mask_col[i_star] = True

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
                        u_vectors.append(_np.zeros(m, dtype=R.dtype)) # 1xm
                        v_vectors.append(_np.zeros(n, dtype=R.dtype)) # 1xn
                    else:
                        return A, None # Old version
                break
            i_star = _np.random.choice(_np.arange(m)[~mask_col])
            continue
        else:
            v = R_row / delta
            aux = 0
            # for l in range(k):
            #     aux += v_vectors[l][j_star] * u_vectors[l]
            if k>0:
                aux = _np.asarray(v_vectors)[:, j_star].T @ _np.asarray(u_vectors)

            # Column of original matrix:
            u = R[:, j_star] - aux # Old version
            mask_row[j_star] = True

            k += 1
            u_vectors.append(u)
            v_vectors.append(v)
        u_abs = _np.abs(u_vectors[-1])
        u_abs[mask_col] = -1
        i_star = _np.argmax(u_abs)

        # Check if compression is still viable:
        if len(u_vectors) * m + len(v_vectors) * n > m * n:
            return A, None # Old version

        # Stopping criterion:
        u = u_vectors[-1]
        v = v_vectors[-1]
        norm_u = _np.linalg.norm(u)
        norm_v = _np.linalg.norm(v)
        Fnorm_square += norm_u**2 * norm_v**2
        # NEW PAPER ============
        aux = 0
        for index in range(k - 1):
            aux += _np.vdot(u_vectors[index], u) * _np.vdot(v_vectors[index], v)
        Fnorm_square += 2 * _np.real(aux)
        # ============ NEW PAPER
        error_rel = norm_u * norm_v / _np.sqrt(Fnorm_square)
        condition = error_rel <= epsilon
        # error_rel2 = norm_u * norm_v / (_np.linalg.norm(u_vectors[0]) * _np.linalg.norm(v_vectors[0]))
        # error_rel3 = _np.linalg.norm(u) * _np.linalg.norm(v)
        # condition = error_rel <= epsilon and error_rel2 <= epsilon and error_rel3 <= epsilon
        if exact_error:
            error_real = _np.linalg.norm(A - _np.asarray(u_vectors).T @ _np.asarray(v_vectors)) / norm_A # Estimación perfecta
            condition = error_real <= epsilon
        if condition or _np.sum(mask_col) == m or _np.sum(mask_row) == n: # or len(u_vectors) == min(m, n): # Epsilon reached or rank completed (checked entire matrix)
            # print("Reached Epsilon")
            break

        # Otras formas de estimar: (se pueden agregar)
        # 1) Literatura original: norm_u * norm_v <= epsilon
        # 2) Literatura original: norm_u * norm_v / (_np.linalg.norm(u_vectors[0]) * _np.linalg.norm(v_vectors[0])) <= epsilon
        # 3) Literatura extra (al inicio del script):
        # for index in range(k - 1):
        #     Fnorm_square += 2 * _np.abs(_np.dot(u_vectors[index], u)) * _np.abs(_np.dot(v_vectors[index], v))

    # Esto es para encontrar bloques que estén ocasionando un error muy grande:
    # if _np.log10(_np.linalg.norm(A - _np.asarray(u_vectors).T @ _np.asarray(v_vectors)) / norm_A) - _np.log10(error_rel) >= 5:
    #     import matplotlib.pyplot as plt
    #     _np.save("A.npy", A)
    #     print("Saved A!")
    #     plt.spy(_np.abs(A))
    #     plt.show()

    return _np.array(u_vectors), _np.array(v_vectors)

def ACAPP_with_assembly(rows, cols, assembler, singular_matrix, epsilon = 0.1, verbose=False, dtype=_np.complex128):
    # rows = _np.asarray(rows)
    # cols = _np.asarray(cols)

    m, n = len(rows), len(cols)
    assembled_values = lil_matrix((m, n), dtype=dtype)
    mask_row = _np.zeros(n, dtype=bool)
    mask_col = _np.zeros(m, dtype=bool)
    k = 0
    i_star = 0

    u_vectors = []
    v_vectors = []

    # Stopping criterion:
    Fnorm_square = 0
    
    while True:
        aux = 0
        if k > 0:
            aux = _np.asarray(u_vectors)[:, i_star].T @ _np.asarray(v_vectors)
        
        # Row of original matrix:
        if mask_col[i_star] != True and _np.sum(~mask_row) > 0:
            assembled_values[i_star, ~mask_row] = assembler(rows[i_star:i_star+1], cols[~mask_row], dtype=dtype) # PIN
            # assembled_values[i_star:i_star+1, ~mask_row] = assembler(rows[i_star:i_star+1], cols[~mask_row], dtype=dtype)
            mask_col[i_star] = True
        R_row = assembled_values[i_star, :].toarray()
        # if singular_matrix is not None:
        #     R_row = _np.array(R_row + singular_matrix[rows[i_star], cols])

        R_row = R_row.flatten()
        R_row -= aux
        R_row_abs = _np.abs(R_row)
        R_row_abs[mask_row] = -1
        j_star = _np.argmax(R_row_abs)
        assert R_row_abs[j_star] != -1, "-1 found in R_row_abs"
        if R_row_abs[j_star] == -1:
            delta = 0
        else:
            delta = R_row[j_star]

        if delta == 0:
            if _np.sum(mask_col) == m: # OJO: acá tenía min(m,n), pero eso no es correcto
                if len(u_vectors) == 0:
                    if m + n <= m * n:
                        u_vectors.append(_np.zeros(m, dtype=R_row.dtype)) # 1xm
                        v_vectors.append(_np.zeros(n, dtype=R_row.dtype)) # 1xn
                    else:
                        matrix = assembled_values.toarray()
                        if _np.sum(~mask_col) > 0 and _np.sum(~mask_row) > 0:
                            meshgrid = _np.meshgrid(_np.arange(m)[~mask_col], _np.arange(n)[~mask_row], indexing="ij")
                            matrix[meshgrid[0], meshgrid[1]] = assembler(rows[~mask_col], cols[~mask_row], dtype=dtype)
                        return matrix, None
                break
            i_star = _np.random.choice(_np.arange(m)[~mask_col])
            continue
        else:
            v = R_row / delta
            aux = 0
            if k>0:
                aux = _np.asarray(v_vectors)[:, j_star].T @ _np.asarray(u_vectors)
            
            # Column of original matrix:
            if mask_row[j_star] != True and _np.sum(~mask_col) > 0:
                assembled_values[~mask_col, j_star] = assembler(rows[~mask_col], cols[j_star:j_star+1], dtype=dtype) # PIN
                # assembled_values[~mask_col, j_star:j_star+1] = assembler(rows[~mask_col], cols[j_star:j_star+1], dtype=dtype)
                mask_row[j_star] = True
            u = assembled_values[:, j_star].toarray()
            # if singular_matrix is not None:
                # u = _np.array(u + singular_matrix[rows, cols[j_star]])
            
            u = u.flatten()
            u -= aux
            k += 1
            u_vectors.append(u)
            v_vectors.append(v)
        u_abs = _np.abs(u_vectors[-1])
        u_abs[mask_col] = -1
        i_star = _np.argmax(u_abs)

        # Check if compression is still viable:
        if len(u_vectors) * m + len(v_vectors) * n > m * n:
            matrix = assembled_values.toarray()
            if _np.sum(~mask_col) > 0 and _np.sum(~mask_row) > 0:
                meshgrid = _np.meshgrid(_np.arange(m)[~mask_col], _np.arange(n)[~mask_row], indexing="ij")
                matrix[meshgrid[0], meshgrid[1]] = assembler(rows[~mask_col], cols[~mask_row], dtype=dtype)
            return matrix, None

        # Stopping criterion:
        u = u_vectors[-1]
        v = v_vectors[-1]
        norm_u = _np.linalg.norm(u)
        norm_v = _np.linalg.norm(v)
        Fnorm_square += norm_u**2 * norm_v**2
        # NEW PAPER ============
        aux = 0
        for index in range(k - 1):
            aux += _np.vdot(u_vectors[index], u) * _np.vdot(v_vectors[index], v)
        Fnorm_square += 2 * _np.real(aux)
        # ============ NEW PAPER
        error_rel = norm_u * norm_v / _np.sqrt(Fnorm_square)
        if error_rel <= epsilon or _np.sum(mask_col) == m or _np.sum(mask_row) == n: # Epsilon reached or rank completed (checked entire matrix)
            # print("Reached Epsilon")
            break

    if verbose:
        print(f"Finished in k={k} out of {m}")
        # print(f"Relative error between matrices: {_np.linalg.norm(A-_np.asarray(u_vectors).T @ _np.asarray(v_vectors)) / _np.linalg.norm(A)}")
    
    return _np.array(u_vectors), _np.array(v_vectors)


if __name__ == "__main__":
    from scipy.linalg import hilbert
    N = 144
    H = hilbert(N)
    u_vectors, v_vectors = ACAPP(H, epsilon=1e-10, verbose=True)
    print("Working...")