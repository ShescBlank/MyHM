import numpy as _np
from scipy.sparse import lil_matrix

# https://www.epfl.ch/labs/anchp/wp-content/uploads/2018/10/lecture4-slides.pdf
# https://web.archive.org/web/20060903235945id_/http://esl.eng.ohio-state.edu/~csg/papers/79.pdf
def ACAPP(A, epsilon = 0.1, verbose=False):
    R = _np.copy(A)
    m, n = R.shape
    assembled_values = lil_matrix((m, n), dtype=R.dtype)
    mask_row = _np.zeros(n, dtype=bool)
    mask_col = _np.zeros(m, dtype=bool)
    I = []
    J = []
    k = 0
    i_star = 0

    u_vectors = []
    v_vectors = []

    # Stopping criterion:
    sum_uv_norm_square = 0
    first_abs_delta = 0
    
    while True:
        aux = 0
        # for l in range(k):
        #     aux += u_vectors[l][i_star] * v_vectors[l]
        if k > 0:
            aux = _np.asarray(u_vectors)[:, i_star].T @ _np.asarray(v_vectors)

        # Row of original matrix:
        # R_row = R[i_star, :] - aux # Old version
        # mask_row[:] = False
        # mask_row[assembled_values[i_star, :].rows[0]] = True
        # if _np.sum(~mask_row) > 0:
        if mask_col[i_star] != True and _np.sum(~mask_row) > 0:
            assembled_values[i_star, ~mask_row] = R[i_star, ~mask_row]
            mask_col[i_star] = True
        R_row = assembled_values[i_star, :].toarray().flatten() - aux

        R_row_copy = _np.copy(R_row)
        R_row_copy[J] = 0
        j_star = _np.argmax(_np.abs(R_row_copy))
        delta = R_row_copy[j_star]
        if first_abs_delta == 0:
            first_abs_delta = abs(delta)

        # if delta == 0 or abs(delta)/first_abs_delta <= 1e-15: # Agregamos un pequeño margen (puede ser complejo)
        if delta == 0:
            if len(I) >= min(m, n): # Para que no siga agregando en caso de ya tener todos los índices
                # print("Not reached Epsilon")
                exit_code = 0
                break
            if len(u_vectors) == 0: # Edge case in which the first row chosen consists of zeros only
                i_star += 1
                if i_star == m: # Everything is zero
                    u_vectors.append(_np.zeros(m, dtype=R.dtype)) # 1xm
                    v_vectors.append(_np.zeros(n, dtype=R.dtype)) # 1xn
                    break
                continue
        else:
            v = R_row / delta
            aux = 0
            # for l in range(k):
            #     aux += v_vectors[l][j_star] * u_vectors[l]
            if k>0:
                aux = _np.asarray(v_vectors)[:, j_star].T @ _np.asarray(u_vectors)

            # Column of original matrix:
            # u = R[:, j_star] - aux # Old version
            # mask_col[:] = False
            # mask_col[assembled_values.T[j_star, :].rows[0]] = True
            # if _np.sum(~mask_col) > 0:
            if mask_row[j_star] != True and _np.sum(~mask_col) > 0:
                assembled_values[~mask_col, j_star] = R[~mask_col, j_star]
                mask_row[j_star] = True
            u = assembled_values[:, j_star].toarray().flatten() - aux

            k += 1
            u_vectors.append(u)
            v_vectors.append(v)
        I.append(i_star)
        J.append(j_star)
        u_copy = _np.copy(u_vectors[-1])
        u_copy[I] = 0
        i_star = _np.argmax(_np.abs(u_copy))

        # Stopping criterion:
        u = u_vectors[-1]
        v = v_vectors[-1]
        norm_u = _np.linalg.norm(u)
        norm_v = _np.linalg.norm(v)
        sum_uv_norm_square += norm_u**2 * norm_v**2
        error_rel = norm_u * norm_v / _np.sqrt(sum_uv_norm_square)
        if error_rel <= epsilon or len(u_vectors) == min(m, n): # Epsilon reached or rank completed
            # print("Reached Epsilon")
            exit_code = 1
            break

    if verbose:
        # print(f"Finished in k={k} out of {m}")
        # print(f"Relative error between matrices: {_np.linalg.norm(A-_np.asarray(u_vectors).T @ _np.asarray(v_vectors)) / _np.linalg.norm(A)}")
    
        print(f"Exit code: {exit_code}")
        if len(u_vectors) > min(m, n):
            print(">"*10)
            print(f"Diff: {len(u_vectors) - min(m, n)}")
            print(R.shape)
            print(len(I), I)
            u, c = _np.unique(I, return_counts=True)
            print(u[c > 1], c[c > 1])
            print(len(J), J)
            u, c = _np.unique(J, return_counts=True)
            print(u[c > 1], c[c > 1])
            print(_np.linalg.norm(A-_np.asarray(u_vectors)[0:min(m, n),:].T @ _np.asarray(v_vectors)[0:min(m, n),:]) / _np.linalg.norm(A))
            print(_np.linalg.norm(A-_np.asarray(u_vectors).T @ _np.asarray(v_vectors)) / _np.linalg.norm(A))
            print("<"*10)

    return _np.array(u_vectors), _np.array(v_vectors)

def ACAPP_with_assembly(rows, cols, boundary_operator, parameters, singular_matrix, epsilon = 0.1, verbose=False, dtype=_np.complex128):
    import MyHM
    if MyHM.ADM_option == 1:
        from MyHM.assembly import partial_dense_assembler as pda
    elif MyHM.ADM_option == 2:
        from MyHM.assembly import partial_dense_assembler2 as pda

    rows = _np.asarray(rows)
    cols = _np.asarray(cols)

    m, n = len(rows), len(cols)
    assembled_values = lil_matrix((m, n), dtype=dtype)
    mask_row = _np.zeros(n, dtype=bool)
    mask_col = _np.zeros(m, dtype=bool)
    I = []
    J = []
    k = 0
    i_star = 0

    u_vectors = []
    v_vectors = []

    # Stopping criterion:
    sum_uv_norm_square = 0
    first_abs_delta = 0
    
    while True:
        aux = 0
        if k > 0:
            aux = _np.asarray(u_vectors)[:, i_star].T @ _np.asarray(v_vectors)
        
        # Row of original matrix:
        # mask_row[:] = False
        # mask_row[assembled_values[i_star, :].rows[0]] = True
        # if len(cols[~mask_row]) > 0:
        if mask_col[i_star] != True and _np.sum(~mask_row) > 0:
            assembled_values[i_star, ~mask_row] = pda(boundary_operator.descriptor, boundary_operator.domain, boundary_operator.dual_to_range, parameters, [rows[i_star]], cols[~mask_row]) # PIN
            # assembled_values[i_star:i_star+1, ~mask_row] = pda(boundary_operator.descriptor, boundary_operator.domain, boundary_operator.dual_to_range, parameters, [rows[i_star]], cols[~mask_row])
            mask_col[i_star] = True
        R_row = assembled_values[i_star, :].toarray()
        # R_row = _np.array(R_row + singular_matrix[rows[i_star], cols])

        R_row = R_row.flatten()
        R_row -= aux
        R_row_copy = _np.copy(R_row)
        R_row_copy[J] = 0
        j_star = _np.argmax(_np.abs(R_row_copy))
        delta = R_row_copy[j_star]
        if first_abs_delta == 0:
            first_abs_delta = abs(delta)

        # if delta == 0 or abs(delta)/first_abs_delta <= 1e-15: # Agregamos un pequeño margen (puede ser complejo)
        if delta == 0:
            if len(I) >= min(m, n): # Para que no siga agregando en caso de ya tener todos los índices
                # print("Not reached Epsilon")
                break
            if len(u_vectors) == 0: # Edge case in which the first row chosen consists of zeros only
                i_star += 1
                if i_star == m: # Everything is zero
                    u_vectors.append(_np.zeros(m, dtype=R_row.dtype)) # 1xm
                    v_vectors.append(_np.zeros(n, dtype=R_row.dtype)) # 1xn
                    break
                continue
        else:
            v = R_row / delta
            aux = 0
            if k>0:
                aux = _np.asarray(v_vectors)[:, j_star].T @ _np.asarray(u_vectors)
            
            # Column of original matrix:
            # mask_col[:] = False
            # mask_col[assembled_values.T[j_star, :].rows[0]] = True
            # if len(rows[~mask_col]) > 0:
            if mask_row[j_star] != True and _np.sum(~mask_col) > 0:
                assembled_values[~mask_col, j_star] = pda(boundary_operator.descriptor, boundary_operator.domain, boundary_operator.dual_to_range, parameters, rows[~mask_col], [cols[j_star]]) # PIN
                # assembled_values[~mask_col, j_star:j_star+1] = pda(boundary_operator.descriptor, boundary_operator.domain, boundary_operator.dual_to_range, parameters, rows[~mask_col], [cols[j_star]])
                mask_row[j_star] = True
            u = assembled_values[:, j_star].toarray()
            # u = _np.array(u + singular_matrix[rows, cols[j_star]])
            
            u = u.flatten()
            u -= aux
            k += 1
            u_vectors.append(u)
            v_vectors.append(v)
        I.append(i_star)
        J.append(j_star)
        u_copy = _np.copy(u_vectors[-1])
        u_copy[I] = 0
        i_star = _np.argmax(_np.abs(u_copy))

        # Stopping criterion:
        u = u_vectors[-1]
        v = v_vectors[-1]
        norm_u = _np.linalg.norm(u)
        norm_v = _np.linalg.norm(v)
        sum_uv_norm_square += norm_u**2 * norm_v**2
        error_rel = norm_u * norm_v / _np.sqrt(sum_uv_norm_square)
        if error_rel <= epsilon or len(u_vectors) == min(m, n): # Epsilon reached or rank completed
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