import numpy as _np
from numba import njit

# Standard ACA: ==================================

@njit()
def ACAPP_numba(rows, cols, info, numba_assembler, epsilon=0.1, dtype=_np.complex128):
    '''Standard ACA with partial pivoting'''
    m, n = len(rows), len(cols)
    i_star = 0 # Arbitrary first pivot

    # Check if it is worth to compress:
    if m*n <= m+n:
        # Return arrays:
        array_u = numba_assembler(rows, cols, info, dtype)
        array_v = _np.array([()], dtype=dtype)
        return array_u, array_v

    # Low-rank approximation vectors:
    u_vectors = []
    v_vectors = []

    # Already used pivots and iteration value:
    mask_row = _np.zeros(n, dtype=_np.bool_) # OJO
    mask_col = _np.zeros(m, dtype=_np.bool_) # OJO
    k = 0

    # Stopping criterion:
    Fnorm_square = 0
    
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
                    # Zero matrix:
                    u_vectors.append(_np.zeros(m, dtype=R_row.dtype)) # 1xm
                    v_vectors.append(_np.zeros(n, dtype=R_row.dtype)) # 1xn
                break
            # Take random next i_star:
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
        
        # Get next i_star:
        u_abs = _np.abs(u_vectors[-1])
        u_abs[mask_col] = -1 # OJO: alternativa al anterior
        i_star = _np.argmax(u_abs)

        # Check if compression is still viable:
        if len(u_vectors) * m + len(v_vectors) * n > m * n:
            # # Return lists:
            # v_vectors.clear()
            # matrix = numba_assembler(rows, cols, info, dtype)
            # u_vectors = [matrix[i,:].flatten() for i in range(len(matrix))]
            # return u_vectors, v_vectors

            # Return arrays:
            array_u = numba_assembler(rows, cols, info, dtype)
            array_v = _np.array([()], dtype=R_row.dtype)
            return array_u, array_v

        # Stopping criterion:
        u = u_vectors[-1]
        v = v_vectors[-1]
        norm_u = _np.linalg.norm(u)
        norm_v = _np.linalg.norm(v)
        Fnorm_square += norm_u**2 * norm_v**2
        # NEW PAPER ============
        aux = 0
        for i in range(k - 1):
            aux += _np.vdot(u_vectors[i], u) * _np.vdot(v_vectors[i], v)
        Fnorm_square += 2 * _np.real(aux)
        # ============ NEW PAPER
        error_rel = norm_u * norm_v / _np.sqrt(Fnorm_square)

        # Check relative error or already checked entire matrix?
        if error_rel <= epsilon or _np.sum(mask_col) == m or _np.sum(mask_row) == n:
            break

    # # Return lists:
    # return u_vectors, v_vectors

    # Return arrays:
    array_u = _np.empty((len(u_vectors), m), dtype=R_row.dtype)
    array_v = _np.empty((len(v_vectors), n), dtype=R_row.dtype)
    for i in range(len(u_vectors)):
        array_u[i] = u_vectors[i]
        array_v[i] = v_vectors[i]
    return array_u, array_v

@njit()
def ACAPP_CCC_numba(rows, cols, info, numba_assembler, epsilon=0.1, dtype=_np.complex128):
    '''Standard ACA with partial pivoting and Combined Convergence Criterion (CCC)'''
    m, n = len(rows), len(cols)
    i_star = 0 # Arbitrary first pivot

    # Check if it is worth to compress:
    if m*n <= m+n:
        # Return arrays:
        array_u = numba_assembler(rows, cols, info, dtype)
        array_v = _np.array([()], dtype=dtype)
        return array_u, array_v

    # Low-rank approximation vectors:
    u_vectors = []
    v_vectors = []

    # Already used pivots and iteration value:
    mask_row = _np.zeros(n, dtype=_np.bool_) # OJO
    mask_col = _np.zeros(m, dtype=_np.bool_) # OJO
    k = 0

    # CCC:
    # === Random:
    # L = m + n
    # extra_pivots = _np.random.choice(m*n, L, replace=False)
    # extra_rows = extra_pivots // n
    # extra_cols = extra_pivots % n
    # === Diagonal:
    L = max(m,n)
    extra_rows = _np.arange(L) % m
    extra_cols = _np.arange(L) % n # -(_np.arange(L) % n) - 1
    # ===
    e = _np.empty(L, dtype=dtype)
    for i in range(L):
        e[i] = numba_assembler(rows[extra_rows[i:i+1]], cols[extra_cols[i:i+1]], info, dtype)[0,0]

    # Stopping criterion:
    Fnorm_square = 0
    
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
        if R_row_abs[j_star] == -1: # En teoría esto nunca debería ocurrir (por la condición de salida más abajo)
            delta = 0
        else:
            delta = R_row[j_star]

        if delta == 0:
            if _np.sum(mask_col) == m: # OJO: acá tenía min(m,n), pero eso no es correcto
                if len(u_vectors) == 0:
                    # Zero matrix:
                    u_vectors.append(_np.zeros(m, dtype=R_row.dtype)) # 1xm
                    v_vectors.append(_np.zeros(n, dtype=R_row.dtype)) # 1xn
                break
            # Take random next i_star:
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
        
        # Update random pivots: 
        for i in range(L):
            e[i] -= u[extra_rows[i]] * v[extra_cols[i]]
        
        # Get next i_star:
        u_abs = _np.abs(u_vectors[-1])
        u_abs[mask_col] = -1 # OJO: alternativa al anterior
        i_star = _np.argmax(u_abs)

        # Check if compression is still viable:
        if len(u_vectors) * m + len(v_vectors) * n > m * n:
            # Return arrays:
            array_u = numba_assembler(rows, cols, info, dtype)
            array_v = _np.array([()], dtype=R_row.dtype)
            return array_u, array_v

        # Already checked entire matrix?
        if _np.sum(mask_col) == m or _np.sum(mask_row) == n:
            break

        # Stopping criterion:
        u = u_vectors[-1]
        v = v_vectors[-1]
        norm_u = _np.linalg.norm(u)
        norm_v = _np.linalg.norm(v)
        Fnorm_square += norm_u**2 * norm_v**2
        # NEW PAPER ============
        aux = 0
        for i in range(k - 1):
            aux += _np.vdot(u_vectors[i], u) * _np.vdot(v_vectors[i], v)
        Fnorm_square += 2 * _np.real(aux)
        # ============ NEW PAPER
        error_rel = norm_u * norm_v / _np.sqrt(Fnorm_square)
        
        # Check relative error:
        if error_rel <= epsilon:
            # CCC:
            if _np.sqrt(_np.mean(_np.abs(e**2)) * m * n) / _np.sqrt(Fnorm_square) <= epsilon:
                # Geometric step?
                break
            else:
                i_star = extra_rows[_np.argmax(_np.abs(e))]
                if mask_col[i_star] == True: # Esto generalmente pasa por pequeños errores de redondeo y e ya fue completamente revisado
                    i_star = _np.random.choice(_np.arange(m)[~mask_col])

    # Return arrays:
    array_u = _np.empty((len(u_vectors), m), dtype=R_row.dtype)
    array_v = _np.empty((len(v_vectors), n), dtype=R_row.dtype)
    for i in range(len(u_vectors)):
        array_u[i] = u_vectors[i]
        array_v[i] = v_vectors[i]
    return array_u, array_v

# ACA+: ==================================

@njit()
def get_a(j_pivot, rows, cols, info, dtype, a_vectors, b_vectors, k, numba_assembler):
    '''ACA+ auxiliary function'''
    # a_pivot = M[:, j_pivot] # Column of pivot
    a_pivot = numba_assembler(rows, cols[j_pivot:j_pivot+1], info, dtype).flatten()
    # a_pivot = a_pivot - _np.asarray(b_vectors)[:, j_pivot].T @ _np.asarray(a_vectors)
    aux2 = _np.zeros_like(a_pivot)
    for l in range(k):
        aux2 += b_vectors[l][j_pivot] * a_vectors[l]
    a_pivot -= aux2
    return a_pivot

@njit()
def get_b(i_pivot, rows, cols, info, dtype, a_vectors, b_vectors, k, numba_assembler):
    '''ACA+ auxiliary function'''
    # b_pivot = M[i_pivot, :] # Row of pivot
    b_pivot = numba_assembler(rows[i_pivot:i_pivot+1], cols, info, dtype).flatten()
    # b_pivot = b_pivot - _np.asarray(a_vectors)[:, i_pivot].T @ _np.asarray(b_vectors)
    aux1 = _np.zeros_like(b_pivot)
    for l in range(k):
        aux1 += a_vectors[l][i_pivot] * b_vectors[l]
    b_pivot -= aux1
    return b_pivot

@njit()
def ACAPlusPP_numba(rows, cols, info, numba_assembler, epsilon=0.1, dtype=_np.complex128):
    '''ACA+ with partial pivoting'''
    m, n = len(rows), len(cols)

    # Check if it is worth to compress:
    if m*n <= m+n:
        # Return arrays:
        array_a = numba_assembler(rows, cols, info, dtype)
        array_b = _np.array([()], dtype=dtype)
        return array_a, array_b
    
    j_ref = 0 # Arbitrary choice of j_ref
    a_ref = numba_assembler(rows, cols[j_ref:j_ref+1], info, dtype).flatten() # Reference column
    i_ref = _np.argmin(_np.abs(a_ref))
    b_ref = numba_assembler(rows[i_ref:i_ref+1], cols, info, dtype).flatten() # Reference row

    # Low-rank approximation vectors:
    a_vectors = []
    b_vectors = []

    # Already used pivots and iteration value:
    P_cols = _np.zeros(n, dtype=_np.bool_) # mask_row
    P_rows = _np.zeros(m, dtype=_np.bool_) # mask_col
    k = 0

    # Stopping criterion:
    Fnorm_square = 0
    
    while True:
        # Cross approximation step:
        a_ref_abs, b_ref_abs = _np.abs(a_ref), _np.abs(b_ref)
        a_ref_abs[P_rows] = -1 # Remove pivots already used
        b_ref_abs[P_cols] = -1 # Remove pivots already used
        i_star = _np.argmax(a_ref_abs)
        j_star = _np.argmax(b_ref_abs)
        assert a_ref_abs[i_star] != -1, "-1 encountered"
        assert b_ref_abs[j_star] != -1, "-1 encountered"

        if a_ref_abs[i_star] > b_ref_abs[j_star]:
            b = get_b(i_star, rows, cols, info, dtype, a_vectors, b_vectors, k, numba_assembler)
            P_rows[i_star] = True

            b_abs = _np.abs(b)
            b_abs[P_cols] = -1 # Remove pivots already used
            j_star = _np.argmax(b_abs)
            assert b_abs[j_star] != -1, "-1 encountered"
            delta = b[j_star]
            assert delta != 0, "delta should be always !=0 in this scope (I think)"
            
            a = get_a(j_star, rows, cols, info, dtype, a_vectors, b_vectors, k, numba_assembler) / delta
            P_cols[j_star] = True
        else:
            a = get_a(j_star, rows, cols, info, dtype, a_vectors, b_vectors, k, numba_assembler)
            P_cols[j_star] = True

            a_abs = _np.abs(a)
            a_abs[P_rows] = -1 # Remove pivots already used
            i_star = _np.argmax(a_abs)
            assert a_abs[i_star] != -1, "-1 encountered"
            delta = a[i_star]

            if delta == 0:
                if _np.sum(P_cols) == n:
                    if len(a_vectors) == 0:
                        # Zero matrix:
                        a_vectors.append(_np.zeros(m, dtype=a_ref.dtype)) # 1xm
                        b_vectors.append(_np.zeros(n, dtype=a_ref.dtype)) # 1xn
                    break
                # TODO: Acá no estoy seguro si debería actualizar j_ref o i_ref o ninguno, pero creo que no es necesario hacer ningún cambio
                continue

            b = get_b(i_star, rows, cols, info, dtype, a_vectors, b_vectors, k, numba_assembler) / delta
            P_rows[i_star] = True

        # Save results of step:
        k += 1
        a_vectors.append(a)
        b_vectors.append(b)

        # Check if compression is still viable:
        if len(a_vectors) * m + len(b_vectors) * n > m * n: # TODO:
            array_a = numba_assembler(rows, cols, info, dtype)
            array_b = _np.array([()], dtype=b_ref.dtype)
            return array_a, array_b

        # Already checked entire matrix?
        if _np.sum(P_rows) == m or _np.sum(P_cols) == n:
            break

        # Check epsilon:
        norm_a = _np.linalg.norm(a)
        norm_b = _np.linalg.norm(b)
        Fnorm_square += norm_a**2 * norm_b**2
        # NEW PAPER ============
        aux = 0
        for i in range(k - 1):
            aux += _np.vdot(a_vectors[i], a) * _np.vdot(b_vectors[i], b)
        Fnorm_square += 2 * _np.real(aux)
        # ============ NEW PAPER
        error_rel = norm_a * norm_b / _np.sqrt(Fnorm_square)

        # Check relative error:
        if error_rel <= epsilon:
            break

        # Update reference elements:
        a_ref = a_ref - (a * b[j_ref])
        b_ref = b_ref - (a[i_ref] * b)
        if i_ref == i_star and j_ref == j_star: # TODO: Esta parte no estoy seguro que sea así..., el comportamiento aleatorio no me gusta
            j_ref = _np.random.choice(_np.arange(n)[~P_cols])
            a_ref = get_a(j_ref, rows, cols, info, dtype, a_vectors, b_vectors, k, numba_assembler)
            a_ref_abs = _np.abs(a_ref)
            a_ref_abs[P_rows] = _np.max(a_ref_abs) + 1
            i_ref = _np.argmin(a_ref_abs)
            b_ref = get_b(i_ref, rows, cols, info, dtype, a_vectors, b_vectors, k, numba_assembler)
        elif i_ref == i_star:
            a_ref_abs = _np.abs(a_ref)
            a_ref_abs[P_rows] = _np.max(a_ref_abs) + 1
            i_ref = _np.argmin(a_ref_abs)
            b_ref = get_b(i_ref, rows, cols, info, dtype, a_vectors, b_vectors, k, numba_assembler)
        elif j_ref == j_star: 
            b_ref_abs = _np.abs(b_ref)
            b_ref_abs[P_cols] = _np.max(b_ref_abs) + 1
            j_ref = _np.argmin(b_ref_abs)
            a_ref = get_a(j_ref, rows, cols, info, dtype, a_vectors, b_vectors, k, numba_assembler)

    # Return arrays:
    array_a = _np.empty((len(a_vectors), m), dtype=dtype)
    array_b = _np.empty((len(b_vectors), n), dtype=dtype)
    for i in range(len(a_vectors)):
        array_a[i] = a_vectors[i]
        array_b[i] = b_vectors[i]
    return array_a, array_b

@njit()
def ACAPlusPP_CCC_numba(rows, cols, info, numba_assembler, epsilon=0.1, dtype=_np.complex128):
    '''ACA+ with partial pivoting and Combined Convergence Criterion (CCC)'''
    m, n = len(rows), len(cols)

    # Check if it is worth to compress:
    if m*n <= m+n:
        # Return arrays:
        array_a = numba_assembler(rows, cols, info, dtype)
        array_b = _np.array([()], dtype=dtype)
        return array_a, array_b
    
    j_ref = 0 # Arbitrary choice of j_ref
    a_ref = numba_assembler(rows, cols[j_ref:j_ref+1], info, dtype).flatten() # Reference column
    i_ref = _np.argmin(_np.abs(a_ref))
    b_ref = numba_assembler(rows[i_ref:i_ref+1], cols, info, dtype).flatten() # Reference row

    # Low-rank approximation vectors:
    a_vectors = []
    b_vectors = []

    # Already used pivots and iteration value:
    P_cols = _np.zeros(n, dtype=_np.bool_) # mask_row
    P_rows = _np.zeros(m, dtype=_np.bool_) # mask_col
    k = 0

    # CCC:
    # === Random:
    # L = m + n
    # extra_pivots = _np.random.choice(m*n, L, replace=False)
    # extra_rows = extra_pivots // n
    # extra_cols = extra_pivots % n
    # === Diagonal:
    L = max(m,n)
    extra_rows = _np.arange(L) % m
    extra_cols = _np.arange(L) % n # -(_np.arange(L) % n) - 1
    # ===
    e = _np.empty(L, dtype=dtype)
    for i in range(L):
        e[i] = numba_assembler(rows[extra_rows[i:i+1]], cols[extra_cols[i:i+1]], info, dtype)[0,0]

    # Stopping criterion:
    Fnorm_square = 0
    
    while True:
        # Cross approximation step:
        a_ref_abs, b_ref_abs = _np.abs(a_ref), _np.abs(b_ref)
        a_ref_abs[P_rows] = -1 # Remove pivots already used
        b_ref_abs[P_cols] = -1 # Remove pivots already used
        i_star = _np.argmax(a_ref_abs)
        j_star = _np.argmax(b_ref_abs)
        assert a_ref_abs[i_star] != -1, "-1 encountered"
        assert b_ref_abs[j_star] != -1, "-1 encountered"

        if a_ref_abs[i_star] > b_ref_abs[j_star]:
            b = get_b(i_star, rows, cols, info, dtype, a_vectors, b_vectors, k, numba_assembler)
            P_rows[i_star] = True

            b_abs = _np.abs(b)
            b_abs[P_cols] = -1 # Remove pivots already used
            j_star = _np.argmax(b_abs)
            assert b_abs[j_star] != -1, "-1 encountered"
            delta = b[j_star]
            assert delta != 0, "delta should be always !=0 in this scope (I think)"
            
            a = get_a(j_star, rows, cols, info, dtype, a_vectors, b_vectors, k, numba_assembler) / delta
            P_cols[j_star] = True
        else:
            a = get_a(j_star, rows, cols, info, dtype, a_vectors, b_vectors, k, numba_assembler)
            P_cols[j_star] = True

            a_abs = _np.abs(a)
            a_abs[P_rows] = -1 # Remove pivots already used
            i_star = _np.argmax(a_abs)
            assert a_abs[i_star] != -1, "-1 encountered"
            delta = a[i_star]

            if delta == 0:
                if _np.sum(P_cols) == n:
                    if len(a_vectors) == 0:
                        # Zero matrix:
                        a_vectors.append(_np.zeros(m, dtype=a_ref.dtype)) # 1xm
                        b_vectors.append(_np.zeros(n, dtype=a_ref.dtype)) # 1xn
                    break
                # TODO: Acá no estoy seguro si debería actualizar j_ref o i_ref o ninguno, pero creo que no es necesario hacer ningún cambio
                continue

            b = get_b(i_star, rows, cols, info, dtype, a_vectors, b_vectors, k, numba_assembler) / delta
            P_rows[i_star] = True

        # Save results of step:
        k += 1
        a_vectors.append(a)
        b_vectors.append(b)

        # Update random pivots: 
        for i in range(L):
            e[i] -= a[extra_rows[i]] * b[extra_cols[i]]

        # Check if compression is still viable:
        if len(a_vectors) * m + len(b_vectors) * n > m * n: # TODO:
            array_a = numba_assembler(rows, cols, info, dtype)
            array_b = _np.array([()], dtype=b_ref.dtype)
            return array_a, array_b

        # Already checked entire matrix?
        if _np.sum(P_rows) == m or _np.sum(P_cols) == n:
            break

        # Check epsilon:
        norm_a = _np.linalg.norm(a)
        norm_b = _np.linalg.norm(b)
        Fnorm_square += norm_a**2 * norm_b**2
        # NEW PAPER ============
        aux = 0
        for i in range(k - 1):
            aux += _np.vdot(a_vectors[i], a) * _np.vdot(b_vectors[i], b)
        Fnorm_square += 2 * _np.real(aux)
        # ============ NEW PAPER
        error_rel = norm_a * norm_b / _np.sqrt(Fnorm_square)
        
        # Check relative error:
        if error_rel <= epsilon:
            # CCC:
            if _np.sqrt(_np.mean(_np.abs(e**2)) * m * n) / _np.sqrt(Fnorm_square) <= epsilon:
                break
            else:
                j_ref = extra_cols[_np.argmax(_np.abs(e))]
                if P_cols[j_ref] == True:
                    j_ref = _np.random.choice(_np.arange(n)[~P_cols])
                a_ref = get_a(j_ref, rows, cols, info, dtype, a_vectors, b_vectors, k, numba_assembler)
                a_ref_abs = _np.abs(a_ref)
                a_ref_abs[P_rows] = _np.max(a_ref_abs) + 1
                i_ref = _np.argmin(a_ref_abs)
                b_ref = get_b(i_ref, rows, cols, info, dtype, a_vectors, b_vectors, k, numba_assembler)
                continue

        # Update reference elements:
        a_ref = a_ref - (a * b[j_ref])
        b_ref = b_ref - (a[i_ref] * b)
        if i_ref == i_star and j_ref == j_star:
            # === Random selection of j_ref:
            # j_ref = _np.random.choice(_np.arange(n)[~P_cols])
            # === Selection from extra pivots:
            j_ref = extra_cols[_np.argmax(_np.abs(e))]
            if P_cols[j_ref] == True:
                j_ref = _np.random.choice(_np.arange(n)[~P_cols])
            # ===
            a_ref = get_a(j_ref, rows, cols, info, dtype, a_vectors, b_vectors, k, numba_assembler)
            a_ref_abs = _np.abs(a_ref)
            a_ref_abs[P_rows] = _np.max(a_ref_abs) + 1
            i_ref = _np.argmin(a_ref_abs)
            b_ref = get_b(i_ref, rows, cols, info, dtype, a_vectors, b_vectors, k, numba_assembler)
        elif i_ref == i_star:
            a_ref_abs = _np.abs(a_ref)
            a_ref_abs[P_rows] = _np.max(a_ref_abs) + 1
            i_ref = _np.argmin(a_ref_abs)
            b_ref = get_b(i_ref, rows, cols, info, dtype, a_vectors, b_vectors, k, numba_assembler)
        elif j_ref == j_star: 
            b_ref_abs = _np.abs(b_ref)
            b_ref_abs[P_cols] = _np.max(b_ref_abs) + 1
            j_ref = _np.argmin(b_ref_abs)
            a_ref = get_a(j_ref, rows, cols, info, dtype, a_vectors, b_vectors, k, numba_assembler)

    # Return arrays:
    array_a = _np.empty((len(a_vectors), m), dtype=dtype)
    array_b = _np.empty((len(b_vectors), n), dtype=dtype)
    for i in range(len(a_vectors)):
        array_a[i] = a_vectors[i]
        array_b[i] = b_vectors[i]
    return array_a, array_b