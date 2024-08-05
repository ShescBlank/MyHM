import numpy as _np

# https://www.epfl.ch/labs/anchp/wp-content/uploads/2018/10/lecture4-slides.pdf
# https://web.archive.org/web/20060903235945id_/http://esl.eng.ohio-state.edu/~csg/papers/79.pdf
def ACAPP(A, epsilon = 0.1, verbose=False):
    R = _np.copy(A)
    m, n = R.shape
    I = []
    J = []
    k = 0
    i_star = 0

    u_vectors = []
    v_vectors = []

    # Stopping criterion:
    sum_uv_norm_square = 0
    
    while True:
        aux = 0
        # for l in range(k):
        #     aux += u_vectors[l][i_star] * v_vectors[l]
        if k > 0:
            aux = _np.asarray(u_vectors)[:, i_star].T @ _np.asarray(v_vectors)
        R_row = R[i_star, :] - aux                               # Se necesita una fila de la matriz original
        j_star = _np.argmax(_np.abs(R_row))
        delta = R_row[j_star]
        if abs(delta) <= 1e-15: # Agregamos un pequeño margen (puede ser complejo)
            if len(I) == min(m, n): # Para que no siga agregando en caso de ya tener todos los índices
                # print("Not reached Epsilon")
                break
        else:
            v = R_row / delta
            aux = 0
            # for l in range(k):
            #     aux += v_vectors[l][j_star] * u_vectors[l]
            if k>0:
                aux = _np.asarray(v_vectors)[:, j_star].T @ _np.asarray(u_vectors)
            u = R[:, j_star] - aux                               # Se necesita una columna de la matriz original
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
        if error_rel <= epsilon:
            # print("Reached Epsilon")
            break

    if verbose:
        print(f"Finished in k={k} out of {m}")
        print(f"Relative error between matrices: {_np.linalg.norm(A-_np.asarray(u_vectors).T @ _np.asarray(v_vectors)) / _np.linalg.norm(A)}")
    
    return _np.array(u_vectors), _np.array(v_vectors)


if __name__ == "__main__":
    from scipy.linalg import hilbert
    N = 144
    H = hilbert(N)
    u_vectors, v_vectors = ACAPP(H, epsilon=1e-10, verbose=True)
    print("Working...")