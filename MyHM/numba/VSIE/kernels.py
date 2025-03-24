import numpy as _np
from numba import njit

@njit()
def single_layer_partial_nb(rows, cols, info, dtype=_np.complex128):
    """
    Partial Single Layer Volume Integral Operator for the same domain.
    Partial version of the single_layer function. Calculates the [rows x cols] submatrix.
    Accelerated with Numba.

    Parameters
    -----------
    wavenumber: float, complex
        Exterior wave number.
    dom: np.ndarray
        Centers of the voxels of the space in order.
        The shape is N x 3, with N the number of voxels.
    vox_size: float
        The size of the length of a voxel.
    alpha: np.ndarray
        Values of alpha in the voxels.
    beta: np.ndarray
        Values of beta in the voxels.
    weights: np.ndarray 
        Volume of each voxel. 
    rows: list, np.ndarray
        Rows to calculate.
    cols: list, np.ndarray
        Cols to calculate.
    dtype: np.dtype
        Data type of the results matrix.

    Returns
    --------
    b_matrix: np.ndarray
        Values of the operator B in each voxel.
        The shape is N x N.
    """
    wavenumber = info.kappa
    dom = info.int_grid
    alpha = info.int_alpha
    beta = info.int_beta
    weights = info.int_w

    n_vox_dom = len(rows)
    n_vox_codom = len(cols)
    b_matrix = _np.empty((n_vox_dom,n_vox_codom), dtype=dtype)
    for row_index in range(n_vox_dom):
        for col_index in range(n_vox_codom):
            i, j = rows[row_index], cols[col_index]
            if i == j:
                r_self = (3/(4*_np.pi) * weights[j])**(1/3)
                self = (1/wavenumber**2 - 1j*r_self/wavenumber)*_np.exp(1j*wavenumber*r_self) - 1/wavenumber**2
                b_matrix[row_index, col_index] = (beta[j] - wavenumber**2*alpha[j]) * self
            else:
                ri_a_rj = dom[i] - dom[j]
                norm = _np.linalg.norm(ri_a_rj)
                b_matrix[row_index, col_index] = weights[j] * _np.exp(1j * wavenumber * norm) / (4*_np.pi * norm)
                b_matrix[row_index, col_index] = (beta[j] - wavenumber**2*alpha[j]) * b_matrix[row_index, col_index]
    return b_matrix

@njit()
def cross_single_layer_partial_nb(rows, cols, info, dtype=_np.complex128):
    """
    Partial Single Layer Volume Integral Operator for disjoint domain and range geometry.
    Partial version of the cross_single_layer function. Calculates the [rows x cols] submatrix.
    Accelerated with numba.

    Parameters
    -----------
    wavenumber: float, complex
        Exterior wave number.
    dom: np.ndarray
        Centers of the voxels of the domain in order.
        The shape is N1 x 3, with N1 the number of voxels.
    codom: np.ndarray
        Centers of the voxels of the range in order.
        The shape is N2 x 3, with N2 the number of voxels.
    alphacod: np.ndarray
        Values of alpha in the voxels of codomain.
    betacod: np.ndarray
        Values of beta in the voxels of codomain.
    weights: np.ndarray 
        Volume of each voxel of the codomain.
    rows: list, np.ndarray
        Rows to calculate.
    cols: list, np.ndarray
        Cols to calculate.
    dtype: np.dtype
        Data type of the results matrix.

    Returns
    --------
    b_matrix: np.ndarray
        Values of the operator B in each voxel.
        The shape is N1 x N2.
    """
    wavenumber = info.kappa
    dom = info.surf_grid
    codom = info.int_grid
    alphacod = info.int_alpha
    betacod = info.int_beta
    weights = info.int_w

    n_vox_dom = len(rows)
    n_vox_codom = len(cols)
    b_matrix = _np.empty((n_vox_dom,n_vox_codom), dtype=dtype)
    for row_index in range(n_vox_dom):
        for col_index in range(n_vox_codom):
            i, j = rows[row_index], cols[col_index]
            ri_a_rj = dom[i] - codom[j]
            norm = _np.linalg.norm(ri_a_rj)
            b_matrix[row_index, col_index] = weights[j] * _np.exp(1j*wavenumber*norm)/(4*_np.pi*norm)
            b_matrix[row_index, col_index] = (betacod[j] - wavenumber**2*alphacod[j]) * b_matrix[row_index, col_index]
    return b_matrix

@njit()
def double_layer_partial_nb(rows, cols, info, dtype=_np.complex128):
    """
    Partial Double Layer Volume Integral Operator for self interactions.
    Partial version of the double_layer function. Calculates the [rows x cols] submatrix.
    Accelerated with numba.

    Parameters
    -----------
    wavenumber: float, complex
        Exterior wave number.
    dom: np.ndarray
        Centers of the voxels of the domain in order.
        The shape is N x 3, with N the number of voxels.
    normals: np.ndarray
        Directions of the normal vectors in each cell of the surface.
    vox_size: float
        The size of the length of a voxel.
    alpha_ext: np.ndarray
        Values of alpha in the voxels of exterior space.
    alpha_int: np.ndarray
        Values of alpha in the voxels of interior space.
    weights: np.ndarray 
        Volume of each voxel. 
    rows: list, np.ndarray
        Rows to calculate.
    cols: list, np.ndarray
        Cols to calculate.
    dtype: np.dtype
        Data type of the results matrix.

    Returns
    --------
    d_matrix: np.ndarray
        Values of the operator D in each voxel.
        The shape is N x N.
    """
    wavenumber = info.kappa
    dom = info.surf_grid
    normals = info.normals
    diff_alpha = info.diff_alpha
    weights = info.surf_w

    n_vox_dom = len(rows)
    n_vox_codom = len(cols)
    d_matrix = _np.empty((n_vox_dom,n_vox_codom), dtype=dtype)
    for row_index in range(n_vox_dom):
        for col_index in range(n_vox_codom):
            i, j = rows[row_index], cols[col_index]
            if i == j:
                d_matrix[row_index, col_index] = 0
            else:
                ri_a_rj = dom[i] - dom[j]
                norm = _np.linalg.norm(ri_a_rj)
                if norm == 0:
                    print(dom[i], dom[j], i, j)
                normal_dot = _np.dot(ri_a_rj, normals[j])
                d_matrix[row_index, col_index] = weights[j] * _np.exp(1j*wavenumber*norm)/(4*_np.pi*(norm**2))\
                            * (1/norm -1j*wavenumber) * normal_dot
                d_matrix[row_index, col_index] = diff_alpha[j] * d_matrix[row_index, col_index]

    return d_matrix

@njit()
def cross_double_layer_partial_nb(rows, cols, info, dtype=_np.complex128):
    """
    Partial Double Layer Volume Integral Operator for cross interactions between
    disjoint domain and range space.
    Partial version of the cross_double_layer function. Calculates the [rows x cols] submatrix.
    Accelerated with numba.

    Parameters
    -----------
    wavenumber: float, complex
        Exterior wave number.
    dom: np.ndarray
        Centers of the voxels of the domain in order.
        The shape is N1 x 3, with N1 the number of voxels.
    codom: np.ndarray
        Centers of the voxels of the range in order.
        The shape is N2 x 3, with N2 the number of voxels.
    normals: np.ndarray
        Directions of the normal vectors in each cell of the surface.
    alpha_ext: np.ndarray
        Values of alpha in the voxels of exterior space.
    alpha_int: np.ndarray
        Values of alpha in the voxels of interior space.
    weights: np.ndarray 
        Volume of each voxel of the codomain. 
    rows: list, np.ndarray
        Rows to calculate.
    cols: list, np.ndarray
        Cols to calculate.
    dtype: np.dtype
        Data type of the results matrix.

    Returns
    --------
    d_matrix: np.ndarray
        Values of the operator D in each voxel.
        The shape is N1 x N2.
    """
    wavenumber = info.kappa
    dom = info.int_grid
    codom = info.surf_grid
    normals = info.normals
    diff_alpha = info.diff_alpha
    weights = info.surf_w
    
    n_vox_dom = len(rows)
    n_vox_codom = len(cols)
    d_matrix = _np.empty((n_vox_dom,n_vox_codom), dtype=dtype)
    for row_index in range(n_vox_dom):
        for col_index in range(n_vox_codom):
            i, j = rows[row_index], cols[col_index]
            ri_a_rj = dom[i] - codom[j]
            norm = _np.linalg.norm(ri_a_rj)
            normal_dot = _np.dot(ri_a_rj, normals[j])
            d_matrix[row_index, col_index] = weights[j] * _np.exp(1j*wavenumber*norm)/(4*_np.pi*(norm**2))*\
                            (1/norm -1j*wavenumber) * normal_dot
            d_matrix[row_index, col_index] = diff_alpha[j] * d_matrix[row_index, col_index]
    return d_matrix

@njit()
def ad_double_layer_partial_nb(rows, cols, info, dtype=_np.complex128):
    """
    Partial Adjoint Double Layer Volume Integral Operator for self interaction.
    Partial version of the ad_double_layer function. Calculates the [rows x cols] submatrix.
    Accelerated with numba.

    Parameters
    -----------
    wavenumber: float, complex
        Exterior wave number.
    dom: np.ndarray
        Centers of the voxels of the space in order.
        The shape is N1 x 3, with N1 the number of voxels.
    codom: np.ndarray
        Centers of the voxels of the range in order.
        The shape is N2 x 3, with N2 the number of voxels.
    vox_size: float
        The size of the length of a voxel.
    grad_alpha: np.ndarray
        Values of the gradient of alpha in the voxels.
    weights: np.ndarray 
        Volume of each voxel. 
    rows: list, np.ndarray
        Rows to calculate.
    cols: list, np.ndarray
        Cols to calculate.
    dtype: np.dtype
        Data type of the results matrix.

    Returns
    --------
    c_matrix: np.ndarray
        Values of the operator C in each voxel.
        The shape is N1 x N2.
    """
    wavenumber = info.kappa
    dom = info.int_grid
    grad_alpha = info.int_grad_alpha
    weights = info.int_w

    n_vox_dom = len(rows)
    n_vox_codom = len(cols)
    c_matrix = _np.empty((n_vox_dom,n_vox_codom), dtype=dtype)
    for row_index in range(n_vox_dom):
        for col_index in range(n_vox_codom):
            i, j = rows[row_index], cols[col_index]
            if i == j:
                c_matrix[row_index, col_index] = 0
            else:
                ri_a_rj = dom[i] - dom[j]
                norm = _np.linalg.norm(ri_a_rj)
                aux = weights[j] * (_np.exp(1j*wavenumber*norm) / (4*_np.pi * norm**2))*(1j*wavenumber - 1/norm)
                c_matrix[row_index, col_index] = aux * (ri_a_rj[0]*grad_alpha[j][0] + ri_a_rj[1]*grad_alpha[j][1] \
                                 + ri_a_rj[2]*grad_alpha[j][2])

    return c_matrix

@njit()
def cross_ad_double_layer_partial_nb(rows, cols, info, dtype=_np.complex128):
    """
    Partial Adjoint Double Layer Volume Integral Operator for cross interaction.
    Partial version of the cross_ad_double_layer function. Calculates the [rows x cols] submatrix.
    Accelerated with numba.

    Parameters
    -----------
    wavenumber: float, complex
        Exterior wave number.
    dom: np.ndarray
        Centers of the voxels of the space in order.
        The shape is N x 3, with N the number of voxels.
    grad_alpha: np.ndarray
        Values of the gradient of alpha in the voxels.
    weights: np.ndarray 
        Volume of each voxel of the codomain. 
    rows, cols: list, np.ndarray
        Rows and columns to calculate.
    dtype: np.dtype
        Data type of the results.

    Returns
    --------
    c_matrix: np.ndarray
        Values of the operator C in each voxel.
        The shape is N x N.
    """
    wavenumber = info.kappa
    dom = info.surf_grid
    codom = info.int_grid
    grad_alpha = info.int_grad_alpha
    weights = info.int_w

    n_vox_dom = len(rows)
    n_vox_codom = len(cols)
    c_matrix = _np.empty((n_vox_dom,n_vox_codom), dtype=dtype)
    for row_index in range(n_vox_dom):
        for col_index in range(n_vox_codom):
            i, j = rows[row_index], cols[col_index]
            ri_a_rj = dom[i] - codom[j]
            norm = _np.linalg.norm(ri_a_rj)
            aux = weights[j] * (_np.exp(1j * wavenumber * norm) /\
                (4*_np.pi * norm**2)) * (1j*wavenumber- 1/norm)
            c_matrix[row_index, col_index] = aux * (ri_a_rj[0]*grad_alpha[j][0] + ri_a_rj[1]*grad_alpha[j][1] \
                              + ri_a_rj[2]*grad_alpha[j][2])

    return c_matrix

def mass_matrix_sp(dom, alpha):
    """
    Mass Matrix in the volume integral formulation.
    Values only in main diagonal (Sparse matrix).

    Parameters
    -----------
    dom: np.ndarray
        Centers of the voxels of the space in order.
        The shape is N x 3, with N the number of voxels.
    alpha: np.ndarray
        Values of alpha in the voxels.

    Returns
    --------
    a_matrix: scipy.sparse
        Values of the mass matrix in each voxel.
        The shape is N x N.
    """
    from scipy.sparse import diags_array

    # n_vox = dom.shape[0]
    # a_matrix = _np.zeros((n_vox,n_vox), dtype=_np.complex128)
    # _np.fill_diagonal(a_matrix, alpha + 1/rho)

    a_matrix = diags_array(alpha + 1)

    return a_matrix