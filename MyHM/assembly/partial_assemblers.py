import numpy as _np
from scipy.sparse import csr_matrix
from MyHM.assembly.numba_kernels import select_numba_assembly_partial, select_numba_assembly_partial2
from MyHM.assembly.utils import get_elements_from_vertices, sort_partial_elements_by_color, create_fun_index_dicts, create_partial_dofs, create_fun_index_masks_and_offsets
from bempp.core.numba_kernels import select_numba_kernels
from bempp.api.utils.helpers import get_type
from bempp.api.integration.triangle_gauss import rule
from bempp.core.singular_assembler import assemble_singular_part

def partial_dense_assembler(
    operator_descriptor, domain, dual_to_range, parameters, rows, cols, dtype=_np.complex128
):  
    """Numba based partial dense assembler."""

    result = _np.zeros((len(rows), len(cols)), dtype=dtype)

    (
        _,
        numba_kernel_function_regular,
    ) = select_numba_kernels(operator_descriptor, mode="regular")

    # Get partial assembly function
    numba_assembly_function_regular = select_numba_assembly_partial(operator_descriptor, mode="regular")

    order = parameters.quadrature.regular
    quad_points, quad_weights = rule(order)

    # Perform Numba assembly always in double precision
    precision = "double"

    data_type = get_type(precision).real

    # Get indices from elements associated to rows and cols vertices:
    test_elements = get_elements_from_vertices(dual_to_range.global2local, rows)
    test_indices, test_color_indexptr = sort_partial_elements_by_color(dual_to_range, test_elements)
    # test_indices, test_color_indexptr = dual_to_range.get_elements_by_color()

    trial_elements = get_elements_from_vertices(domain.global2local, cols)
    trial_indices, trial_color_indexptr = sort_partial_elements_by_color(domain, trial_elements)
    # trial_indices, trial_color_indexptr = domain.get_elements_by_color()
    
    number_of_test_colors = len(test_color_indexptr) - 1

    # Get function index dictionaries and partial dofs
    test_global_dofs = dual_to_range.local2global
    trial_global_dofs = domain.local2global
    # test_fun_index_dict, trial_fun_index_dict = create_fun_index_dicts(
    #     rows, cols, test_indices, trial_indices, test_global_dofs, trial_global_dofs
    # )
    test_fun_index_dict, trial_fun_index_dict = create_fun_index_dicts(
        rows, cols, test_indices, trial_indices, test_global_dofs, trial_global_dofs
    )
    test_partial_dofs, trial_partial_dofs = create_partial_dofs(rows, cols, test_global_dofs, trial_global_dofs)

    nshape_test = dual_to_range.number_of_shape_functions
    nshape_trial = domain.number_of_shape_functions
    grids_identical = domain.grid == dual_to_range.grid

    for test_color_index in range(number_of_test_colors):
        # print(numba_assembly_function_regular.__name__)
        numba_assembly_function_regular(
            dual_to_range.grid.data(precision), #.data en mp
            domain.grid.data(precision), #.data en mp
            nshape_test,
            nshape_trial,
            test_indices[
                test_color_indexptr[test_color_index] : test_color_indexptr[
                    1 + test_color_index
                ]
            ],
            trial_indices,
            dual_to_range.local_multipliers.astype(data_type),
            domain.local_multipliers.astype(data_type),
            test_partial_dofs, # NEW
            trial_partial_dofs, # NEW
            test_fun_index_dict, # NEW
            trial_fun_index_dict, # NEW
            dual_to_range.normal_multipliers,
            domain.normal_multipliers,
            quad_points.astype(data_type),
            quad_weights.astype(data_type),
            numba_kernel_function_regular,
            _np.array(operator_descriptor.options, dtype=data_type),
            grids_identical,
            dual_to_range.shapeset.evaluate,
            domain.shapeset.evaluate,
            result,
        )
    
    return result

def singular_assembler_sparse(
    device_interface, operator_descriptor, domain, dual_to_range, parameters
):  
    grids_identical = domain.grid == dual_to_range.grid
    
    if grids_identical:
        trial_local2global = domain.local2global.ravel()
        test_local2global = dual_to_range.local2global.ravel()
        trial_multipliers = domain.local_multipliers.ravel()
        test_multipliers = dual_to_range.local_multipliers.ravel()

        singular_rows, singular_cols, singular_values = assemble_singular_part(
            domain.localised_space,
            dual_to_range.localised_space,
            parameters,
            operator_descriptor,
            device_interface,
        )

        singular_rows_global = test_local2global[singular_rows]
        singular_cols_global = trial_local2global[singular_cols]
        values = (
            singular_values
            * trial_multipliers[singular_cols]
            * test_multipliers[singular_rows]
        )

        # _np.add.at(result, (singular_rows_global, singular_cols_global), values)

        nrow = dual_to_range.global_dof_count
        ncol = domain.global_dof_count
        sparse_matrix = csr_matrix((values, (singular_rows_global, singular_cols_global)), shape=(nrow, ncol), dtype=values.dtype)

        # Return sparse matrix to distribute values throughout the tree:
        return sparse_matrix
        
        # # Return all the results to distribute them throughout the tree:
        # return singular_rows_global, singular_cols_global, values
        

def partial_dense_assembler2(
    operator_descriptor, domain, dual_to_range, parameters, rows, cols, dtype=_np.complex128
):  
    """Numba based partial dense assembler."""

    (
        _,
        numba_kernel_function_regular,
    ) = select_numba_kernels(operator_descriptor, mode="regular")

    # Get partial assembly function
    numba_assembly_function_regular = select_numba_assembly_partial2(operator_descriptor, mode="regular")

    order = parameters.quadrature.regular
    quad_points, quad_weights = rule(order)

    # Perform Numba assembly always in double precision
    precision = "double"

    data_type = get_type(precision).real

    # Get indices from elements associated to rows and cols vertices:
    test_elements = get_elements_from_vertices(dual_to_range.global2local, rows)
    trial_elements = get_elements_from_vertices(domain.global2local, cols)

    # Si los ordenamos por colores, deberíamos obtener el mismo resultado que la versión 1: (se altera el orden de la suma)
    # test_elements, test_color_indexptr = sort_partial_elements_by_color(dual_to_range, test_elements)
    # trial_elements, trial_color_indexptr = sort_partial_elements_by_color(domain, trial_elements)

    # Get function index dictionaries and partial dofs
    test_partial_dofs = dual_to_range.local2global[test_elements]
    trial_partial_dofs = domain.local2global[trial_elements]
    (
        test_fun_index_mask,
        trial_fun_index_mask,
        test_offsets,
        trial_offsets 
    ) = create_fun_index_masks_and_offsets(
        rows, cols, len(test_elements), len(trial_elements), test_partial_dofs, trial_partial_dofs
    )
    
    result = _np.zeros((test_offsets[-1], trial_offsets[-1]), dtype=dtype)

    nshape_test = dual_to_range.number_of_shape_functions
    nshape_trial = domain.number_of_shape_functions
    grids_identical = domain.grid == dual_to_range.grid

    numba_assembly_function_regular(
        dual_to_range.grid.data(precision), #.data en mp
        domain.grid.data(precision), #.data en mp
        nshape_test,
        nshape_trial,
        test_elements,
        trial_elements,
        dual_to_range.local_multipliers.astype(data_type),
        domain.local_multipliers.astype(data_type),
        test_partial_dofs,
        trial_partial_dofs,
        test_fun_index_mask,
        trial_fun_index_mask,
        test_offsets,
        trial_offsets,
        dual_to_range.normal_multipliers,
        domain.normal_multipliers,
        quad_points.astype(data_type),
        quad_weights.astype(data_type),
        numba_kernel_function_regular,
        _np.array(operator_descriptor.options, dtype=data_type),
        grids_identical,
        dual_to_range.shapeset.evaluate,
        domain.shapeset.evaluate,
        result,
    )

    global_rows = test_partial_dofs[test_fun_index_mask]
    global_cols = trial_partial_dofs[trial_fun_index_mask]
    nrow = dual_to_range.global_dof_count
    ncol = domain.global_dof_count
    sparse_matrix = csr_matrix((result.ravel(), (_np.repeat(global_rows, len(global_cols)), _np.tile(global_cols, len(global_rows)))), shape=(nrow, ncol), dtype=result.dtype)
    
    meshgrid = _np.meshgrid(rows, cols, indexing="ij")
    return sparse_matrix[meshgrid[0], meshgrid[1]].toarray() # PIN
    # return sparse_matrix[meshgrid[0], meshgrid[1]]

if __name__ == "__main__":
    import bempp.api
    grid = bempp.api.shapes.sphere(h=0.2)
    print(grid.vertices.shape)
    space = bempp.api.function_space(grid, "P", 1)
    k = 7
    dlp = bempp.api.operators.boundary.helmholtz.double_layer(space, space, space, k)
    boundary_operator = dlp

    # Inputs:
    parameters = bempp.api.GLOBAL_PARAMETERS
    device_interface = "opencl"
    cols = list(range(4,10))
    rows = list(range(0,4))
    result = _np.zeros((len(rows), len(cols)), dtype=_np.complex128)

    # Assemble dense:
    partial_dense_assembler(boundary_operator.descriptor, boundary_operator.domain, boundary_operator.dual_to_range, parameters, result, cols=cols, rows=rows)
    singular_sm = singular_assembler_sparse(device_interface, boundary_operator.descriptor, boundary_operator.domain, boundary_operator.dual_to_range, parameters) 
    
    print("Working...") 
