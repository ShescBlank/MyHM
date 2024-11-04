import numba as _numba
import numpy as _np
from bempp.core.numba_kernels import get_normals, get_global_points, elements_adjacent

def select_numba_assembly_partial(operator_descriptor, mode="regular"):
    """Select the Numba kernels."""
    assembly_functions_regular = {
        "default_scalar": default_scalar_regular_kernel_partial,
        "helmholtz_hypersingular": helmholtz_hypersingular_regular_partial,
        "modified_helmholtz_hypersingular": modified_helmholtz_hypersingular_regular_partial,
    }
    # La parte singular no la modifiqué, por lo que utilizo la original de bempp
    # assembly_functions_singular = {
        # "default_scalar": default_scalar_singular_kernel,
        # "helmholtz_hypersingular": helmholtz_hypersingular_singular,
        # "modified_helmholtz_hypersingular": modified_helmholtz_hypersingular_singular,
    # }

    # default_sparse_kernel la utiliza bempp.api.operators.boundary.helmholtz.osrc_ntd(space, k)

    if mode == "regular":
        return assembly_functions_regular[operator_descriptor.assembly_type]
    # elif mode == "singular":
    #     return assembly_functions_singular[operator_descriptor.assembly_type]
    else:
        raise ValueError("Unknown mode.")

def select_numba_assembly_partial2(operator_descriptor, mode="regular"):
    # TODO: Todavía hay optimizaciones que se pueden hacer a estos kernels2, en particular,
    # sería bueno quitar todas las indexaciones sobre test_fun_index y trial_fun_index (hay cálculos de más en esas partes)
    # y también los arrays creados con tamaño 3.
    # De hecho, podría hacer cálculos intermedios y luego reutilizarlos ene stos kernels (evitando así operaciones repetidas)
    """Select the Numba kernels."""
    assembly_functions_regular = {
        "default_scalar": default_scalar_regular_kernel_partial2,
        "helmholtz_hypersingular": helmholtz_hypersingular_regular_partial2,
        "modified_helmholtz_hypersingular": modified_helmholtz_hypersingular_regular_partial2,
    }

    if mode == "regular":
        return assembly_functions_regular[operator_descriptor.assembly_type]
    else:
        raise ValueError("Unknown mode.")

@_numba.jit(
    nopython=True, parallel=True, error_model="numpy", fastmath=True, boundscheck=False, cache=True
)
def default_scalar_regular_kernel_partial(
    test_grid_data,
    trial_grid_data,
    nshape_test,
    nshape_trial,
    test_elements,
    trial_elements,
    test_multipliers,
    trial_multipliers,
    test_partial_dofs,
    trial_partial_dofs,
    test_fun_index_dict,
    trial_fun_index_dict,
    test_normal_multipliers,
    trial_normal_multipliers,
    quad_points,
    quad_weights,
    kernel_evaluator,
    kernel_parameters,
    grids_identical,
    test_shapeset,
    trial_shapeset,
    result,
):
    """Evaulate default scalar kernel."""
    # Compute global points
    result_type = result.dtype
    n_quad_points = len(quad_weights)
    n_test_elements = len(test_elements)
    n_trial_elements = len(trial_elements)

    local_test_fun_values = test_shapeset(quad_points)
    local_trial_fun_values = trial_shapeset(quad_points)
    trial_normals = get_normals(
        trial_grid_data, n_quad_points, trial_elements, trial_normal_multipliers
    )
    trial_global_points = get_global_points(
        trial_grid_data, trial_elements, quad_points
    )

    factors = _np.empty(
        n_quad_points * n_trial_elements, dtype=trial_global_points.dtype
    )
    for trial_element_index in range(n_trial_elements):
        for trial_point_index in range(n_quad_points):
            factors[n_quad_points * trial_element_index + trial_point_index] = (
                quad_weights[trial_point_index]
                * trial_grid_data.integration_elements[
                    trial_elements[trial_element_index]
                ]
            )

    for i in _numba.prange(n_test_elements):
        test_element = test_elements[i]
        local_result = _np.zeros(
            (n_trial_elements, nshape_test, nshape_trial), dtype=result_type
        )
        test_global_points = test_grid_data.local2global(test_element, quad_points)
        test_normal = (
            test_grid_data.normals[test_element] * test_normal_multipliers[test_element]
        )
        local_factors = _np.empty(
            n_trial_elements * n_quad_points, dtype=test_global_points.dtype
        )
        tmp = _np.empty(n_trial_elements * n_quad_points, dtype=result_type)
        is_adjacent = _np.zeros(n_trial_elements, dtype=_np.bool_)

        for trial_element_index in range(n_trial_elements):
            trial_element = trial_elements[trial_element_index]
            if grids_identical and elements_adjacent(
                test_grid_data.elements, test_element, trial_element
            ):
                is_adjacent[trial_element_index] = True

        for index in range(n_trial_elements * n_quad_points):
            local_factors[index] = (
                factors[index] * test_grid_data.integration_elements[test_element]
            )
        for test_point_index in range(n_quad_points):
            test_global_point = test_global_points[:, test_point_index]
            kernel_values = kernel_evaluator(
                test_global_point,
                trial_global_points,
                test_normal,
                trial_normals,
                kernel_parameters,
            )
            for index in range(n_trial_elements * n_quad_points):
                tmp[index] = kernel_values[index] * (
                    local_factors[index] * quad_weights[test_point_index]
                )

            for trial_element_index in range(n_trial_elements):
                if is_adjacent[trial_element_index]:
                    continue
                trial_element = trial_elements[trial_element_index]
                # for test_fun_index in range(nshape_test): # EDIT
                # for test_fun_index in test_fun_index_dict[test_element]: # EDIT2
                for test_fun_index in _np.arange(nshape_test)[test_fun_index_dict[test_element]]:
                    # for trial_fun_index in range(nshape_trial): # EDIT
                    # for trial_fun_index in trial_fun_index_dict[trial_element]: # EDIT2
                    for trial_fun_index in _np.arange(nshape_trial)[trial_fun_index_dict[trial_element]]:
                        for quad_point_index in range(n_quad_points):
                            local_result[
                                trial_element_index, test_fun_index, trial_fun_index
                            ] += (
                                tmp[
                                    trial_element_index * n_quad_points
                                    + quad_point_index
                                ]
                                * local_trial_fun_values[
                                    0, trial_fun_index, quad_point_index
                                ]
                                * local_test_fun_values[
                                    0, test_fun_index, test_point_index
                                ]
                            )

        for trial_element_index in range(n_trial_elements):
            trial_element = trial_elements[trial_element_index]
            # for test_fun_index in range(nshape_test): # EDIT
            # for test_fun_index in test_fun_index_dict[test_element]: # EDIT2
            for test_fun_index in _np.arange(nshape_test)[test_fun_index_dict[test_element]]:
                # for trial_fun_index in range(nshape_trial): # EDIT
                # for trial_fun_index in trial_fun_index_dict[trial_element]: # EDIT2
                for trial_fun_index in _np.arange(nshape_trial)[trial_fun_index_dict[trial_element]]:
                    result[
                        # test_global_dofs[test_element, test_fun_index], # EDIT
                        test_partial_dofs[test_element, test_fun_index],
                        # trial_global_dofs[trial_element, trial_fun_index], # EDIT
                        trial_partial_dofs[trial_element, trial_fun_index],
                    ] += (
                        local_result[
                            trial_element_index, test_fun_index, trial_fun_index
                        ]
                        * test_multipliers[test_element, test_fun_index]
                        * trial_multipliers[trial_element, trial_fun_index]
                    )

@_numba.jit(
    nopython=True, parallel=True, error_model="numpy", fastmath=True, boundscheck=False, cache=True
)
def default_scalar_regular_kernel_partial2(
    test_grid_data,
    trial_grid_data,
    nshape_test,
    nshape_trial,
    test_elements,
    trial_elements,
    test_multipliers,
    trial_multipliers,
    test_partial_dofs,
    trial_partial_dofs,
    test_fun_index_mask,    # New
    trial_fun_index_mask,   # New
    test_offsets,           # New
    trial_offsets,          # New
    test_normal_multipliers,
    trial_normal_multipliers,
    quad_points,
    quad_weights,
    kernel_evaluator,
    kernel_parameters,
    grids_identical,
    test_shapeset,
    trial_shapeset,
    result,
):
    """Evaulate default scalar kernel."""
    # Compute global points
    result_type = result.dtype
    n_quad_points = len(quad_weights)
    n_test_elements = len(test_elements)
    n_trial_elements = len(trial_elements)

    local_test_fun_values = test_shapeset(quad_points)
    local_trial_fun_values = trial_shapeset(quad_points)
    trial_normals = get_normals(
        trial_grid_data, n_quad_points, trial_elements, trial_normal_multipliers
    )
    trial_global_points = get_global_points(
        trial_grid_data, trial_elements, quad_points
    )

    factors = _np.empty(
        n_quad_points * n_trial_elements, dtype=trial_global_points.dtype
    )
    for trial_element_index in range(n_trial_elements):
        for trial_point_index in range(n_quad_points):
            factors[n_quad_points * trial_element_index + trial_point_index] = (
                quad_weights[trial_point_index]
                * trial_grid_data.integration_elements[
                    trial_elements[trial_element_index]
                ]
            )

    for test_element_index in _numba.prange(n_test_elements): # New: test_element_index and remove local_result
        test_element = test_elements[test_element_index]
        test_global_points = test_grid_data.local2global(test_element, quad_points)
        test_normal = (
            test_grid_data.normals[test_element] * test_normal_multipliers[test_element]
        )
        local_factors = _np.empty(
            n_trial_elements * n_quad_points, dtype=test_global_points.dtype
        )
        tmp = _np.empty(n_trial_elements * n_quad_points, dtype=result_type)
        is_adjacent = _np.zeros(n_trial_elements, dtype=_np.bool_)

        for trial_element_index in range(n_trial_elements):
            trial_element = trial_elements[trial_element_index]
            if grids_identical and elements_adjacent(
                test_grid_data.elements, test_element, trial_element
            ):
                is_adjacent[trial_element_index] = True

        for index in range(n_trial_elements * n_quad_points):
            local_factors[index] = (
                factors[index] * test_grid_data.integration_elements[test_element]
            )
        for test_point_index in range(n_quad_points):
            test_global_point = test_global_points[:, test_point_index]
            kernel_values = kernel_evaluator(
                test_global_point,
                trial_global_points,
                test_normal,
                trial_normals,
                kernel_parameters,
            )
            for index in range(n_trial_elements * n_quad_points):
                tmp[index] = kernel_values[index] * (
                    local_factors[index] * quad_weights[test_point_index]
                )

            for trial_element_index in range(n_trial_elements):
                if is_adjacent[trial_element_index]:
                    continue
                trial_element = trial_elements[trial_element_index] # New: enumerate, result and mutipliers
                for i, test_fun_index in enumerate(_np.arange(nshape_test)[test_fun_index_mask[test_element_index]]):
                    for j, trial_fun_index in enumerate(_np.arange(nshape_trial)[trial_fun_index_mask[trial_element_index]]):
                        for quad_point_index in range(n_quad_points):
                            result[
                                test_offsets[test_element_index] + i, trial_offsets[trial_element_index] + j
                            ] += (
                                tmp[
                                    trial_element_index * n_quad_points
                                    + quad_point_index
                                ]
                                * local_trial_fun_values[
                                    0, trial_fun_index, quad_point_index
                                ]
                                * local_test_fun_values[
                                    0, test_fun_index, test_point_index
                                ]
                            )
                        result[
                            test_offsets[test_element_index] + i, trial_offsets[trial_element_index] + j
                        ] *= (
                            test_multipliers[test_element, test_fun_index]
                            * trial_multipliers[trial_element, trial_fun_index]
                        )

@_numba.jit(
    nopython=True, parallel=True, error_model="numpy", fastmath=True, boundscheck=False, cache=True
)
def helmholtz_hypersingular_regular_partial(
    test_grid_data,
    trial_grid_data,
    nshape_test,
    nshape_trial,
    test_elements,
    trial_elements,
    test_multipliers,
    trial_multipliers,
    test_partial_dofs,
    trial_partial_dofs,
    test_fun_index_dict,
    trial_fun_index_dict,
    test_normal_multipliers,
    trial_normal_multipliers,
    quad_points,
    quad_weights,
    kernel_evaluator,
    kernel_parameters,
    grids_identical,
    test_shapeset,
    trial_shapeset,
    result,
):
    """Evaluate Helmholtz hypersingular kernel."""
    # Compute global points
    wavenumber = kernel_parameters[0] + 1j * kernel_parameters[1]
    dtype = test_grid_data.vertices.dtype
    result_type = result.dtype
    n_quad_points = len(quad_weights)
    n_test_elements = len(test_elements)
    n_trial_elements = len(trial_elements)

    local_test_fun_values = test_shapeset(quad_points)
    local_trial_fun_values = trial_shapeset(quad_points)
    trial_normals = get_normals(
        trial_grid_data, n_quad_points, trial_elements, trial_normal_multipliers
    )
    trial_global_points = get_global_points(
        trial_grid_data, trial_elements, quad_points
    )

    factors = _np.empty(
        n_quad_points * n_trial_elements, dtype=trial_global_points.dtype
    )
    for trial_element_index in range(n_trial_elements):
        for trial_point_index in range(n_quad_points):
            factors[n_quad_points * trial_element_index + trial_point_index] = (
                quad_weights[trial_point_index]
                * trial_grid_data.integration_elements[
                    trial_elements[trial_element_index]
                ]
            )

    reference_gradient = _np.array([[-1, 1, 0], [-1, 0, 1]], dtype=dtype)

    test_surface_curls_trans = _np.empty((n_test_elements, 3, 3), dtype=dtype)
    trial_surface_curls = _np.empty((n_trial_elements, 3, 3), dtype=dtype)

    for test_index in range(n_test_elements):
        test_element = test_elements[test_index]
        test_surface_gradients = (
            test_grid_data.jac_inv_trans[test_element] @ reference_gradient
        )
        for i in range(3):
            test_surface_curls_trans[test_index, i, :] = (
                _np.cross(
                    test_grid_data.normals[test_element], test_surface_gradients[:, i]
                )
                * test_normal_multipliers[test_element]
            )

    for trial_index in range(n_trial_elements):
        trial_element = trial_elements[trial_index]
        trial_surface_gradients = (
            trial_grid_data.jac_inv_trans[trial_element] @ reference_gradient
        )
        for i in range(3):
            trial_surface_curls[trial_index, :, i] = (
                _np.cross(
                    trial_grid_data.normals[trial_element],
                    trial_surface_gradients[:, i],
                )
                * trial_normal_multipliers[trial_element]
            )

    for i in _numba.prange(n_test_elements):
        test_element = test_elements[i]
        local_result = _np.zeros(
            (n_trial_elements, nshape_test, nshape_trial), dtype=result_type
        )
        test_global_points = test_grid_data.local2global(test_element, quad_points)
        test_normal = (
            test_grid_data.normals[test_element] * test_normal_multipliers[test_element]
        )
        local_factors = _np.empty(
            n_trial_elements * n_quad_points, dtype=test_global_points.dtype
        )
        tmp = _np.empty(n_trial_elements * n_quad_points, dtype=result_type)
        is_adjacent = _np.zeros(n_trial_elements, dtype=_np.bool_)

        for trial_element_index in range(n_trial_elements):
            trial_element = trial_elements[trial_element_index]
            if grids_identical and elements_adjacent(
                test_grid_data.elements, test_element, trial_element
            ):
                is_adjacent[trial_element_index] = True

        for index in range(n_trial_elements * n_quad_points):
            local_factors[index] = (
                factors[index] * test_grid_data.integration_elements[test_element]
            )
        for test_point_index in range(n_quad_points):
            test_global_point = test_global_points[:, test_point_index]
            kernel_values = kernel_evaluator(
                test_global_point,
                trial_global_points,
                test_normal,
                trial_normals,
                kernel_parameters,
            )
            for index in range(n_trial_elements * n_quad_points):
                tmp[index] = kernel_values[index] * (
                    local_factors[index] * quad_weights[test_point_index]
                )

            for trial_element_index in range(n_trial_elements):
                if is_adjacent[trial_element_index]:
                    continue
                trial_element = trial_elements[trial_element_index]
                trial_normal = (
                    trial_grid_data.normals[trial_element]
                    * trial_normal_multipliers[trial_element]
                )
                normal_prod = _np.dot(test_normal, trial_normal)
                curl_product = (
                    test_surface_curls_trans[i]
                    @ trial_surface_curls[trial_element_index]
                )
                # for test_fun_index in range(nshape_test): # EDIT
                # for test_fun_index in test_fun_index_dict[test_element]: # EDIT2
                for test_fun_index in _np.arange(nshape_test)[test_fun_index_dict[test_element]]:
                    # for trial_fun_index in range(nshape_trial): # EDIT
                    # for trial_fun_index in trial_fun_index_dict[trial_element]: # EDIT2
                    for trial_fun_index in _np.arange(nshape_trial)[trial_fun_index_dict[trial_element]]:
                        for quad_point_index in range(n_quad_points):
                            local_result[
                                trial_element_index, test_fun_index, trial_fun_index
                            ] += tmp[
                                trial_element_index * n_quad_points + quad_point_index
                            ] * (
                                curl_product[test_fun_index, trial_fun_index]
                                - wavenumber
                                * wavenumber
                                * local_test_fun_values[
                                    0, test_fun_index, test_point_index
                                ]
                                * local_trial_fun_values[
                                    0, trial_fun_index, quad_point_index
                                ]
                                * normal_prod
                            )

        for trial_element_index in range(n_trial_elements):
            trial_element = trial_elements[trial_element_index]
            # for test_fun_index in range(nshape_test): # EDIT
            # for test_fun_index in test_fun_index_dict[test_element]: # EDIT2
            for test_fun_index in _np.arange(nshape_test)[test_fun_index_dict[test_element]]:
                # for trial_fun_index in range(nshape_trial): # EDIT
                # for trial_fun_index in trial_fun_index_dict[trial_element]: # EDIT2
                for trial_fun_index in _np.arange(nshape_trial)[trial_fun_index_dict[trial_element]]:
                    result[
                        # test_global_dofs[test_element, test_fun_index], # EDIT
                        test_partial_dofs[test_element, test_fun_index],
                        # trial_global_dofs[trial_element, trial_fun_index], # EDIT
                        trial_partial_dofs[trial_element, trial_fun_index],
                    ] += (
                        local_result[
                            trial_element_index, test_fun_index, trial_fun_index
                        ]
                        * test_multipliers[test_element, test_fun_index]
                        * trial_multipliers[trial_element, trial_fun_index]
                    )

@_numba.jit(
    nopython=True, parallel=True, error_model="numpy", fastmath=True, boundscheck=False, cache=True
)
def helmholtz_hypersingular_regular_partial2(
    test_grid_data,
    trial_grid_data,
    nshape_test,
    nshape_trial,
    test_elements,
    trial_elements,
    test_multipliers,
    trial_multipliers,
    test_partial_dofs,
    trial_partial_dofs,
    test_fun_index_mask,    # New
    trial_fun_index_mask,   # New
    test_offsets,           # New
    trial_offsets,          # New
    test_normal_multipliers,
    trial_normal_multipliers,
    quad_points,
    quad_weights,
    kernel_evaluator,
    kernel_parameters,
    grids_identical,
    test_shapeset,
    trial_shapeset,
    result,
):
    """Evaluate Helmholtz hypersingular kernel."""
    # Compute global points
    wavenumber = kernel_parameters[0] + 1j * kernel_parameters[1]
    dtype = test_grid_data.vertices.dtype
    result_type = result.dtype
    n_quad_points = len(quad_weights)
    n_test_elements = len(test_elements)
    n_trial_elements = len(trial_elements)

    local_test_fun_values = test_shapeset(quad_points)
    local_trial_fun_values = trial_shapeset(quad_points)
    trial_normals = get_normals(
        trial_grid_data, n_quad_points, trial_elements, trial_normal_multipliers
    )
    trial_global_points = get_global_points(
        trial_grid_data, trial_elements, quad_points
    )

    factors = _np.empty(
        n_quad_points * n_trial_elements, dtype=trial_global_points.dtype
    )
    for trial_element_index in range(n_trial_elements):
        for trial_point_index in range(n_quad_points):
            factors[n_quad_points * trial_element_index + trial_point_index] = (
                quad_weights[trial_point_index]
                * trial_grid_data.integration_elements[
                    trial_elements[trial_element_index]
                ]
            )

    reference_gradient = _np.array([[-1, 1, 0], [-1, 0, 1]], dtype=dtype)

    test_surface_curls_trans = _np.empty((n_test_elements, 3, 3), dtype=dtype)
    trial_surface_curls = _np.empty((n_trial_elements, 3, 3), dtype=dtype)

    for test_index in range(n_test_elements):
        test_element = test_elements[test_index]
        test_surface_gradients = (
            test_grid_data.jac_inv_trans[test_element] @ reference_gradient
        )
        for i in range(3):
            test_surface_curls_trans[test_index, i, :] = (
                _np.cross(
                    test_grid_data.normals[test_element], test_surface_gradients[:, i]
                )
                * test_normal_multipliers[test_element]
            )

    for trial_index in range(n_trial_elements):
        trial_element = trial_elements[trial_index]
        trial_surface_gradients = (
            trial_grid_data.jac_inv_trans[trial_element] @ reference_gradient
        )
        for i in range(3):
            trial_surface_curls[trial_index, :, i] = (
                _np.cross(
                    trial_grid_data.normals[trial_element],
                    trial_surface_gradients[:, i],
                )
                * trial_normal_multipliers[trial_element]
            )

    for test_element_index in _numba.prange(n_test_elements):
        test_element = test_elements[test_element_index]
        test_global_points = test_grid_data.local2global(test_element, quad_points)
        test_normal = (
            test_grid_data.normals[test_element] * test_normal_multipliers[test_element]
        )
        local_factors = _np.empty(
            n_trial_elements * n_quad_points, dtype=test_global_points.dtype
        )
        tmp = _np.empty(n_trial_elements * n_quad_points, dtype=result_type)
        is_adjacent = _np.zeros(n_trial_elements, dtype=_np.bool_)

        for trial_element_index in range(n_trial_elements):
            trial_element = trial_elements[trial_element_index]
            if grids_identical and elements_adjacent(
                test_grid_data.elements, test_element, trial_element
            ):
                is_adjacent[trial_element_index] = True

        for index in range(n_trial_elements * n_quad_points):
            local_factors[index] = (
                factors[index] * test_grid_data.integration_elements[test_element]
            )
        for test_point_index in range(n_quad_points):
            test_global_point = test_global_points[:, test_point_index]
            kernel_values = kernel_evaluator(
                test_global_point,
                trial_global_points,
                test_normal,
                trial_normals,
                kernel_parameters,
            )
            for index in range(n_trial_elements * n_quad_points):
                tmp[index] = kernel_values[index] * (
                    local_factors[index] * quad_weights[test_point_index]
                )

            for trial_element_index in range(n_trial_elements):
                if is_adjacent[trial_element_index]:
                    continue
                trial_element = trial_elements[trial_element_index]
                trial_normal = (
                    trial_grid_data.normals[trial_element]
                    * trial_normal_multipliers[trial_element]
                )
                normal_prod = _np.dot(test_normal, trial_normal)
                curl_product = (
                    test_surface_curls_trans[test_element_index]
                    @ trial_surface_curls[trial_element_index]
                )
                for i, test_fun_index in enumerate(_np.arange(nshape_test)[test_fun_index_mask[test_element_index]]):
                    for j, trial_fun_index in enumerate(_np.arange(nshape_trial)[trial_fun_index_mask[trial_element_index]]):
                        for quad_point_index in range(n_quad_points):
                            result[
                                test_offsets[test_element_index] + i, trial_offsets[trial_element_index] + j
                            ] += tmp[
                                trial_element_index * n_quad_points + quad_point_index
                            ] * (
                                curl_product[test_fun_index, trial_fun_index]
                                - wavenumber
                                * wavenumber
                                * local_test_fun_values[
                                    0, test_fun_index, test_point_index
                                ]
                                * local_trial_fun_values[
                                    0, trial_fun_index, quad_point_index
                                ]
                                * normal_prod
                            )
                        result[
                            test_offsets[test_element_index] + i, trial_offsets[trial_element_index] + j
                        ] *= (
                            test_multipliers[test_element, test_fun_index]
                            * trial_multipliers[trial_element, trial_fun_index]
                        )

@_numba.jit(
    nopython=True, parallel=True, error_model="numpy", fastmath=True, boundscheck=False, cache=True
)
def modified_helmholtz_hypersingular_regular_partial(
    test_grid_data,
    trial_grid_data,
    nshape_test,
    nshape_trial,
    test_elements,
    trial_elements,
    test_multipliers,
    trial_multipliers,
    test_partial_dofs,
    trial_partial_dofs,
    test_fun_index_dict,
    trial_fun_index_dict,
    test_normal_multipliers,
    trial_normal_multipliers,
    quad_points,
    quad_weights,
    kernel_evaluator,
    kernel_parameters,
    grids_identical,
    test_shapeset,
    trial_shapeset,
    result,
):
    """Evaluate Modified Helmholtz hypersingular kernel."""
    # Compute global points
    wavenumber = kernel_parameters[0]
    dtype = test_grid_data.vertices.dtype
    result_type = result.dtype
    n_quad_points = len(quad_weights)
    n_test_elements = len(test_elements)
    n_trial_elements = len(trial_elements)

    local_test_fun_values = test_shapeset(quad_points)
    local_trial_fun_values = trial_shapeset(quad_points)
    trial_normals = get_normals(
        trial_grid_data, n_quad_points, trial_elements, trial_normal_multipliers
    )
    trial_global_points = get_global_points(
        trial_grid_data, trial_elements, quad_points
    )

    factors = _np.empty(
        n_quad_points * n_trial_elements, dtype=trial_global_points.dtype
    )
    for trial_element_index in range(n_trial_elements):
        for trial_point_index in range(n_quad_points):
            factors[n_quad_points * trial_element_index + trial_point_index] = (
                quad_weights[trial_point_index]
                * trial_grid_data.integration_elements[
                    trial_elements[trial_element_index]
                ]
            )

    reference_gradient = _np.array([[-1, 1, 0], [-1, 0, 1]], dtype=dtype)

    test_surface_curls_trans = _np.empty((n_test_elements, 3, 3), dtype=dtype)
    trial_surface_curls = _np.empty((n_trial_elements, 3, 3), dtype=dtype)

    for test_index in range(n_test_elements):
        test_element = test_elements[test_index]
        test_surface_gradients = (
            test_grid_data.jac_inv_trans[test_element] @ reference_gradient
        )
        for i in range(3):
            test_surface_curls_trans[test_index, i, :] = (
                _np.cross(
                    test_grid_data.normals[test_element], test_surface_gradients[:, i]
                )
                * test_normal_multipliers[test_element]
            )

    for trial_index in range(n_trial_elements):
        trial_element = trial_elements[trial_index]
        trial_surface_gradients = (
            trial_grid_data.jac_inv_trans[trial_element] @ reference_gradient
        )
        for i in range(3):
            trial_surface_curls[trial_index, :, i] = (
                _np.cross(
                    trial_grid_data.normals[trial_element],
                    trial_surface_gradients[:, i],
                )
                * trial_normal_multipliers[trial_element]
            )

    for i in _numba.prange(n_test_elements):
        test_element = test_elements[i]
        local_result = _np.zeros(
            (n_trial_elements, nshape_test, nshape_trial), dtype=result_type
        )
        test_global_points = test_grid_data.local2global(test_element, quad_points)
        test_normal = (
            test_grid_data.normals[test_element] * test_normal_multipliers[test_element]
        )
        local_factors = _np.empty(
            n_trial_elements * n_quad_points, dtype=test_global_points.dtype
        )
        tmp = _np.empty(n_trial_elements * n_quad_points, dtype=result_type)
        is_adjacent = _np.zeros(n_trial_elements, dtype=_np.bool_)

        for trial_element_index in range(n_trial_elements):
            trial_element = trial_elements[trial_element_index]
            if grids_identical and elements_adjacent(
                test_grid_data.elements, test_element, trial_element
            ):
                is_adjacent[trial_element_index] = True

        for index in range(n_trial_elements * n_quad_points):
            local_factors[index] = (
                factors[index] * test_grid_data.integration_elements[test_element]
            )
        for test_point_index in range(n_quad_points):
            test_global_point = test_global_points[:, test_point_index]
            kernel_values = kernel_evaluator(
                test_global_point,
                trial_global_points,
                test_normal,
                trial_normals,
                kernel_parameters,
            )
            for index in range(n_trial_elements * n_quad_points):
                tmp[index] = kernel_values[index] * (
                    local_factors[index] * quad_weights[test_point_index]
                )

            for trial_element_index in range(n_trial_elements):
                if is_adjacent[trial_element_index]:
                    continue
                trial_element = trial_elements[trial_element_index]
                trial_normal = (
                    trial_grid_data.normals[trial_element]
                    * trial_normal_multipliers[trial_element]
                )
                normal_prod = _np.dot(test_normal, trial_normal)
                curl_product = (
                    test_surface_curls_trans[i]
                    @ trial_surface_curls[trial_element_index]
                )
                # for test_fun_index in range(nshape_test): # EDIT
                # for test_fun_index in test_fun_index_dict[test_element]: # EDIT2
                for test_fun_index in _np.arange(nshape_test)[test_fun_index_dict[test_element]]:
                    # for trial_fun_index in range(nshape_trial): # EDIT
                    # for trial_fun_index in trial_fun_index_dict[trial_element]: # EDIT2
                    for trial_fun_index in _np.arange(nshape_trial)[trial_fun_index_dict[trial_element]]:
                        for quad_point_index in range(n_quad_points):
                            local_result[
                                trial_element_index, test_fun_index, trial_fun_index
                            ] += tmp[
                                trial_element_index * n_quad_points + quad_point_index
                            ] * (
                                curl_product[test_fun_index, trial_fun_index]
                                + wavenumber
                                * wavenumber
                                * local_test_fun_values[
                                    0, test_fun_index, test_point_index
                                ]
                                * local_trial_fun_values[
                                    0, trial_fun_index, quad_point_index
                                ]
                                * normal_prod
                            )

        for trial_element_index in range(n_trial_elements):
            trial_element = trial_elements[trial_element_index]
            # for test_fun_index in range(nshape_test): # EDIT
            # for test_fun_index in test_fun_index_dict[test_element]: # EDIT2
            for test_fun_index in _np.arange(nshape_test)[test_fun_index_dict[test_element]]:
                # for trial_fun_index in range(nshape_trial): # EDIT
                # for trial_fun_index in trial_fun_index_dict[trial_element]: # EDIT2
                for trial_fun_index in _np.arange(nshape_trial)[trial_fun_index_dict[trial_element]]:
                    result[
                        # test_global_dofs[test_element, test_fun_index], # EDIT
                        test_partial_dofs[test_element, test_fun_index],
                        # trial_global_dofs[trial_element, trial_fun_index], # EDIT
                        trial_partial_dofs[trial_element, trial_fun_index],
                    ] += (
                        local_result[
                            trial_element_index, test_fun_index, trial_fun_index
                        ]
                        * test_multipliers[test_element, test_fun_index]
                        * trial_multipliers[trial_element, trial_fun_index]
                    )

@_numba.jit(
    nopython=True, parallel=True, error_model="numpy", fastmath=True, boundscheck=False, cache=True
)
def modified_helmholtz_hypersingular_regular_partial2(
    test_grid_data,
    trial_grid_data,
    nshape_test,
    nshape_trial,
    test_elements,
    trial_elements,
    test_multipliers,
    trial_multipliers,
    test_partial_dofs,
    trial_partial_dofs,
    test_fun_index_mask,    # New
    trial_fun_index_mask,   # New
    test_offsets,           # New
    trial_offsets,          # New
    test_normal_multipliers,
    trial_normal_multipliers,
    quad_points,
    quad_weights,
    kernel_evaluator,
    kernel_parameters,
    grids_identical,
    test_shapeset,
    trial_shapeset,
    result,
):
    """Evaluate Modified Helmholtz hypersingular kernel."""
    # Compute global points
    wavenumber = kernel_parameters[0]
    dtype = test_grid_data.vertices.dtype
    result_type = result.dtype
    n_quad_points = len(quad_weights)
    n_test_elements = len(test_elements)
    n_trial_elements = len(trial_elements)

    local_test_fun_values = test_shapeset(quad_points)
    local_trial_fun_values = trial_shapeset(quad_points)
    trial_normals = get_normals(
        trial_grid_data, n_quad_points, trial_elements, trial_normal_multipliers
    )
    trial_global_points = get_global_points(
        trial_grid_data, trial_elements, quad_points
    )

    factors = _np.empty(
        n_quad_points * n_trial_elements, dtype=trial_global_points.dtype
    )
    for trial_element_index in range(n_trial_elements):
        for trial_point_index in range(n_quad_points):
            factors[n_quad_points * trial_element_index + trial_point_index] = (
                quad_weights[trial_point_index]
                * trial_grid_data.integration_elements[
                    trial_elements[trial_element_index]
                ]
            )

    reference_gradient = _np.array([[-1, 1, 0], [-1, 0, 1]], dtype=dtype)

    test_surface_curls_trans = _np.empty((n_test_elements, 3, 3), dtype=dtype)
    trial_surface_curls = _np.empty((n_trial_elements, 3, 3), dtype=dtype)

    for test_index in range(n_test_elements):
        test_element = test_elements[test_index]
        test_surface_gradients = (
            test_grid_data.jac_inv_trans[test_element] @ reference_gradient
        )
        for i in range(3):
            test_surface_curls_trans[test_index, i, :] = (
                _np.cross(
                    test_grid_data.normals[test_element], test_surface_gradients[:, i]
                )
                * test_normal_multipliers[test_element]
            )

    for trial_index in range(n_trial_elements):
        trial_element = trial_elements[trial_index]
        trial_surface_gradients = (
            trial_grid_data.jac_inv_trans[trial_element] @ reference_gradient
        )
        for i in range(3):
            trial_surface_curls[trial_index, :, i] = (
                _np.cross(
                    trial_grid_data.normals[trial_element],
                    trial_surface_gradients[:, i],
                )
                * trial_normal_multipliers[trial_element]
            )

    for test_element_index in _numba.prange(n_test_elements):
        test_element = test_elements[test_element_index]
        test_global_points = test_grid_data.local2global(test_element, quad_points)
        test_normal = (
            test_grid_data.normals[test_element] * test_normal_multipliers[test_element]
        )
        local_factors = _np.empty(
            n_trial_elements * n_quad_points, dtype=test_global_points.dtype
        )
        tmp = _np.empty(n_trial_elements * n_quad_points, dtype=result_type)
        is_adjacent = _np.zeros(n_trial_elements, dtype=_np.bool_)

        for trial_element_index in range(n_trial_elements):
            trial_element = trial_elements[trial_element_index]
            if grids_identical and elements_adjacent(
                test_grid_data.elements, test_element, trial_element
            ):
                is_adjacent[trial_element_index] = True

        for index in range(n_trial_elements * n_quad_points):
            local_factors[index] = (
                factors[index] * test_grid_data.integration_elements[test_element]
            )
        for test_point_index in range(n_quad_points):
            test_global_point = test_global_points[:, test_point_index]
            kernel_values = kernel_evaluator(
                test_global_point,
                trial_global_points,
                test_normal,
                trial_normals,
                kernel_parameters,
            )
            for index in range(n_trial_elements * n_quad_points):
                tmp[index] = kernel_values[index] * (
                    local_factors[index] * quad_weights[test_point_index]
                )

            for trial_element_index in range(n_trial_elements):
                if is_adjacent[trial_element_index]:
                    continue
                trial_element = trial_elements[trial_element_index]
                trial_normal = (
                    trial_grid_data.normals[trial_element]
                    * trial_normal_multipliers[trial_element]
                )
                normal_prod = _np.dot(test_normal, trial_normal)
                curl_product = (
                    test_surface_curls_trans[test_element_index]
                    @ trial_surface_curls[trial_element_index]
                )
                for i, test_fun_index in enumerate(_np.arange(nshape_test)[test_fun_index_mask[test_element_index]]):
                    for j, trial_fun_index in enumerate(_np.arange(nshape_trial)[trial_fun_index_mask[trial_element_index]]):
                        for quad_point_index in range(n_quad_points):
                            result[
                                test_offsets[test_element_index] + i, trial_offsets[trial_element_index] + j
                            ] += tmp[
                                trial_element_index * n_quad_points + quad_point_index
                            ] * (
                                curl_product[test_fun_index, trial_fun_index]
                                + wavenumber
                                * wavenumber
                                * local_test_fun_values[
                                    0, test_fun_index, test_point_index
                                ]
                                * local_trial_fun_values[
                                    0, trial_fun_index, quad_point_index
                                ]
                                * normal_prod
                            )
                        result[
                            test_offsets[test_element_index] + i, trial_offsets[trial_element_index] + j
                        ] *= (
                            test_multipliers[test_element, test_fun_index]
                            * trial_multipliers[trial_element, trial_fun_index]
                        )