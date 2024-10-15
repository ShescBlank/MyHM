import numpy as _np

# TODO: Quizás sea importante acá revisar el requires_dof_transformation ???
def get_elements_from_vertices(global2local, vertices):
    '''
    Obtain the elements associated to an array of vertices of the grid
    '''
    elements = []
    for vertex in vertices:
        # elements.extend(_np.asarray(global2local[vertex])[:,0])
        elements.extend([item[0] for item in global2local[vertex]])
    return _np.unique(elements)

def sort_partial_elements_by_color(space, target_elements):
    '''
    Sort the subset of elements (target_elements) based on the space predefined colormap
    '''
    color_map = space.color_map[target_elements]

    sorted_indices = _np.empty(len(target_elements), dtype="uint32")
    ncolors = 1 + max(space.color_map)
    # indexptr = _np.zeros(1 + ncolors, dtype="uint32")
    indexptr = [0]

    count = 0
    for index, color in enumerate(_np.arange(ncolors, dtype="uint32")):
        colors = _np.where(color_map == color)[0]
        colors_length = len(colors)
        if colors_length > 0:
            sorted_indices[count : count + colors_length] = target_elements[colors]
            count += colors_length
            # indexptr[index + 1] = count
            indexptr.append(count)
    return sorted_indices, _np.array(indexptr, dtype="uint32")

def create_fun_index_dicts(rows, cols, test_indices, trial_indices, test_global_dofs, trial_global_dofs):
    '''
        Creates dictionaries that indicate the vertices to be calculated for each element when using
        a kernel, avoiding to operate on others that do not belong to the target columns/rows.
    '''
    
    from numba.typed import Dict

    test_fun_index_mask = _np.isin(test_global_dofs[test_indices], rows)
    trial_fun_index_mask = _np.isin(trial_global_dofs[trial_indices], cols)
    test_fun_index_dict = Dict(zip(test_indices, test_fun_index_mask))
    trial_fun_index_dict = Dict(zip(trial_indices, trial_fun_index_mask))

    return test_fun_index_dict, trial_fun_index_dict

def create_partial_dofs(rows, cols, test_global_dofs, trial_global_dofs):
    ''' 
        Indicates where to put the partial results of the kernel
        associated to the target columns/rows (This is the translation of
        the positions of the complete matrix into one of the right size)
    '''
    test_partial_dofs = _np.zeros_like(test_global_dofs, dtype = int) - 1
    for i in range(len(rows)):
        test_partial_dofs[_np.where(test_global_dofs == rows[i])] = i
    
    trial_partial_dofs = _np.zeros_like(trial_global_dofs, dtype = int) - 1
    for i in range(len(cols)):
        trial_partial_dofs[_np.where(trial_global_dofs == cols[i])] = i

    return test_partial_dofs, trial_partial_dofs

def create_fun_index_masks_and_offsets(rows, cols, n_test_elements, n_trial_elements, test_partial_dofs, trial_partial_dofs):
    test_fun_index_mask = _np.isin(test_partial_dofs, rows)
    trial_fun_index_mask = _np.isin(trial_partial_dofs, cols)

    test_offsets = _np.empty(n_test_elements + 1, dtype="uint32")   #TODO: uint64?
    trial_offsets = _np.empty(n_trial_elements + 1, dtype="uint32") #TODO: uint64?
    test_offsets[0] = 0
    trial_offsets[0] = 0
    test_offsets[1:] = _np.cumsum(_np.sum(test_fun_index_mask, axis=1))
    trial_offsets[1:] = _np.cumsum(_np.sum(trial_fun_index_mask, axis=1))

    return test_fun_index_mask, trial_fun_index_mask, test_offsets, trial_offsets