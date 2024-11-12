# Para correr este código en el contenedor de Docker:
# Abrir una terminal desde el notebook y correr la siguiente línea de código
# py-spy record -o NAME_SVG.svg -- python3 profiler.py
# --subprocesses (para medir los subprocesos también) 
# --rate 10 (para modificar el sampling rate)

# Ejecución completa con speedscope:
# py-spy record --rate 300 --subprocesses -f speedscope -o speedscope2.json -- python3 profiler.py
# Página para revisar resultados:
# https://www.speedscope.app/

import bempp.api
import numpy as np
import MyHM.structures as stt
from MyHM.compression.aca import ACAPP_with_assembly, ACAPP

grid = bempp.api.shapes.sphere(h=0.3)
bempp.api.DEFAULT_DEVICE_INTERFACE = 'numba'

bbox = grid.bounding_box
vertices = grid.vertices
space = bempp.api.function_space(grid, "P", 1)
if not space.requires_dof_transformation:
    dof_indices = list(range(vertices.shape[1]))
else:
    # TODO:
    raise NotImplementedError
    # space.dof_transformation.indices

octree = stt.Octree(vertices.T, dof_indices, bbox, max_depth=4)
octree.generate_tree()

tree_3d = stt.Tree3D(octree, octree, dtype=np.complex128)
tree_3d.generate_adm_tree()

k = 7
dlp = bempp.api.operators.boundary.helmholtz.double_layer(space, space, space, k)
hyp = bempp.api.operators.boundary.helmholtz.hypersingular(space, space, space, k)
boundary_operator = dlp
parameters = bempp.api.GLOBAL_PARAMETERS
device_interface = "numba"

# Complete matrix:
matrix_wf = boundary_operator.weak_form()
matrix_wf = boundary_operator.weak_form() # Without compiling
A = np.array(matrix_wf.A)

# Compressed matrix in tree:
tree_3d.add_compressed_matrix(ACAPP_with_assembly, device_interface, boundary_operator, parameters, epsilon=1e-3, verbose=False)