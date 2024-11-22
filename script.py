import bempp.api
import numpy as np
import MyHM.structures as stt
from MyHM.compression.aca import ACAPP_with_assembly, ACAPP
import MyHM
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from numba import config
from time import time
import os

def convert_to_preferred_format(sec):
   sec = sec % (24 * 3600)
   hour = sec // 3600
   sec %= 3600
   min = sec // 60
   sec %= 60
   return "%02d:%02d:%02d" % (hour, min, sec)

os.environ['NUMBA_DEBUG_CACHE'] = "1"

grid_name = "-h-4"

grid = bempp.api.import_grid(f'grids/ribcage4{grid_name}.msh')

print(f"\n{'='*50}")
print("GRID NAME:", grid_name)
print("CPUS:", config.NUMBA_DEFAULT_NUM_THREADS)
print(f"{'='*50}\n")

# bempp.api.DEFAULT_DEVICE_INTERFACE = 'opencl'
bempp.api.DEFAULT_DEVICE_INTERFACE = 'numba'

bbox = grid.bounding_box
space = bempp.api.function_space(grid, "P", 1)
print(f"Global DOF count: {space.global_dof_count}")
print(f"# Vertices: {grid.vertices.shape[1]}")
if not space.requires_dof_transformation:
    dof_indices = list(range(space.global_dof_count))
    excluded_vertices = np.unique(grid.elements[(space.local_multipliers == 0).T])
    mask = np.ones(grid.vertices.shape[1], dtype=bool)
    mask[excluded_vertices] = False
    vertices = grid.vertices[:, mask]
else:
    # TODO:
    raise NotImplementedError
    # space.dof_transformation.indices
print(f"# Vertices DOFs: {vertices.shape[1]}")

t0 = time()
octree = stt.Octree(vertices.T, dof_indices, bbox, grid.maximum_element_diameter, max_depth=4)
octree.generate_tree()
print(f"Generate Octree time: {time()-t0}")
print(f"Max Depth: {octree.max_depth}")

t0 = time()
tree_3d = stt.Tree3D(octree, octree, dtype=np.complex128)
tree_3d.generate_adm_tree()
print(f"Generate Tree3D time: {time()-t0}")

k = 7
dlp = bempp.api.operators.boundary.helmholtz.double_layer(space, space, space, k)
hyp = bempp.api.operators.boundary.helmholtz.hypersingular(space, space, space, k)
boundary_operator = dlp
# parameters = bempp.api.GLOBAL_PARAMETERS
parameters = boundary_operator.parameters
device_interface = "numba"

# Complete matrix:
if os.path.isfile(f"Inputs/A{grid_name}.npy"):
    A = np.load(f"Inputs/A{grid_name}.npy")
    b = np.load(f"Inputs/b{grid_name}.npy")
    print("Loaded A and b")
else:
    t0 = time()
    A = np.array(boundary_operator.weak_form().A)
    tf = time()
    print(f"Generate complete matrix A time: {tf-t0} s")
    print("=>", convert_to_preferred_format(tf-t0))
    b = np.random.rand(A.shape[1])
    np.save(f"Inputs/A{grid_name}.npy", A)
    np.save(f"Inputs/b{grid_name}.npy", b)
    print("Saved A and b")

print("Shapes A and b:", A.shape, b.shape)
result1 = A@b
tree_3d.add_vector(b)


print("\n========================")
print("Assemblers compilation:")
import MyHM.assembly as asb
rows = [0]
cols = [0]
t0 = time()
# asb.partial_dense_assembler(boundary_operator.descriptor, boundary_operator.domain, boundary_operator.dual_to_range, parameters, rows, cols)
asb.partial_dense_assembler2(boundary_operator.descriptor, boundary_operator.domain, boundary_operator.dual_to_range, parameters, rows, cols)
print(f"Compilation time: {time()-t0}")
t0 = time()
singular_matrix =  asb.singular_assembler_sparse(device_interface, boundary_operator.descriptor, boundary_operator.domain, boundary_operator.dual_to_range, parameters)
print(f"Singular matrix calculation time: {time()-t0}")
print(f"Singular adm leaves? {tree_3d.check_singular_adm_leaves(singular_matrix)}")
from functools import partial
assembler = partial(asb.partial_dense_assembler2, boundary_operator.descriptor, boundary_operator.domain, boundary_operator.dual_to_range, parameters)
print("========================")

epsilons = [2**(-1*i) for i in range(1,50,4)][:3]
formatted_epsilons = [np.format_float_scientific(e, precision=3) for e in epsilons]
errors2 = []
used_storages = []
total_storage_without_compression = tree_3d.calculate_matrix_storage_without_compression()

for i in range(len(epsilons)):
    if i < 10:
        string_i = f"0{i}"
    else:
        string_i = f"{i}"
    print()
    print("="*50)
    print(f"\n{string_i}) Epsilon: {formatted_epsilons[i]}\n")

    t0 = time()
    tree_3d.add_matrix_with_ACA(A, ACAPP, epsilon=epsilons[i], verbose=False)
    tf = time()
    print(f"Time of compression w/o assembler and w/assembled_values: {tf-t0} s")
    print("=>", convert_to_preferred_format(tf-t0))
    t0 = time()
    aux_result = tree_3d.matvec_compressed()
    tf = time()
    print(f"Time of matvec: {tf-t0} s")
    print("=>", convert_to_preferred_format(tf-t0))
    print("Relative error:", np.linalg.norm(result1 - aux_result) / np.linalg.norm(result1))
    # print(np.linalg.norm(A - tree_3d.get_matrix_from_compression()) / np.linalg.norm(A))
    print()
    tree_3d.clear_compression()

    t0 = time()
    tree_3d.add_compressed_matrix(ACAPP_with_assembly, assembler, singular_matrix, epsilon=epsilons[i], verbose=False)
    tf = time()
    print(f"Time of compression w/assembler: {tf-t0} s")
    print("=>", convert_to_preferred_format(tf-t0))
    t0 = time()
    aux_result = tree_3d.matvec_compressed()
    tf = time()
    print(f"Time of matvec: {tf-t0} s")
    print("=>", convert_to_preferred_format(tf-t0))
    print("Relative error:", np.linalg.norm(result1 - aux_result) / np.linalg.norm(result1))
    print()

    errors2.append(np.linalg.norm(result1 - aux_result) / np.linalg.norm(result1))
    used_storages.append(tree_3d.calculate_compressed_matrix_storage())
    tree_3d.pairplot(save=True, name=f"Results/Pairplot_{string_i}{grid_name}.png", extra_title=f"(epsilon = {formatted_epsilons[i]})")
    tree_3d.plot_storage_per_level(save=True, name=f"Results/Storage_per_level_{string_i}{grid_name}.png", extra_title=f"(epsilon = {formatted_epsilons[i]})")
    tree_3d.compression_imshow(save=True, name=f"Results/Imshow_{string_i}{grid_name}.png", extra_title=f"(epsilon = {formatted_epsilons[i]})")

    tree_3d.clear_compression()

plt.plot(epsilons, errors2, "o-", label="Relative error")
plt.plot(epsilons, epsilons, "or--", label="Epsilon")
plt.gca().invert_xaxis()
plt.yscale("log")
plt.xscale("log")
plt.xlabel("Epsilon")
plt.ylabel("Relative error")
plt.title("Relative errors in matvec operation")
plt.legend()
plt.savefig(f"Results/Relative_errors{grid_name}.png")
plt.close()

df = pd.DataFrame({"Used storage": used_storages, "Epsilon": formatted_epsilons})
ax = sns.barplot(df, x="Epsilon", y="Used storage", errorbar=None, gap=0.3)
bar_labels = np.round(np.asarray(used_storages) / total_storage_without_compression * 100, decimals=1)
ax.bar_label(ax.containers[0], fontsize=10, labels=map(lambda x: f"{x}%", bar_labels))
xlim = plt.xlim()
xlim = (xlim[0] - 0.5, xlim[1] + 0.5)
plt.xlim(xlim)
plt.axhline(y=total_storage_without_compression, label="Total storage\nw/o compression", linestyle="--", color='r') 
plt.title("Used storage progression with epsilon")
plt.ylabel("Used storage (Floating point units)")
plt.xticks(rotation=45)
plt.legend()
plt.savefig(f"Results/Final_storages{grid_name}.png", bbox_inches="tight")
plt.close()

# for i in range(2):
#     for j in range(2):
#         print((i,j))
#         assembler = partial(asb.partial_dense_assembler, boundary_operator.descriptor, boundary_operator.domain, boundary_operator.dual_to_range, parameters)
#         assembler = partial(asb.partial_dense_assembler2, boundary_operator.descriptor, boundary_operator.domain, boundary_operator.dual_to_range, parameters)
#         t0 = time()
#         tree_3d.add_compressed_matrix(ACAPP_with_assembly, assembler, singular_matrix, epsilon=epsilons[i], verbose=False)
#         print(f"Time of compression: {time()-t0}")
#         t0 = time()
#         aux_result = tree_3d.matvec_compressed()
#         tf = time()
#         print(f"Time of matvec: {tf-t0} s")
#         print("Relative error:", np.linalg.norm(result1 - aux_result) / np.linalg.norm(result1))
#         print()
#         np.save(f"Results/{i}_{j}.npy", np.array([tf-t0]))