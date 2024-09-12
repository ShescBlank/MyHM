import bempp.api
import numpy as np
import MyHM.structures as stt
from MyHM.compression.aca import ACAPP_with_assembly, ACAPP
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from time import time

# grid = bempp.api.import_grid('grids/ribcage4-h-4.msh')
# grid = bempp.api.import_grid('grids/ribcage3-h-1.msh')
grid = bempp.api.import_grid('grids/ribcage4-h-0.5.msh')

# bempp.api.DEFAULT_DEVICE_INTERFACE = 'opencl'
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

t0 = time()
octree = stt.Octree(vertices.T, dof_indices, bbox, grid.maximum_element_diameter, max_depth=4)
octree.generate_tree()
print(f"Generate Octree time: {time()-t0}")

t0 = time()
tree_3d = stt.Tree3D(octree)
tree_3d.generate_adm_tree()
print(f"Generate Tree3D time: {time()-t0}")

k = 7
dlp = bempp.api.operators.boundary.helmholtz.double_layer(space, space, space, k)
hyp = bempp.api.operators.boundary.helmholtz.hypersingular(space, space, space, k)
boundary_operator = dlp
parameters = bempp.api.GLOBAL_PARAMETERS
device_interface = "numba"

# Complete matrix:
t0 = time()
A = np.array(boundary_operator.weak_form().A)
print(f"Generate complete matrix A time: {time()-t0}")
b = np.random.rand(A.shape[1])

np.save("Inputs/A.npy", A)
np.save("Inputs/b.npy", b)
print("Saved A and b")

result1 = A@b
tree_3d.add_vector(b)

print(f"Singular adm leaves? {tree_3d.check_singular_adm_leaves(device_interface, boundary_operator, parameters)}")

errors2 = []
epsilons = [2**(-1*i) for i in range(1,50,4)]
used_storages = []
total_storage_without_compression = tree_3d.calculate_matrix_storage_without_compression()

for i in range(len(epsilons)):
    if i < 10:
        string_i = f"0{i}"
    else:
        string_i = f"{i}"

    print(f"\n{string_i}) Epsilon:{epsilons[i]}")
    t0 = time()
    # tree_3d.add_compressed_matrix(ACAPP_with_assembly, device_interface, boundary_operator, parameters, epsilon=epsilons[i], verbose=False)
    tree_3d.add_matrix_with_ACA(A, ACAPP, epsilon=epsilons[i], verbose=False)
    print(f"Time of compression: {time()-t0}")
    t0 = time()
    aux_result = tree_3d.matvec_compressed(dtype=np.complex128)
    print(f"Time of matvec: {time()-t0}")
    errors2.append(np.linalg.norm(result1 - aux_result) / np.linalg.norm(result1))
    used_storages.append(tree_3d.calculate_compressed_matrix_storage())
    tree_3d.pairplot(save=True, name=f"Results/Pairplot_{string_i}")
    tree_3d.plot_leaves_stats(save=True, name=f"Results/Storage_per_level_{string_i}")
    tree_3d.compression_imshow(save=True, name=f"Results/Imshow_{string_i}")

plt.plot(epsilons, errors2, "o-", label="Relative error")
plt.plot(epsilons, epsilons, "or--", label="Epsilon")
plt.gca().invert_xaxis()
plt.yscale("log")
plt.xscale("log")
plt.xlabel("Epsilon")
plt.ylabel("Relative error")
plt.title("Relative errors in matvec operation")
plt.legend()
plt.savefig("Results/Relative_errors")
plt.close()

formatted_epsilons = [np.format_float_scientific(e, precision=3) for e in epsilons]
df = pd.DataFrame({"Used storage": used_storages, "Epsilon": formatted_epsilons})
ax = sns.barplot(df, x="Epsilon", y="Used storage", errorbar=None, gap=0.3)
bar_labels = np.round(np.asarray(used_storages) / total_storage_without_compression * 100, decimals=1)
ax.bar_label(ax.containers[0], fontsize=10, labels=map(lambda x: f"{x}%", bar_labels))
xlim = plt.xlim()
xlim = (xlim[0] - 0.5, xlim[1] + 0.5)
plt.xlim(xlim)
plt.axhline(y=total_storage_without_compression, label="Total storage\nw/o compression", linestyle="--", color='r') 
plt.title("Used storage progression with epsilon")
plt.xticks(rotation=45)
plt.legend()
plt.savefig("Results/Final_storages", bbox_inches="tight")
plt.close()



