import numpy as np
from vsie.operators.cross_interaction import cross_single_layer, cross_double_layer, cross_ad_double_layer
from vsie.operators.self_interaction import single_layer, double_layer, ad_double_layer, mass_matrix
from vsie.operators.wave_op import incident_plane
from vsie.geometry.grid import concentric_cubes
from vsie.geometry.space_fun import physical_functions, discont_physical_functions, anal_grad, surface_dif, max_wavenumber
from vsie.post_processing.graphs import create_slice
from scipy.sparse.linalg import gmres

'''           Parameters           '''
l_cube_ext = 7                    # Size exterior cube
l_cube_int = 5                    # Size interior cube
hl = l_cube_int / 2               # half the lenght of interior cube
wavespeed = 1482300               # Exterior wavespeed
frec = wavespeed / 4              # Exterior frequency
lambda_ext = wavespeed / frec     # Exterior wavelength
kappa = 2 * np.pi / lambda_ext    # Exterior wavenumber
rho_0 = 1                         # Exterior density


# Space functions
gl = l_cube_int/2
xi = np.pi / gl
am = 0.8

# We define the parameters as functions.
int_density = lambda x, y, z: am * np.sin((x - gl)*xi) * np.sin((y - gl)*xi) * np.sin((z-gl)*0.5*xi) + 2
ext_density = lambda x, y, z: rho_0
# We define the density on the surface as the mean value of the traces.
surf_density = lambda x, y, z: 2*int_density(x,y,z)*ext_density(x,y,z) /(int_density(x,y,z) + ext_density(x,y,z))
int_wave = lambda x, y, z: 1.2 * kappa
ext_wave = lambda x, y, z: 1.2 * kappa

dom_lims = [[-gl, gl], [-gl, gl], [-gl, gl]]
k_max = max_wavenumber(dom_lims, int_wave) # We get the maximum wave number

# Obtain the size of a voxel
lambda_min = 2 * np.pi / (k_max) # Define the smallest wavelength

'''         Grid         '''
# Creates the grid
vox_per_wave = 14
space, idx_int, idx_ext, surf_grid, w, sw, normals = concentric_cubes(l_cube_int, l_cube_ext, lambda_min, vox_per_wave=vox_per_wave)
dx = abs(space[0, 2] - space[1, 2])


# Evaluate the space functions
alpha, beta = discont_physical_functions(kappa, rho_0, int_wave, ext_wave, int_density, ext_density, space, idx_int, idx_ext)
alpha_g, beta_g = physical_functions(kappa, rho_0, ext_wave, surf_density, surf_grid)
diff_alpha = surface_dif(rho_0, ext_density, int_density, surf_grid)

ext_cube, int_cube = space[idx_ext], space[idx_int]
ext_alpha, int_alpha = alpha[idx_ext], alpha[idx_int]
ext_beta, int_beta = beta[idx_ext], beta[idx_int]
ext_w, int_w = w[idx_ext], w[idx_int]

# We get the gradients of alpha
int_density_x = lambda x, y, z: am * xi * np.sin((y-gl)*xi) * np.sin((z-gl)*0.5*xi) * np.cos((x-gl)*xi)
int_density_y = lambda x, y, z: am * xi * np.sin((x-gl)*xi) * np.sin((z-gl)*0.5*xi) * np.cos((y-gl)*xi)
int_density_z = lambda x, y, z: am * 0.5 * xi * np.sin((x-gl)*xi) * np.sin((y-gl)*xi) * np.cos((z-gl)*0.5*xi)

ext_density_x = lambda x, y, z: 0
ext_density_y = lambda x, y, z: 0
ext_density_z = lambda x, y, z: 0

int_grad_alpha = anal_grad(int_cube, int_density, int_density_x, int_density_y, int_density_z)
ext_grad_alpha = anal_grad(ext_cube, ext_density, ext_density_x, ext_density_y, ext_density_z)

'''        Operators       '''

# Evaluate operators
# mass_op_ext = mass_matrix(ext_cube, ext_alpha, rho_0)
# mass_op_int = mass_matrix(int_cube, int_alpha, rho_0)
# mass_op_bdy = mass_matrix(surf_grid, alpha_g, rho_0)
sl_op_ext = single_layer(kappa, ext_cube, ext_alpha, ext_beta, ext_w)
sl_op_int = single_layer(kappa, int_cube, int_alpha, int_beta, int_w)
sl_op_bdy_ext = cross_single_layer(kappa, surf_grid, ext_cube, ext_alpha, ext_beta, ext_w)
sl_op_bdy_int = cross_single_layer(kappa, surf_grid, int_cube, int_alpha, int_beta, int_w)
sl_op_ext_int = cross_single_layer(kappa, ext_cube, int_cube, int_alpha, int_beta, int_w)
sl_op_int_ext = cross_single_layer(kappa, int_cube, ext_cube, ext_alpha, ext_beta, ext_w)
dl_op_ext_bdy = cross_double_layer(kappa, ext_cube, surf_grid, normals, diff_alpha, sw)
dl_op_int_bdy = cross_double_layer(kappa, int_cube, surf_grid, normals, diff_alpha, sw)
dl_op_bdy = double_layer(kappa, surf_grid, normals, diff_alpha, sw)
ad_dl_op_ext = ad_double_layer(kappa, ext_cube, ext_grad_alpha, ext_w)
ad_dl_op_int = ad_double_layer(kappa, int_cube, int_grad_alpha, int_w)
ad_dl_op_bdy_ext = cross_ad_double_layer(kappa, surf_grid, ext_cube, ext_grad_alpha, ext_w)
ad_dl_op_bdy_int = cross_ad_double_layer(kappa, surf_grid, int_cube, int_grad_alpha, int_w)
ad_dl_op_ext_int = cross_ad_double_layer(kappa, ext_cube, int_cube, int_grad_alpha, int_w) 
ad_dl_op_int_ext = cross_ad_double_layer(kappa, int_cube, ext_cube, ext_grad_alpha, ext_w)

import bempp.api
import numpy as np
import MyHM.structures as stt
from MyHM.compression.aca import ACAPP
from time import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

print()
octrees = []

spaces = [int_cube, ext_cube, surf_grid]
space_names = ["int_cube", "ext_cube", "surf_grid"]
for i in range(len(spaces)):
    # Vertices:
    vertices = spaces[i]
    print(f"Space name: {space_names[i]}")
    print("Shape vertices:", vertices.shape)
    
    # Bbox:
    wavelength = lambda_min
    vox_length = wavelength/vox_per_wave
    print("Vox length:", vox_length)
    bbox = np.array([np.min(vertices, axis=0) - vox_length/2, np.max(vertices, axis=0) + vox_length/2]).T
    print("Bbox:", bbox)
    
    # DOF indices:
    dof_indices = list(range(vertices.shape[0]))
    print("DOF indices:", len(dof_indices))
    
    # Octree:
    maximum_element_diameter = vox_length
    octree = stt.Octree(vertices, dof_indices, bbox, maximum_element_diameter, max_depth=None)
    ti = time()
    octree.generate_tree()
    tf = time()
    octrees.append(octree)
    print("Max depth octree:", octree.max_depth)
    print(f"Generate Octree time: {tf-ti}")
    print()

octree_int, octree_ext, octree_bdy = octrees

# Tree3D:
print("Int Int:")
tree_3d_int_int = stt.Tree3D(octree_int, octree_int)
t0 = time()
tree_3d_int_int.generate_adm_tree()
print(f"Time of generation: {time()-t0} s")
print()

print("Ext Ext:")
tree_3d_ext_ext = stt.Tree3D(octree_ext, octree_ext)
t0 = time()
tree_3d_ext_ext.generate_adm_tree()
print(f"Time of generation: {time()-t0} s")
print()

print("Bdy Bdy:")
tree_3d_bdy_bdy = stt.Tree3D(octree_bdy, octree_bdy)
t0 = time()
tree_3d_bdy_bdy.generate_adm_tree()
print(f"Time of generation: {time()-t0} s")
print()

print("Int Ext:")
tree_3d_int_ext = stt.Tree3D(octree_int, octree_ext)
t0 = time()
tree_3d_int_ext.generate_adm_tree()
print(f"Time of generation: {time()-t0} s")
print()

print("Ext Int:")
tree_3d_ext_int = stt.Tree3D(octree_ext, octree_int)
t0 = time()
tree_3d_ext_int.generate_adm_tree()
print(f"Time of generation: {time()-t0} s")
print()

print("Bdy Int:")
tree_3d_bdy_int = stt.Tree3D(octree_bdy, octree_int)
t0 = time()
tree_3d_bdy_int.generate_adm_tree()
print(f"Time of generation: {time()-t0} s")
print()

print("Bdy Ext:")
tree_3d_bdy_ext = stt.Tree3D(octree_bdy, octree_ext)
t0 = time()
tree_3d_bdy_ext.generate_adm_tree()
print(f"Time of generation: {time()-t0} s")
print()

print("Int Bdy:")
tree_3d_int_bdy = stt.Tree3D(octree_int, octree_bdy)
t0 = time()
tree_3d_int_bdy.generate_adm_tree()
print(f"Time of generation: {time()-t0} s")
print()

print("Ext Bdy:")
tree_3d_ext_bdy = stt.Tree3D(octree_ext, octree_bdy)
t0 = time()
tree_3d_ext_bdy.generate_adm_tree()
print(f"Time of generation: {time()-t0} s")
print()

print(tree_3d_int_int.stats)
print(tree_3d_ext_ext.stats)
print(tree_3d_bdy_bdy.stats)
print()

# Matrix and vector:
As = [sl_op_ext, sl_op_int, sl_op_bdy_ext, sl_op_bdy_int, sl_op_ext_int, sl_op_int_ext, dl_op_ext_bdy, dl_op_int_bdy, dl_op_bdy, ad_dl_op_ext, ad_dl_op_int, ad_dl_op_bdy_ext, ad_dl_op_bdy_int, ad_dl_op_ext_int, ad_dl_op_int_ext]
trees = [tree_3d_ext_ext, tree_3d_int_int, tree_3d_bdy_ext, tree_3d_bdy_int, tree_3d_ext_int, tree_3d_int_ext, tree_3d_ext_bdy, tree_3d_int_bdy, tree_3d_bdy_bdy, tree_3d_ext_ext, tree_3d_int_int, tree_3d_bdy_ext, tree_3d_bdy_int, tree_3d_ext_int, tree_3d_int_ext]
names = ["sl_op_ext", "sl_op_int", "sl_op_bdy_ext", "sl_op_bdy_int", "sl_op_ext_int", "sl_op_int_ext", "dl_op_ext_bdy", "dl_op_int_bdy", "dl_op_bdy", "ad_dl_op_ext", "ad_dl_op_int", "ad_dl_op_bdy_ext", "ad_dl_op_bdy_int", "ad_dl_op_ext_int", "ad_dl_op_int_ext"]

for index in range(len(As)):
    os.makedirs(f"Results/{names[index]}")
    A = As[index]
    tree_3d = trees[index]
    b = np.random.rand(A.shape[1])
    print(f"{index}) {names[index]}")
    print("Shape A:\t", A.shape)
    print("Shape Tree3D:\t", tree_3d.shape())
    print("Spy A:")
    plt.spy(A)
    plt.title(f"Matrix: {names[index]}")
    # plt.show()
    plt.savefig(f"Results/{names[index]}/Spy.png")
    plt.close()
    print()
    
    result1 = A@b
    tree_3d.add_vector(b)
    
    epsilons = [2**(-1*i) for i in range(1,50,4)][:4]
    formatted_epsilons = [np.format_float_scientific(e, precision=3) for e in epsilons]
    errors2 = []
    used_storages = []
    total_storage_without_compression = tree_3d.calculate_matrix_storage_without_compression()
    
    for i in range(len(epsilons)):
        if i < 10:
            string_i = f"0{i}"
        else:
            string_i = f"{i}"
        # print()
        # print("="*50)
        print(f"\n{string_i}) Epsilon: {formatted_epsilons[i]}\n")
    
        t0 = time()
        tree_3d.add_matrix_with_ACA(A, ACAPP, epsilon=epsilons[i], verbose=False)
        tf = time()
        print(f"Time of compression w/o assembler and w/assembled_values: {tf-t0} s")
        t0 = time()
        aux_result = tree_3d.matvec_compressed(dtype=np.complex128)
        tf = time()
        print(f"Time of matvec: {tf-t0} s")
        print("Relative error:", np.linalg.norm(result1 - aux_result) / np.linalg.norm(result1))
        print()
    
        errors2.append(np.linalg.norm(result1 - aux_result) / np.linalg.norm(result1))
        used_storages.append(tree_3d.calculate_compressed_matrix_storage())
        tree_3d.pairplot(save=True, name=f"Results/{names[index]}/Pairplot_{string_i}.png", extra_title=f"(epsilon = {formatted_epsilons[i]})")
        tree_3d.plot_storage_per_level(save=True, name=f"Results/{names[index]}/Storage_per_level_{string_i}.png", extra_title=f"(epsilon = {formatted_epsilons[i]})")
        tree_3d.compression_imshow(save=True, name=f"Results/{names[index]}/Imshow_{string_i}.png", extra_title=f"(epsilon = {formatted_epsilons[i]})")

    plt.plot(epsilons, errors2, "o-", label="Relative error")
    plt.plot(epsilons, epsilons, "or--", label="Epsilon")
    plt.gca().invert_xaxis()
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Epsilon")
    plt.ylabel("Relative error")
    plt.title("Relative errors in matvec operation")
    plt.legend()
    # plt.show()
    plt.savefig(f"Results/{names[index]}/Relative_errors.png")
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
    # plt.show()
    plt.savefig(f"Results/{names[index]}/Final_storages.png", bbox_inches="tight")
    plt.close()

    print(f"\n{'='*100}\n")

    tree_3d.clear_compression()
    del A, b
