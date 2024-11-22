
import numpy as np
from vsie.operators.cross_interaction import cross_single_layer, cross_double_layer, cross_ad_double_layer
from vsie.operators.self_interaction import single_layer, double_layer, ad_double_layer, mass_matrix
from vsie.operators.cross_interaction import cross_single_layer_partial, cross_double_layer_partial, cross_ad_double_layer_partial
from vsie.operators.self_interaction import single_layer_partial, double_layer_partial, ad_double_layer_partial
from vsie.operators.wave_op import incident_plane
# from vsie.geometry.grid import concentric_slabs
# from vsie.geometry.space_fun import physical_functions, discont_physical_functions, anal_grad, surface_dif, max_wavenumber
from vsie.post_processing.graphs import create_slice
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt
from time import time

# OJO (dependiendo del ejemplo que se quiera correr): 
# No olvidar cambiar los max_depth, los np.load del inicio (puntos, densidades y velocidades) y los dx,dy,dz
# Los cálculos de los operadores y la lista de As (y no indexarla después)
# MD: 4

# Cargamos los puntos de los voxels, además de las densidades y velocidades:
# - Skull slab:
points = np.load("Voxels skull slab/voxel_points.npy")
densities = np.load("Voxels skull slab/density_of_sound.npy")
speeds = np.load("Voxels skull slab/speed_of_sound.npy")
# - Ribs:
# points = np.load("Voxels skull slab/voxel_points_ribs.npy")
# mask_ribs = points[0,:] > -0.068
# points = points[:, mask_ribs]
# densities = np.load("Voxels skull slab/density_of_sound_ribs.npy")[mask_ribs]
# speeds = np.load("Voxels skull slab/speed_of_sound_ribs.npy")[mask_ribs]
# print("Points shape:", points.shape)

# Definimos dx, dy, dz de los voxels: (suponiendo que todos los voxels tienen los mismos tamaños)
# (Estos fallan para las costillas)
dx = np.max(np.diff(points[0,:]))
dy = np.max(np.diff(points[1,:]))
dz = np.max(np.diff(points[2,:]))
# dx = np.min(np.abs(np.diff(points[0,:]))[np.abs(np.diff(points[0,:])) > 0]) # Este funciona en todos los casos
# dy = np.min(np.abs(np.diff(points[1,:]))[np.abs(np.diff(points[1,:])) > 0])
# dz = np.min(np.abs(np.diff(points[2,:]))[np.abs(np.diff(points[2,:])) > 0])
vox_size = np.array([dx, dy, dz])
print("Vox size:", vox_size)

# Definimos la bounding box que encierra a todos los puntos:
mins = np.min(points, axis=1)
maxs = np.max(points, axis=1)
bbox = np.array([mins - (vox_size/2), maxs + (vox_size/2)]).T
print("Bounding box:", bbox)

# Cantidad de voxels si la bbox estuviera llena:
nx, ny, nz = np.ceil((bbox[:,1] - bbox[:,0]) / vox_size) # TODO: decidir con cuál me quedo
# nx, ny, nz = np.round((bbox[:,1] - bbox[:,0]) / vox_size)
print("(nx, ny, nz):",((bbox[:,1] - bbox[:,0]) / vox_size)[0], ((bbox[:,1] - bbox[:,0]) / vox_size)[1], ((bbox[:,1] - bbox[:,0]) / vox_size)[2])
nx, ny, nz = int(nx), int(ny), int(nz)
print(f"Voxels en cada dimensión: {nx} * {ny} * {nz} = {nx*ny*nz}")
print("=>", (nx, ny, nz))

# Ahora, ordenaremos los puntos, densidades y velocidades en un arreglo 3D:
# Creamos una función para ordenar todos los datos
def reshape_densities(nx, ny, nz, bbox, vox_size, points, densities, speeds):
    # Primero, ordenemos los puntos respecto al orden 'x, y, z':
    indices = np.lexsort((points[2,:], points[1,:], points[0,:]))  
    # print(indices.shape, points.shape[1]) 

    # Creamos el arreglo 3D:
    A = np.zeros((nx,ny,nz), dtype=np.float64) # Densidades
    B = np.zeros((nx+2,ny+2,nz+2), dtype=np.float64) # Mask para encontrar el borde
    C = np.zeros((nx,ny,nz), dtype=np.float64) # Velocidades
    space3D = np.zeros((nx,ny,nz,3), dtype=np.float64)

    # La idea es ir recorriendo los puntos con el orden 'x,y,z' e ir buscando su posición en A.
    # Literal vamos a ir pasando por todas las posibles posiciones de la bbox y verificando si es la ubicación correcta del punto (points[:, indices[pos_index]]).
    pos_index = 0

    vec = np.zeros(3, dtype=np.float64)
    vec[0] = bbox[0,0] + vox_size[0]/2
    for i in range(nx):
        vec[1] = bbox[1,0] + vox_size[1]/2
        for j in range(ny):
            vec[2] = bbox[2,0] + vox_size[2]/2
            for k in range(nz):
                if pos_index < densities.shape[0] and np.linalg.norm(vec - points[:, indices[pos_index]]) < 1e-14:
                    # Agregamos la densidad respectiva a su posición en el arreglo:                   
                    A[i,j,k] = densities[indices[pos_index]]
                    C[i,j,k] = speeds[indices[pos_index]]
                    space3D[i,j,k] = points[:, indices[pos_index]]
                    B[i+1,j+1,k+1] = 1
                    pos_index += 1
                    # Stop condition:
                    # if pos_index == densities.shape[0]:
                    #     return A, B, C, space3D
                else:
                    space3D[i,j,k] = vec
                vec[2] += vox_size[2]
            vec[1] += vox_size[1]
        vec[0] += vox_size[0]

    if pos_index < densities.shape[0]:
        print("Bad execution")
        return
    return A, B, C, space3D

A, B, C, space3D = reshape_densities(nx, ny, nz, bbox, vox_size, points, densities, speeds)

# Verificamos que todos los puntos tengan una posición en A:
print("Check if all values are in 3D array:", A.nonzero()[0].shape, densities.shape)

# Identificamos los bordes del hueso y sus orientaciones:
diff0 = np.diff(B, axis=0)
diff1 = np.diff(B, axis=1)
diff2 = np.diff(B, axis=2)
diff0_surf_left  = (diff0 ==  1)[:-1, 1:-1, 1:-1]
diff0_surf_right = (diff0 == -1)[1:, 1:-1, 1:-1]
diff1_surf_left  = (diff1 ==  1)[1:-1, :-1, 1:-1]
diff1_surf_right = (diff1 == -1)[1:-1, 1:, 1:-1]
diff2_surf_left  = (diff2 ==  1)[1:-1, 1:-1, :-1]
diff2_surf_right = (diff2 == -1)[1:-1, 1:-1, 1:]

# Definimos la máscara de la superficie:
surface = diff0_surf_left + diff0_surf_right + diff1_surf_left + diff1_surf_right + diff2_surf_left + diff2_surf_right

# Definimos la máscara del interior:
interior = A != 0
interior[surface] = False

# Comparamos que se estén considerando todos los puntos:
print("Check if all points are considered in the masks.:", A.nonzero()[0].shape, interior.sum() + surface.sum())
print()

# Por último, cargamos los gradientes:
# gradients2 = np.load("Voxels skull slab/gradients.npy")

# O los calculamos: (técnica de extrapolación estudiada en el notebook notebook_skull_slab.ipynb)
gradients2 = []

A_padded = np.pad(A, [(1, 1), (1, 1), (1, 1)], mode='constant')
mask = np.logical_xor(A != 0, A_padded[:-2, 1:-1, 1:-1] != 0)
mask[mask.nonzero()[0][A[mask] == 0], mask.nonzero()[1][A[mask] == 0], mask.nonzero()[2][A[mask] == 0]] = False
# print(mask.sum())
A_padded[:-2, 1:-1, 1:-1][mask] = A[mask]
mask = np.logical_xor(A != 0, A_padded[2:, 1:-1, 1:-1] != 0)
mask[mask.nonzero()[0][A[mask] == 0], mask.nonzero()[1][A[mask] == 0], mask.nonzero()[2][A[mask] == 0]] = False
# print(mask.sum())
A_padded[2:, 1:-1, 1:-1][mask] = A[mask]
gradients2.append(np.gradient(A_padded, dx, axis=0, edge_order=2)[1:-1,1:-1,1:-1])

A_padded = np.pad(A, [(1, 1), (1, 1), (1, 1)], mode='constant')
mask = np.logical_xor(A != 0, A_padded[1:-1:, :-2, 1:-1] != 0)
mask[mask.nonzero()[0][A[mask] == 0], mask.nonzero()[1][A[mask] == 0], mask.nonzero()[2][A[mask] == 0]] = False
# print(mask.sum())
A_padded[1:-1:, :-2, 1:-1][mask] = A[mask]
mask = np.logical_xor(A != 0, A_padded[1:-1:, 2:, 1:-1] != 0)
mask[mask.nonzero()[0][A[mask] == 0], mask.nonzero()[1][A[mask] == 0], mask.nonzero()[2][A[mask] == 0]] = False
# print(mask.sum())
A_padded[1:-1:, 2:, 1:-1][mask] = A[mask]
gradients2.append(np.gradient(A_padded, dy, axis=1, edge_order=2)[1:-1,1:-1,1:-1])

A_padded = np.pad(A, [(1, 1), (1, 1), (1, 1)], mode='constant')
mask = np.logical_xor(A != 0, A_padded[1:-1:, 1:-1, :-2] != 0)
mask[mask.nonzero()[0][A[mask] == 0], mask.nonzero()[1][A[mask] == 0], mask.nonzero()[2][A[mask] == 0]] = False
# print(mask.sum())
A_padded[1:-1:, 1:-1, :-2][mask] = A[mask]
mask = np.logical_xor(A != 0, A_padded[1:-1:, 1:-1, 2:] != 0)
mask[mask.nonzero()[0][A[mask] == 0], mask.nonzero()[1][A[mask] == 0], mask.nonzero()[2][A[mask] == 0]] = False
# print(mask.sum())
A_padded[1:-1:, 1:-1, 2:][mask] = A[mask]
gradients2.append(np.gradient(A_padded, dz, axis=2, edge_order=2)[1:-1,1:-1,1:-1])

# =====================================================
# ==================== VIE system: ====================
# =====================================================

# Funciones:
def data_grad(dom, dom_density, density_x, density_y, density_z):
    num_voxels = dom.shape[0]
    grad_alpha = np.zeros((num_voxels, 3), dtype=np.complex128)
    grad_alpha[:, 0] = -density_x / (dom_density)**2
    grad_alpha[:, 1] = -density_y / (dom_density)**2
    grad_alpha[:, 2] = -density_z / (dom_density)**2

    return grad_alpha

def data_physical_functions(wavenumber, ext_density, dom_wave, dom_density, dom):
    alpha = 1/dom_density - 1/ext_density
    beta = dom_wave**2/dom_density - wavenumber**2/ext_density
    
    return alpha, beta


# Explicación de algunas variables:
# gradients2[i][(interior + surface)] # => gradientes de las densidades con i \in {x, y, z}
# A[(interior + surface)] # => densidades en 3D
# C[(interior + surface)] # => velocidades en 3D

# Definición de características físicas:
wavespeed = 1500                  # Exterior wavespeed
frec = 500000                     # Exterior frequency
lambda_ext = wavespeed / frec     # Exterior wavelength
kappa = 2 * np.pi / lambda_ext    # Exterior wavenumber
rho_0 = 1000                      # Exterior density
W = 2 * np.pi * frec / C          # Interior wavenumber


# Variables importantes:
# rho_0 = rho_0
# kappa = kappa
int_w = np.array([dx*dy*dz] * (interior.sum() + surface.sum()))
int_slab = space3D[interior + surface]
int_grad_alpha = data_grad(int_slab, A[(interior + surface)], gradients2[0][(interior + surface)],
                           gradients2[1][(interior + surface)], gradients2[2][(interior + surface)])
int_alpha3D, int_beta3D = data_physical_functions(kappa, rho_0, W, A, int_slab)
int_alpha, int_beta = int_alpha3D[(interior+surface)], int_beta3D[(interior+surface)]

# Recordando definición de las orientaciones de los bordes:
# diff0_surf_left  = (diff0 ==  1)[:-1, 1:-1, 1:-1]
# diff0_surf_right = (diff0 == -1)[1:, 1:-1, 1:-1]
# diff1_surf_left  = (diff1 ==  1)[1:-1, :-1, 1:-1]
# diff1_surf_right = (diff1 == -1)[1:-1, 1:, 1:-1]
# diff2_surf_left  = (diff2 ==  1)[1:-1, 1:-1, :-1]
# diff2_surf_right = (diff2 == -1)[1:-1, 1:-1, 1:]

# Cálculo del resto de variables importantes:
normals = []
surf_grid = []
sw = []
diff_alpha = []

# diff0_surf_left
normals.extend([np.array([-1.,0.,0.]) for i in range(diff0_surf_left.sum())])
surf_grid.extend(space3D[diff0_surf_left] + np.array([-dx/2, 0, 0]))
sw.extend([dy*dz for i in range(diff0_surf_left.sum())])
diff_alpha.extend(int_alpha3D[diff0_surf_left])

# diff0_surf_right
normals.extend([np.array([1.,0.,0.]) for i in range(diff0_surf_right.sum())])
surf_grid.extend(space3D[diff0_surf_right] + np.array([dx/2, 0, 0]))
sw.extend([dy*dz for i in range(diff0_surf_right.sum())])
diff_alpha.extend(int_alpha3D[diff0_surf_right])

# diff1_surf_left
normals.extend([np.array([0.,-1.,0.]) for i in range(diff1_surf_left.sum())])
surf_grid.extend(space3D[diff1_surf_left] + np.array([0, -dy/2, 0]))
sw.extend([dx*dz for i in range(diff1_surf_left.sum())])
diff_alpha.extend(int_alpha3D[diff1_surf_left])

# diff1_surf_right
normals.extend([np.array([0.,1.,0.]) for i in range(diff1_surf_right.sum())])
surf_grid.extend(space3D[diff1_surf_right] + np.array([0, dy/2, 0]))
sw.extend([dx*dz for i in range(diff1_surf_right.sum())])
diff_alpha.extend(int_alpha3D[diff1_surf_right])

# diff2_surf_left
normals.extend([np.array([0.,0.,-1.]) for i in range(diff2_surf_left.sum())])
surf_grid.extend(space3D[diff2_surf_left] + np.array([0, 0, -dz/2]))
sw.extend([dx*dy for i in range(diff2_surf_left.sum())])
diff_alpha.extend(int_alpha3D[diff2_surf_left])

# diff2_surf_right
normals.extend([np.array([0.,0.,1.]) for i in range(diff2_surf_right.sum())])
surf_grid.extend(space3D[diff2_surf_right] + np.array([0, 0, dz/2]))
sw.extend([dx*dy for i in range(diff2_surf_right.sum())])
diff_alpha.extend(int_alpha3D[diff2_surf_right])

normals, surf_grid, sw, diff_alpha = np.array(normals), np.array(surf_grid), np.array(sw), np.array(diff_alpha)
alpha_g = diff_alpha / 2
normals.shape, surf_grid.shape, sw.shape, diff_alpha.shape, alpha_g.shape

# Resumen resto de variables importantes:
# sw = sw
# normals = normals
# surf_grid = surf_grid
# diff_alpha = diff_alpha # valor de alpha en la superficie
# alpha_g = alpha_g # valor de alpha/2 en la superficie

# ======================== Creación de árboles para compresión: ========================
import bempp.api
import numpy as np
import MyHM.structures as stt
from MyHM.compression.aca import ACAPP, ACAPP_with_assembly
from time import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

print()
octrees = []
spaces = [int_slab, surf_grid]
space_names = ["int_slab", "surf_grid"]
for i in range(len(spaces)):
    # Vertices:
    vertices = spaces[i]
    print(f"Space name: {space_names[i]}")
    print("Shape vertices:", vertices.shape)
    
    # Bbox:
    mins = np.min(vertices, axis=0)
    maxs = np.max(vertices, axis=0)
    bbox_space = np.array([mins - (vox_size/2), maxs + (vox_size/2)]).T
    print("Bbox space:", bbox_space)
    
    # DOF indices:
    dof_indices = list(range(vertices.shape[0]))
    print("DOF indices:", len(dof_indices))
    
    # Octree:
    if i == 0:
        maximum_element_diameter = np.sqrt(np.sum(vox_size**2))
    else:
        maximum_element_diameter = np.sqrt(vox_size[0]**2 + vox_size[2]**2)
    octree = stt.Octree(vertices, dof_indices, bbox_space, maximum_element_diameter, max_depth=4) # TODO:
    ti = time()
    octree.generate_tree()
    tf = time()
    octrees.append(octree)
    print("Max depth octree:", octree.max_depth)
    print(f"Generate Octree time: {tf-ti}")
    print()

octree_int, octree_bdy = octrees

# Tree3D:
print("Int Int:")
tree_3d_int_int = stt.Tree3D(octree_int, octree_int)
t0 = time()
tree_3d_int_int.generate_adm_tree()
print(f"Time of generation: {time()-t0} s")
print(tree_3d_int_int.stats)
print()

print("Bdy Bdy:")
tree_3d_bdy_bdy = stt.Tree3D(octree_bdy, octree_bdy)
t0 = time()
tree_3d_bdy_bdy.generate_adm_tree()
print(f"Time of generation: {time()-t0} s")
print(tree_3d_bdy_bdy.stats)
print()

print("Bdy Int:")
tree_3d_bdy_int = stt.Tree3D(octree_bdy, octree_int)
t0 = time()
tree_3d_bdy_int.generate_adm_tree()
print(f"Time of generation: {time()-t0} s")
print(tree_3d_bdy_int.stats)
print()

print("Int Bdy:")
tree_3d_int_bdy = stt.Tree3D(octree_int, octree_bdy)
t0 = time()
tree_3d_int_bdy.generate_adm_tree()
print(f"Time of generation: {time()-t0} s")
print(tree_3d_int_bdy.stats)
print()

# # ======================== Cálculo de operadores: ========================
size_bdy = np.int64(surf_grid.shape[0])
size_int = np.int64(interior.sum()) + np.int64(surface.sum())
print("Constructing M2...")
M2 = np.zeros((size_bdy + size_int, size_bdy + size_int), dtype=np.complex128)
print("Created M2 with zeros")

precision = 1e-14

print("\nCalculating operators...")

t0 = time()
mass_op_int = mass_matrix(int_slab, int_alpha, rho_0)
tf = time()
plt.spy(mass_op_int, precision=precision)
plt.savefig(f"Results/spy_mass_op_int_{precision}.png")
plt.close()
print(f"mass_op_int calculation time: {tf-t0} s")
# np.save(f"/tmp0/alberto.almuna/mass_op_int.npy", mass_op_int)
M2[size_bdy:,size_bdy:] += mass_op_int
del mass_op_int

t0 = time()
mass_op_bdy = mass_matrix(surf_grid, alpha_g, rho_0)
tf = time()
plt.spy(mass_op_bdy, precision=precision)
plt.savefig(f"Results/spy_mass_op_bdy_{precision}.png")
plt.close()
print(f"mass_op_bdy calculation time: {tf-t0} s")
# np.save(f"/tmp0/alberto.almuna/mass_op_bdy.npy", mass_op_bdy) 
M2[:size_bdy,:size_bdy] += mass_op_bdy
del mass_op_bdy

t0 = time()
sl_op_int = single_layer(kappa, int_slab, int_alpha, int_beta, int_w)
tf = time()
plt.spy(sl_op_int, precision=precision)
plt.savefig(f"Results/spy_sl_op_int_{precision}.png")
plt.close()
print(f"sl_op_int calculation time: {tf-t0} s")
# np.save(f"/tmp0/alberto.almuna/sl_op_int.npy", sl_op_int)
M2[size_bdy:,size_bdy:] -= sl_op_int
del sl_op_int

t0 = time()
sl_op_bdy_int = cross_single_layer(kappa, surf_grid, int_slab, int_alpha, int_beta, int_w)
tf = time()
plt.spy(sl_op_bdy_int, precision=precision)
plt.savefig(f"Results/spy_sl_op_bdy_int_{precision}.png")
plt.close()
print(f"sl_op_bdy_int calculation time: {tf-t0} s")
# np.save(f"/tmp0/alberto.almuna/sl_op_bdy_int.npy", sl_op_bdy_int)
M2[:size_bdy,size_bdy:] -= sl_op_bdy_int
del sl_op_bdy_int

t0 = time()
dl_op_int_bdy = cross_double_layer(kappa, int_slab, surf_grid, normals, diff_alpha, sw)
tf = time()
plt.spy(dl_op_int_bdy, precision=precision)
plt.savefig(f"Results/spy_dl_op_int_bdy_{precision}.png")
plt.close()
print(f"dl_op_int_bdy calculation time: {tf-t0} s")
# np.save(f"/tmp0/alberto.almuna/dl_op_int_bdy.npy", dl_op_int_bdy)
M2[size_bdy:,:size_bdy] += dl_op_int_bdy
del dl_op_int_bdy

t0 = time()
dl_op_bdy = double_layer(kappa, surf_grid, normals, diff_alpha, sw)
tf = time()
plt.spy(dl_op_bdy, precision=precision)
plt.savefig(f"Results/spy_dl_op_bdy_{precision}.png")
plt.close()
print(f"dl_op_bdy calculation time: {tf-t0} s")
# np.save(f"/tmp0/alberto.almuna/dl_op_bdy.npy", dl_op_bdy)
M2[:size_bdy,:size_bdy] += dl_op_bdy
del dl_op_bdy

t0 = time()
ad_dl_op_int = ad_double_layer(kappa, int_slab, int_grad_alpha, int_w)
tf = time()
plt.spy(ad_dl_op_int, precision=precision)
plt.savefig(f"Results/spy_ad_dl_op_int_{precision}.png")
plt.close()
print(f"ad_dl_op_int calculation time: {tf-t0} s")
# np.save(f"/tmp0/alberto.almuna/ad_dl_op_int.npy", ad_dl_op_int)
M2[size_bdy:,size_bdy:] += ad_dl_op_int
del ad_dl_op_int

t0 = time()
ad_dl_op_bdy_int = cross_ad_double_layer(kappa, surf_grid, int_slab, int_grad_alpha, int_w)
tf = time()
plt.spy(ad_dl_op_bdy_int, precision=precision)
plt.savefig(f"Results/spy_ad_dl_op_bdy_int_{precision}.png")
plt.close()
print(f"ad_dl_op_bdy_int calculation time: {tf-t0} s")
# np.save(f"/tmp0/alberto.almuna/ad_dl_op_bdy_int.npy", ad_dl_op_bdy_int)
M2[:size_bdy,size_bdy:] += ad_dl_op_bdy_int
del ad_dl_op_bdy_int

# ======================== Testeos de compresión en árboles: ========================

# ======================== Cada operador por separado: 
# # As =    [dl_op_bdy,       sl_op_bdy_int,   ad_dl_op_bdy_int,   dl_op_int_bdy,   sl_op_int,       ad_dl_op_int   ]
# names = ["dl_op_bdy",     "sl_op_bdy_int", "ad_dl_op_bdy_int", "dl_op_int_bdy", "sl_op_int",     "ad_dl_op_int" ]
# trees = [tree_3d_bdy_bdy, tree_3d_bdy_int, tree_3d_bdy_int,    tree_3d_int_bdy, tree_3d_int_int, tree_3d_int_int]

# from functools import partial
# partial_assemblers = [
#     partial(double_layer_partial, kappa, surf_grid, normals, diff_alpha, sw),
#     partial(cross_single_layer_partial, kappa, surf_grid, int_slab, int_alpha, int_beta, int_w),
#     partial(cross_ad_double_layer_partial, kappa, surf_grid, int_slab, int_grad_alpha, int_w),
#     partial(cross_double_layer_partial, kappa, int_slab, surf_grid, normals, diff_alpha, sw),
#     partial(single_layer_partial, kappa, int_slab, int_alpha, int_beta, int_w),
#     partial(ad_double_layer_partial, kappa, int_slab, int_grad_alpha, int_w),
# ]

# full_assemblers = [
#     partial(double_layer, kappa, surf_grid, normals, diff_alpha, sw),
#     partial(cross_single_layer, kappa, surf_grid, int_slab, int_alpha, int_beta, int_w),
#     partial(cross_ad_double_layer, kappa, surf_grid, int_slab, int_grad_alpha, int_w),
#     partial(cross_double_layer, kappa, int_slab, surf_grid, normals, diff_alpha, sw),
#     partial(single_layer, kappa, int_slab, int_alpha, int_beta, int_w),
#     partial(ad_double_layer, kappa, int_slab, int_grad_alpha, int_w),
# ]

# ======================== Bloques de M2: 
As =    [M2[:size_bdy, :size_bdy], M2[:size_bdy, size_bdy:], M2[size_bdy:,:size_bdy], M2[size_bdy:,size_bdy:]]
names = ["M2_NW",                  "M2_NE",                  "M2_SW",                 "M2_SE"                ]
trees = [tree_3d_bdy_bdy,          tree_3d_bdy_int,          tree_3d_int_bdy,         tree_3d_int_int        ]

np.random.seed(0)
for index in range(len(trees)):
    print(f"{index}) {names[index]}")
    os.makedirs(f"Results/{names[index]}")
    t0 = time()
    # A = np.load(f"/tmp0/alberto.almuna/{names[index]}.npy")
    A = As[index]
    # A = full_assemblers[index]()
    tf = time()
    print(f"Obtained {names[index]} matrix! (Time: {tf-t0}s)")
    print()
    tree_3d = trees[index]
    # partial_assembler = partial_assemblers[index]
    b = np.random.rand(tree_3d.shape[1])
    print("Shape A:\t", A.shape)
    print("Shape Tree3D:\t", tree_3d.shape)
    # print("Spy A:")
    # precision = 1e-14
    # plt.spy(A, precision=precision)
    # plt.title(f"Matrix: {names[index]}")
    # # plt.show()
    # plt.savefig(f"Results/{names[index]}/Spy_{precision}.png")
    # plt.close()
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
        # tree_3d.add_compressed_matrix(ACAPP_with_assembly, partial_assembler, None, epsilon=epsilons[i], verbose=False)
        tf = time()
        print(f"Time of compression (w/o assembly): {tf-t0} s")
        # print(f"Time of compression (assembly): {tf-t0} s")
        t0 = time()
        aux_result = tree_3d.matvec_compressed()
        tf = time()
        print(f"Time of matvec: {tf-t0} s")
        print("Relative error:", np.linalg.norm(result1 - aux_result) / np.linalg.norm(result1))
        print()
    
        errors2.append(np.linalg.norm(result1 - aux_result) / np.linalg.norm(result1))
        used_storages.append(tree_3d.calculate_compressed_matrix_storage())
        tree_3d.pairplot(save=True, name=f"Results/{names[index]}/Pairplot_{string_i}.png", extra_title=f"(epsilon = {formatted_epsilons[i]})")
        tree_3d.plot_storage_per_level(save=True, name=f"Results/{names[index]}/Storage_per_level_{string_i}.png", extra_title=f"(epsilon = {formatted_epsilons[i]})")
        tree_3d.compression_imshow(save=True, name=f"Results/{names[index]}/Imshow_{string_i}.png", extra_title=f"(epsilon = {formatted_epsilons[i]})")

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
 
    del A, b

# ======================== Creación del vector del lado derecho: ========================
# => Onda plana:
# t0 = time()
# direction = np.array((2., 1., 0.))
# direction = direction/np.linalg.norm(direction)
# uinc_int = incident_plane(kappa, int_slab, direction, rho_0)
# uinc_bdy = incident_plane(kappa, surf_grid, direction, rho_0)
# uinc2 = np.block([uinc_bdy, uinc_int])
# tf = time()
# print(f"uinc2 calculation time: {tf-t0} s")
# np.save(f"/tmp0/alberto.almuna/uinc2.npy", uinc2)

# Carga del vector del lado derecho:
# uinc2 = np.load(f"/tmp0/alberto.almuna/uinc2.npy")

# ======================== Construción de la matriz por bloques: ========================
# Construcción con todos los bloques en memoria:
# print("Constructing M2...")
# t0 = time()
# M2 = np.block([
#     [mass_op_bdy+dl_op_bdy,  -sl_op_bdy_int + ad_dl_op_bdy_int     ],
#     [dl_op_int_bdy        ,  mass_op_int - sl_op_int + ad_dl_op_int]
# ])
# tf = time()
# print(f"M2 construction time: {tf-t0} s")

# Construcción con cargas puntuales de los operadores: (menos memoria necesaria en total)
# print("Constructing M2...")
# t0 = time()
# size_bdy = np.int64(surf_grid.shape[0])
# size_int = np.int64(interior.sum()) + np.int64(surface.sum())
# M2 = np.zeros((size_bdy + size_int, size_bdy + size_int), dtype=np.complex128)
# print("Created M2 with zeros")
# M2[:size_bdy,:size_bdy] += np.load("/tmp0/alberto.almuna/mass_op_bdy.npy")
# print("Loaded mass_op_bdy matrix!")
# M2[:size_bdy,:size_bdy] += np.load("/tmp0/alberto.almuna/dl_op_bdy.npy")
# print("Loaded dl_op_bdy matrix!")
# M2[:size_bdy,size_bdy:] -= np.load("/tmp0/alberto.almuna/sl_op_bdy_int.npy")
# print("Loaded sl_op_bdy_int matrix!")
# M2[:size_bdy,size_bdy:] += np.load("/tmp0/alberto.almuna/ad_dl_op_bdy_int.npy")
# print("Loaded ad_dl_op_bdy_int matrix!")
# M2[size_bdy:,:size_bdy] += np.load("/tmp0/alberto.almuna/dl_op_int_bdy.npy")
# print("Loaded dl_op_int_bdy matrix!")
# M2[size_bdy:,size_bdy:] += np.load("/tmp0/alberto.almuna/mass_op_int.npy")
# print("Loaded mass_op_int matrix!")
# M2[size_bdy:,size_bdy:] -= np.load("/tmp0/alberto.almuna/sl_op_int.npy")
# print("Loaded sl_op_int matrix!")
# M2[size_bdy:,size_bdy:] += np.load("/tmp0/alberto.almuna/ad_dl_op_int.npy")
# print("Loaded ad_dl_op_int matrix!")
# tf = time()
# print(f"M2 construction time: {tf-t0} s")

# Guardamos la matriz en disco duro:
# np.save(f"/tmp0/alberto.almuna/M2.npy", M2)
# print("Saved M2!")

# Cargamos la matriz del disco duro:
# M2 = np.load(f"/tmp0/alberto.almuna/M2.npy")

# ======================== Resolvemos el sistema lineal: ========================
# iteration_count = 0
# def counter(r):
#     global iteration_count
#     iteration_count += 1

#     if iteration_count % 50 == 0:
#         print("GMRES current iteration:", iteration_count)
# TODO: Puedo ir agregando el residuo para después graficarlo

# print("\nStarting GMRES...")
# t0 = time()
# # sol2, info = gmres(M2, uinc2, rtol=1e-5, callback=counter, callback_type='pr_norm')
# sol2, info = gmres(M2, uinc2, rtol=1e-5, callback=counter, callback_type='pr_norm', restart=1000, maxiter=1000)
# tf = time()
# print(f"GMRES time: {tf-t0} s")
# print("GMRES iterations:", iteration_count)
# np.save(f"/tmp0/alberto.almuna/sol2.npy", sol2)
# print("Saved solution!")
# print("READY :D")

# ======================== Gráfico solución: ========================
# (Esto en general lo vamos a hacer después, localmente)
# Sol2 = sol2[size_bdy:]

# info_plot = [
#     ("Absolute Value", np.abs, [0, 2.5], 'viridis', [], ('$x$ $(m)$', '$y$ $(m)$')),
#     ("Real Part", np.real, [-3.5, 3.5], 'seismic', [], ('$x$ $(m)$', '$y$ $(m)$')),
#     (19,7)
# ]

# Sol3D = np.zeros(space3D.shape[:-1], dtype=np.complex128)
# Sol3D[interior + surface] =  Sol2
# # slice = create_slice(int_slab, Sol2, plot_info=info_plot, plane='xy', value_slice=0, interpolate=True)
# slice = create_slice(space3D, Sol3D, plot_info=info_plot, plane='yz', value_slice=None, interpolate=True)
# plt.savefig(f"Results/Plot_solution_skull_slab.png")
# plt.close()
