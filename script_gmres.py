# import bempp.api
# import numpy as np
# import MyHM.structures as stt
# from MyHM.compression.aca import ACAPP_with_assembly, ACAPP
# import MyHM
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# from numba import config
# from time import time
# import os
# from scipy.sparse.linalg import gmres

# def convert_to_preferred_format(sec):
#    sec = sec % (24 * 3600)
#    hour = sec // 3600
#    sec %= 3600
#    min = sec // 60
#    sec %= 60
#    return "%02d:%02d:%02d" % (hour, min, sec)

# os.environ['NUMBA_DEBUG_CACHE'] = "1"

# grid_name = "-h-4"

# grid = bempp.api.import_grid(f'grids/ribcage4{grid_name}.msh')

# print(f"\n{'='*50}")
# print("GRID NAME:", grid_name)
# print("CPUS:", config.NUMBA_DEFAULT_NUM_THREADS)
# print(f"{'='*50}\n")

# # bempp.api.DEFAULT_DEVICE_INTERFACE = 'opencl'
# bempp.api.DEFAULT_DEVICE_INTERFACE = 'numba'

# bbox = grid.bounding_box
# space = bempp.api.function_space(grid, "P", 1)
# print(f"Global DOF count: {space.global_dof_count}")
# print(f"# Vertices: {grid.vertices.shape[1]}")
# if not space.requires_dof_transformation:
#     dof_indices = list(range(space.global_dof_count))
#     excluded_vertices = np.unique(grid.elements[(space.local_multipliers == 0).T])
#     mask = np.ones(grid.vertices.shape[1], dtype=bool)
#     mask[excluded_vertices] = False
#     vertices = grid.vertices[:, mask]
# else:
#     # TODO:
#     raise NotImplementedError
#     # space.dof_transformation.indices
# print(f"# Vertices DOFs: {vertices.shape[1]}")

# t0 = time()
# octree = stt.Octree(vertices.T, dof_indices, bbox, grid.maximum_element_diameter)
# octree.generate_tree()
# print(f"Generate Octree time: {time()-t0}")
# print(f"Max Depth: {octree.max_depth}")

# t0 = time()
# tree_3d = stt.Tree3D(octree, octree, dtype=np.complex128)
# tree_3d.generate_adm_tree()
# print(f"Generate Tree3D time: {time()-t0}")

# k = 7
# slp = bempp.api.operators.boundary.helmholtz.single_layer(space, space, space, k)
# dlp = bempp.api.operators.boundary.helmholtz.double_layer(space, space, space, k)
# hyp = bempp.api.operators.boundary.helmholtz.hypersingular(space, space, space, k)
# parameters = bempp.api.GLOBAL_PARAMETERS
# device_interface = "numba"

# operators = [slp, dlp, hyp]
# epsilons = [2**(-1*i) for i in range(1,50,4)][:3]
# formatted_epsilons = [np.format_float_scientific(e, precision=3) for e in epsilons]
# all_errors = []
# for index_operator in range(len(operators)):
#     print(f"\n{index_operator})\n")
#     boundary_operator = operators[index_operator]
#     t0 = time()
#     A = np.array(boundary_operator.weak_form().A)
#     tf = time()
#     print(f"Generate complete matrix A time: {tf-t0} s")
#     print("=>", convert_to_preferred_format(tf-t0))
#     # b = np.random.rand(A.shape[1])
#     b = np.ones(A.shape[1])
#     print("Calculated A and b")
#     print("Shapes A and b:", A.shape, b.shape)

#     # GMRES:
#     print("GMRES:") # Este GMRES se está tardando bastante
#     t0 = time()
#     sol1, info = gmres(A, b, rtol=1e-5)
#     tf = time()
#     print(f"Time GMRES: {tf-t0} s\n")

#     # Epsilons:
#     errors = []
#     for i in range(len(epsilons)):
#         print(f"{i}) Epsilon: {formatted_epsilons[i]}\n")
#         t0 = time()
#         tree_3d.add_matrix_with_ACA(A, ACAPP, epsilon=epsilons[i], verbose=False)
#         tf = time()
#         print(f"Time of compression w/o assembler and w/assembled_values: {tf-t0} s")
#         print("=>", convert_to_preferred_format(tf-t0))
    
#         blocked_tree = stt.BlockedTree((tree_3d.shape[0],), (tree_3d.shape[1],), dtype=tree_3d.dtype)
#         blocked_tree.add(tree_3d)
#         linear_op = blocked_tree1.linearoperator()
#         sol2, info = gmres(linear_op, b, rtol=1e-5)

#         errors.append(np.linalg.norm(sol1 - sol2) / np.linalg.norm(sol1))
#         tree_3d.clear_compression()
#     all_errors.append(errors)

# # Plots:
# plt.plot(epsilons, epsilons, "or--")
# for i in range(len(all_errors)):
#     plt.plot(epsilons, all_errors[i], "o-")
# plt.gca().invert_xaxis()
# plt.yscale("log")
# plt.xscale("log")
# plt.xlabel("Epsilon")
# plt.ylabel("Relative error")
# plt.title("Relative errors in matvec operation")
# plt.legend(['Epsilon', 'slp', 'dlp', 'hyp'])
# plt.savefig(f"Results/Relative_errors{grid_name}.png")
# plt.close()

# VSIE:
import numpy as np
from vsie.operators.cross_interaction import cross_single_layer, cross_double_layer, cross_ad_double_layer
from vsie.operators.self_interaction import single_layer, double_layer, ad_double_layer, mass_matrix
from vsie.operators.wave_op import incident_plane
from vsie.geometry.grid import concentric_cubes
from vsie.geometry.space_fun import physical_functions, discont_physical_functions, anal_grad, surface_dif, max_wavenumber
from vsie.post_processing.graphs import create_slice
from scipy.sparse.linalg import gmres

# MyHM:
import bempp.api
import numpy as np
import MyHM.structures as stt
from MyHM.compression.aca import ACAPP
from time import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_sol(sol, space, ext_cube, surf_grid, int_cube, name):
    '''         Plot           '''
    from matplotlib import pyplot as plt
    import matplotlib

    matplotlib.rcParams.update({'font.size': 16})
    plt.rc('font', family='serif')
    # plt.rc('text', usetex=True)

    num_points = round(space.shape[0]**(1/3))
    sol_ext = sol[:len(ext_cube)]
    sol_bdy = sol[len(ext_cube):len(ext_cube)+len(surf_grid)]
    sol_int = sol[-len(int_cube):]

    Sol = np.zeros(space.shape[0], dtype=np.complex128)
    Sol[idx_ext] = sol_ext
    Sol[idx_int] = sol_int

    square1 = plt.Rectangle((-hl, -hl), 
                            l_cube_int, l_cube_int,
                                color='black', fill=False, linestyle=':')
    square2 = plt.Rectangle((-hl, -hl), 
                            l_cube_int, l_cube_int,
                                color='black', fill=False, linestyle=':')

    info_plot = [
        ("Absolute Value", np.abs, [0, 2.5], 'viridis', [square1], ('$x$ $(m)$', '$y$ $(m)$')),
        ("Real Part", np.real, [-3.5, 3.5], 'seismic', [square2], ('$x$ $(m)$', '$y$ $(m)$')),
        (19,7)
    ]

    slice = create_slice(space, Sol, plot_info=info_plot, plane='xy', value_slice=0, interpolate=True)
    plt.savefig(f"Results/Plot_solution_{name}.png")
    plt.close()


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
vox_per_wave = 16
space, idx_int, idx_ext, surf_grid, w, sw, normals = concentric_cubes(l_cube_int, l_cube_ext, lambda_min, vox_per_wave=vox_per_wave)
dx = abs(space[0, 2] - space[1, 2])
print("Space shape:", space.shape)


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
print("Evaluating operators...\n")
t0 = time()
mass_op_ext = mass_matrix(ext_cube, ext_alpha, rho_0)
mass_op_int = mass_matrix(int_cube, int_alpha, rho_0)
mass_op_bdy = mass_matrix(surf_grid, alpha_g, rho_0)
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
tf = time()
print(f"Time: {tf-t0} s")

'''        Matrix & RHS          '''
# Create Matrix
M = np.block([
    [mass_op_ext - sl_op_ext + ad_dl_op_ext ,  dl_op_ext_bdy        ,  -sl_op_ext_int + ad_dl_op_ext_int     ],
    [-sl_op_bdy_ext + ad_dl_op_bdy_ext      ,  mass_op_bdy+dl_op_bdy,  -sl_op_bdy_int + ad_dl_op_bdy_int     ],
    [-sl_op_int_ext + ad_dl_op_int_ext      ,  dl_op_int_bdy        ,  mass_op_int - sl_op_int + ad_dl_op_int]
])

# Create RHS vector
direction = np.array((2., 1., 0.))
direction = direction/np.linalg.norm(direction)
uinc_ext = incident_plane(kappa, ext_cube, direction, rho_0)
uinc_int = incident_plane(kappa, int_cube, direction, rho_0)
uinc_bdy = incident_plane(kappa, surf_grid, direction, rho_0)

uinc = np.block([uinc_ext, uinc_bdy, uinc_int])

'''     Solve linear problem      '''

# Solve the linear system
iteration_count = 0
def counter(r):
    global iteration_count
    iteration_count += 1

print("\nGMRES:")
t0 = time()
sol1, info = gmres(M, uinc, rtol=1e-5, callback=counter, callback_type='pr_norm')
tf = time()
print(f"Time: {tf-t0} s")
print("Iterations:", iteration_count, "\n")

plot_sol(sol1, space, ext_cube, surf_grid, int_cube, "original")


'''             Trees and compression           '''

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
    maximum_element_diameter = vox_length * np.sqrt(2) # TODO: chequear cómo se comporta con el raíz de 2
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

# Matrix and vector:
from copy import deepcopy
tree_3d_int_int2 = deepcopy(tree_3d_int_int)
tree_3d_bdy_int2 = deepcopy(tree_3d_bdy_int)
tree_3d_ext_int2 = deepcopy(tree_3d_ext_int)
As      = [sl_op_ext,       sl_op_int,       sl_op_bdy_ext,   sl_op_bdy_int,   sl_op_ext_int,   sl_op_int_ext,   dl_op_ext_bdy,   dl_op_int_bdy,   dl_op_bdy,       ad_dl_op_ext,    ad_dl_op_int,     ad_dl_op_bdy_ext,   ad_dl_op_bdy_int,    ad_dl_op_ext_int,   ad_dl_op_int_ext]
trees   = [tree_3d_ext_ext, tree_3d_int_int, tree_3d_bdy_ext, tree_3d_bdy_int, tree_3d_ext_int, tree_3d_int_ext, tree_3d_ext_bdy, tree_3d_int_bdy, None,            None,            tree_3d_int_int2, None,               tree_3d_bdy_int2,    tree_3d_ext_int2,    None]
names   = ["sl_op_ext",     "sl_op_int",     "sl_op_bdy_ext", "sl_op_bdy_int", "sl_op_ext_int", "sl_op_int_ext", "dl_op_ext_bdy", "dl_op_int_bdy", "None",          "None",          "ad_dl_op_int",   "None",             "ad_dl_op_bdy_int",  "ad_dl_op_ext_int", "None"]
# trees = [tree_3d_ext_ext, tree_3d_int_int, tree_3d_bdy_ext, tree_3d_bdy_int, tree_3d_ext_int, tree_3d_int_ext, tree_3d_ext_bdy, tree_3d_int_bdy, tree_3d_bdy_bdy, tree_3d_ext_ext, tree_3d_int_int, tree_3d_bdy_ext,     tree_3d_bdy_int,     tree_3d_ext_int,    tree_3d_int_ext]

# empty: ad_dl_op_bdy_ext, ad_dl_op_ext, ad_dl_op_int_ext
# blocks with zeros: dl_op_bdy

# Plot for each epsilon:
epsilons = [2**(-1*i) for i in range(1,50,4)][:4]
formatted_epsilons = [np.format_float_scientific(e, precision=3) for e in epsilons]
errors = []
print("=> Epsilons:")
for i in range(len(epsilons)):
    if i < 10:
        string_i = f"0{i}"
    else:
        string_i = f"{i}"
    print(f"\n{string_i}) Epsilon: {formatted_epsilons[i]}\n")

    for index in range(len(As)):
        if trees[index] is None:
            continue

        A = As[index]
        tree_3d = trees[index]
        print(f"{index}) {names[index]}")
        print("Shape A:\t", A.shape)
        print("Shape Tree3D:\t", tree_3d.shape)

        t0 = time()
        tree_3d.add_matrix_with_ACA(A, ACAPP, epsilon=epsilons[i], verbose=False)
        tf = time()
        print(f"Time of compression w/o assembler and w/assembled_values: {tf-t0} s")

    blocked_tree = stt.BlockedTree((mass_op_ext.shape[0], sl_op_bdy_ext.shape[0], sl_op_int_ext.shape[0]), (mass_op_ext.shape[1], dl_op_ext_bdy.shape[1], sl_op_ext_int.shape[1]), dtype=np.complex128)
    blocked_tree.add(mass_op_ext, (0,0), 1.0) # mass_op_ext
    blocked_tree.add(tree_3d_ext_ext, (0,0), -1.0) # sl_op_ext
    blocked_tree.add(ad_dl_op_ext, (0,0), 1.0) # ad_dl_op_ext
    blocked_tree.add(tree_3d_ext_bdy, (0,1), 1.0) # dl_op_ext_bdy
    blocked_tree.add(tree_3d_ext_int, (0,2), -1.0) # sl_op_ext_int
    blocked_tree.add(tree_3d_ext_int2, (0,2), 1.0) # ad_dl_op_ext_int
    blocked_tree.add(tree_3d_bdy_ext, (1,0), -1.0) # sl_op_bdy_ext
    blocked_tree.add(ad_dl_op_bdy_ext, (1,0), 1.0) # ad_dl_op_bdy_ext
    blocked_tree.add(mass_op_bdy, (1,1), 1.0) # mass_op_bdy
    blocked_tree.add(dl_op_bdy, (1,1), 1.0) # dl_op_bdy
    blocked_tree.add(tree_3d_bdy_int, (1,2), -1.0) # sl_op_bdy_int
    blocked_tree.add(tree_3d_bdy_int2, (1,2), 1.0) # ad_dl_op_bdy_int
    blocked_tree.add(tree_3d_int_ext, (2,0), -1.0) # sl_op_int_ext
    blocked_tree.add(ad_dl_op_int_ext, (2,0), 1.0) # ad_dl_op_int_ext
    blocked_tree.add(tree_3d_int_bdy, (2,1), 1.0) # dl_op_int_bdy
    blocked_tree.add(mass_op_int, (2,2), 1.0) # mass_op_int
    blocked_tree.add(tree_3d_int_int, (2,2), -1.0) # sl_op_int
    blocked_tree.add(tree_3d_int_int2, (2,2), 1.0) # ad_dl_op_int
    linear_op = blocked_tree.linearoperator()

    print("GMRES (custom linear operator):")
    iteration_count = 0
    t0 = time()
    sol2, info = gmres(linear_op, uinc, rtol=1e-5, callback=counter, callback_type='pr_norm')
    tf = time()
    print(f"Time: {tf-t0} s")
    print("Iterations:", iteration_count, "\n") 
    errors.append(np.linalg.norm(sol1 - sol2) / np.linalg.norm(sol1))

    plot_sol(sol2, space, ext_cube, surf_grid, int_cube, f"{formatted_epsilons[i]}")

    for index in range(len(As)):
        if trees[index] is None:
            continue
        trees[index].clear_compression()

# Plots:
plt.plot(epsilons, epsilons, "or--", label="Epsilon")
plt.plot(epsilons, errors, "o-", label="Relative error")
plt.gca().invert_xaxis()
plt.yscale("log")
plt.xscale("log")
plt.xlabel("Epsilon")
plt.ylabel("Relative error")
plt.title("Relative errors GMRES")
plt.legend()
plt.savefig(f"Results/Relative_errors.png", bbox_inches="tight")
plt.close()
