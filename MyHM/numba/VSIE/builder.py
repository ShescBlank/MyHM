import numpy as np
from MyHM.numba.VSIE.info_class import Info_VSIE

# TODO: Translate this file
# TODO: Recheck the code

def points_to_3D(nx, ny, nz, bbox, vox_size, points):
    # Primero, ordenemos los puntos respecto al orden 'x, y, z':
    indices = np.lexsort((points[2,:], points[1,:], points[0,:]))  
    # print(indices.shape, points.shape[1]) 

    # Creamos los arreglos 3D a retornar
    space3D = np.zeros((nx,ny,nz,3), dtype=np.float64)
    mask_ones3D = np.zeros((nx+2,ny+2,nz+2), dtype=np.float64) # Mask para encontrar el borde
    mask_to_3d = np.zeros((3,points.shape[1]), dtype=int)

    # La idea es ir recorriendo los puntos con el orden 'x,y,z' e ir buscando su posición en points.
    # Literal vamos a ir pasando por todas las posibles posiciones de la bbox y verificando si es la ubicación correcta del punto (points[:, indices[pos_index]]).
    pos_index = 0

    vec = np.zeros(3, dtype=np.float64)
    vec[0] = bbox[0,0] + vox_size[0]/2
    for i in range(nx):
        vec[1] = bbox[1,0] + vox_size[1]/2
        for j in range(ny):
            vec[2] = bbox[2,0] + vox_size[2]/2
            for k in range(nz):
                if pos_index < points.shape[1] and np.linalg.norm(vec - points[:, indices[pos_index]]) < 1e-14:                 
                    space3D[i,j,k] = points[:, indices[pos_index]]
                    mask_ones3D[i+1,j+1,k+1] = 1
                    mask_to_3d[0, indices[pos_index]] = i
                    mask_to_3d[1, indices[pos_index]] = j
                    mask_to_3d[2, indices[pos_index]] = k
                    pos_index += 1
                else:
                    space3D[i,j,k] = vec
                vec[2] += vox_size[2]
            vec[1] += vox_size[1]
        vec[0] += vox_size[0]

    if pos_index < points.shape[1]:
        print("Bad execution")
        return
    return space3D, mask_ones3D, mask_to_3d

def data_grad(dom, dom_density, density_dx, density_dy, density_dz, ext_density):
    num_voxels = dom.shape[0]
    grad_alpha = np.zeros((num_voxels, 3), dtype=np.float64)
    grad_alpha[:, 0] = -ext_density*density_dx / (dom_density)**2
    grad_alpha[:, 1] = -ext_density*density_dy / (dom_density)**2
    grad_alpha[:, 2] = -ext_density*density_dz / (dom_density)**2

    return grad_alpha

def data_physical_functions(wavenumber, ext_density, dom_wave, dom_density, dom):
    alpha = ext_density/dom_density - 1
    beta = ext_density*dom_wave**2/dom_density - wavenumber**2
    
    return alpha, beta

def build_system_VSIE(points, densities, speeds, frequency=None):
    print("Points shape:", points.shape)

    dx = np.min(np.abs(np.diff(points[0,:]))[np.abs(np.diff(points[0,:])) > 0]) # Este funciona en todos los casos
    dy = np.min(np.abs(np.diff(points[1,:]))[np.abs(np.diff(points[1,:])) > 0])
    dz = np.min(np.abs(np.diff(points[2,:]))[np.abs(np.diff(points[2,:])) > 0])
    vox_size = np.array([dx, dy, dz])
    print("Vox size:", vox_size)

    # Definimos la bounding box que encierra a todos los puntos:
    mins = np.min(points, axis=1)
    maxs = np.max(points, axis=1)
    bbox = np.array([mins - (vox_size/2), maxs + (vox_size/2)]).T
    print("Bounding box:")
    print(bbox)

    # Cantidad de voxels si la bbox estuviera llena:
    # nx, ny, nz = np.ceil((bbox[:,1] - bbox[:,0]) / vox_size) # TODO: decidir con cuál me quedo
    nx, ny, nz = np.round((bbox[:,1] - bbox[:,0]) / vox_size)
    print("(nx, ny, nz):",((bbox[:,1] - bbox[:,0]) / vox_size)[0], ((bbox[:,1] - bbox[:,0]) / vox_size)[1], ((bbox[:,1] - bbox[:,0]) / vox_size)[2])
    nx, ny, nz = int(nx), int(ny), int(nz)
    print(f"Voxels in each dimension: {nx} * {ny} * {nz} = {nx*ny*nz}")

    # Ahora, ordenaremos los puntos, densidades y velocidades en un arreglo 3D:
    space3D, mask_ones3D, mask_to_3d = points_to_3D(nx, ny, nz, bbox, vox_size, points)
    densities3D = np.zeros((nx,ny,nz), dtype=np.float64)
    densities3D[mask_to_3d[0],mask_to_3d[1],mask_to_3d[2]] = densities
    speeds3D = np.zeros((nx,ny,nz), dtype=np.float64)
    speeds3D[mask_to_3d[0],mask_to_3d[1],mask_to_3d[2]] = speeds

    # Verificamos que todos los puntos tengan una posición en densities3D:
    print("Check if all values are in 3D array:", densities3D.nonzero()[0].shape, densities.shape)

    # Identificamos los bordes del hueso y sus orientaciones:
    diff0 = np.diff(mask_ones3D, axis=0)
    diff1 = np.diff(mask_ones3D, axis=1)
    diff2 = np.diff(mask_ones3D, axis=2)
    diff0_surf_left  = (diff0 ==  1)[:-1, 1:-1, 1:-1]
    diff0_surf_right = (diff0 == -1)[1:, 1:-1, 1:-1]
    diff1_surf_left  = (diff1 ==  1)[1:-1, :-1, 1:-1]
    diff1_surf_right = (diff1 == -1)[1:-1, 1:, 1:-1]
    diff2_surf_left  = (diff2 ==  1)[1:-1, 1:-1, :-1]
    diff2_surf_right = (diff2 == -1)[1:-1, 1:-1, 1:]

    # Definimos la máscara de la superficie:
    surface = diff0_surf_left + diff0_surf_right + diff1_surf_left + diff1_surf_right + diff2_surf_left + diff2_surf_right

    # Definimos la máscara del interior:
    interior = densities3D != 0
    interior[surface] = False

    # Comparamos que se estén considerando todos los puntos:
    print("Check if all points are considered in the masks.:", densities3D.nonzero()[0].shape, interior.sum() + surface.sum())

    # Calculamos los gradientes: (técnica de extrapolación estudiada en el notebook notebook_skull_slab.ipynb)
    gradients = []

    A_padded = np.pad(densities3D, [(1, 1), (1, 1), (1, 1)], mode='constant')
    mask = np.logical_xor(densities3D != 0, A_padded[:-2, 1:-1, 1:-1] != 0)
    mask[mask.nonzero()[0][densities3D[mask] == 0], mask.nonzero()[1][densities3D[mask] == 0], mask.nonzero()[2][densities3D[mask] == 0]] = False
    # print(mask.sum())
    A_padded[:-2, 1:-1, 1:-1][mask] = densities3D[mask]
    mask = np.logical_xor(densities3D != 0, A_padded[2:, 1:-1, 1:-1] != 0)
    mask[mask.nonzero()[0][densities3D[mask] == 0], mask.nonzero()[1][densities3D[mask] == 0], mask.nonzero()[2][densities3D[mask] == 0]] = False
    # print(mask.sum())
    A_padded[2:, 1:-1, 1:-1][mask] = densities3D[mask]
    gradients.append(np.gradient(A_padded, dx, axis=0, edge_order=2)[1:-1,1:-1,1:-1])

    A_padded = np.pad(densities3D, [(1, 1), (1, 1), (1, 1)], mode='constant')
    mask = np.logical_xor(densities3D != 0, A_padded[1:-1:, :-2, 1:-1] != 0)
    mask[mask.nonzero()[0][densities3D[mask] == 0], mask.nonzero()[1][densities3D[mask] == 0], mask.nonzero()[2][densities3D[mask] == 0]] = False
    # print(mask.sum())
    A_padded[1:-1:, :-2, 1:-1][mask] = densities3D[mask]
    mask = np.logical_xor(densities3D != 0, A_padded[1:-1:, 2:, 1:-1] != 0)
    mask[mask.nonzero()[0][densities3D[mask] == 0], mask.nonzero()[1][densities3D[mask] == 0], mask.nonzero()[2][densities3D[mask] == 0]] = False
    # print(mask.sum())
    A_padded[1:-1:, 2:, 1:-1][mask] = densities3D[mask]
    gradients.append(np.gradient(A_padded, dy, axis=1, edge_order=2)[1:-1,1:-1,1:-1])

    A_padded = np.pad(densities3D, [(1, 1), (1, 1), (1, 1)], mode='constant')
    mask = np.logical_xor(densities3D != 0, A_padded[1:-1:, 1:-1, :-2] != 0)
    mask[mask.nonzero()[0][densities3D[mask] == 0], mask.nonzero()[1][densities3D[mask] == 0], mask.nonzero()[2][densities3D[mask] == 0]] = False
    # print(mask.sum())
    A_padded[1:-1:, 1:-1, :-2][mask] = densities3D[mask]
    mask = np.logical_xor(densities3D != 0, A_padded[1:-1:, 1:-1, 2:] != 0)
    mask[mask.nonzero()[0][densities3D[mask] == 0], mask.nonzero()[1][densities3D[mask] == 0], mask.nonzero()[2][densities3D[mask] == 0]] = False
    # print(mask.sum())
    A_padded[1:-1:, 1:-1, 2:][mask] = densities3D[mask]
    gradients.append(np.gradient(A_padded, dz, axis=2, edge_order=2)[1:-1,1:-1,1:-1])

    # =====================================================
    # ==================== VIE system: ====================
    # =====================================================

    # Explicación de algunas variables:
    # gradients[i][(interior + surface)] # => gradientes de las densidades con i \in {x, y, z}
    # densities3D[(interior + surface)] # => densidades en 3D
    # speeds3D[(interior + surface)] # => velocidades en 3D
    
    # Para evitar warnings de divisiones por cero:
    densities3D[~(interior + surface)] = 1
    speeds3D[~(interior + surface)] = 1

    # Definición de características físicas:
    wavespeed = 1500 # 1482.3                # Exterior wavespeed (water)
    rho_0 = 1000 # 994.035                   # Exterior density (water)
    if frequency is None:
        frequency = 500000                   # Frequency (500kHz)
    lambda_ext = wavespeed / frequency       # Exterior wavelength
    kappa = 2 * np.pi / lambda_ext           # Exterior wavenumber
    W = 2 * np.pi * frequency / speeds3D     # Interior wavenumber

    print("Frequency:", frequency, "Hz")

    # Variables importantes:
    # rho_0 = rho_0
    # kappa = kappa
    int_w = np.array([dx*dy*dz] * (interior.sum() + surface.sum()))
    int_grid = space3D[interior + surface]
    int_grad_alpha = data_grad(int_grid, densities3D[(interior + surface)], gradients[0][(interior + surface)],
                               gradients[1][(interior + surface)], gradients[2][(interior + surface)], rho_0)
    int_alpha3D, int_beta3D = data_physical_functions(kappa, rho_0, W, densities3D, int_grid)
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
    surf_w = []
    diff_alpha = []

    # diff0_surf_left
    normals.extend([np.array([-1.,0.,0.]) for i in range(diff0_surf_left.sum())])
    surf_grid.extend(space3D[diff0_surf_left] + np.array([-dx/2, 0, 0]))
    surf_w.extend([dy*dz for i in range(diff0_surf_left.sum())])
    diff_alpha.extend(int_alpha3D[diff0_surf_left])

    # diff0_surf_right
    normals.extend([np.array([1.,0.,0.]) for i in range(diff0_surf_right.sum())])
    surf_grid.extend(space3D[diff0_surf_right] + np.array([dx/2, 0, 0]))
    surf_w.extend([dy*dz for i in range(diff0_surf_right.sum())])
    diff_alpha.extend(int_alpha3D[diff0_surf_right])

    # diff1_surf_left
    normals.extend([np.array([0.,-1.,0.]) for i in range(diff1_surf_left.sum())])
    surf_grid.extend(space3D[diff1_surf_left] + np.array([0, -dy/2, 0]))
    surf_w.extend([dx*dz for i in range(diff1_surf_left.sum())])
    diff_alpha.extend(int_alpha3D[diff1_surf_left])

    # diff1_surf_right
    normals.extend([np.array([0.,1.,0.]) for i in range(diff1_surf_right.sum())])
    surf_grid.extend(space3D[diff1_surf_right] + np.array([0, dy/2, 0]))
    surf_w.extend([dx*dz for i in range(diff1_surf_right.sum())])
    diff_alpha.extend(int_alpha3D[diff1_surf_right])

    # diff2_surf_left
    normals.extend([np.array([0.,0.,-1.]) for i in range(diff2_surf_left.sum())])
    surf_grid.extend(space3D[diff2_surf_left] + np.array([0, 0, -dz/2]))
    surf_w.extend([dx*dy for i in range(diff2_surf_left.sum())])
    diff_alpha.extend(int_alpha3D[diff2_surf_left])

    # diff2_surf_right
    normals.extend([np.array([0.,0.,1.]) for i in range(diff2_surf_right.sum())])
    surf_grid.extend(space3D[diff2_surf_right] + np.array([0, 0, dz/2]))
    surf_w.extend([dx*dy for i in range(diff2_surf_right.sum())])
    diff_alpha.extend(int_alpha3D[diff2_surf_right])

    normals, surf_grid, surf_w, diff_alpha = np.array(normals), np.array(surf_grid), np.array(surf_w), np.array(diff_alpha)
    alpha_g = diff_alpha / 2
    normals.shape, surf_grid.shape, surf_w.shape, diff_alpha.shape, alpha_g.shape

    # Return:
    info = Info_VSIE(rho_0, kappa, int_grid, surf_grid, int_alpha, int_beta, int_w, 
                     surf_w, normals, alpha_g, diff_alpha, int_grad_alpha)

    return info, vox_size