import numpy as _np
from scipy.spatial.distance import cdist

def bcube_from_bbox(bbox):
    min_value , max_value = _np.min(bbox), _np.max(bbox)
    return _np.array([[min_value, max_value]] * 3)

def tuple2index(t):
    id_x, id_y, id_z = t
    return id_x + id_y * 2 + id_z * 4

def index2tuple(i):
    id_z = i // 4
    id_y = (i % 4) // 2
    id_x = ((i % 4) % 2)
    return (id_x, id_y, id_z)

def admissibility(bbox1, bbox2, max_element_diameter):
    # Diam:
    diam = _np.max(_np.linalg.norm([bbox1[:,1] - bbox1[:,0], bbox2[:,1] - bbox2[:,0]], axis=1))
    # print(diam)

    # Dist: (min of distance between all vertices)
    vertex_bbox1 = []
    for x in bbox1[0,:]:
        for y in bbox1[1,:]:
            for z in bbox1[2,:]:
                vertex_bbox1.append([x,y,z])
    
    vertex_bbox2 = []
    for x in bbox2[0,:]:
        for y in bbox2[1,:]:
            for z in bbox2[2,:]:
                vertex_bbox2.append([x,y,z])
    dist = _np.min(cdist(vertex_bbox1, vertex_bbox2))
    # print(dist)

    # La razón del factor 2 es que con un max_element_diameter de distancia entre dos bloques todavía puede haber aristas adjacentes,
    # lo que se traduce en interacción singular. Luego, nos aseguramos de que esto no ocurra con un factor 2.
    return diam <= dist and 2*max_element_diameter < dist # A partir de 1.6 obtuve ya no más hojas admisibles singulares
    # return diam <= dist

class WrapperGridData():
    """
        Wrapper to pickle Bempp's GridDataDouble or GridDataSingle class and use it in Multiprocessing
    """

    def __init__(
            self,
            precision,
            vertices,
            elements,
            edges,
            element_edges,
            volumes,
            normals,
            jacobians,
            jac_inv_trans,
            diameters,
            integration_elements,
            centroids,
            domain_indices,
            vertex_on_boundary,
            element_neighbor_indices,
            element_neighbor_indexptr,
        ):
        from bempp.api.grid.grid import GridDataFloat, GridDataDouble

        self.precision = precision
        self.vertices = vertices
        self.elements = elements
        self.edges = edges
        self.element_edges = element_edges
        self.volumes = volumes
        self.normals = normals
        self.jacobians = jacobians
        self.jac_inv_trans = jac_inv_trans
        self.diameters = diameters
        self.integration_elements = integration_elements
        self.centroids = centroids
        self.domain_indices = domain_indices
        self.vertex_on_boundary = vertex_on_boundary
        self.element_neighbor_indices = element_neighbor_indices
        self.element_neighbor_indexptr = element_neighbor_indexptr

        if self.precision == "double":
            self.data = GridDataDouble(
                self.vertices,
                self.elements,
                self.edges,
                self.element_edges,
                self.volumes,
                self.normals,
                self.jacobians,
                self.jac_inv_trans,
                self.diameters,
                self.integration_elements,
                self.centroids,
                self.domain_indices,
                self.vertex_on_boundary,
                self.element_neighbor_indices,
                self.element_neighbor_indexptr,
            )
        elif self.precision == "single":
            self.data = GridDataFloat(
                self.vertices,
                self.elements,
                self.edges,
                self.element_edges,
                self.volumes,
                self.normals,
                self.jacobians,
                self.jac_inv_trans,
                self.diameters,
                self.integration_elements,
                self.centroids,
                self.domain_indices,
                self.vertex_on_boundary,
                self.element_neighbor_indices,
                self.element_neighbor_indexptr,
            )


    def __reduce__(self):
        return (self.__class__, (
                    self.precision,
                    self.vertices, 
                    self.elements,
                    self.edges,
                    self.element_edges,
                    self.volumes,
                    self.normals,
                    self.jacobians,
                    self.jac_inv_trans,
                    self.diameters,
                    self.integration_elements,
                    self.centroids,
                    self.domain_indices,
                    self.vertex_on_boundary,
                    self.element_neighbor_indices,
                    self.element_neighbor_indexptr,
                ))

def wrap_classes(boundary_operator):
    prev_domain_grid_data_double = boundary_operator.domain.grid._grid_data_double
    prev_dual_to_range_grid_data_double = boundary_operator.dual_to_range.grid._grid_data_double
    prev_domain_grid_data_single = boundary_operator.domain.grid._grid_data_single
    prev_dual_to_range_grid_data_single = boundary_operator.dual_to_range.grid._grid_data_single

    boundary_operator.domain.grid._grid_data_double = WrapperGridData(
        "double",
        prev_domain_grid_data_double.vertices,
        prev_domain_grid_data_double.elements,
        prev_domain_grid_data_double.edges,
        prev_domain_grid_data_double.element_edges,
        prev_domain_grid_data_double.volumes,
        prev_domain_grid_data_double.normals,
        prev_domain_grid_data_double.jacobians,
        prev_domain_grid_data_double.jac_inv_trans,
        prev_domain_grid_data_double.diameters,
        prev_domain_grid_data_double.integration_elements,
        prev_domain_grid_data_double.centroids,
        prev_domain_grid_data_double.domain_indices,
        prev_domain_grid_data_double.vertex_on_boundary,
        prev_domain_grid_data_double.element_neighbor_indices,
        prev_domain_grid_data_double.element_neighbor_indexptr,
    )
    boundary_operator.dual_to_range.grid._grid_data_double = WrapperGridData(
        "double",
        prev_dual_to_range_grid_data_double.vertices,
        prev_dual_to_range_grid_data_double.elements,
        prev_dual_to_range_grid_data_double.edges,
        prev_dual_to_range_grid_data_double.element_edges,
        prev_dual_to_range_grid_data_double.volumes,
        prev_dual_to_range_grid_data_double.normals,
        prev_dual_to_range_grid_data_double.jacobians,
        prev_dual_to_range_grid_data_double.jac_inv_trans,
        prev_dual_to_range_grid_data_double.diameters,
        prev_dual_to_range_grid_data_double.integration_elements,
        prev_dual_to_range_grid_data_double.centroids,
        prev_dual_to_range_grid_data_double.domain_indices,
        prev_dual_to_range_grid_data_double.vertex_on_boundary,
        prev_dual_to_range_grid_data_double.element_neighbor_indices,
        prev_dual_to_range_grid_data_double.element_neighbor_indexptr,
    )
    boundary_operator.domain.grid._grid_data_single = WrapperGridData(
        "single",
        prev_domain_grid_data_single.vertices,
        prev_domain_grid_data_single.elements,
        prev_domain_grid_data_single.edges,
        prev_domain_grid_data_single.element_edges,
        prev_domain_grid_data_single.volumes,
        prev_domain_grid_data_single.normals,
        prev_domain_grid_data_single.jacobians,
        prev_domain_grid_data_single.jac_inv_trans,
        prev_domain_grid_data_single.diameters,
        prev_domain_grid_data_single.integration_elements,
        prev_domain_grid_data_single.centroids,
        prev_domain_grid_data_single.domain_indices,
        prev_domain_grid_data_single.vertex_on_boundary,
        prev_domain_grid_data_single.element_neighbor_indices,
        prev_domain_grid_data_single.element_neighbor_indexptr,
    )
    boundary_operator.dual_to_range.grid._grid_data_single = WrapperGridData(
        "single",
        prev_dual_to_range_grid_data_single.vertices,
        prev_dual_to_range_grid_data_single.elements,
        prev_dual_to_range_grid_data_single.edges,
        prev_dual_to_range_grid_data_single.element_edges,
        prev_dual_to_range_grid_data_single.volumes,
        prev_dual_to_range_grid_data_single.normals,
        prev_dual_to_range_grid_data_single.jacobians,
        prev_dual_to_range_grid_data_single.jac_inv_trans,
        prev_dual_to_range_grid_data_single.diameters,
        prev_dual_to_range_grid_data_single.integration_elements,
        prev_dual_to_range_grid_data_single.centroids,
        prev_dual_to_range_grid_data_single.domain_indices,
        prev_dual_to_range_grid_data_single.vertex_on_boundary,
        prev_dual_to_range_grid_data_single.element_neighbor_indices,
        prev_dual_to_range_grid_data_single.element_neighbor_indexptr,
    )

    return (
        prev_domain_grid_data_double,
        prev_dual_to_range_grid_data_double,
        prev_domain_grid_data_single,
        prev_dual_to_range_grid_data_single,
    )

def unwrap_classes(prev_grid_datas):
    boundary_operator.domain.grid._grid_data_double = prev_grid_datas[0]
    boundary_operator.dual_to_range.grid._grid_data_double = prev_grid_datas[1]
    boundary_operator.domain.grid._grid_data_single = prev_grid_datas[2]
    boundary_operator.dual_to_range.grid._grid_data_single = prev_grid_datas[3]

if __name__ == "__main__":
    bbox = _np.array([[-91.67590332, -34.7798996 ],
                      [-78.86689758,  51.68640137],
                      [-26.42860031,  45.42950058]])
    bcube_from_bbox(bbox)
    index2tuple(0)
    admissibility(bbox, bbox)
    print("Working...")