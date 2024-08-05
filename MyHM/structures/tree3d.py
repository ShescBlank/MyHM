import numpy as _np
from MyHM.structures.octree import Node
from MyHM.structures.utils import admissibility

# Tree 3D:
class Node3D:
    pass

class Node3D:
    def __init__(self, parent: Node3D, node1: Node, node2: Node, level: int, adm: bool, leaf: bool):
        """
        Parameters:
        - 
        """
        self.parent = parent
        self.node1 = node1
        self.node2 = node2
        self.level = level
        self.adm = adm
        self.leaf = leaf
        self.children = _np.empty((8,8), dtype=Node3D) # Type: numpy.ndarray

        # New:
        self.matrix_block = None   # Type: _np.array
        self.u_vectors = None      # Type: _np.array
        self.v_vectors = None      # Type: _np.array
        self.vector_segment = None # Type: _np.array

class Tree3D:
    def __init__(self, octree_root: Node, max_depth: int = 4):
        assert max_depth > 0, "max_depth parameter must be greater than 0"
        self.root = Node3D(parent=None, node1=octree_root, node2=octree_root, level=0, adm=False, leaf=False)
        self.max_depth = max_depth
        self.number_of_adm_nodes = 0
        self.number_of_nodes = 0
        self.number_of_leaves = 0

    def generate_adm_tree(self, adm_fun=admissibility):
        nodes_3d_to_add = [self.root]

        while nodes_3d_to_add:
            node_3d = nodes_3d_to_add.pop()
            children1 = node_3d.node1.children.flatten()
            children2 = node_3d.node2.children.flatten()

            # Acá se puede hacer una optimización sacando a los hijos None de inmediato
            # y obtener su posición mediante los ids de nodo1 y nodo2
            for i, child1 in enumerate(children1):
                if not child1: continue
                for j, child2 in enumerate(children2):
                    if not child2: continue
                    adm = adm_fun(child1.bbox, child2.bbox)
                    new_node_3d = Node3D(parent=node_3d, node1=child1, node2=child2, level=child1.level, adm=adm, leaf=adm)
                    node_3d.children[i,j] = new_node_3d
                    # If admissible, we stop constructing the branch.
                    # If not admissible, add to the stack (only if max_depth is not yet reached):
                    if adm:
                        self.number_of_adm_nodes += 1
                        self.number_of_leaves += 1
                    else:
                        if new_node_3d.level < self.max_depth:
                            nodes_3d_to_add.append(new_node_3d)
                        elif new_node_3d.level == self.max_depth:
                            new_node_3d.leaf = True
                            self.number_of_leaves += 1
                    self.number_of_nodes += 1

    # New:
    def add_matrix(self, A):
        nodes_3d_to_check = [self.root]
        while nodes_3d_to_check:
            node_3d = nodes_3d_to_check.pop()
            if node_3d.leaf:
                rows = node_3d.node1.dof_indices
                cols = node_3d.node2.dof_indices
                mesh = _np.meshgrid(rows, cols, indexing="ij")
                node_3d.matrix_block = A[mesh[0], mesh[1]]
            else:
                if node_3d.level < self.max_depth:
                    nodes_3d_to_check.extend(node_3d.children[node_3d.children != None].flatten())

    # New
    def add_matrix_with_ACA(self, A, epsilon=1e-3, verbose=False):
        # TODO: add compression algorithm as parameter of function
        nodes_3d_to_check = [self.root]
        while nodes_3d_to_check:
            node_3d = nodes_3d_to_check.pop()
            if node_3d.leaf:
                rows = node_3d.node1.dof_indices
                cols = node_3d.node2.dof_indices
                mesh = _np.meshgrid(rows, cols, indexing="ij")
                if node_3d.adm:
                    node_3d.u_vectors, node_3d.v_vectors = ACAPP(A[mesh[0], mesh[1]], epsilon=epsilon, method=2, verbose=verbose)
                else:
                    node_3d.matrix_block = A[mesh[0], mesh[1]]
            else:
                if node_3d.level < self.max_depth:
                    nodes_3d_to_check.extend(node_3d.children[node_3d.children != None].flatten())

    # New:
    def get_matrix(self):
        m = len(self.root.node1.points)
        n = len(self.root.node2.points)
        A = _np.zeros((m, n))
        nodes_3d_to_check = [self.root]
        while nodes_3d_to_check:
            node_3d = nodes_3d_to_check.pop()
            if node_3d.leaf:
                rows = node_3d.node1.dof_indices
                cols = node_3d.node2.dof_indices
                mesh = _np.meshgrid(rows, cols, indexing="ij")
                A[mesh[0], mesh[1]] = node_3d.matrix_block
            else:
                if node_3d.level < self.max_depth:
                    nodes_3d_to_check.extend(node_3d.children[node_3d.children != None].flatten())
        return A

    # New:
    def add_vector(self, b):
        nodes_3d_to_check = [self.root]
        while nodes_3d_to_check:
            node_3d = nodes_3d_to_check.pop()
            if node_3d.leaf:
                cols = node_3d.node2.dof_indices
                node_3d.vector_segment = b[cols]
            else:
                if node_3d.level < self.max_depth:
                    nodes_3d_to_check.extend(node_3d.children[node_3d.children != None].flatten())

    # New:
    def get_vector(self):
        n = len(self.root.node2.points)
        b = _np.zeros(n)
        nodes_3d_to_check = [self.root]
        while nodes_3d_to_check:
            node_3d = nodes_3d_to_check.pop()
            if node_3d.leaf:
                cols = node_3d.node2.dof_indices
                b[cols] = node_3d.vector_segment
            else:
                if node_3d.level < self.max_depth:
                    nodes_3d_to_check.extend(node_3d.children[node_3d.children != None].flatten())
        return b

    # New:
    def matvec(self, dtype=None):
        m = len(self.root.node1.points)
        result_vector = _np.zeros(m, dtype=dtype)
        nodes_3d_to_check = [self.root]
        while nodes_3d_to_check:
            node_3d = nodes_3d_to_check.pop()
            if node_3d.leaf:
                rows = node_3d.node1.dof_indices
                result_vector[rows] += (node_3d.matrix_block @ node_3d.vector_segment)
            else:
                if node_3d.level < self.max_depth:
                    nodes_3d_to_check.extend(node_3d.children[node_3d.children != None].flatten())
        return result_vector

    # New
    def matvec_compressed(self, dtype=None):
        m = len(self.root.node1.points)
        result_vector = _np.zeros(m, dtype=dtype)
        nodes_3d_to_check = [self.root]
        while nodes_3d_to_check:
            node_3d = nodes_3d_to_check.pop()
            if node_3d.leaf:
                rows = node_3d.node1.dof_indices
                if node_3d.adm:
                    result_vector[rows] += (node_3d.u_vectors.T @ node_3d.v_vectors @ node_3d.vector_segment)
                else:
                    result_vector[rows] += (node_3d.matrix_block @ node_3d.vector_segment)
            else:
                if node_3d.level < self.max_depth:
                    nodes_3d_to_check.extend(node_3d.children[node_3d.children != None].flatten())
        return result_vector

    def check_valid_tree(self, node_3d):
        # Se me ocurre hacer
        pass

    # New
    def calculate_compressed_matrix_storage(self, size_dtype):
        # size_dtype: size of dtype in bit (_np.complex128 = 128)
        total_sum = 0
        nodes_3d_to_check = [self.root]
        while nodes_3d_to_check:
            node_3d = nodes_3d_to_check.pop()
            if node_3d.leaf:
                if node_3d.adm:
                    total_sum += _np.prod(node_3d.u_vectors.shape)
                    total_sum += _np.prod(node_3d.v_vectors.shape)
                else:
                    total_sum += _np.prod(node_3d.matrix_block.shape)
            else:
                if node_3d.level < self.max_depth:
                    nodes_3d_to_check.extend(node_3d.children[node_3d.children != None].flatten())
        return total_sum * size_dtype / 8

    def search_node(self, tuple_of_ids):
        assert type(tuple_of_ids) == tuple, "tuple_of_ids must be a tuple two of strings"
        assert len(tuple_of_ids) == 2, "tuple_of_ids must be a tuple two of strings"
        assert len(tuple_of_ids[0]) == len(tuple_of_ids[1]), "The strings must have the same size"

        id1, id2 = tuple_of_ids
        if id1[0] != "0" or id2[0] != "0":
            return

        node_3d = self.root
        for ids in zip(id1[1:], id2[1:]):
            node_3d = node_3d.children[int(ids[0]), int(ids[1])]
            if not node_3d:
                return
        return node_3d

    def print_tree(self, node_3d, file=None):
        print("|\t"*node_3d.level + f"({node_3d.node1.id}, {node_3d.node2.id}) - Leaf: {node_3d.leaf}", file=file)
        for child in node_3d.children.flatten():
            if child:
                self.print_tree(child, file)
    
    # New:
    def print_tree_with_matrix(self, node_3d, file=None):
        print("|\t"*node_3d.level + f"({node_3d.node1.id}, {node_3d.node2.id}) - Leaf: {node_3d.leaf}", file=file)
        if node_3d.leaf:
            print("|\t"*node_3d.level + f"{node_3d.matrix_block}", file=file)
        for child in node_3d.children.flatten():
            if child:
                self.print_tree_with_matrix(child, file)

    def plot_node_adm(self, target_node, points):
        assert target_node != None, "Node is Null"
    
        import plotly.graph_objects as go
    
        # First, search for all children with the same level of target_node
        nodes_3d_to_check = [self.root]
        nodes_to_plot = []
        while nodes_3d_to_check:
            node_3d = nodes_3d_to_check.pop()
            length_id = node_3d.level+1
            target_node_id = target_node.id[:length_id]
            node1_id = node_3d.node1.id[:length_id]
            node2_id = node_3d.node2.id[:length_id]
            if target_node_id == node1_id or target_node_id == node2_id:
                if node_3d.adm:
                    if target_node_id == node1_id and node_3d.node2 not in nodes_to_plot:
                        nodes_to_plot.append(node_3d.node2)
                    elif target_node_id == node2_id and node_3d.node1 not in nodes_to_plot:
                        nodes_to_plot.append(node_3d.node1)
                elif node_3d.level < target_node.level:
                    nodes_3d_to_check.extend(node_3d.children[node_3d.children != None].flatten())
    
        # Now, we plot the admissibility of our node:
        all_vertices = _np.asarray(points).T
        bbox = target_node.bbox
        plots_data = [
            go.Scatter3d(x=all_vertices[0,:], y=all_vertices[1,:], z=all_vertices[2,:],
                         mode='markers', marker=dict(size=1)
            ),
            go.Mesh3d(
                x=[bbox[0,0], bbox[0,0], bbox[0,1], bbox[0,1], bbox[0,0], bbox[0,0], bbox[0,1], bbox[0,1]],
                y=[bbox[1,0], bbox[1,1], bbox[1,1], bbox[1,0], bbox[1,0], bbox[1,1], bbox[1,1], bbox[1,0]],
                z=[bbox[2,0], bbox[2,0], bbox[2,0], bbox[2,0], bbox[2,1], bbox[2,1], bbox[2,1], bbox[2,1]],
                i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                opacity=0.2,
                flatshading = True,
                color='#EF553B',
            )
        ]
        for node in nodes_to_plot:
            bbox = node.bbox
            plots_data.append(
                go.Mesh3d(
                    x=[bbox[0,0], bbox[0,0], bbox[0,1], bbox[0,1], bbox[0,0], bbox[0,0], bbox[0,1], bbox[0,1]],
                    y=[bbox[1,0], bbox[1,1], bbox[1,1], bbox[1,0], bbox[1,0], bbox[1,1], bbox[1,1], bbox[1,0]],
                    z=[bbox[2,0], bbox[2,0], bbox[2,0], bbox[2,0], bbox[2,1], bbox[2,1], bbox[2,1], bbox[2,1]],
                    i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                    j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                    k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                    opacity=0.2,
                    flatshading = True,
                    color='#00CC96',
                )
            )
    
        fig = go.Figure(data=plots_data)
        fig.update_layout(
            autosize=False,
            width=800,
            height=800,
            showlegend=False
        )
        fig.show()

if __name__ == "__main__":
    import bempp.api
    from MyHM.structures.octree import Octree
    grid = bempp.api.shapes.sphere(h=0.2)
    bbox = grid.bounding_box
    vertices = grid.vertices
    dof_indices = list(range(vertices.shape[1]))
    octree = Octree(vertices.T, dof_indices, bbox, max_depth=4)
    octree.generate_tree()
    tree_3d = Tree3D(octree.root, octree.max_depth)
    tree_3d.generate_adm_tree()
    print("Working...")