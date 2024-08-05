import numpy as _np
from MyHM.structures.utils import tuple2index, index2tuple

# Node and Octree:
class Node:
    pass

class Node:
    def __init__(self, parent: Node, id: str, level: int, bbox: _np.ndarray):
        """
        Parameters:
        - 
        """
        self.parent = parent
        self.id = id
        self.level = level
        self.bbox = bbox
        self.children = _np.empty((8), dtype=Node) # Type: numpy.ndarray
        self.points = [] # Type: list (n x 3) # Prefiero lista por la eficiencia del append de puntos
        
        self.dof_indices = [] # Type: list (n) # Corresponde a los índices de los puntos

    def add_child(self, child: Node):
        self.children[int(child.id[-1])] = child


class Octree:
    def __init__(self, points: _np.ndarray, dof_indices: list, bbox: _np.ndarray, max_depth=4, min_block_size=2):
        # TODO: Add min_size to the generation of the tree
        assert max_depth > 0, "max_depth parameter must be greater than 0"
        self.root = Node(parent=None, id="0", level=0, bbox=bbox)
        self.root.points = list(points)
        self.root.dof_indices = dof_indices
        self.max_depth = max_depth
        self.min_block_size = min_block_size

    def generate_tree(self):
        # Hacer versión iterativa o recursiva
        nodes_to_add = [self.root]

        while nodes_to_add:
            node = nodes_to_add.pop()

            x_min = node.bbox[0,0]
            x_max = node.bbox[0,1]
            y_min = node.bbox[1,0]
            y_max = node.bbox[1,1]
            z_min = node.bbox[2,0]
            z_max = node.bbox[2,1]
            delta_x = (x_max - x_min) / 2
            delta_y = (y_max - y_min) / 2
            delta_z = (z_max - z_min) / 2
    
            assert delta_x > 0, "Wrong x-dimension for bbox"
            assert delta_y > 0, "Wrong y-dimension for bbox"
            assert delta_z > 0, "Wrong z-dimension for bbox"
    
            # From each node, eight children are created
            # Assign each point to a child and create the child if it does not exist
            # for point in node.points:
            for i in range(len(node.points)):
                point = node.points[i]
                dof_index = node.dof_indices[i]
                id_x = int((point[0] - x_min) >= delta_x)
                id_y = int((point[1] - y_min) >= delta_y)
                id_z = int((point[2] - z_min) >= delta_z)
                index = tuple2index((id_x, id_y, id_z))
                if not node.children[index]:
                    sub_bbox_x = [x_min + delta_x * id_x, x_max + delta_x * (id_x - 1)]
                    sub_bbox_y = [y_min + delta_y * id_y, y_max + delta_y * (id_y - 1)]
                    sub_bbox_z = [z_min + delta_z * id_z, z_max + delta_z * (id_z - 1)]
                    sub_bbox = _np.array([sub_bbox_x, sub_bbox_y, sub_bbox_z])
                    node.add_child(Node(parent=node, id=node.id + str(index), level=node.level + 1, bbox=sub_bbox))
                node.children[index].points.append(point)
                node.children[index].dof_indices.append(dof_index)
    
            # Repeat process until max_depth is reached and with valid children:
            for child in node.children:
                # if child and child.level < self.max_depth and child.points:
                if child and child.level < self.max_depth:
                    nodes_to_add.append(child)

    def check_valid_tree(self, node):
        total = len(node.points)
        sum = 0
        for child in node.children:
            # if child and child.points:
            if child:
                sum += len(child.points)
        if sum != total:
            return False
        for child in node.children:
            if child and child.level < self.max_depth and not self.check_valid_tree(child):
                return False
        return True

    def print_tree(self, node, file=None):
        if node.points:
            print("|\t"*node.level + node.id + f": (Level {node.level}, {len(node.points)} Points)", file=file)
        else:
            print("|\t"*node.level + node.id + f": (Level {node.level}, 0 Points)", file=file)
        for child in node.children:
            if child:
                self.print_tree(child, file)

    def search_node(self, id):
        assert type(id) == str, "id must be a string"

        if id[0] != "0":
            return

        node = self.root
        for s in id[1:]:
            node = node.children[int(s)]
            if not node:
                return
        return node
        

    def plot_node(self, node):
        assert node != None, "Node is Null"
        
        import plotly.graph_objects as go
        
        all_vertices = _np.asarray(self.root.points).T
        node_vertices = _np.asarray(node.points).T
        bbox = node.bbox

        fig = go.Figure(data=[
            go.Scatter3d(x=all_vertices[0,:], y=all_vertices[1,:], z=all_vertices[2,:],
                         mode='markers', marker=dict(size=1)
            ),
            go.Scatter3d(x=node_vertices[0,:], y=node_vertices[1,:], z=node_vertices[2,:],
                         mode='markers', marker=dict(size=2)
            ),
            go.Mesh3d(
                x=[bbox[0,0], bbox[0,0], bbox[0,1], bbox[0,1], bbox[0,0], bbox[0,0], bbox[0,1], bbox[0,1]],
                y=[bbox[1,0], bbox[1,1], bbox[1,1], bbox[1,0], bbox[1,0], bbox[1,1], bbox[1,1], bbox[1,0]],
                z=[bbox[2,0], bbox[2,0], bbox[2,0], bbox[2,0], bbox[2,1], bbox[2,1], bbox[2,1], bbox[2,1]],
                i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                opacity=0.2,
                # color='#DC143C',
                flatshading = True,
            )
            ])
        fig.update_layout(
            autosize=False,
            width=800,
            height=800,
            showlegend=False
        )
        fig.show()

    def plot_children(self, node):
        assert node != None, "Node is Null"
        
        import plotly.graph_objects as go

        all_vertices = _np.asarray(self.root.points).T
        plots_data = [
            go.Scatter3d(x=all_vertices[0,:], y=all_vertices[1,:], z=all_vertices[2,:],
                         mode='markers', marker=dict(size=1)
            )
        ]

        for child in node.children:
            if child:
                bbox = child.bbox
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
                    )
                )
                node_vertices = _np.asarray(child.points).T
                plots_data.append(
                    go.Scatter3d(x=node_vertices[0,:], y=node_vertices[1,:], z=node_vertices[2,:],
                         mode='markers', marker=dict(size=2)
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
    grid = bempp.api.shapes.sphere(h=0.2)
    bbox = grid.bounding_box
    vertices = grid.vertices
    dof_indices = list(range(vertices.shape[1]))
    octree = Octree(vertices.T, dof_indices, bbox, max_depth=4)
    octree.generate_tree()
    print("Working...")