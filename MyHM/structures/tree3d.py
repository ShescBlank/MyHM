import numpy as _np
from time import time
from MyHM.structures.octree import Node, Octree
from MyHM.structures.utils import admissibility

def compress_node_mp(node_3d, method, boundary_operator, parameters, singular_sm, epsilon=1e-3, verbose=False):
    from MyHM.assembly import partial_dense_assembler as pda
    from MyHM.assembly import singular_assembler_sparse as sas 

    rows = node_3d.node1.dof_indices
    cols = node_3d.node2.dof_indices
    meshgrid = _np.meshgrid(rows, cols, indexing="ij")
    if node_3d.adm:
        node_3d.u_vectors, node_3d.v_vectors = method(rows, cols, boundary_operator, parameters, singular_sm, epsilon=epsilon, verbose=verbose)
    else:
        node_3d.matrix_block = pda(boundary_operator.descriptor, boundary_operator.domain, boundary_operator.dual_to_range, parameters, rows, cols)
        node_3d.matrix_block = _np.array(node_3d.matrix_block + singular_sm[meshgrid[0], meshgrid[1]])

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
        self.stats = {
            "compression_storage": 0,
            "full_storage": 0,
            "compression_time": 0,
            "matvec_time": 0,
        }

class Tree3D:
    def __init__(self, octree: Octree):
        self.root = Node3D(parent=None, node1=octree.root, node2=octree.root, level=0, adm=False, leaf=False)
        self.max_depth = octree.max_depth
        self.min_block_size = octree.min_block_size
        self.max_element_diameter = octree.max_element_diameter
        self.adm_leaves = []
        self.nadm_leaves = []
        self.stats = {
            "number_of_nodes": 1,
            "number_of_leaves": 0,
            "number_of_not_adm_leaves": 0,
            "number_of_adm_leaves": 0,
        }

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
                    adm = adm_fun(child1.bbox, child2.bbox, self.max_element_diameter)
                    new_node_3d = Node3D(parent=node_3d, node1=child1, node2=child2, level=child1.level, adm=adm, leaf=adm)
                    node_3d.children[i,j] = new_node_3d
                    # If admissible, stop constructing the branch.
                    # If not admissible, add to the stack (only if max_depth is not yet reached):
                    # TODO: Review this if statement when min_block_size is implemented (quité el if debido al cambio en la admisibilidad)
                    # If minimum size is not met, stop constructing the branch 
                    # if len(child1.points) < self.min_block_size or len(child2.points) < self.min_block_size:
                    #     new_node_3d.adm = False
                    #     new_node_3d.leaf = True
                    #     self.nadm_leaves.append(new_node_3d)
                    #     self.stats["number_of_leaves"] += 1
                    #     self.stats["number_of_not_adm_leaves"] += 1
                    if adm:
                        self.adm_leaves.append(new_node_3d)
                        self.stats["number_of_leaves"] += 1
                        self.stats["number_of_adm_leaves"] += 1
                    else:
                        if new_node_3d.level < self.max_depth:
                            nodes_3d_to_add.append(new_node_3d)
                        elif new_node_3d.level == self.max_depth:
                            new_node_3d.leaf = True
                            self.nadm_leaves.append(new_node_3d)
                            self.stats["number_of_leaves"] += 1
                            self.stats["number_of_not_adm_leaves"] += 1
                    self.stats["number_of_nodes"] += 1

    # New:
    def add_matrix(self, A):
        """
        Adds the uncompressed matrix from the full matrix
        """
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
    def add_matrix_with_ACA(self, A, method, epsilon=1e-3, verbose=False):
        """
        Adds the compressed matrix from the full matrix
        """
        nodes_3d_to_check = [self.root]
        while nodes_3d_to_check:
            node_3d = nodes_3d_to_check.pop()
            if node_3d.leaf:
                rows = node_3d.node1.dof_indices
                cols = node_3d.node2.dof_indices
                mesh = _np.meshgrid(rows, cols, indexing="ij")
                if node_3d.adm:
                    t0_compression = time()
                    node_3d.u_vectors, node_3d.v_vectors = method(A[mesh[0], mesh[1]], epsilon=epsilon, verbose=verbose)
                    tf_compression = time()
                    node_3d.stats["compression_time"] = tf_compression - t0_compression
                    node_3d.stats["compression_storage"] = _np.prod(node_3d.u_vectors.shape) + _np.prod(node_3d.v_vectors.shape)
                else:
                    node_3d.matrix_block = A[mesh[0], mesh[1]]
                node_3d.stats["full_storage"] = len(rows) * len(cols)
            else:
                if node_3d.level < self.max_depth:
                    nodes_3d_to_check.extend(node_3d.children[node_3d.children != None].flatten())

    def add_compressed_matrix(self, method, device_interface, boundary_operator, parameters, epsilon=1e-3, verbose=False):
        """
        Adds the compressed matrix using a custom assembler (does not require the full matrix)
        """
        from MyHM.assembly import partial_dense_assembler as pda
        from MyHM.assembly import singular_assembler_sparse as sas 
        
        # Obtain singular part:
        singular_sm = sas(device_interface, boundary_operator.descriptor, boundary_operator.domain, boundary_operator.dual_to_range, parameters) 

        # Traverse tree:
        nodes_3d_to_check = [self.root]
        while nodes_3d_to_check:
            node_3d = nodes_3d_to_check.pop()
            if node_3d.leaf:
                rows = node_3d.node1.dof_indices
                cols = node_3d.node2.dof_indices
                meshgrid = _np.meshgrid(rows, cols, indexing="ij")
                if node_3d.adm:
                    t0_compression = time()
                    node_3d.u_vectors, node_3d.v_vectors = method(rows, cols, boundary_operator, parameters, singular_sm, epsilon=epsilon, verbose=verbose)
                    tf_compression = time()
                    node_3d.stats["compression_time"] = tf_compression - t0_compression
                    node_3d.stats["compression_storage"] = _np.prod(node_3d.u_vectors.shape) + _np.prod(node_3d.v_vectors.shape)
                else:
                    # node_3d.matrix_block = _np.zeros((len(rows), len(cols)), dtype=_np.complex128)
                    node_3d.matrix_block = pda(boundary_operator.descriptor, boundary_operator.domain, boundary_operator.dual_to_range, parameters, rows, cols)
                    node_3d.matrix_block = _np.array(node_3d.matrix_block + singular_sm[meshgrid[0], meshgrid[1]])
                node_3d.stats["full_storage"] = len(rows) * len(cols)
            else:
                if node_3d.level < self.max_depth:
                    nodes_3d_to_check.extend(node_3d.children[node_3d.children != None].flatten())

    def add_compressed_matrix_mp(self, method, device_interface, boundary_operator, parameters, epsilon=1e-3, verbose=False):
        from MyHM.assembly import singular_assembler_sparse as sas 
        from MyHM.structures.utils import wrap_classes, unwrap_classes

        from joblib import Parallel
        from joblib import delayed

        # from multiprocessing import Pool
        # from itertools import repeat

        # Obtain singular part:
        singular_sm = sas(device_interface, boundary_operator.descriptor, boundary_operator.domain, boundary_operator.dual_to_range, parameters) 

        # Wrap classes to pickle:
        prev_grid_datas = wrap_classes(boundary_operator)
        
        # # Joblib:
        parallel_compression = delayed(compress_node_mp)
        parallel_tasks_adm = [parallel_compression(self.adm_leaves[i], method, boundary_operator, parameters, singular_sm, epsilon, verbose) for i in range(len(self.adm_leaves))]
        parallel_tasks_nadm = [parallel_compression(self.nadm_leaves[i], method, boundary_operator, parameters, singular_sm, epsilon, verbose) for i in range(len(self.nadm_leaves))]
        with Parallel(n_jobs=-1, verbose=10) as parallel_pool:
            parallel_pool(parallel_tasks_adm)
            parallel_pool(parallel_tasks_nadm)

        # # Multiprocessing:
        # pool = Pool(1)
        # pool.starmap(
        #     compress_node_mp,
        #     zip(
        #         self.adm_leaves,
        #         repeat(method),
        #         repeat(boundary_operator),
        #         repeat(parameters),
        #         repeat(singular_sm),
        #         repeat(epsilon),
        #         repeat(verbose),
        #     ),
        # )

        # Unwrap classes:
        unwrap_classes(prev_grid_datas)

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

    def get_matrix_from_compression(self, dtype=_np.complex128):
        m = len(self.root.node1.points)
        n = len(self.root.node2.points)
        A = _np.zeros((m, n), dtype=dtype)
        nodes_3d_to_check = [self.root]
        while nodes_3d_to_check:
            node_3d = nodes_3d_to_check.pop()
            if node_3d.leaf:
                rows = node_3d.node1.dof_indices
                cols = node_3d.node2.dof_indices
                mesh = _np.meshgrid(rows, cols, indexing="ij")

                if not node_3d.adm:
                    A[mesh[0], mesh[1]] = node_3d.matrix_block
                else:
                    A[mesh[0], mesh[1]] = _np.asarray(node_3d.u_vectors).T @ _np.asarray(node_3d.v_vectors)
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
        # Se me ocurre quizás, cuando ya tenga hechas las listas de hojas (admisibles y no admisibles), 
        # chequear que efectivamente estén todas las hojas del árbol en las listas.
        pass

    def check_singular_adm_leaves(self, device_interface, boundary_operator, parameters):
        from MyHM.assembly import singular_assembler_sparse as sas 
        
        # Obtain singular part:
        singular_sm = sas(device_interface, boundary_operator.descriptor, boundary_operator.domain, boundary_operator.dual_to_range, parameters) 

        for node_3d in self.adm_leaves:
            rows = node_3d.node1.dof_indices
            cols = node_3d.node2.dof_indices
            meshgrid = _np.meshgrid(rows, cols, indexing="ij")
            if singular_sm[meshgrid[0], meshgrid[1]].nnz != 0:
                return True
        return False


    # New
    def calculate_compressed_matrix_storage(self, verbose=False):
        total_storage_in_use = 0
        # aux = 0.0

        for node_3d in self.adm_leaves:
            total_storage_in_use += node_3d.stats["compression_storage"]
            # aux += node_3d.stats["full_storage"]

        for node_3d in self.nadm_leaves: 
            total_storage_in_use += node_3d.stats["full_storage"]
            # aux += node_3d.stats["full_storage"]

        if verbose:
            total_storage_without_compression = self.calculate_matrix_storage_without_compression()
            print(f"Total storage in use: \t\t\t{total_storage_in_use:>10} [floating point units]")
            print(f"Total storage without compression: \t{total_storage_without_compression:>10} [floating point units]")
            print(f"Percentage: \t\t\t\t{_np.round(total_storage_in_use / total_storage_without_compression * 100, decimals=2):>10} %")

        return total_storage_in_use

    def calculate_matrix_storage_without_compression(self):
        total_storage_without_compression = len(self.root.node1.dof_indices) * len(self.root.node2.dof_indices)
        return total_storage_without_compression

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

    def plot_storage_per_level(self, save=False, name="Results/Storage_per_level"):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        compression_storages = [0.0] * (self.max_depth + 1)
        full_storages = [0.0] * (self.max_depth + 1)
        levels = list(range(self.max_depth + 1))

        for node_3d in self.adm_leaves:
            compression_storages[node_3d.level] += node_3d.stats["compression_storage"]
            full_storages[node_3d.level] += node_3d.stats["full_storage"]
        
        df = pd.DataFrame({"Floating point units": compression_storages + full_storages, "Level": levels + levels,
                           "Type": ["Compression"] * (self.max_depth + 1) + ["Full"] * (self.max_depth + 1)})
        ax = sns.barplot(df, x="Level", y="Floating point units", hue="Type", errorbar=None, gap=0)
        bar_labels = _np.divide(compression_storages, full_storages, where=_np.asarray(full_storages)!=0, out=_np.zeros_like(compression_storages)) * 100
        bar_labels = _np.round(bar_labels, decimals=2)
        ax.bar_label(ax.containers[0], fontsize=10, labels=map(lambda x: f"{x}%", bar_labels))
        plt.title("Compression storage comparison (Adm leaves)")
        sns.move_legend(ax, "upper left")
        if save:
            plt.savefig(name)
            plt.close()
        else:
            plt.show()

    def pairplot(self, save=False, name="Results/Pairplot"):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        block_sizes = []
        block_ranks = []
        block_levels = []
        nodes_3d_to_check = [self.root]
        while nodes_3d_to_check:
            node_3d = nodes_3d_to_check.pop()
            if node_3d.leaf and node_3d.adm:
                block_sizes.append(
                    min(len(node_3d.node1.dof_indices), len(node_3d.node2.dof_indices))
                )
                block_ranks.append(node_3d.u_vectors.shape[0])
                block_levels.append(node_3d.level)
            else:
                if node_3d.level < self.max_depth:
                    nodes_3d_to_check.extend(node_3d.children[node_3d.children != None].flatten())

        df = pd.DataFrame({"Size": block_sizes, "Rank": block_ranks, "Ratio": _np.asarray(block_ranks)/_np.asarray(block_sizes), "Level": block_levels})

        fig, axes = plt.subplots(3, 3, figsize=(10, 8))
        fig.suptitle("Admissible leaves information")
        sns.histplot(df["Size"], ax=axes[0, 0], stat='count')
        sns.histplot(df["Rank"], ax=axes[1, 1], stat='count')
        sns.histplot(df["Ratio"], ax=axes[2, 2], stat='count')
        sns.scatterplot(x=df["Size"], y=df["Rank"], hue=df["Level"], ax=axes[1, 0])
        sns.scatterplot(x=df["Size"], y=df["Ratio"], hue=df["Level"], ax=axes[2, 0])
        sns.scatterplot(x=df["Rank"], y=df["Ratio"], hue=df["Level"], ax=axes[2, 1])

        # New y-axis to the right
        for i in range(3):
            ax2 = axes[i, i].twinx()
            # sns.histplot(df["Size"], ax=ax2, bins=4, stat='count')
            ax2.set_ylim(axes[i, i].get_ylim())
            ax2.set_yticks(axes[i, i].get_yticks()) 
            ax2.set_ylabel(axes[i, i].get_ylabel())

            axes[i, i].tick_params(axis='y',length=0)
            axes[i, i].set_yticklabels([])
            axes[i, i].set_ylabel("")
        axes[0, 0].set_ylabel(axes[2, 0].get_xlabel())

        # Remove shared tick labels:
        axes[1, 0].set_xticklabels([])
        axes[0, 0].set_xticklabels([])
        axes[1, 1].set_xticklabels([])
        axes[2, 1].set_yticklabels([])

        # Remove shared axes labels:
        axes[0, 0].set_xlabel("")
        axes[1, 0].set_xlabel("")
        axes[1, 1].set_xlabel("")
        # axes[1, 1].set_ylabel("")
        axes[2, 1].set_ylabel("")
        # axes[2, 2].set_ylabel("")

        # Legends:
        fig.legend(*axes[1, 0].get_legend_handles_labels(), title='Level', bbox_to_anchor=axes[0, 2].get_position()) # loc='upper right'
        axes[1, 0].get_legend().remove()
        axes[2, 0].get_legend().remove()
        axes[2, 1].get_legend().remove()

        # Remove upper diagonal plots:
        hide_indices = _np.triu_indices_from(axes, 1)
        for i, j in zip(*hide_indices):
            axes[i, j].remove()
            axes[i, j] = None
            # axes[i, j].set_visible(False)

        plt.tight_layout(w_pad=-4)
        if save:
            plt.savefig(name)
            plt.close()
        else:
            plt.show()

    def compression_imshow(self, save=False, name="Results/Imshow"):
        import matplotlib.pyplot as plt

        m = len(self.root.node1.points)
        n = len(self.root.node2.points)
        A = _np.zeros((m, n), dtype=float)
        sorted_rows = []
        sorted_cols = []

        nodes_3d_to_check = [self.root]
        while nodes_3d_to_check:
            node_3d = nodes_3d_to_check.pop(0)
            if node_3d.leaf:
                rows = node_3d.node1.dof_indices
                cols = node_3d.node2.dof_indices
                mesh = _np.meshgrid(rows, cols, indexing="ij")

                # for i in rows:
                #     if i not in sorted_rows:
                #         sorted_rows.append(i)
                # for i in cols:
                #     if i not in sorted_cols:
                #         sorted_cols.append(i)

                if node_3d.adm:
                    size = min(len(node_3d.node1.dof_indices), len(node_3d.node2.dof_indices))
                    rank = node_3d.u_vectors.shape[0]
                    A[mesh[0], mesh[1]] = rank / size
                else:
                    A[mesh[0], mesh[1]] = 1.0
            else:
                if node_3d.level < self.max_depth:
                    nodes_3d_to_check.extend(node_3d.children[node_3d.children != None].flatten())

        plt.imshow(A, cmap="Blues", vmin=0.0, vmax=1.0)
        plt.title("Image of compression")
        plt.colorbar()
        if save:
            plt.savefig(name)
            plt.close()
        else:
            plt.show()

        # plt.imshow(A[:, sorted_cols][sorted_rows, :], cmap="Blues", vmin=0.0, vmax=1.0)
        # plt.title("Image of compression (sorted)")
        # plt.colorbar()
        # plt.show()

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