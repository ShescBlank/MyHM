import numpy as _np

class BlockedTree:
    def __init__(self, row_dims, col_dims, dtype=_np.complex128):
        assert isinstance(row_dims, tuple) and isinstance(col_dims, tuple), "'row_dims' and 'col_dims' must be a n-tuple of integers"
        self.row_dims, self.col_dims = row_dims, col_dims
        self.full_shape = (_np.sum(row_dims), _np.sum(col_dims))
        self.shape = (len(row_dims), len(col_dims))
        self.dtype = dtype

        self.blocks = [[[] for _ in range(self.shape[1])] for _ in range(self.shape[0])]
        self.factors = [[[] for _ in range(self.shape[1])] for _ in range(self.shape[0])]

    def add(self, operator, position=None, value=1.0):
        assert callable(getattr(operator, 'dot', None)), "'operator' must have a dot() method implemented"
        assert (isinstance(position, tuple) and len(position) == 2) or position == None, "'position' must be a 2-tuple of integers or None"
        if self.shape != (1, 1) and position == None:
            raise TypeError(f"'position' must be 2-tuple of integers in a multidimensional {type(self).__name__}")

        if position is None:
            position = (0, 0)
            blocks, factors = self[position]
        else:
            blocks, factors = self[position]

        if operator.shape != (self.row_dims[position[0]], self.col_dims[position[1]]):
            raise ValueError(f"dimension mismatch between operator shape {operator.shape} and block shape {(self.row_dims[position[0]], self.col_dims[position[1]])}")
        blocks.append(operator)
        factors.append(value)

    def dot(self, b):
        assert b.shape[0] == self.full_shape[1], f"dimension mismatch between shape of b ({b.shape[0]}) and number of columns of block ({self.full_shape[1]})"

        result_vector = _np.zeros(self.full_shape[0], dtype=self.dtype)
        start_row = 0
        end_row = 0
        for i in range(self.shape[0]):
            start_col = 0
            end_col = 0
            end_row += self.row_dims[i]
            for j in range(self.shape[1]):
                end_col += self.col_dims[j]
                blocks, factors = self[i, j]
                for l in range(len(blocks)):
                    result_vector[start_row:end_row] += factors[l] * (blocks[l].dot(b[start_col:end_col]))
                start_col += self.col_dims[j]
            start_row += self.row_dims[i]

        return result_vector

    def linearoperator(self):
        from scipy.sparse.linalg import LinearOperator
        return LinearOperator(self.full_shape, matvec=self.dot, dtype=self.dtype)

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key)==2:
            i = key[0] % self.shape[0]
            j = key[1] % self.shape[1]
            return self.blocks[i][j], self.factors[i][j]
        else:
            raise TypeError(f"{type(self).__name__} indices must be a 2-tuple of integers, not {type(key).__name__}")

if __name__ == "__main__":
    # A = _np.array([[1,2],[3,4]])
    # b = _np.array([5,6])
    # blocked_tree1 = BlockedTree((2,), (2,))
    # print(blocked_tree1.shape)
    # print(blocked_tree1.blocks)
    # blocked_tree1.add(A)
    # print(blocked_tree1.blocks)
    # print(blocked_tree1.factors)
    # print(A.dot(b))
    # print(blocked_tree1.dot(b))

    A1 = _np.random.rand(10, 10)
    A2 = _np.random.rand(10, 10)
    A = A1 - A2
    b = _np.random.rand(10)
    blocked_tree1 = BlockedTree((2,5,3), (5,1,4), dtype=_np.float64)
    blocked_tree1.add(A1[:2,:5], (0,0))
    blocked_tree1.add(A1[:2,5:6], (0,1))
    blocked_tree1.add(A1[:2,6:10], (0,2))
    blocked_tree1.add(A1[2:7,:5], (1,0))
    blocked_tree1.add(A1[2:7,5:6], (1,1))
    blocked_tree1.add(A1[2:7,6:10], (1,2))
    blocked_tree1.add(A1[7:10,:5], (2,0))
    blocked_tree1.add(A1[7:10,5:6], (2,1))
    blocked_tree1.add(A1[7:10,6:10], (2,2))
    blocked_tree1.add(A2[:2,:5], (0,0), -1.0)
    blocked_tree1.add(A2[:2,5:6], (0,1), -1.0)
    blocked_tree1.add(A2[:2,6:10], (0,2), -1.0)
    blocked_tree1.add(A2[2:7,:5], (1,0), -1.0)
    blocked_tree1.add(A2[2:7,5:6], (1,1), -1.0)
    blocked_tree1.add(A2[2:7,6:10], (1,2), -1.0)
    blocked_tree1.add(A2[7:10,:5], (2,0), -1.0)
    blocked_tree1.add(A2[7:10,5:6], (2,1), -1.0)
    blocked_tree1.add(A2[7:10,6:10], (2,2), -1.0)
    result1 = A.dot(b)
    result2 = blocked_tree1.dot(b)
    # print(result1)
    # print(result2)
    print("Relative error matvec:")
    print(_np.linalg.norm(result1 - result2) / _np.linalg.norm(result1))

    from scipy.sparse.linalg import gmres
    print("\nGmres:")
    # A:
    sol1, info = gmres(A, b, rtol=1e-5)
    print("Full A:", _np.linalg.norm(A.dot(sol1) - b) / _np.linalg.norm(b))

    # Custom LinearOperator:
    linear_op = blocked_tree1.linearoperator()
    sol2, info = gmres(linear_op, b, rtol=1e-5)
    print("Blocked Tree", _np.linalg.norm(blocked_tree1.dot(sol2) - b) / _np.linalg.norm(b))

    print("Norm of difference:", _np.linalg.norm(sol2 - sol1))