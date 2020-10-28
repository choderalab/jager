# =============================================================================
# IMPORTS
# =============================================================================
import abc
import jax
from jax import numpy as jnp

# =============================================================================
# BASE CLASS
# =============================================================================
class SparseMatrix(abc.ABC):
    """ Base class for sparse matrix. """
    def __init__(self):
        super(SparseMatrix, self).__init__()
        # implement more general sparse matrix

# =============================================================================
# MODULE CLASS
# =============================================================================
class SparseMatrix2D(SparseMatrix):
    """ Two-dimensional sparse matrix. """
    def __init__(self, index=None, data=None, shape=None):
        super(SparseMatrix2D, self).__init__()
        self.index = jnp.atleast_2d(index)
        self.data = jnp.asarray(data)
        self._shape = shape

    def to_dense(self):
        dense = jnp.zeros(self.shape, self.dtype)
        return dense.at[tuple(self.index)].add(self.data)

    @classmethod
    def from_dense(cls, x):
        x = jnp.asarray(x)
        nz = (x != 0)
        return cls(jnp.where(nz), x[nz], x.shape)

    @property
    def nnz(self):
        return self.data.shape[0]

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self):
        return tuple(self._shape)

    @property
    def ndim(self):
        return len(self._shape)

    @staticmethod
    @partial(jax.jit, static_argnums=(3,))
    def _matmul(x, index, data, num_segments):
        # (n_entries, )
        rows = index[0, :]
        cols = index[1, :]

        # (n_entries, x_shape[1])
        in_ = x.take(cols, axis=0)

        # data: shape=(n_entries)
        prod = in_ * jax.lax.broadcast(data, in_.shape[1:][::-1]).transpose(
            list(range(len(in_.shape)))[::-1]
        )

        return jax.ops.segment_sum(prod, rows, num_segments)

    @jax.jit
    def __matmul__(self, x):
        if not self.ndim == 2:
            raise NotImplementedError

        assert self.shape[1] == x.shape[0]

        return self._matmul(x, self.index, self.data, self.shape[0])

    def __repr__(self):
        return 'Sparse Matrix with shape=%s, indices=%s, and data=%s' % (
            self.shape,
            self.index,
            self.data,
        )

    def __eq__(self, x):
        if not isinstance(x, type(self)):
            return False
        else:
            return self.to_dense() == x.to_dense()

# =============================================================================
# REGISTER WITH JAX
# =============================================================================
def _flatten_SparseMatrix2D(x):
    return x.index, x.data, x.shape

def _unflatten_SparseMatrix2D(index, data, shape):
    return SparseMatrix2D(index=index, data=data, shape=shape)

jax.tree_util.register_pytree_node(
    SparseMatrix2D,
    _flatten_SparseMatrix2D,
    _unflatten_SparseMatrix2D
)
