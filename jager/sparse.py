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
        return dense.at[self.index].add(self.data)

    @classmethod
    def from_dense(cls, x):
        x = jnp.asarray(x)
        nz = (x != 0)
        return cls(jnp.where(nz), x[nz], x.shape)

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self):
        return tuple(self._shape)

    @property
    def ndim(self):
        return len(self._shape)

    # @jax.jit
    def __matmul__(self, x):
        if not self.ndim == 2 and x.ndim == 2:
            raise NotImplementedError

        assert self.shape[1] == x.shape[0]

        # (n_entries, )
        rows = self.index[0, :]
        cols = self.index[1, :]

        # (n_entries, x_shape[1])
        in_ = x.take(cols, axis=0)

        # data: shape=(n_entries)
        prod = in_ * self.data[:, None]

        return jax.ops.segment_sum(prod, rows, self.shape[0])

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
