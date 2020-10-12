import pytest

def test_import():
    import jager
    from jager.sparse import SparseMatrix2D

def test_init_from_indices():
    from jager.sparse import SparseMatrix2D
    x = SparseMatrix2D(
        index=[
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3]
        ],
        data=[1, 1, 1, 1],
        shape=[4, 4]
    )

def test_init_from_dense():
    import jax.numpy as jnp
    from jager.sparse import SparseMatrix2D
    x = SparseMatrix2D.from_dense(
        jnp.identity(4),
    )


def test_init_dense_indices_agree():
    from jager.sparse import SparseMatrix2D
    import jax.numpy as jnp

    x_0 = SparseMatrix2D(
        index=[
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3]
        ],
        data=jnp.ones(4),
        shape=[4, 4]
    )

    x_1 = SparseMatrix2D.from_dense(
        jnp.identity(4),
    )


    assert jnp.all(x_0 == x_1)
