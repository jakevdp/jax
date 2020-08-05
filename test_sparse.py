import numpy as np
from jax import jit
from jax.experimental.sparse import CSRMatrix
from scipy.sparse import csr_matrix

num_cols = 10
num_rows = 10
nnz = 12

rng = np.random.RandomState(0)

data = rng.uniform(size=nnz).astype('float32')
ind = rng.randint(0, num_cols, size=len(data)).astype('int32')
indptr = np.concatenate([np.zeros(1),
                         np.sort(rng.randint(0, len(ind), size=num_rows - 1)),
                         np.array([len(ind)])]).astype('int32')

A = CSRMatrix((num_rows, num_cols), data, ind, indptr)
v = np.arange(A.shape[1])
print(np.allclose(A @ v, A.toarray() @ v))

#jit(lambda x: x)(A)
# jit(scale)(3., A)