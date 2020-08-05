from jax import core, partial
from jax import numpy as jnp
from jax.interpreters import xla
from scipy.sparse import csr_matrix

from jax.lib import xla_bridge as xb
from jax.lib import xla_client as xc

class CSRMatrix:
  def __init__(self, shape, data, ind, indptr):
    self.ind = ind
    self.indptr = indptr
    self.data = data

    self.shape = shape
    self.dtype = data.dtype

  @property
  def nnz(self):
    return len(self.data)

  def to_scipy_csr(self):
    return csr_matrix((self.data, self.ind, self.indptr),
                       shape=self.shape)

  def toarray(self):
    return self.to_scipy_csr().toarray()

  def __matmul__(self, v):
    assert v.ndim == 1
    assert len(v) == self.shape[1]
    i = jnp.cumsum(jnp.zeros_like(self.ind).at[self.indptr].add(1))
    return jnp.zeros(self.shape[0], self.dtype).at[i - 1].add(self.data * v[self.ind])

  def __repr__(self):
    return repr(self.to_scipy_csr())

  def __str__(self):
    return str(self.to_scipy_csr())


def scale(a, mat):
  return scale_p.bind(a, mat)

scale_p = core.Primitive('scale')

@scale_p.def_impl
def scale_impl(a, mat):
  return CSRMatrix(a * mat.data, mat.ind, mat.indptr)


class ShapedCSRMatrix(core.AbstractValue):
  def __init__(self, shape, dtype, nnz):
    self.shape = shape
    self.dtype = dtype
    self.nnz = nnz

def shaped_csr_matrix(csr_matrix):
  return ShapedCSRMatrix(csr_matrix.shape, csr_matrix.dtype,
                         csr_matrix.nnz)

core.pytype_aval_mappings[CSRMatrix] = shaped_csr_matrix
xla.pytype_aval_mappings[CSRMatrix] = shaped_csr_matrix
xla.canonicalize_dtype_handlers[CSRMatrix] = lambda x: x

# TODO without tuples, may need to turn a single jax value into multiple XLA
# values
xla.xla_shape_handlers[ShapedCSRMatrix] = \
    lambda mat: xc.Shape.tuple_shape(
        xc.Shape.array_shape(mat.data.dtype, mat.data.shape),
        xc.Shape.array_shape(mat.ind.dtype, mat.ind.shape),
        xc.Shape.array_shape(mat.indptr.dtype, mat.indptr.shape))

xla.xla_result_handlers[ShapedCSRMatrix] = \
  lambda device, aval: \
    partial(CSRMatrix, xla.raise_to_shaped(aval), device, xla.lazy.array(aval.shape))

xla.device_put_handlers[CSRMatrix] = \
   lambda x, device: \
      xla.xb.get_device_backend(device).buffer_from_pyval(x, device)

# TODO device_put handlers
# TODO result handlers (given output triple of DeviceBuffers, produce a
# CSRMatrix, by constructing three DeviceArrays to wrap the buffers and then
# putting them together in a CSRMatrix)