# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import absltest, parameterized

from jax import test_util as jtu
import jax.numpy as jnp
from jax import core, jit, lax, lazy
from jax.interpreters import xla
from jax.lib import xla_client
xops = xla_client.ops

from jax.config import config
config.parse_flags_with_absl()

# Define a sparse array data structure. The important feature here is that
# it is a jaxpr object that is backed by two device buffers.
class SparseArray:
  """Simple sparse COO array data structure."""
  def __init__(self, aval, shape, data, indices):
    self.aval = aval
    self.shape = shape
    self._data = data
    self._indices = indices

  @property
  def index_dtype(self):
    return self.indices.dtype

  @property
  def dtype(self):
    return self.data.dtype

  @property
  def nnz(self):
    return self.data.shape[0]

  @property
  def data(self):
    return self._data

  @property
  def indices(self):
    return self._indices

  def __repr__(self):
    return repr(list((tuple(ind), d) for ind, d in zip(self.indices, self.data)))


class AbstractSparseArray(core.ShapedArray):
  __slots__ = ['index_dtype', 'nnz', '_data', '_indices']
  _num_buffers = 2

  def __init__(self, shape, dtype, index_dtype, nnz):
    super(AbstractSparseArray, self).__init__(shape, dtype)
    self.index_dtype = index_dtype
    self.nnz = nnz
    self._data = core.ShapedArray((self.nnz,), self.dtype)
    self._indices = core.ShapedArray((self.nnz, len(self.shape)), self.index_dtype)

  @core.aval_property
  def data(self):
    return sp_data_p.bind(self)

  @core.aval_property
  def indices(self):
    return sp_indices_p.bind(self)

def abstract_sparse_array(arr):
  return AbstractSparseArray(arr.shape, arr.dtype, arr.index_dtype, arr.nnz)

def sparse_array_result_handler(device, aval):
  def build_sparse_array(data_buf, indices_buf):
    data = xla.DeviceArray(aval._data, device, lazy.array(aval._data.shape), data_buf)
    indices = xla.DeviceArray(aval._indices, device, lazy.array(aval._indices.shape), indices_buf)
    return SparseArray(aval, aval.shape, data, indices)
  return build_sparse_array

def sparse_array_shape_handler(a):
  return (
    xla.xc.Shape.array_shape(a._data.dtype, a._data.shape),
    xla.xc.Shape.array_shape(a._indices.dtype, a._indices.shape),
  )

def sparse_array_device_put_handler(a, device):
  return (
    xla.xb.get_device_backend(device).buffer_from_pyval(a.data, device),
    xla.xb.get_device_backend(device).buffer_from_pyval(a.indices, device)
  )

core.pytype_aval_mappings[SparseArray] = abstract_sparse_array
core.raise_to_shaped_mappings[AbstractSparseArray] = lambda aval, _: aval
xla.pytype_aval_mappings[SparseArray] = abstract_sparse_array
xla.canonicalize_dtype_handlers[SparseArray] = lambda x: x
xla.device_put_handlers[SparseArray] = sparse_array_device_put_handler
xla.xla_result_handlers[AbstractSparseArray] = sparse_array_result_handler
xla.xla_shape_handlers[AbstractSparseArray] = sparse_array_shape_handler


sp_indices_p = core.Primitive('sp_indices')

@sp_indices_p.def_impl
def _sp_indices_impl(mat):
  return mat.indices

@sp_indices_p.def_abstract_eval
def _sp_indices_abstract_eval(mat):
  return core.ShapedArray((mat.nnz, len(mat.shape)), mat.index_dtype)

def _sp_indices_translation_rule(c, data, indices):
  return indices

xla.translations[sp_indices_p] = _sp_indices_translation_rule

sp_data_p = core.Primitive('sp_data')

@sp_data_p.def_impl
def _sp_data_impl(mat):
  return mat.data

@sp_data_p.def_abstract_eval
def _sp_data_abstract_eval(mat):
  return core.ShapedArray((mat.nnz,), mat.dtype)

def _sp_data_translation_rule(c, data, indices):
  return data

xla.translations[sp_data_p] = _sp_data_translation_rule

def identity(x):
  return identity_p.bind(x)

identity_p = core.Primitive('identity')

@identity_p.def_impl
def _identity_impl(mat):
  return SparseArray(mat.aval, mat.shape, mat.data, mat.indices)

@identity_p.def_abstract_eval
def _identity_abstract_eval(mat):
  return mat

def _identity_translation_rule(c, data, indices):
  return xops.Tuple(c, (data, indices))

xla.translations[identity_p] = _identity_translation_rule


def make_sparse_array():
    data = jnp.arange(5.0)
    indices = jnp.arange(5).reshape(5, 1)
    shape = (10,)
    aval = AbstractSparseArray(shape, data.dtype, indices.dtype, len(indices))
    return SparseArray(aval, shape, data, indices)


class CustomObjectTest(jtu.JaxTestCase):

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_compile={}_primitive={}".format(compile, primitive),
       "compile": compile, "primitive": primitive}
      for primitive in [True, False]
      for compile in [True, False]))
  def testIdentity(self, compile, primitive):
    f = identity if primitive else (lambda x: x)
    f = jit(f) if compile else f
    M = make_sparse_array()
    M2 = f(M)
    self.assertEqual(M.dtype, M2.dtype)
    self.assertEqual(M.index_dtype, M2.index_dtype)
    self.assertAllClose(M.data, M2.data)
    self.assertAllClose(M.indices, M2.indices)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_compile={}_primitive={}".format(compile, primitive),
       "compile": compile, "primitive": primitive}
      for primitive in [True, False]
      for compile in [True, False]))
  def testLaxLoop(self, compile, primitive):
    f = identity if primitive else (lambda x: x)
    f = jit(f) if compile else f
    body_fun = lambda _, A: f(A)
    lax.fori_loop(0, 10, body_fun, make_sparse_array())

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_attr={}".format(attr), "attr": attr}
      for attr in ["data", "indices"]))
  def testAttrAccess(self, attr):
    args_maker = lambda: [make_sparse_array()]
    f = lambda x: getattr(x, attr)
    self._CompileAndCheck(f, args_maker)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())

  # from jax import core
  # core.skip_checks = False
  # def f(x): return (x.indices, x.data)
  # M = make_sparse_array()
  # print(f(M))
  # print(jit(f)(M))