# Copyright 2023 The JAX Authors.
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
"""More general Batched CSR matrix."""

from functools import partial
from typing import NamedTuple, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import lax
# from jax import tree_util
from jax._src.typing import Array
from jax._src.util import safe_zip
from jax.experimental.sparse import bcoo
# from jax.experimental.sparse._base import JAXSparse
from jax.experimental.sparse.util import SparseInfo, Shape, _count_stored_elements_per_batch, nfold_vmap


class BCSRProperties(NamedTuple):
  n_batch: int
  n_sparse: int
  n_dense: int
  nse: int


def _compatible(shape1: Sequence[int], shape2: Sequence[int]) -> bool:
  return all(s1 in (1, s2) for s1, s2 in safe_zip(shape1, shape2))


def _validate_bcsr_indices(indices: jnp.ndarray, indptr: jnp.ndarray,
                           shape: Sequence[int], compressed_dim: Optional[int]) -> BCSRProperties:
  assert jnp.issubdtype(indices.dtype, jnp.integer)
  assert jnp.issubdtype(indptr.dtype, jnp.integer)
  shape = tuple(shape)

  *batch_shape, nse, n_sparse = indices.shape
  n_sparse += (compressed_dim is not None)
  n_batch = len(batch_shape)
  assert n_batch >= 0

  nse = indices.shape[-2]
  n_sparse = indices.shape[-1] + (compressed_dim is not None)
  n_dense = len(shape) - n_batch - n_sparse
  assert n_dense >= 0

  if not _compatible(indices.shape[:n_batch], shape[:n_batch]):
    raise ValueError(f"indices batch dimensions not compatible for {indices.shape=}, {shape=}")
  if not _compatible(indptr.shape[:n_batch], shape[:n_batch]):
    raise ValueError(f"indptr batch dimensions not compatible for {indptr.shape=}, {shape=}")
  if compressed_dim is None:
    if indptr.shape[n_batch:] != (2,):
      raise ValueError("with no compressed dimension, indptr should be have a trailing dimensions of 2")
  else:
    if not 0 <= compressed_dim <= n_sparse:
      raise ValueError("compressed_dim must be None or between 0 and n_sparse.")
    if indptr.shape[n_batch:] != (shape[n_batch + compressed_dim] + 1,):
      raise ValueError("indptr shape must match the compressed shape plus 1.")

  return BCSRProperties(n_batch=n_batch, n_sparse=n_sparse, n_dense=n_dense, nse=nse)


def _validate_bcsr(data: jnp.ndarray, indices: jnp.ndarray,
                   indptr: jnp.ndarray, shape: Sequence[int],
                   compressed_dim: Optional[int]) -> BCSRProperties:
  props = _validate_bcsr_indices(indices, indptr, shape, compressed_dim)
  shape = tuple(shape)
  n_batch, n_sparse, n_dense, nse = props.n_batch, props.n_sparse, props.n_dense, props.nse
  if not _compatible(data.shape[:n_batch], shape[:n_batch]):
    raise ValueError(f"data batch dimensions not compatible for {data.shape=}, {shape=}")
  if data.shape[n_batch:] != (nse,) + shape[n_batch + n_sparse:]:
    raise ValueError(f"Invalid {data.shape=} for {nse=}, {n_batch=}, {n_dense=}")
  return props


def bcsr_fromdense(mat, *, n_batch=0, n_dense=0, compressed_dim=0, nse=None, index_dtype='int32'):
  shape = mat.shape
  nse_per_batch = _count_stored_elements_per_batch(mat, n_batch=n_batch, n_dense=n_dense)
  if nse is None:
    nse = int(nse_per_batch.max())
  if compressed_dim is None:
    mat = lax.expand_dims(mat, [n_batch])
  else:
    mat = jnp.rollaxis(mat, n_batch + compressed_dim, n_batch)
  data, indices = bcoo._bcoo_fromdense(mat, n_batch=n_batch, n_dense=n_dense, nse=nse)
  m = mat.shape[n_batch]
  @partial(nfold_vmap, N=n_batch, broadcasted=False)
  def _make_indptr(indices):
    return jnp.zeros(mat.shape[n_batch] + 1, dtype=index_dtype).at[1:].set(
      jnp.cumsum(jnp.bincount(indices, length=m).astype(index_dtype)))
  indptr = _make_indptr(indices[..., 0])
  indices = indices[..., 1:]
  return BCSR((data, indices, indptr), shape=shape, compressed_dim=compressed_dim,
              indices_sorted=True, unique_indices=True)


def bcsr_todense(data, indices, indptr, *, shape, compressed_dim=0):
  props = _validate_bcsr(data, indices, indptr, shape, compressed_dim)
  if compressed_dim is None:
    # TODO: zero-out out-of-bound data
    # data = jnp.where(jnp.arange(props.nse) < indptr[..., 1:], data, 0)
    return bcoo._bcoo_todense(data, indices, spinfo=bcoo.SparseInfo(shape=shape))
  @partial(nfold_vmap, N=props.n_batch, broadcasted=False)
  def _make_indices(indptr, indices):
    new_indices = jnp.cumsum(jnp.zeros(indices.shape[0], indices.dtype).at[indptr].add(1)) - 1
    return jnp.column_stack([indices[:, :compressed_dim], new_indices, indices[:, compressed_dim:]])
  indices = _make_indices(indptr, indices)
  return bcoo._bcoo_todense(data, indices, spinfo=bcoo.SparseInfo(shape=shape))


# @tree_util.register_pytree_node_class
class BCSR: #(JAXSparse):

  data: jnp.ndarray
  indices: jnp.ndarray
  indptr: jnp.ndarray
  shape: Shape
  compressed_dim: Optional[int]

  nse = property(lambda self: self.indices.shape[-1])
  dtype = property(lambda self: self.data.dtype)
  n_batch = property(lambda self: self.indices.ndim - 1)
  n_sparse = property(lambda self: self.indices.shape[-1] - (self.compressed_dim is None))
  n_dense = property(lambda self: self.data.ndim - self.indices.ndim)
  indices_sorted: bool
  unique_indices: bool
  _bufs = property(lambda self: (self.data, self.indices, self.indptr))
  _info = property(lambda self: SparseInfo(self.shape, self.indices_sorted,
                                           self.unique_indices))

  def __init__(self, args: Tuple[Array, Array, Array], *, shape: Sequence[int],
               compressed_dim=0, indices_sorted: bool = False, unique_indices: bool = False):
    self.data, self.indices, self.indptr = map(jnp.asarray, args)
    self.compressed_dim = compressed_dim
    self.indices_sorted = indices_sorted
    self.unique_indices = unique_indices
    self.shape = tuple(shape)
    # super().__init__(args, shape=shape)
    _validate_bcsr(self.data, self.indices, self.indptr, self.shape, self.compressed_dim)

  @classmethod
  def fromdense(cls, mat, n_batch=0, n_dense=0, compressed_dim=0):
    return bcsr_fromdense(mat, n_batch=n_batch, n_dense=n_dense, compressed_dim=compressed_dim)

  def todense(self):
    return bcsr_todense(self.data, self.indices, self.indptr,
                        shape=self.shape, compressed_dim=self.compressed_dim)
