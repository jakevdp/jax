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

import functools
import itertools

import numpy as np

import jax
from jax.experimental.sparse import JAXSparse
from jax.experimental.sparse.util import Shape
from jax.experimental.sparse import bcoo
from jax._src.typing import Array

import jax.numpy as jnp
from jax import lax


def _validate_block_bcoo(data, indices, batch_nse, shape):
  *batch_shape, nse, n_sparse = indices.shape
  batch_shape = tuple(batch_shape)
  n_batch = len(batch_shape)

  block_shape = data.shape[n_batch + 1:]
  assert len(block_shape) == n_sparse
  assert len(shape) == n_batch + n_sparse

  # TODO(jakevdp): allow broadcasted batches?
  assert data.shape[:n_batch + 1] == indices.shape[:n_batch + 1]
  assert batch_nse.shape == batch_shape
  assert shape[:n_batch] == batch_shape

  if any(s % bs for s, bs in zip(shape[n_batch:], block_shape)):
    raise ValueError("block shape must evenly divide sparse shape.")

  return n_batch, n_sparse, nse


def block_bcoo_todense(mat):
  todense = functools.partial(_block_bcoo_todense_unbatched, shape=mat.shape[mat.n_batch:])
  for i in range(mat.n_batch):
    todense = jax.vmap(todense)
  return todense(mat.data, mat.indices, mat.batch_nse)


def _block_bcoo_todense_unbatched(data, indices, nse, shape):
  n_batch, n_sparse, nse = _validate_block_bcoo(data, indices, nse, shape=shape)
  assert n_batch == 0
  block_shape = data.shape[1:]

  # zero-out out-of-range data & indices
  mask = jnp.arange(data.shape[0]) < nse
  indices = jnp.where(lax.expand_dims(mask, [1]), indices, 0)
  data = jnp.where(lax.expand_dims(mask, range(1, 1 + n_sparse)), data, 0)

  num_blocks = [s // bs for s, bs in zip(shape, block_shape)]
  mat = jnp.zeros((*num_blocks, *block_shape)).at[tuple(indices.T)].add(data)
  return lax.reshape(mat,
    new_sizes=shape,
    dimensions=tuple(itertools.chain.from_iterable(
      (i, i + n_sparse) for i in range(n_sparse))))


def block_bcoo_fromdense(mat, *, block_shape, nse=None, index_dtype=np.int32):
  n_sparse = len(block_shape)
  n_batch = mat.ndim - n_sparse
  shape = mat.shape
  assert not any(s % bs for s, bs in zip(mat.shape[n_batch:], block_shape))
  block_size = tuple(s // bs for s, bs in zip(mat.shape[n_batch:], block_shape))
  mat = mat.reshape(*mat.shape[:n_batch],
                    *itertools.chain.from_iterable(zip(block_size, block_shape)))
  mat = mat.transpose(*range(n_batch),
                      *range(n_batch, n_batch + 2 * n_sparse, 2),
                      *range(n_batch + 1, n_batch + 2 * n_sparse, 2))
  mat_bcoo = bcoo.BCOO.fromdense(mat, n_batch=n_batch, n_dense=len(block_shape),
                                 nse=nse, index_dtype=index_dtype)
  nse = jnp.full(mat.shape[:n_batch], mat_bcoo.nse, dtype=int)
  return BlockBCOO((mat_bcoo.data, mat_bcoo.indices, nse), shape=shape, indices_sorted=True, unique_indices=True)


class BlockBCOO: #(JAXSparse):
  data: Array
  indices: Array
  batch_nse: Array
  shape: Shape

  n_batch = property(lambda self: self.indices.ndim - 2)
  n_sparse = property(lambda self: self.indices.shape[-1])

  def __init__(self, args, *, shape, indices_sorted=False, unique_indices=False):
    self.data, self.indices, self.batch_nse = map(jnp.asarray, args)
    self.indices_sorted = indices_sorted
    self.unique_indices = unique_indices
    self.shape = tuple(shape)
    # super().__init__(args, shape=tuple(shape))
    _validate_block_bcoo(self.data, self.indices, self.batch_nse, self.shape)

  def todense(self):
    return block_bcoo_todense(self)

  @classmethod
  def fromdense(self, a, *, block_shape, nse=None, index_dtype=np.int32):
    return block_bcoo_fromdense(a, block_shape=block_shape, nse=nse, index_dtype=index_dtype)
