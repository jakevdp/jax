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

from typing import NamedTuple, Optional, Sequence, Iterator

from absl.testing import absltest
import numpy as np

from jax._src import test_util as jtu
from jax.experimental.sparse import new_bcsr


class SparseLayout(NamedTuple):
  n_batch: int
  n_dense: int
  n_sparse: int
  compressed_dim: Optional[int]


def iter_sparse_layouts(shape: Sequence[int], min_n_batch=0) -> Iterator[SparseLayout]:
  for n_batch in range(min_n_batch, len(shape) + 1):
    for n_dense in range(len(shape) + 1 - n_batch):
      n_sparse = len(shape) - n_batch - n_dense
      for compressed_dim in [None, *range(n_sparse)]:
        yield SparseLayout(n_batch=n_batch, n_sparse=n_sparse,
                           n_dense=n_dense, compressed_dim=compressed_dim)


def rand_sparse(rng, nse=0.5, post=lambda x: x, rand_method=jtu.rand_default):
  def _rand_sparse(shape, dtype, nse=nse):
    rand = rand_method(rng)
    size = np.prod(shape).astype(int)
    if 0 <= nse < 1:
      nse = nse * size
    nse = min(size, int(nse))
    M = rand(shape, dtype)
    indices = rng.choice(size, size - nse, replace=False)
    M.flat[indices] = 0
    return post(M)
  return _rand_sparse


class BCSRTest(jtu.JaxTestCase):
  @jtu.sample_product(
    [dict(shape=shape, layout=layout)
      for shape in [(5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for layout in iter_sparse_layouts(shape)
    ],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  def test_bcsr_dense_round_trip(self, shape, dtype, layout):
    rng = rand_sparse(self.rng())
    mat = rng(shape, dtype)
    mat_bcsr = new_bcsr.BCSR.fromdense(mat, n_batch=layout.n_batch, n_dense=layout.n_dense,
                                       compressed_dim=layout.compressed_dim)
    mat_out = mat_bcsr.todense()
    self.assertArraysEqual(mat, mat_out)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
