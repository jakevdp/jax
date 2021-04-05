# Copyright 2021 Google LLC
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

import jax.numpy as jnp
from jax import tree_util

#--------------------------------------------------------------------------------
# Sparse Vector definition

@tree_util.register_pytree_node_class
class SparseVector:
  """Basic 1D Sparse Vector"""
  def __init__(self, data, indices=None, shape=None, dtype=None):
    if indices is None:
        assert data.ndim == 1
        indices = jnp.nonzero(data)
        data = data[indices]
    self.data = jnp.asarray(data, dtype)
    self.indices = jnp.asarray(indices)
    self.shape = tuple(shape) if shape is not None else (self.indices.max() + 1,)
    assert len(self.shape) == 1

  @property
  def dtype(self):
    return self.data.dtype

  @property
  def ndim(self):
      return len(self.shape)

  def to_array(self):
      return jnp.zeros(self.shape, self.dtype).at[self.indices].set(self.data)

  def tree_flatten(self):
    return (self.data, self.indices), (self.shape, self.dtype)

  @classmethod
  def tree_unflatten(cls, aux, data):
    shape, dtype = aux
    data, indices = data
    return cls(data, indices, shape=shape, dtype=dtype)

  def __repr__(self):
    return f"{self.__class__.__name__}({self.to_array()})"


# Some Primitives
from jax import core


def axpyi(a, x, y):
  return axpyi_p.bind(a, x, y)

def _axpyi_abstract_eval(*args):
  breakpoint()

def _axpyi_impl(a, x, y):
  return y.at[x.indices].add(a * x.data)


axpyi_p = core.Primitive('axpyi')
axpyi_p.def_abstract_eval(_axpyi_abstract_eval)
axpyi_p.def_impl(_axpyi_impl)


if __name__ == '__main__':
  from jax import jit

  vec = SparseVector(jnp.array([0, 1, 0, 2, 0, 0, 3], dtype=float))
  print(vec)
  # print(tree_util.tree_flatten(vec))

  y = jnp.ones(vec.shape)
  print(axpyi(3.0, vec, y))
  print(jit(axpyi)(3.0, vec, y))
