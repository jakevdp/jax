# Copyright 2022 Google LLC
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

from typing import cast

from absl.testing import absltest

import jax
import jax.numpy as jnp
from jax._src.array import Array
from jax.core import Tracer
from jax._src import test_util as jtu


@jtu.with_config(jax_array=True)
class TypingTest(jtu.JaxTestCase):

  def testArrayInstanceIdentity(self):
    x = jnp.arange(4)
    self.assertIs(type(x), Array)

  def testArrayInstanceChecks(self):
    def f(x):
      return isinstance(x, Array)

    x = jnp.arange(3)
    assert f(x)
    assert jax.jit(f)(x)
    assert jax.vmap(f)(x).all()

  def testArrayAnnotations(self):
    def g(x: Array) -> Array:
      assert isinstance(x, Array)
      return x

    @jax.jit
    def g_jit(x: Array) -> Array:
      assert isinstance(x, Tracer)
      return cast(Tracer, x)  # Explicit cast because jit strips annotations.

    @jax.vmap
    def g_vmap(x: Array) -> Array:
      assert isinstance(x, Tracer)
      return cast(Tracer, x)  # Explicit cast because vmap strips annotations.

    x = jnp.arange(4)

    out: Array = g(x)
    out_jit: Array = g_jit(x)
    out_vmap: Array = g_vmap(x)

    self.assertArraysEqual(out, x)
    self.assertArraysEqual(out_jit, x)
    self.assertArraysEqual(out_vmap, x)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
