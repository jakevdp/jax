# Copyright 2024 The JAX Authors.
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

_deprecations = {
    # Finalized 2025-03-31
    "ffi_call": (
        "jax.extend.ffi.ffi_call was removed in JAX v0.6.0, use jax.ffi.ffi_call instead.",
        None,
    ),
    "ffi_lowering": (
        "jax.extend.ffi.ffi_lowering was removed in JAX v0.6.0, use jax.ffi.ffi_lowering instead.",
        None,
    ),
    "include_dir": (
        "jax.extend.ffi.include_dir was removed in JAX v0.6.0, use jax.ffi.include_dir instead.",
        None,
    ),
    "pycapsule": (
        "jax.extend.ffi.pycapsule was removed in JAX v0.6.0, use jax.ffi.pycapsule instead.",
        None,
    ),
    "register_ffi_target": (
        "jax.extend.ffi.register_ffi_target was removed in JAX v0.6.0, use jax.ffi.register_ffi_target instead.",
        None,
    ),
}

import typing
if typing.TYPE_CHECKING:
  pass
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del typing
