# Copyright 2018 Google LLC
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
from __future__ import annotations

from collections import namedtuple
import types
from typing import Any

from jax._src.config import config
from jax._src.errors import TracerIntegerConversionError, ConcretizationTypeError, TracerArrayConversionError
import jax._src.pretty_printer as pp

Trace = Any


class Tracer:
  __array_priority__ = 1000
  __slots__ = ['_trace', '__weakref__', '_line_info']

  def __array__(self, *args, **kw):
    raise TracerArrayConversionError(self)

  def __dlpack__(self, *args, **kw):
    raise ConcretizationTypeError(self,
      f"The __dlpack__() method was called on the JAX Tracer object {self}")

  def __index__(self):
    raise TracerIntegerConversionError(self)

  def __init__(self, trace: Trace):
    self._trace = trace

  def __iter__(self):
    return iter(self.aval._iter(self))

  def __len__(self):
    return self.aval._len(self)

  @property
  def aval(self):
    raise NotImplementedError("must override")

  def _assert_live(self) -> None:
    pass  # Override for liveness checking

  def get_referent(self) -> Any:
    return self  # Override for object equivalence checking

  # Python looks up special methods only on classes, not instances. This means
  # these methods needs to be defined explicitly rather than relying on
  # __getattr__.
  def __neg__(self): return self.aval._neg(self)
  def __pos__(self): return self.aval._pos(self)
  def __eq__(self, other): return self.aval._eq(self, other)
  def __ne__(self, other): return self.aval._ne(self, other)
  def __lt__(self, other): return self.aval._lt(self, other)
  def __le__(self, other): return self.aval._le(self, other)
  def __gt__(self, other): return self.aval._gt(self, other)
  def __ge__(self, other): return self.aval._ge(self, other)
  def __abs__(self): return self.aval._abs(self)
  def __add__(self, other): return self.aval._add(self, other)
  def __radd__(self, other): return self.aval._radd(self, other)
  def __sub__(self, other): return self.aval._sub(self, other)
  def __rsub__(self, other): return self.aval._rsub(self, other)
  def __mul__(self, other): return self.aval._mul(self, other)
  def __rmul__(self, other): return self.aval._rmul(self, other)
  def __div__(self, other): return self.aval._div(self, other)
  def __rdiv__(self, other): return self.aval._rdiv(self, other)
  def __truediv__(self, other): return self.aval._truediv(self, other)
  def __rtruediv__(self, other): return self.aval._rtruediv(self, other)
  def __floordiv__(self, other): return self.aval._floordiv(self, other)
  def __rfloordiv__(self, other): return self.aval._rfloordiv(self, other)
  def __divmod__(self, other): return self.aval._divmod(self, other)
  def __rdivmod__(self, other): return self.aval._rdivmod(self, other)
  def __mod__(self, other): return self.aval._mod(self, other)
  def __rmod__(self, other): return self.aval._rmod(self, other)
  def __pow__(self, other): return self.aval._pow(self, other)
  def __rpow__(self, other): return self.aval._rpow(self, other)
  def __matmul__(self, other): return self.aval._matmul(self, other)
  def __rmatmul__(self, other): return self.aval._rmatmul(self, other)
  def __and__(self, other): return self.aval._and(self, other)
  def __rand__(self, other): return self.aval._rand(self, other)
  def __or__(self, other): return self.aval._or(self, other)
  def __ror__(self, other): return self.aval._ror(self, other)
  def __xor__(self, other): return self.aval._xor(self, other)
  def __rxor__(self, other): return self.aval._rxor(self, other)
  def __invert__(self): return self.aval._invert(self)
  def __lshift__(self, other): return self.aval._lshift(self, other)
  def __rlshift__(self, other): return self.aval._rlshift(self, other)
  def __rshift__(self, other): return self.aval._rshift(self, other)
  def __rrshift__(self, other): return self.aval._rrshift(self, other)
  def __getitem__(self, idx): return self.aval._getitem(self, idx)
  def __nonzero__(self): return self.aval._nonzero(self)
  def __bool__(self): return self.aval._bool(self)
  def __int__(self): return self.aval._int(self)
  def __long__(self): return self.aval._long(self)
  def __hex__(self): return self.aval._hex(self)
  def __oct__(self): return self.aval._oct(self)
  def __float__(self): return self.aval._float(self)
  def __complex__(self): return self.aval._complex(self)
  def __copy__(self): return self.aval._copy(self)
  def __deepcopy__(self, memo): return self.aval._deepcopy(self, memo)

  # raises a useful error on attempts to pickle a Tracer.
  def __reduce__(self):
    raise ConcretizationTypeError(
      self, ("The error occurred in the __reduce__ method, which may "
             "indicate an attempt to serialize/pickle a traced value."))

  # raises the better error message from ShapedArray
  def __setitem__(self, idx, val): return self.aval._setitem(self, idx, val)

  # NumPy also only looks up special methods on classes.
  def __array_module__(self, types): return self.aval._array_module(self, types)

  def __getattr__(self, name):
    # if the aval property raises an AttributeError, gets caught here
    assert not config.jax_enable_checks or name != "aval"

    try:
      attr = getattr(self.aval, name)
    except KeyError as err:
      raise AttributeError(
          f"{self.__class__.__name__} has no attribute {name}"
      ) from err
    else:
      t = type(attr)
      if t is aval_property:
        return attr.fget(self)
      elif t is aval_method:
        return types.MethodType(attr.fun, self)
      else:
        return attr

  def _pretty_print(self):
    base = pp.text(f'Traced<{self.aval}>with<{self._trace}>')
    contents = [(name, attr._pretty_print() if isinstance(attr, Tracer)
                 else pp.text(repr(attr))) for name, attr in self._contents()]
    if contents:
      base = pp.group(pp.nest(2, pp.concat([
        base, pp.text(' with'), pp.brk(), pp.join(pp.brk(), [
          pp.text(f'{name} = ') + pp_payload
          for name, pp_payload in contents])
      ])))
    return base

  def __repr__(self):
    return self._pretty_print().format()

  def _contents(self):
    try:
      return [(name, getattr(self, name)) for name in self.__slots__]
    except AttributeError:
      return ()

  def _origin_msg(self) -> str:
    return ""


# these can be used to set up forwarding of properties and instance methods from
# Tracer instances to the underlying avals
aval_property = namedtuple("aval_property", ["fget"])
aval_method = namedtuple("aval_method", ["fun"])
