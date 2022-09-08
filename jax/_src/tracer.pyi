from typing import Any, NamedTuple
from jax._src.array import Array

# Let tracer inherit from Array, so that tracer instances will be compatible
# with Array annotations.
class Tracer(Array): ...

class aval_property(NamedTuple):
  fget: Any

class aval_method(NamedTuple):
  fun: Any
