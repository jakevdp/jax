---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# How to think in JAX

JAX provides a simple and powerful API for writing accelerated numerical code,
but working effectively in JAX sometimes requires extra consideration.
This document is meant to help build a ground-up understanding of how JAX operates,
so that you can use it more effectively.

## JAX vs. NumPy

**Key Concepts:**

- JAX provides a NumPy-inspired interface for convenience.
- Through duck-typing, JAX arrays can often be used as drop-in replacements of NumPy arrays.
- Unlike NumPy arrays, JAX arrays are always immutable.

NumPy provides a well-known, powerful API for working with numerical data.
For convenience, JAX includes the {mod}`jax.numpy` module, which closely mirrors the
NumPy API and provides easy entry into JAX. Here are some operations you might do in ``numpy``:

```{code-cell}
import matplotlib.pyplot as plt

import numpy as np
x = np.linspace(0, 10, 1000)
y = 2 * np.sin(x) * np.cos(x)
plt.plot(x, y);
```

And here is the equivalent with {mod}`jax.numpy`:

```{code-cell}
import jax.numpy as jnp
x = jnp.linspace(0, 10, 1000)
y = 2 * jnp.sin(x) * jnp.cos(x)
plt.plot(x, y);
```

The code blocks are identical aside from replacing ``np`` with ``jnp``, and the results are the same.
As we can see, JAX arrays can often be used directly in place of NumPy arrays for things like plotting.

The arrays themselves are implemented as different Python types:

```{code-cell}
type(np.arange(10))
```

```{code-cell}
type(jnp.arange(10))
```

Python's [duck-typing](https://en.wikipedia.org/wiki/Duck_typing) allows JAX arrays and NumPy arrays
to be used interchangeably in many places.

However, there is one important difference between JAX and NumPy arrays: JAX arrays are immutable,
meaning that once created their contents cannot be changed.

Here is an example of mutating an array in Numpy:

```{code-cell}
x = np.arange(10)  # Numpy: mutable arrays
x[0] = 10
x
```

The equivalent in JAX results in an error, as JAX arrays are immutable:

```{code-cell}
:tags: [raises-exception]
x = jnp.arange(10)  # JAX: immutable arrays
x[0] = 10
```

For updating individual elements, JAX provides an
[indexed update syntax](https://jax.readthedocs.io/en/latest/jax.ops.html#syntactic-sugar-for-indexed-update-operators)
that returns an updated copy:

```{code-cell}
x.at[0].set(10)
```

## NumPy, lax & XLA: JAX's layers of API

**Key Concepts:**

- {mod}`jax.numpy` is a high-level wrapper that provides a familiar interface.
- {mod}`jax.lax` is a lower-level API that is stricter and often more powerful.
- All JAX operations are implemented in terms of operations in
  [XLA](https://www.tensorflow.org/xla/)  – the Accelerated Linear Algebra compiler.

If you look at the source of {mod}`jax.numpy`, you'll see that all the operations are eventually
expressed in terms of functions defined in {mod}`jax.lax`.
You can think of {mod}`jax.lax` as a stricter, but often more powerful, API for working with
multi-dimensional arrays.

For example, {mod}`jax.numpy` will implicitly promote arguments to allow operations
between mixed data types:

```{code-cell}
import jax.numpy as jnp
jnp.add(1, 1.0)  # jax.numpy API implicitly promotes mixed types.
```

... and {mod}`jax.lax` will not:

```{code-cell}
:tags: [raises-exception]
from jax import lax
lax.add(1, 1.0)  # jax.lax API requires explicit type promotion.
```

If using {mod}`jax.lax` directly, you'll have to do type promotion explicitly in such cases:

```{code-cell}
from jax import lax
lax.add(jnp.float32(1), 1.0)
```

Along with this strictness, {mod}`jax.lax` also provides efficient APIs for some more
general operations than are supported by NumPy.

For example, consider a 1D convolution, which can be expressed in NumPy this way:

```{code-cell}
x = jnp.array([1, 2, 1])
y = jnp.ones(10)
jnp.convolve(x, y)
```

Under the hood, this NumPy operation is translated to a much more general convolution implemented by
{func}`jax.lax.conv_general_dilated`:

```{code-cell}

result = lax.conv_general_dilated(
    x.reshape(1, 1, 3).astype(float),  # note: explicit promotion
    y.reshape(1, 1, 10),
    window_strides=(1,),
    padding=[(len(y) - 1, len(y) - 1)])  # equivalent of padding='full' in numpy
result[0, 0]
```

This is a batched convolution operation designed to be efficient for the types of convolutions often
used in deep neural nets. It requires much more boilerplate, but is far more flexible and scalable
than the convolution provided by NumPy (See
[JAX Sharp Bits: Convolutions](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#%F0%9F%94%AA-Convolutions)
for more detail on JAX convolutions).

## To JIT or not to JIT

**Key Concepts:**

- By default JAX executes operations one at a time, in sequence.
- Using a just-in-time (JIT) compilation decorator, sequences of operations can be optimized
  together and run at once.
- Not all JAX code can be JIT compiled, as it requires array shapes to be static and known at
  compile time.

The fact that all JAX operations are expressed in terms of XLA allows JAX to use the XLA
compiler to execute blocks of code very efficiently.

For example, consider this function that normalizes the rows of a 2D matrix,
expressed in terms of {mod}`jax.numpy` operations:

```{code-cell}
def norm(X):
    X = X - X.mean(0)
    return X / X.std(0)
```

A just-in-time compiled version of the function can be created using the {func}`jax.jit` transform:

```{code-cell}
from jax import jit
norm_compiled = jit(norm)
```

This function returns the same results as the original, up to standard floating-point accuracy:

```{code-cell}
np.random.seed(1701)
X = jnp.array(np.random.rand(10000, 10))
np.allclose(norm(X), norm_compiled(X), atol=1E-6)
```

But due to the compilation (which includes fusing of operations, avoidance of allocating temporary
arrays, and a host of other tricks), execution times can be orders of magnitude faster in the 
JIT-compiled case (note the use of {meth}`~jax.xla.DeviceArray.block_until_ready`
to account for JAX's {ref}`async-dispatch`):

```{code-cell}
%timeit norm(X).block_until_ready()
```
```{code-cell}
%timeit norm_compiled(X).block_until_ready()
```

That said, {func}`jax.jit` does have limitations: in particular, it requires all arrays to have
static shapes. That means that some JAX operations are incompatible with JIT compilation.

For example, this operation can be executed in op-by-op mode:

```{code-cell}
def get_negatives(x):
    return x[x < 0]
x = jnp.array(np.random.randn(10))

get_negatives(x)
```

But it returns an error if you attempt to execute it in jit mode:

```{code-cell}
:tags: [raises-exception]
jit(get_negatives)(x)
```

This is because the function generates an array whose shape is not known at compile time:
the size of the output depends on the values of the input array, and so it is not compatible with JIT.

## JIT mechanics: tracing and static variables

**Key Concepts:**

- JIT and other JAX transforms work by *tracing* a function to determine its effect on inputs
  of a specific shape and type.

- Variables that you don't want to be traced can be marked as *static*

To use {func}`jax.jit` effectively, it is useful to understand how it works.
Let's put a few ``print()`` statements within a JIT-compiled function and see what we find:

```{code-cell}
@jit
def f(x, y):
  print("Running f():")
  print(f"  x = {x}")
  print(f"  y = {y}")
  result = jnp.dot(x + 1, y + 1)
  print(f"  result = {result}")
  return result
```

The first time this function is run, what is printed is not the data we passed to the function,
but rather *tracer* objects that stand-in for them:

```{code-cell}
x = np.random.randn(3, 4)
y = np.random.randn(4)
f(x, y)
```

These tracer objects are what {func}`jax.jit` uses to extract the sequence of operations specified
by the function. Basic tracers are stand-ins that encode the **shape** and **dtype** of the arrays,
but are agnostic to the values. This recorded sequence of computations can then be efficiently
applied within XLA to new inputs with the same shape and dtype, without having to re-execute the
Python code.

When we call the compiled fuction again on matching inputs, no re-compilation is required:

```{code-cell}
x2 = np.random.randn(3, 4)
y2 = np.random.randn(4)
f(x2, y2)
```

The extracted sequence of operations is encoded in a JAX expression, or *jaxpr* for short.
You can view the jaxpr using the {func}`jax.make_jaxpr` transformation:

```{code-cell}
from jax import make_jaxpr
def f(x, y):
    return jnp.dot(x + 1, y + 1)

make_jaxpr(f)(x, y)
```

Note one consequence of this: because JIT compilation is done *without* information on the content
of the array, control flow statements in the function cannot depend on traced values.
For example, this fails:

```{code-cell}
:tags: [raises-exception]
@jit
def f(x, neg):
  return -x if neg else x

f(1, True)
```

If there are variables that you would not like to be traced,
they can be marked as static for the purposes of JIT compilation:

```{code-cell}
from functools import partial
   
@partial(jit, static_argnums=(1,))
def f(x, neg):
  return -x if neg else x

f(1, True)
```

Note that calling a JIT-compiled function with a different static argument results in re-compilation,
so the function still works as expected:

```{code-cell}
f(1, False)
```

Understanding which values and operations will be static and which will be traced
is a key part of using {func}`jax.jit` effectively.

## Static vs Traced Operations

**Key Concepts:**

- Just as values can be either static or traced, operations can be static or traced.

- Static operations are evaluated at compile-time in Python; traced operations are compiled
  and evaluated at run-time in XLA.

- Use ``numpy`` for operations that you want to be static; use {mod}`jax.numpy` for operations
  that you want to be traced.

This distinction between static and traced values makes it important to think about how to keep
a static value static. Consider this function:

```{code-cell}
:tags: [raises-exception]
@jit
def f(x):
  return x.reshape(jnp.array(x.shape).prod())

x = jnp.ones((2, 3))
f(x)
```

This fails with an error specifying that a tracer was found in {func}`jax.numpy.reshape`.
Let's add some print statements to the function to understand why this is happening:

```{code-cell}
@jit
def f(x):
  print(f"x = {x}")
  print(f"x.shape = {x.shape}")
  print(f"jnp.array(x.shape).prod() = {jnp.array(x.shape).prod()}")
  # comment this out to avoid the error:
  # return x.reshape(jnp.array(x.shape).prot())

f(x)
```

Notice that although ``x`` is traced, ``x.shape`` is a static value. However, when we use
{func}`jnp.array` and {func}`jnp.prod` on this static value, it becomes a traced value, at
which point it cannot be used in a function like {func}`jax.numpy.reshape` that requires a
static input (recall: array shapes must be static).

A useful pattern is to use ``numpy`` for operations that should be static (i.e. done at compile-time),
and use {mod}`jax.numpy` for operations that should be traced
(i.e. compiled and executed at run-time). For this function, it might look like this:

```{code-cell}
import jax.numpy as jnp
import numpy as np

@jit
def f(x):
  return x.reshape((np.prod(x.shape),))

f(x)
```

For this reason, a standard convention in JAX programs is to ``import numpy as np`` and
``import jax.numpy as jnp`` so that both interfaces are available for finer control over
whether operations are performed in a static matter (with ``numpy``, once at compile-time) 
or a traced manner (with {mod}`jax.numpy`, optimized at run-time).
