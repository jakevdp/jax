(core-concepts)=
# Core concepts

This section of the tutorial gives a quick overview of the core concepts in JAX,
to give the context for the deeper dives in the following sections.

## JAX Arrays

The main array object type in JAX is the {class}`jax.Array`.
Typically arrays are created in JAX using array creation routines available in {mod}`jax.numpy`; for example:
```python
>>> import jax.numpy as jnp

>>> x = jnp.arange(10)  # Similar to numpy.arange

>>> print(x)
[0 1 2 3 4 5 6 7 8 9]
```
Other array creation routines include:

- {func}`jax.numpy.array`: create a JAX array from a NumPy array, Python list, or other data source.
- {func}`jax.numpy.zeros`: create a JAX array full of zeros.
- {func}`jax.numpy.from_dlpack`: create a JAX array from a dlpack buffer.
- {func}`jax.random.uniform`: create a JAX array containing uniformly-distributed pseudorandom numbers.
- and many more (See {mod}`jax.numpy` and {mod}`jax.random` for descriptions of others)

JAX arrays have an interface inspired by NumPy's {class}`numpy.ndarray`, including familiar array attributes related to the shape, size, and data type:
```
>>> x.shape
(10,)

>>> x.size
10

>>> x.ndim
1

>>> x.dtype
dtype('int32')
```
Unlike NumPy, JAX arrays in general may be backed by multiple buffers on one or more devices,
revealed via the {meth}`~jax.Array.devices` method and {attr}`~jax.Array.sharding` attribute.
For `x` created above, it is backed by a single buffer on a single CPU:
```
>>> x.devices()
{CpuDevice(id=0)}

>>> x.sharding
SingleDeviceSharding(device=CpuDevice(id=0))
```
In general, though, arrays may be stored on one or more CPUs, GPUs, or TPUs.
We will explore more sophisticated shardings and device layouts in {ref}`parallelism`.

## Transformations
JAX goes beyond just offering tools for operating on arrays; it also includes higher-order APIs for transforming functions which operate on arrays.
Examples of these transformations are

- {func}`jax.jit`: just-in-time compilation of a JAX function via the [XLA compiler](https://github.com/openxla/xla) (see {ref}`jit-compilation`)
- {func}`jax.grad`: automatic differentiation of JAX functions (see {ref}`automatic-differentiation`)
- {func}`jax.vmap`: automatic vectorization of JAX functions (see {ref}`automatic-vectorization`)
- {func}`jax.shard_map`: parallelization of JAX functions (see {ref}`parallelism`)

You should think of a transformation as a function that accepts a function as input, and returns a transformed function as output.
For example, here is a `grad` transformation applied to a simple trigonometric function:
```
>>> import jax

>>> def f(x):
...   return jnp.sin(x) * jnp.cos(x)
...

>>> f_grad = jax.grad(f)

>>> print(f_grad(1.0))
-0.4161468
```
In general, JAX transformations can be composed; for example, here is a vectorized and JIT-compiled version of the gradient of this function applied elementwise to several values:
```
>>> jax.jit(jax.vmap(f_grad))(jnp.arange(5.0))
Array([ 1.        , -0.4161468 , -0.65364355,  0.96017027, -0.14550006],      dtype=float32)
```

## Tracing



## PyTrees