---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(core-concepts)=
# Core concepts

This section of the tutorial gives a quick overview of the core concepts in JAX,
to give the context for the deeper dives in the following sections.

## JAX Arrays

The main array object type in JAX is the {class}`jax.Array`.
Typically arrays are created in JAX using array creation routines available in {mod}`jax.numpy`; for example:
```{code-cell} ipython3
import jax.numpy as jnp
x = jnp.arange(10)  # Similar to numpy.arange
print(x)
```
Other array creation routines include:

- {func}`jax.numpy.array`: create a JAX array from a NumPy array, Python list, or other data source.
- {func}`jax.numpy.zeros`: create a JAX array full of zeros.
- {func}`jax.random.uniform`: create a JAX array containing uniformly-distributed pseudorandom numbers.
- etc.

Like NumPy arrays, 