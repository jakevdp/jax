JAX: High-Performance Array Computing
=====================================

JAX is Autograd_ and XLA_, brought together for high-performance numerical computing.

.. grid:: 3

   .. grid-item::

      .. card:: Familiar API
         :class-card: key-ideas
         :shadow: None

         JAX provides a familiar NumPy-style API for ease of adoption by researchers and engineers. 

   .. grid-item::

      .. card:: Transformations
         :class-card: key-ideas
         :shadow: None

         JAX includes composable function transformations for compilation, batching, automatic differentiation, and parallelization.

   .. grid-item::

      .. card:: Run Anywhere
         :class-card: key-ideas
         :shadow: None

         The same code executes on multiple backends, including CPU, GPU, & TPU

.. note::
   JAX 0.4.1 introduces new parallelism APIs, including breaking changes to :func:`jax.experimental.pjit` and a new unified ``jax.Array`` type.
   Please see `Distributed arrays and automatic parallelization <https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html>`_ tutorial and the :ref:`jax-array-migration`
   guide for more information.


.. grid:: 3

    .. grid-item::

      .. card:: :material-regular:`rocket_launch;2em` Getting Started
         :link: beginner_guide
         :link-type: ref
         :class-card: getting-started

    .. grid-item::

      .. card:: :material-regular:`library_books;2em` User Guides
         :link: advanced_guide
         :link-type: ref
         :class-card: user-guides

    .. grid-item::

      .. card:: :material-regular:`laptop_chromebook;2em` Developer Docs
         :link: contributor_guide
         :link-type: ref
         :class-card: developer-docs

Installation
------------

.. code-block:: bash

   pip install "jax[cpu]"

For installation on accelerators (GPU, TPU) and other installation options,
see the `Install Guide`_ in the project README.


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   installation
   notebooks/quickstart
   notebooks/thinking_in_jax
   notebooks/Common_Gotchas_in_JAX
   jax-101/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Reference Documentation

   faq
   async_dispatch
   aot
   jaxpr
   notebooks/convolutions
   pytrees
   jax_array_migration
   type_promotion
   errors
   transfer_guard
   glossary
   changelog

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Advanced JAX Tutorials

   notebooks/autodiff_cookbook
   multi_process
   notebooks/Distributed_arrays_and_automatic_parallelization
   notebooks/vmapped_log_probs
   notebooks/neural_network_with_tfds_data
   notebooks/Custom_derivative_rules_for_Python_code
   notebooks/How_JAX_primitives_work
   notebooks/Writing_custom_interpreters_in_Jax
   notebooks/Neural_Network_and_Data_Loading
   notebooks/xmap_tutorial
   notebooks/external_callbacks
   Custom_Operation_for_GPUs


.. toctree::
   :hidden:
   :maxdepth: 2

   debugging/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API documentation

   jax


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Developer documentation

   contributing
   developer
   jax_internal_api
   autodidax
   jep/index


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Notes

   api_compatibility
   deprecation
   concurrency
   gpu_memory_allocation
   profiling
   device_memory_profiling
   rank_promotion_warning


.. _Autograd: https://github.com/hips/autograd
.. _XLA: https://www.tensorflow.org/xla
.. _Install Guide: https://github.com/google/jax#installation