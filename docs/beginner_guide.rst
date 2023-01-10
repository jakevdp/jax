:orphan:

.. _beginner_guide:


Beginner Guide
==============

Welcome to the beginners guide for JAX. 
On this page we will introduce you to the key ideas of JAX,
show you how to get JAX running
and provide you some tutorials to get started.

If looking to jump straight in take a look at the jax quickstart.

.. toctree::
   :maxdepth: 1

   notebooks/quickstart

For most users starting out the key functionalities of JAX to become familiar with are

- :func:`jax.jit` 
- :func:`jax.grad` 
- :func:`jax.vmap` 

For introductions to specific topics take a look at the tutorials contained in Jax-101

.. toctree::
   :maxdepth: 2

   jax-101/index

If you prefer a video format here is an introduction from Jake Vanderplas .

.. raw:: html

	<iframe width="640" height="360" src="https://www.youtube.com/embed/WdTeDXsOSj4"
	 title="Intro to JAX: Accelerating Machine Learning research"
	frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
	allowfullscreen></iframe>

Installing JAX
==============
Installation instructions are available on the `Install Guide <https://github.com/google/jax#installation>`_
Alternatively Jax comes preinstalled on `Google Colab <https://colab.research.google.com>`_.