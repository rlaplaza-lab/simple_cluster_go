SCGO Documentation
===================

.. raw:: html

   <div style="text-align: center; margin: 30px 0 20px 0;">
       <img src="_static/scgo_logo.svg" alt="SCGO" style="width: 200px;">
   </div>

**SCGO** finds low-energy atomic structures with global optimization. It uses
Basin Hopping, Genetic Algorithms, NEB transition-state search, and machine
learning potentials (MACE, UMA, UPET) through ASE and TorchSim.

What SCGO works with
--------------------

Three building blocks cover every workflow:

- **Cluster** (also called the **core** when molecules are present): the metal
  nanoparticle whose shape you are searching.
- **Adsorbate**: a small molecule or fragment (for example OH or CO) attached
  to the cluster or the surface.
- **Slab**: a periodic surface the cluster or adsorbate sits on. You can also
  search the top layers of the slab itself, with the bottom layers held still.

The six system types combine these pieces (see :doc:`/quickstart` for the
full table): ``gas_cluster``, ``surface_cluster``, ``gas_cluster_adsorbate``,
``surface_cluster_adsorbate``, bare-slab ``surface``, and ``surface_adsorbate``.

See :doc:`/quickstart` to get started.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   surface_slab_guide
   output_layout
   parameters
   uniqueness
   validation_and_constraints
   benchmarks

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/runner_api
   api/runners
   api/scgo
   api/calculators
   api/database
   api/metadata
   api/exceptions
   api/initialization
   api/surface
   api/cluster_adsorbate
   api/param_presets
   api/validation_and_constraints_api
   api/system_types_guide
   api/system_types
   api/utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
