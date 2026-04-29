SCGO Documentation
===================

.. raw:: html

   <div style="text-align: center; margin: 30px 0 20px 0;">
       <img src="_static/scgo_logo.svg" alt="SCGO" style="width: 200px;">
   </div>

SCGO: Simple Cluster Global Optimization
=========================================

A modern Python package for global optimization of atomic clusters using ASE with Basin Hopping and Genetic Algorithms.
Designed for researchers in computational chemistry and materials science.

Key Features
------------

- **Multiple Optimization Algorithms**: Basin Hopping (BH) and Genetic Algorithm (GA)
- **MLIP Support**: MACE and UMA (fairchem) for GPU-accelerated optimization
- **Surface Workflows**: Support for slab-supported clusters with adsorbates
- **Transition State Search**: NEB-based TS search with automated pair selection
- **Flexible API**: High-level runners and low-level control for custom workflows

Quick Start
-----------

.. code-block:: python

   from scgo import run_go
   from scgo.param_presets import get_testing_params
   
   results = run_go(
       ["Pt"] * 4,
       params=get_testing_params(),
       seed=42,
       system_type="gas_cluster",
   )

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/runner_api
   api/param_presets
   api/system_types

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`