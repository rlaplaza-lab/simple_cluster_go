Quick Start
===========

SCGO provides several example scripts in the `examples/` directory that demonstrate different use cases. This guide explains how these examples work and how to adapt them for your own research.

Example Structure
-----------------

All examples follow a similar pattern:

1. Import necessary modules
2. Define system parameters (composition, seed, system type, etc.)
3. Build parameter dictionaries using presets
4. Customize parameters as needed
5. Call the appropriate `run_*` function

Basic Gas-Phase Cluster (example_pt5_gas.py)
---------------------------------------------

This example demonstrates global optimization and transition state search for a simple gas-phase Pt5 cluster.

**Key Features:**
- `system_type="gas_cluster"` - no slab, no adsorbates
- Uses `run_go_ts` for combined GO + TS pipeline
- Customizes GA parameters (niter, population_size)
- Limits TS search with max_pairs

**Code:**

.. literalinclude:: ../../examples/example_pt5_gas.py
   :language: python
   :linenos:

**How to Run:**

.. code-block:: bash

   cd examples
   python example_pt5_gas.py

**Output:**
- Results saved in `examples/results/pt5_gas_*/`
- Contains GO minima and TS search results

Surface-Supported Cluster (example_pt5_graphite.py)
---------------------------------------------------

This example shows how to optimize a Pt5 cluster supported on a graphite surface.

**Key Features:**
- `system_type="surface_cluster"` - supported cluster with slab
- Uses `make_graphite_surface_config()` for surface setup
- Includes surface-specific GA parameters (batch_size)
- Demonstrates surface-adsorbate interaction

**Code:**

.. literalinclude:: ../../examples/example_pt5_graphite.py
   :language: python
   :linenos:

**Key Differences from Gas-Phase:**
- Requires `surface_config` parameter
- GA uses `batch_size` for surface calculations
- Different output structure due to slab presence

Gas-Phase Cluster with Adsorbate (example_pt5_oh_gas.py)
---------------------------------------------------------

This example demonstrates a gas-phase Pt5 cluster with OH adsorbate.

**Key Features:**
- `system_type="gas_cluster_adsorbate"` - gas-phase with adsorbates
- Uses ASE Atoms object to define adsorbate fragment
- Shows hierarchical adsorbate placement
- Demonstrates adsorbate-cluster interaction

**Code:**

.. literalinclude:: ../../examples/example_pt5_oh_gas.py
   :language: python
   :linenos:

**Adsorbate Setup:**

.. code-block:: python

   ADSORBATES = [Atoms(symbols=["O", "H"], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]])]

Surface-Supported Cluster with Multiple Adsorbates (example_pt5_2oh_graphite.py)
----------------------------------------------------------------------------------

This advanced example shows a surface-supported Pt5 cluster with two OH adsorbates.

**Key Features:**
- `system_type="surface_cluster_adsorbate"` - most complex system type
- Combines surface support with multiple adsorbates
- Demonstrates advanced TS search parameter customization
- Shows how to handle complex chemical environments

**Code:**

.. literalinclude:: ../../examples/example_pt5_2oh_graphite.py
   :language: python
   :linenos:

**Advanced TS Parameters:**

.. code-block:: python

   ts_params["energy_gap_threshold"] = 1.0
   ts_params["neb_n_images"] = 7
   ts_params["neb_steps"] = 800

Creating Your Own Examples
---------------------------

To create your own SCGO workflow:

1. **Choose the right system type:**
   - `gas_cluster` - simple gas-phase clusters
   - `surface_cluster` - surface-supported clusters
   - `gas_cluster_adsorbate` - gas-phase with adsorbates
   - `surface_cluster_adsorbate` - surface-supported with adsorbates

2. **Start with presets:**

   .. code-block:: python

      from scgo.param_presets import get_torchsim_ga_params, get_ts_search_params
      
      go_params = get_torchsim_ga_params(seed=42)
      ts_params = get_ts_search_params(system_type="gas_cluster", seed=42)

3. **Customize parameters:**

   .. code-block:: python

      go_params["optimizer_params"]["ga"].update(niter=10, population_size=50)
      ts_params["max_pairs"] = 15

4. **Run the workflow:**

   .. code-block:: python

      from scgo.runner_api import run_go_ts
      
      results = run_go_ts(
          ["Pt"] * 5,
          go_params=go_params,
          ts_params=ts_params,
          seed=42,
          system_type="gas_cluster",
      )

Parameter Customization Guide
------------------------------

**Global Optimization Parameters:**
- `niter` - Number of GA iterations
- `population_size` - GA population size
- `batch_size` - Batch size for surface calculations (surface systems only)
- `calculator` - "MACE" or other supported calculators

**Transition State Search Parameters:**
- `max_pairs` - Maximum number of minima pairs to search
- `energy_gap_threshold` - Energy threshold for pair selection
- `neb_n_images` - Number of NEB images
- `neb_steps` - Maximum NEB optimization steps

**Output Control:**
- `output_root` - Base output directory
- `output_stem` - Output directory name stem
- `seed` - Random seed for reproducibility

Best Practices
--------------

1. **Start small:** Begin with small test systems (e.g., Pt4) before scaling up
2. **Use presets:** Always start with parameter presets, then customize
3. **Set seeds:** Use fixed seeds (`seed=42`) for reproducible results
4. **Monitor resources:** Surface systems and TS searches can be resource-intensive
5. **Check outputs:** Examine the output directories to understand results structure

Understanding Output Structure
-------------------------------

SCGO creates a structured output directory:

.. code-block:: text

   {output_stem}_*/
   ├── go_results/
   │   ├── trial_*/
   │   │   ├── minima.db
   │   │   └── params.json
   │   └── results_summary.json
   └── ts_results/
       ├── ts_*/
       │   ├── neb_metadata.json
       │   └── trajectory_files/
       └── ts_search_summary.json

- `minima.db` - SQLite database of optimized structures
- `params.json` - Run parameters and metadata
- `results_summary.json` - Summary of GO results
- `ts_search_summary.json` - Summary of TS search results

Next Steps
----------

- Explore the :doc:`API Reference </api/runner_api>` for detailed function documentation
- Learn about :doc:`parameter presets </api/param_presets>` for different use cases
- Check the :doc:`system types </api/system_types>` documentation for system configuration
- Run the examples and examine their output to understand SCGO's workflow
