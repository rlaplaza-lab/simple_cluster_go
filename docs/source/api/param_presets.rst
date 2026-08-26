Parameter Presets
==================

GO (``params`` / ``go_params``) and TS (``ts_params``) dicts. Merge rules:
:doc:`/parameters`.

Preset Functions
----------------

**Global Optimization:**

.. list-table:: GO Presets
   :widths: 35 65

   * - ``get_testing_params()``
     - Fast EMT-based parameters for testing (``calculator="EMT"``, small ``niter`` / ``population_size`` everywhere)
   * - ``get_default_params()``
     - Canonical MACE production defaults (baseline for GO merge)
   * - ``get_minimal_ga_params(seed, model_name)``
     - Compact GA parameters that run sequentially (``n_jobs_* = 1``; easier to debug)
   * - ``get_torchsim_ga_params(*, system_type, surface_config, seed, model_name)``
     - MACE benchmark GA stack + TorchSim relaxer for GPU acceleration. Requires ``scgo[mace]``. For surface types stamps top-level ``surface_config`` only.
   * - ``get_low_effort_torchsim_ga_params(*, system_type, surface_config, seed, model_name)``
     - Reduced-budget (~25%) variant of ``get_torchsim_ga_params`` for demos and CI. Same calculator and relaxer; ~25% GA budget; sequential; no early stopping or timing JSON. Surface types clamp local relaxation up to 400 steps at run time.
   * - ``get_low_effort_upet_ga_params(*, system_type, surface_config, seed, model_name, version)``
     - Reduced-budget (~25%) UPET GO. TorchSim relaxer attached after ``model_name`` / ``version`` so the PES matches the ASE calculator; sequential; same surface clamp.
   * - ``get_low_effort_uma_ga_params(*, system_type, surface_config, seed, model_name, uma_task)``
     - Reduced-budget (~25%) UMA GO. FairChem TorchSim relaxer attached after ``model_name`` / ``uma_task``; sequential; same surface clamp. (UMA is omitted from the Kaggle GPU matrix.)
   * - ``get_default_uma_params()``
     - Default UMA (fairchem) parameters with auto local-step budget
   * - ``get_uma_ga_benchmark_params(seed, *, model_name, uma_task)``
     - UMA + autobatcher for benchmarking campaigns (``expected_max_atoms=600``)
   * - ``get_default_upet_params()``
     - Default UPET (metatomic) parameters. Requires ``scgo[upet]``.
   * - ``get_upet_ga_benchmark_params(seed, *, model_name)``
     - UPET + TorchSim benchmark GA parameters
   * - ``get_diversity_params(reference_db_glob, max_references, update_interval)``
     - Bias exploration toward diverse structures (``fitness_strategy="diversity"``)
   * - ``get_high_energy_params()``
     - Bias exploration toward high-energy structures (BH Metropolis scale raised)

**Transition State Search:**

.. list-table:: TS Presets
   :widths: 35 65

   * - ``get_ts_search_params(calculator, calculator_kwargs, *, system_type, surface_config, seed)``
     - Full flat TS dict for one ``system_type`` (NEB, calculator, pairing). Requires ``system_type``; surfaces also require ``surface_config``. Default calculator ``"MACE"``. Baseline for TS merge.
   * - ``get_low_effort_ts_search_params(calculator, calculator_kwargs, *, system_type, surface_config, seed)``
     - Reduced-budget (~25%, floored) variant for demos and CI. Every NEB physics knob is inherited unchanged; only ``neb_steps`` / ``torchsim_max_steps`` shrink (per-type floor 1000). Covers MACE, UMA, and UPET uniformly via its ``calculator`` arguments — there is no separate per-calculator TS wrapper. ``max_pairs`` is left uncapped for the caller.
   * - ``low_effort_neb_steps(system_type)``
     - The ``neb_steps`` budget used by :func:`~scgo.param_presets.get_low_effort_ts_search_params` for one system type.
   * - ``get_ts_defaults(system_type)``
     - NEB knob defaults for one system type (used internally by :func:`~scgo.param_presets.get_ts_search_params`; prefer ``get_ts_search_params`` in user code)

.. note::
   Canonical signatures are rendered by the ``automodule`` block below; the
   summary above is a convenience view.

Parameter reference
-------------------

See :doc:`/parameters` for the full GO, TS, surface, and adsorbate parameter tables.

Available Models
----------------

**MACE models:** ``"mace_matpes_0"``, ``"mace_mp_small"``, ``"mace_mpa_medium"``, ``"mace_off_small"``

**UMA models:** ``"uma-s-1p2"``, ``"uma-s-1p1"``, ``"uma-m-1p1"``

**UPET models:** ``"pet-mad-s"``, ``"pet-mad-xs"``, ``"pet-oam-xl"``, ``"pet-omat-s"``, ``"pet-spice-s"``

Usage Examples
--------------

**Start from a preset:**

.. code-block:: python

   from scgo.param_presets import get_default_params

   params = get_default_params()
   params["calculator_kwargs"]["model_name"] = "mace_mp_small"
   params["optimizer_params"]["ga"]["population_size"] = 100

**Build TS params:**

.. code-block:: python

   from scgo import make_graphite_surface_config
   from scgo.param_presets import get_ts_search_params

   surface_config = make_graphite_surface_config(slab_layers=3)

   ts_params = get_ts_search_params(
       system_type="surface_cluster",
       surface_config=surface_config,
       seed=42,
   )
   ts_params["max_pairs"] = 20
   ts_params["neb_n_images"] = 7

**Combined GO + TS:**

.. code-block:: python

   from scgo import make_graphite_surface_config
   from scgo.param_presets import get_torchsim_ga_params, get_ts_search_params

   surface_config = make_graphite_surface_config(slab_layers=3)

   go_params = get_torchsim_ga_params(
       system_type="surface_cluster",
       surface_config=surface_config,
       seed=42,
   )

   ts_params = get_ts_search_params(
       system_type="surface_cluster",
       surface_config=surface_config,
       seed=42,
   )

**Low-effort GO + TS (demos, examples, CI):**

Same physics, ~25% of the budget. This is what every script in ``examples/``
and the Kaggle GPU test matrix uses, so the two cannot drift apart.

.. code-block:: python

   from scgo import (
       get_low_effort_torchsim_ga_params,
       get_low_effort_ts_search_params,
       make_hopg_5x5_graphite_surface_config,
   )

   surface_config = make_hopg_5x5_graphite_surface_config()

   go_params = get_low_effort_torchsim_ga_params(
       system_type="surface_cluster",
       surface_config=surface_config,
       seed=42,
   )

   ts_params = get_low_effort_ts_search_params(
       system_type="surface_cluster",
       surface_config=surface_config,
       seed=42,
   )
   # max_pairs is the dominant TS cost lever and is left to the caller.
   ts_params["max_pairs"] = 6

``get_low_effort_ts_search_params`` already covers MACE, UMA, and UPET uniformly
via its ``calculator`` / ``calculator_kwargs`` arguments. There is no separate
per-calculator TS wrapper.

**Low-effort UPET GO:**

.. code-block:: python

   from scgo import (
       get_low_effort_upet_ga_params,
       get_low_effort_ts_search_params,
       make_hopg_5x5_graphite_surface_config,
   )

   surface_config = make_hopg_5x5_graphite_surface_config()

   go_params = get_low_effort_upet_ga_params(
       system_type="surface_cluster",
       surface_config=surface_config,
       seed=42,
       model_name="pet-mad-s",
       version="1.5.0",
   )

See :doc:`/quickstart` for complete workflow examples and :doc:`/parameters` for the full parameter list.

Module Reference
----------------

.. automodule:: scgo.param_presets
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: _* TS_DEFAULTS_BY_SYSTEM_TYPE
