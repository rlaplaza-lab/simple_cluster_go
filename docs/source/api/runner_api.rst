API Reference
=============

High-level functions for running global optimization and transition state searches.

Main Functions
--------------

**Single composition:**

.. list-table::
   :widths: 40 60

   * - ``run_go(...)`` → ``list[tuple[float, Atoms]]``
     - Optimize one composition, return list of (energy, Atoms) tuples
   * - ``run_ts_search(...)`` → ``list[dict[str, Any]]``
     - Find transition states for one composition
   * - ``run_go_ts(...)`` → ``dict[str, Any]``
     - Run GO then TS for one composition

**Multiple compositions (campaigns):**

.. list-table::
   :widths: 40 60

   * - ``run_go_campaign(...)`` → ``dict[str, list[tuple[float, Atoms]]]``
     - Optimize multiple compositions; keys are on-disk ``path_key`` values
       (formula for bare gas clusters). Failed compositions are logged and
       skipped (empty list for that key). See :doc:`/api/initialization`.
   * - ``run_ts_campaign(...)`` → ``dict[str, list[dict[str, Any]]]``
     - Find TS for multiple compositions
   * - ``run_go_ts_campaign(...)`` → ``dict[str, dict[str, Any]]``
     - Run GO+TS for multiple compositions

All functions accept:

- ``composition``: formula string (``"Pt5"``), symbol list (``["Pt"]*5``), or ASE Atoms
- ``params`` / ``go_params``: GO parameter dictionary (``None`` or partial dict; merged with :func:`~scgo.param_presets.get_default_params` at run time)
- ``ts_params``: TS parameter dictionary (``None`` or partial dict; merged with :func:`~scgo.param_presets.get_ts_search_params` at run time)
- ``seed``: random seed for reproducibility (must agree across ``seed=``, ``go_params['seed']``, and ``ts_params['seed']`` when more than one is set)
- ``system_type``: ``"gas_cluster"``, ``"surface_cluster"``, ``"gas_cluster_adsorbate"``, ``"surface_cluster_adsorbate"``, ``"surface"``, or ``"surface_adsorbate"`` (run argument only; not inside preset dicts or optimizer slots)
- ``surface_config``: required for surface system types (run argument preferred;
  a top-level key in ``go_params`` / ``ts_params`` is enough when the run
  argument is omitted, and must agree when both are set)
- ``adsorbates``: ASE Atoms or list of Atoms, required for adsorbate system types
- ``verbosity``: 0 quiet … 3 trace (progress bars when ``verbosity >= 1``)
- ``calculator_for_global_optimization``: optional pre-warmed ASE/MLIP calculator
  for ``run_go`` / ``run_go_campaign``. When supplied, the campaign reuses it
  across every composition instead of reloading the model per run. The keyword
  sits immediately after ``output_dir``, so purely positional calls that pass
  arguments past ``output_dir`` shift by one position.

``scgo.runner_api`` is the public facade (implementation split across
``runner_composition``, ``runner_params``, ``runner_go``, ``runner_ts``).

See :doc:`/parameters` for merge rules.

Output directories
------------------

``output_dir`` semantics differ by runner. See :doc:`/output_layout` for the
full table and directory-tree examples.

Complete Examples
-----------------

.. code-block:: python

   from scgo import run_go
   from scgo.param_presets import get_testing_params

   results = run_go(
       "Pt5",
       params=get_testing_params(),
       seed=42,
       system_type="gas_cluster",
   )

See :doc:`/quickstart` for surface, adsorbate, TS, campaign, and output-layout examples.

Utility Functions
-----------------

.. list-table::
   :widths: 40 60

   * - ``build_one_element_compositions(element, min_atoms, max_atoms)``
     - Symbol lists for mono-element size scans (e.g. ``[["Pt", "Pt"], ["Pt", "Pt", "Pt"]]``)
   * - ``build_two_element_compositions(el1, el2, min_atoms, max_atoms)``
     - Symbol lists for bimetallic size scans (all ``el1``/``el2`` splits per atom count)
   * - ``parse_composition_arg(comp_str)``
     - Parse a compact formula with chemical capitalization (``"Pt5"``, ``"HO2Ru9W2"``),
       or comma-separated symbols (``"Pt,Pt,Au"``; unambiguous, case-insensitive)
   * - ``resolve_workflow_seed(*, seed_kw, go_params, ts_params)``
     - Ensure seed consistency across params
   * - ``log_go_ts_summary(logger, summary, *, wall_time_s=None)``
     - Print summary of a GO+TS run

Timing and Profiling
--------------------

**Per-run timing**: set in ``params`` / ``go_params`` under
``optimizer_params['ga']`` or ``bh``:

- ``write_timing_json=True``: write ``{run_dir}/timing.json`` (alongside ``metadata.json``)
- ``detailed_timing=True``: add ``per_generation`` rows (requires ``write_timing_json=True``)

**TS timing**: set ``write_timing_json`` in ``ts_params`` for ``{ts_run_dir}/timing.json``.

**GO+TS pipeline rollup**: when timing JSON is enabled in ``go_params`` and/or
``ts_params``, ``run_go_ts`` also writes ``go_ts_timing.json`` at the campaign root.

See :doc:`/api/utils` for timing JSON layout and output-path helpers. On-disk layout and
provenance: :doc:`/output_layout`.

Module Reference
----------------

.. automodule:: scgo.runner_api
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: _*
