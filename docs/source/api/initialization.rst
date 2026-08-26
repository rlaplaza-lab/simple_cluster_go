Cluster initialization
========================

Gas-phase and deposition workflows build starting structures through
:mod:`scgo.initialization`. The GA population uses the same engine via
``ClusterStartGenerator`` (``smart`` mode by default).

Initialization modes
--------------------

Pass ``mode`` to :func:`~scgo.initialization.create_initial_cluster` or
``init_mode`` on :class:`~scgo.surface.config.SurfaceSystemConfig` for surface
deposition:

.. list-table::
   :widths: 22 78
   :header-rows: 1

   * - Mode
     - Behavior
   * - ``smart`` (default)
     - Metropolis allocation across templates, seed+growth, and random_spherical.
       Batch generation discovers strategies once, then assigns per-structure
       seeds for reproducible parallel runs.
   * - ``seed+growth``
     - Grow from low-energy candidates in prior ``*.db`` searches (Boltzmann
       sampling by composition counts). Chain: DB combine+grow → random-seed
       growth → ``random_spherical`` when earlier stages yield no suitable
       structure.
   * - ``random_spherical``
     - Iterative random placement with clash and connectivity checks; retries
       relax placement radii within user bounds.
   * - ``template``
     - Icosahedral / decahedral / octahedral templates when available for the
       target size. Exact magic-number shells are preferred; near-match
       adjustment from the nearest magic number is only attempted when the
       size is within ``MAGIC_NUMBER_TOLERANCE`` (default ±2). Falls back to
       ``random_spherical`` when no usable template is produced.

Atom ordering (multi-element GA)
--------------------------------

Genetic-algorithm cut-and-splice crossover requires parents to share identical
per-index atomic numbers (``a1.numbers == a2.numbers``), not merely the same
composition.

SCGO therefore:

- Keeps the **campaign composition list** as the canonical symbol order (e.g.
  ``["Ir", "O", "O", "O"]``, not alphabetical ``O``-first).
- Reorders validated structures with
  :func:`~scgo.initialization.geometry_helpers.reorder_cluster_to_composition`
  when ``validate_cluster(..., sort_atoms=True, composition=...)`` runs.
- Applies the same reordering when inserting gas-phase candidates into the GA
  database.

All structures in a batch for one composition therefore share the same
``.numbers`` vector, which avoids stoichiometry pairing errors in multi-element
runs.

Placement order and diversity
-----------------------------

For ``random_spherical`` and seed growth, atoms are added one at a time. The
order is sampled on each attempt (mass-biased by default, exploratory otherwise);
see :mod:`scgo.initialization.initialization_config`.

- **Mass-biased** (default ~65% of attempts): heavier element groups are placed
  first (ASE atomic masses); order within each element group is shuffled. This
  favours metal-first growth for oxides and bimetallics without fixing the same
  sequence for every structure.
- **Exploratory** (~35%): mixed growth-order strategies (random shuffle,
  size-based, composition-aware, and similar) preserve batch diversity.

The bias probability is ``MASS_FIRST_PLACEMENT_PROB`` in
:mod:`scgo.initialization.initialization_config` (not exposed in GO presets).

Reproducibility
---------------

All placement randomness flows through a single ``numpy.random.Generator``:

- **Single structure:** pass ``rng`` to :func:`~scgo.initialization.create_initial_cluster`
  or set ``seed`` on ``run_go`` / ``run_go_campaign`` (converted to a generator
  at the API boundary).
- **Batch / GA population:** ``create_initial_cluster_batch`` derives an
  independent, deterministic per-structure seed from the parent generator, so
  ``n_jobs=1`` and parallel workers produce identical populations for the same
  parent seed.
- **Campaigns:** ``run_go_campaign`` draws a reproducible per-composition seed
  from the campaign generator; failed compositions are logged and skipped (see
  below) without aborting the rest of the scan.

Use the same ``seed`` everywhere it appears (``seed=``, ``go_params['seed']``,
``ts_params['seed']``) when more than one is set.

Connectivity and steric checks
------------------------------

- ``connectivity_factor`` (default ``1.4`` in GO presets) is the shared
  connectivity spec for initialization, post-operator GA checks, per-minimum
  algorithm gates, the ``run_trials`` final structural gate, and TS. It accepts
  a global float or a dict of per-element and/or per-pair multipliers (see
  :doc:`/validation_and_constraints` and :doc:`/parameters`).
- Placement clash tables use ``BLMIN_RATIO_DEFAULT`` (``0.7``), aligned with GA
  ``blmin`` tables via :func:`~scgo.initialization.atomic_radii.build_blmin`.

Difficult stoichiometries (e.g. O-rich oxides) may fail initialization; filter
composition scans or relax ``connectivity_factor`` / placement parameters rather
than expecting every binary grid point to succeed.

Module reference
----------------

.. automodule:: scgo.initialization
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: _*, initialization_config

.. autoclass:: scgo.initialization.BatchInitPlan
   :members:
   :show-inheritance:
   :exclude-members: _*
   :no-index:

.. automodule:: scgo.initialization.initialization_config
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: _*

.. autofunction:: scgo.initialization.geometry_helpers.reorder_cluster_to_composition

.. autoclass:: scgo.initialization.geometry_helpers.ClusterExtent
   :members:
   :undoc-members:

