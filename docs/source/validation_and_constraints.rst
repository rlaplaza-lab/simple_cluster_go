Validation and Constraints
==========================

SCGO checks two related things: which atoms may move (constraints), and which
structures are legal (validation). Outcome toggles control how strict the
surface search is when a structure splits or detaches.

What constraints and validation do
----------------------------------

- **Constraints** define what atoms may move: slab fixation, core freezing, and
  adsorbate rigidity.
- **Validation** rejects structures that violate the allowed geometry: clashes,
  connectivity, penetration, and fragmentation rules.
- **Outcome toggles** decide how permissive the surface search can be when a
  structure is partially detached or fragmented.

A relaxed structure is only legal relative to the constraints that defined the
search. Read them together.

What counts as connected
------------------------

Gas systems (``gas_cluster`` and ``gas_cluster_adsorbate``) must form a single
connected component. A fragmented bare cluster or a gas adsorbate detached from
its core is rejected.

Surface systems (``surface_cluster`` and the ``*_adsorbate`` variants) require
connectivity and slab contact for each mobile subgroup, through
:func:`~scgo.system_types.validate_connectivity_policy`.

The bare ``surface`` type skips the supported-deposit connectivity gate. It
still checks the slab-prefix layout through
:func:`~scgo.surface.partition.validate_slab_search_config`.

All global optimizers use the same validation rule set when a ``system_type`` is
supplied. Placement uses looser steric floors than the final check, so the
search can explore broadly while illegal candidates are still dropped before the
final set.

Outcome toggles
---------------

Defaults come from :doc:`/parameters` and
:class:`~scgo.system_types.GLOptimizerParams`.

.. list-table::
   :widths: 38 10 30 32
   :header-rows: 1

   * - Toggle
     - Default
     - Gas (``gas_cluster_adsorbate``)
     - Surface (``surface_cluster_adsorbate``)
   * - ``allow_cluster_fragmentation``
     - ``False``
     - n/a (no slab)
     - Multiple core/mixed subgroups OK, each must touch slab
   * - ``allow_adsorbate_surface_detachment``
     - ``False``
     - n/a
     - Adsorbate-only subgroups on slab allowed
   * - ``enforce_adsorbate_subgraph_integrity``
     - ``True``
     - Fragments kept connected
     - Fragments kept connected
   * - ``freeze_adsorbate_internal_geometry``
     - ``False``
     - Rigid fragment (``FixBondLengths``)
     - Rigid fragment (``FixBondLengths``)
   * - ``validate_combined_structure`` (``ClusterAdsorbateConfig``)
     - ``True``
     - Pre/post-relax combined check
     - Pre/post-relax combined check

.. note::

   ``allow_cluster_fragmentation`` and ``allow_adsorbate_surface_detachment`` are
   surface-only. Gas paths ignore them because there is no slab. Surface
   defaults require a single connected mobile component touching the slab.

How validation runs during global optimization
----------------------------------------------

Every candidate minimum goes through
:func:`~scgo.system_types.validate_minimum_structure` (a thin wrapper over
:func:`~scgo.system_types.validate_structure_for_system_type`):

- Basin Hopping validates each relaxed trial. The initial seed is checked
  softly: a failure is logged as a warning and the run continues (later moves
  and the final gate still reject disconnected minima).
- The genetic algorithm validates each child before relaxation and uses the same
  helper for storage.
- ``simple_go`` validates its single relaxed structure when a ``system_type`` is
  given and returns an empty list if it is invalid.

After global optimization,
:func:`~scgo.minima_search.core.run_trials` applies the same structural gate to
deduplicated unique candidates before the Hessian/vibration check. See
:doc:`/uniqueness`. Surface candidates are checked against the prepared slab
search config when the slab is the search target, and the frozen bottom-layer
prefix is passed as the deposit boundary (``n_slab_deposit``) so a detached /
migrated search-mobile sheet is rejected exactly like at GA storage time. The
basin-hopping inline gates apply the same deposit-aware boundary to the relaxed
seed and every trial.

.. note::

   For slab-search adsorbate types (``surface_adsorbate``, and any
   ``*_adsorbate`` type where the slab top layers are the search core), metal
   cores are rejected up front: the mobile core **is** the supported slab top.
   Pass adsorbates only (``core_symbols=[]``).

The connectivity factor resolves the same way at every gate: explicit
``connectivity_factor``, then ``ClusterAdsorbateConfig``, then
``SurfaceSystemConfig``, then ``1.4``. The value may be a float or a
per-element/pair dict.

Minima pair selection
---------------------

Before NEB, endpoints are chosen by
:func:`~scgo.ts_search.transition_state_io.select_structure_pairs`. Hard gates
(``energy_gap_threshold``, ``max_endpoint_mismatch``, ``pair_core_rms_max``) and
soft ranking (``pair_score_*``) are documented under **Pair selection** in
:doc:`/parameters`. Defaults come from
:func:`~scgo.pair_selection_defaults.pair_selection_param_defaults`.

``max_pairs`` is the NEB budget. Adsorbate searches may oversample the select
pool (then re-rank) via
:func:`~scgo.ts_search.transition_state_io.resolve_ts_pair_select_cap`. Bare
types do not. See **Budget and oversampling** in :doc:`/parameters`.

Pairing regimes per system type (intentional):

- ``surface_cluster`` uses the **bare-surface** regime: no core-RMS gate;
  distinctness and mismatch gates only.
- On ``surface_adsorbate`` runs with an inferred top-layer mobile block, the
  inferred block is gated by ``pair_core_rms_max`` (default ``2.0`` Å). Whole
  layer shifts can legitimately exceed this; raise the knob to relax.

.. note::

   The pre-screen geometry gates (images, clash distance, displacement) are
   fixed-size knobs — they do not scale with the number of mobile atoms. Large
   search-mobile layers may need manual retuning of
   ``neb_prescreen_clash_distance`` / ``max_endpoint_mismatch``.

.. note::

   Optimized NEB bands are compared with plain Cartesian image differences;
   minimum-image distances apply at interpolation and at the gates only.

NEB pre-screen gates
--------------------

Before any NEB optimization,
:func:`~scgo.ts_search.transition_state.validate_initial_neb_path` runs for
every system type.
:func:`~scgo.ts_search.transition_state.validate_initial_neb_energy_profile`
runs only when ``max_endpoint_mismatch`` is set (bare ``gas_cluster`` leaves it
``None``):

- ``validate_initial_neb_path`` always checks that the aligned endpoints are
  actually distinct (maximum mobile displacement at least
  ``MIN_NEB_ENDPOINT_DISPLACEMENT_A`` = 0.3 Å; a label-swap or symmetry image of
  the same minimum has no transition path) and that interior images do not
  clash (minimum mobile pairwise distance vs ``neb_prescreen_clash_distance``).
  An upper aligned endpoint-displacement gate also runs when
  ``max_endpoint_mismatch`` is set.
- ``validate_initial_neb_energy_profile`` runs only when
  ``max_endpoint_mismatch`` is set (barrier cap ``neb_max_spurious_barrier``;
  drift and ``min_saddle_prominence`` checks when endpoint energies are
  available).

Per-system-type defaults:

.. list-table::
   :widths: 30 22 22 22 22
   :header-rows: 1

   * - Preset
     - ``neb_prescreen_clash_distance``
     - ``min_saddle_prominence``
     - ``neb_max_spurious_barrier``
     - ``max_endpoint_mismatch``
   * - Bare gas
     - 1.0
     - 0.10
     - 8.0
     - ``None`` (clash always; energy-profile skipped)
   * - Surface cluster
     - 0.7
     - 0.40
     - 8.0
     - 2.5
   * - Bare surface
     - 0.35
     - 0.40
     - 50.0
     - 3.0
   * - Gas adsorbate
     - 0.7
     - 0.40
     - 8.0
     - 1.25
   * - Surface cluster adsorbate
     - 0.7
     - 0.40
     - 8.0
     - 1.5
   * - Surface adsorbate
     - 0.7
     - 0.40
     - 8.0
     - 3.0

Constraint model
----------------

Slab fixation
~~~~~~~~~~~~~

Configured via :class:`~scgo.surface.config.SurfaceSystemConfig`:

- ``fix_all_slab_atoms=True`` (default),
- ``n_relax_top_slab_layers=N``, or
- ``n_fix_bottom_slab_layers=L-N``.

Applied by :func:`~scgo.surface.constraints.attach_slab_constraints`. Existing
non-``FixAtoms`` constraints (for example adsorbate ``FixBondLengths``) are kept
when slab fixation is refreshed.

Core freeze
~~~~~~~~~~~

``fix_core`` in
:func:`~scgo.cluster_adsorbate.relax.relax_metal_cluster_with_adsorbate`:
when ``True``, it freezes core indices ``0..n_core-1`` via ``FixAtoms``
(gas-phase only; periodic cores keep their cell).

Adsorbate rigidity
~~~~~~~~~~~~~~~~~~

``freeze_adsorbate_internal_geometry`` triggers
:func:`~scgo.cluster_adsorbate.constraints.attach_adsorbate_internal_geometry_constraints`,
which appends one multi-pair ``FixBondLengths`` per fragment. Those constraints
survive slab re-freezing.

Tunables and floors
-------------------

Placement and steric room
~~~~~~~~~~~~~~~~~~~~~~~~~

- ``height_min`` / ``height_max``: default ``0.9`` / ``2.2`` Å; placement
  height range in :mod:`scgo.cluster_adsorbate.config`.
- ``blmin_ratio``: default ``0.7``; clash threshold with a floor of ``0.55``.
- ``structure_min_distance_factor``: default ``0.4``; floor ``0.3``.

Connectivity and legal topology
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``connectivity_factor`` / ``structure_connectivity_factor``: default ``1.4``.
  This is the main structural legality check during initialization, after GA
  operators, at per-minimum algorithm gates, at the final structural gate, and
  in TS. Precedence (resolved via
  :func:`~scgo.system_types.resolve_connectivity_factor`): explicit
  ``connectivity_factor``, then
  ``ClusterAdsorbateConfig.structure_connectivity_factor``, then
  ``SurfaceSystemConfig.structure_connectivity_factor``, then ``1.4``.
- The value may be a global float or a dict:

  - float ``f``: bonded if ``d <= (r_i + r_j) * f``
  - element dict (e.g. ``{"Pt": 1.8, "C": 1.4}``): bonded if
    ``d <= r_i*f_i + r_j*f_j`` (missing symbols use ``1.4``)
  - pair keys ``"Pt-C"`` or ``("Pt", "C")``: bonded if ``d <= (r_i + r_j) * f_ij``;
    pair entries override element-derived thresholds

  Example for Pt on graphite: ``{"Pt": 1.4, "C": 1.4, "Pt-C": 1.8}``.

  Example (Pt on graphite): ``{"Pt": 1.4, "C": 1.4, "Pt-C": 1.8}``.

Surface contact and penetration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Penetration tolerance: ``0.1`` Å.
- Soft H-contact threshold: ``1.15`` Å (not tunable). It separates weak
  H-bond-like contacts from newly formed covalent bonds between fragments.

Why placement and validation differ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Placement uses a looser steric floor (for example ``blmin_ratio=0.7``), while
validation runs at ``connectivity_factor=1.4``. That gives the search room to
explore, then rejects borderline disconnected or topologically illegal
candidates before they reach the final set.

MIC semantics
-------------

Gas always uses ``use_mic=False``. Surface derives it from
``SurfaceSystemConfig.comparator_use_mic``, resolved by
:func:`~scgo.system_types.resolve_structure_mic`. It returns ``False`` for
non-surface systems and raises if the system is surface-based but
``surface_config`` is ``None``.
