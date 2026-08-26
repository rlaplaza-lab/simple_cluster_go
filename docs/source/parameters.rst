All Parameters
==============

This page lists all parameters you can use in SCGO. For preset functions and
their defaults, see :doc:`/api/param_presets`.

Parameter resolution
--------------------

All high-level ``run_*`` functions share the same contract:

1. **Safe defaults**: pass ``params=None``, ``go_params=None``, or
   ``ts_params=None`` to use full preset defaults.
2. **Partial overrides**: pass a dict with only the keys you want to change;
   runners merge with defaults before execution.
3. **Presets**: start from a :doc:`/api/param_presets` builder, edit what you
   need, then pass to ``run_*``.

**Merge rules**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Dict
     - Merge behavior
   * - ``params`` / ``go_params``
     - Deep-merge onto :func:`~scgo.param_presets.get_default_params` via
       :func:`~scgo.utils.run_helpers.initialize_params`. Nested dicts (for
       example ``optimizer_params["ga"]``) merge recursively; user keys win.
       Changing ``calculator`` from the MACE default replaces
       ``calculator_kwargs`` wholesale (new-calculator defaults from
       :func:`~scgo.param_presets.default_calculator_kwargs`, then any user
       kwargs) so MACE keys do not leak into EMT/UMA/UPET.
   * - ``ts_params``
     - Deep-merge onto :func:`~scgo.param_presets.get_ts_search_params` via
       :func:`~scgo.utils.run_helpers.initialize_ts_params`. Not merged with GO
       defaults. For ``run_go_ts*``, calculator settings align with merged
       ``go_params`` unless ``ts_params`` sets ``calculator``, in which case
       ``calculator_kwargs`` are replaced wholesale.
   * - Forbidden in dicts
     - Top-level ``system_type`` in ``go_params`` / ``ts_params`` (use the run
       ``system_type=`` argument). Identity keys
       (``system_type``, ``surface_config``, ``adsorbate_definition``,
       ``adsorbate_fragment_template``, ``cluster_adsorbate_config``) are also
       forbidden inside ``optimizer_params`` slots. Those slots hold algorithm
       hyperparameters only.
   * - Run kwargs
     - ``system_type``, ``surface_config``, ``adsorbates``, ``seed``,
       ``verbosity``, ``output_*`` belong on the ``run_*`` call.
       Top-level ``surface_config`` / adsorbate keys in ``go_params`` (or
       ``ts_params`` for ``surface_config``) are enough when the run argument
       is omitted, and must agree when both are set.

**Logging** (``verbosity >= 1``): SCGO logs the defaults source and a flat list
of user overrides, then the resolved GO optimizer settings or TS NEB
configuration. See :doc:`/api/utils`.

Verbosity levels (``run_*`` ``verbosity=`` argument):

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Level
     - Behavior
   * - 0
     - Warnings and errors only; no progress bars
   * - 1
     - Normal operation: parameter merge logs, timing summaries, campaign
       progress, GA phase summaries, and one-line TorchSim / TorchInductor
       summaries when GPU probing or compile-cache locking ran
   * - 2
     - Per-individual GA and initialization detail; per-pair NEB detail;
       third-party loggers still suppressed in HPC mode
   * - 3
     - TRACE-level diagnostics (deepest SCGO logging)

Configure the root logger with :func:`~scgo.configure_logging`. Set
``SCGO_LOCAL_DEV=1`` for milder third-party log suppression during local
development (see :doc:`/installation`).

**Workflow**

.. code-block:: python

   from scgo import run_go_ts
   from scgo.param_presets import get_default_params, get_ts_search_params

   go_params = get_default_params()
   go_params["optimizer_params"]["ga"]["niter"] = 8

   ts_params = get_ts_search_params(system_type="gas_cluster")
   ts_params["max_pairs"] = 12

   summary = run_go_ts(
       "Pt5",
       go_params=go_params,
       ts_params=ts_params,
       system_type="gas_cluster",
       seed=7,
   )

GO Parameters
-------------

Passed as ``params`` or ``go_params`` to ``run_go``, ``run_go_campaign``, ``run_go_ts``, etc.

**Algorithm selection**

Runners call :func:`~scgo.runner_api.select_scgo_minima_algorithm` automatically:

- ``gas_cluster`` only, ≤2 mobile atoms → ``simple`` (``simple_go.db``)
- 3 atoms, no adsorbate → Basin Hopping (``bh_go.db``)
- 3 atoms, adsorbate system types → Genetic Algorithm (``ga_go.db``)
- ≥4 atoms → Genetic Algorithm (``ga_go.db``)

**Top-Level:**

.. list-table::
   :widths: 25 10 65

   * - ``calculator``
     - ``"MACE"``
     - Calculator: ``"MACE"``, ``"UMA"``, ``"UPET"``, or ``"EMT"``
   * - ``calculator_kwargs``
     - ``{"model_name": "mace_matpes_0"}``
     - Calculator options. Changing ``calculator`` replaces this dict with that
       calculator's defaults (:func:`~scgo.param_presets.default_calculator_kwargs`).
       Unsupported ``device`` values raise ``SCGOValidationError``.
   * - ``seed``
     - ``None``
     - Random seed (function argument overrides)
   * - ``fitness_strategy``
     - ``"low_energy"``
     - ``"low_energy"``, ``"high_energy"``, or ``"diversity"``
   * - ``diversity_reference_db``
     - ``None``
     - Glob pattern for reference DBs (for diversity mode)
   * - ``diversity_max_references``
     - ``100``
     - Cap on reference structures loaded for diversity scoring
   * - ``diversity_update_interval``
     - ``5``
     - Refresh diversity references every N BH iterations / GA generations.
       BH/GA slot copies default to ``None`` (inherit these top-level values).
   * - ``connectivity_factor``
     - ``1.4``
     - Connectivity threshold for initialization, post-operator GA checks,
       per-minimum algorithm gates, the ``run_trials`` final structural gate, and
       TS. Accepts a global float or a dict of per-element and/or per-pair
       multipliers. Bonded means distance ≤ threshold; see
       :doc:`/validation_and_constraints` for the full spec and precedence rules.
   * - ``allow_cluster_fragmentation``
     - ``False``
     - Allow cluster to split (surface only)
   * - ``allow_adsorbate_surface_detachment``
     - ``False``
     - Allow adsorbates without cluster contact
   * - ``enforce_adsorbate_subgraph_integrity``
     - ``True``
     - Keep adsorbate fragments connected
   * - ``freeze_adsorbate_internal_geometry``
     - ``False``
     - Keep adsorbate fragments rigid
   * - ``surface_config``
     - ``None``
     - Required for surface runs. Prefer the run-function ``surface_config=``;
       a top-level key in ``go_params`` is enough when that argument is omitted.
   * - ``cluster_adsorbate_config``
     - ``None``
     - Adsorbate placement knobs (in ``go_params`` only)
   * - ``n_jobs``
     - ``1``
     - Single CPU parallelism knob. ``1`` = sequential; ``-1`` = all CPUs;
       ``-2`` = all but one CPU; or a positive worker count. Inherited by GA
       population init, GA offspring, and post-GO validation unless those
       stages are set explicitly.
   * - ``validation_n_jobs``
     - (optional)
     - Parallel workers for post-GO Hessian/force validation. ``None`` (default)
       inherits the top-level ``n_jobs``; an explicit value overrides it.
   * - ``validate_with_hessian``
     - ``False``
     - Run vibrational analysis
   * - ``tag_final_minima``
     - ``True``
     - Mark final structures in database
   * - ``fmax_threshold``
     - ``0.05``
     - Force threshold for validation (eV/\ :math:`\AA`)
   * - ``check_hessian``
     - ``True``
     - Check Hessian during validation
   * - ``imag_freq_threshold``
     - ``50.0``
     - Imaginary frequency cutoff (cm\ :sup:`-1`)

The subsections below list **algorithm hyperparameters** only
(``optimizer_params["simple"|"bh"|"ga"]``). Do not put ``system_type``,
``surface_config``, or adsorbate identity keys in these slots. See
*Parameter resolution* above.

Uniqueness knobs are documented in :doc:`/uniqueness`.

**Simple** (``optimizer_params["simple"]``), used for 1-2 atom gas clusters only:

.. list-table::
   :widths: 25 10 65

   * - ``optimizer``
     - ``"FIRE"``
     - Local optimizer name
   * - ``fmax``
     - ``0.05``
     - Force convergence (eV/\ :math:`\AA`)
   * - ``niter``
     - ``1``
     - Relaxation steps
   * - ``niter_local_relaxation``
     - ``"auto"``
     - Local relaxation budget
   * - ``energy_tolerance``
     - ``0.02`` eV
     - Energy window for campaign uniqueness (see :doc:`/uniqueness`)
   * - ``comparator_tol``
     - ``0.015``
     - Overall shape mismatch still counted as the same isomer
   * - ``comparator_pair_cor_max``
     - ``0.7`` Å
     - Largest allowed difference in any one interatomic distance
   * - ``comparator_n_top``
     - ``None``
     - Leave ``None``; uses the moving atoms for this system type
       (see :doc:`/uniqueness`)

**GA** (``optimizer_params["ga"]``):

Parallelism is driven by the top-level ``params["n_jobs"]`` knob (default ``1``,
sequential). Set ``-2`` or ``-1`` to parallelize population initialization,
offspring construction, and post-GO validation together. Per-stage keys
(``n_jobs_population_init``, ``n_jobs_offspring``, ``validation_n_jobs``)
override that default when set. See :doc:`/installation` for full semantics.
The TorchSim/UMA/UPET benchmark presets default to ``-2``.

.. code-block:: python

   params = get_default_params()
   params["n_jobs"] = -2  # one switch parallelizes population init, offspring, and validation

.. list-table::
   :widths: 25 10 65

   * - ``population_size``
     - ``"auto"``
     - Number of structures in population
   * - ``niter``
     - ``"auto"``
     - Number of generations
   * - ``mutation_probability``
     - ``0.4``
     - Probability of mutating each structure
   * - ``offspring_fraction``
     - ``0.5``
     - Fraction of population replaced each generation
   * - ``fmax``
     - ``0.05``
     - Force convergence (eV/\ :math:`\AA`)
   * - ``vacuum``
     - ``10.0``
     - Vacuum around clusters (\ :math:`\AA`)
   * - ``energy_tolerance``
     - ``0.02`` eV
     - Energy window for in-search and campaign uniqueness (see :doc:`/uniqueness`)
   * - ``comparator_tol``
     - ``0.015`` (``0.010`` for supported clusters)
     - Overall shape mismatch still counted as the same isomer
   * - ``comparator_pair_cor_max``
     - ``0.7`` Å (``0.45`` for supported clusters)
     - Largest allowed difference in any one interatomic distance
   * - ``comparator_n_top``
     - ``None``
     - Leave ``None``; uses the moving atoms for this system type
       (see :doc:`/uniqueness`)
   * - ``comparator_component_weights``
     - ``None``
     - Per-role block weights, e.g. ``{"mobile_slab": 0.2}`` (see :doc:`/uniqueness`)
   * - ``comparator_cross_weight``
     - ``1.0``
     - Weight of cross-block binding-geometry distance terms
   * - ``use_adaptive_mutations``
     - ``True``
     - Auto-adjust mutation rate
   * - ``early_stopping_niter``
     - ``10``
     - Stop if no improvement for N generations
   * - ``n_jobs_population_init``
     - ``None`` (inherits ``n_jobs``)
     - Workers for population initialization. ``None`` inherits the top-level
       ``params["n_jobs"]``; pass ``-1``, ``-2``, or a positive worker count to
       override.
   * - ``n_jobs_offspring``
     - ``None`` (inherits ``n_jobs``)
     - Workers for offspring construction. Same semantics as
       ``n_jobs_population_init``.
   * - ``write_timing_json``
     - ``False``
     - Write ``{run_dir}/timing.json``; enables ``go_ts_timing.json`` rollup in ``run_go_ts``
   * - ``detailed_timing``
     - ``False``
     - Include per-generation timing
   * - ``stagnation_trigger``
     - ``4``
     - Generations without improvement before adaptive mutation boost
   * - ``stagnation_full_trigger``
     - ``8``
     - Stronger stagnation threshold
   * - ``recovery_window``
     - ``2``
     - Generations to watch after a mutation boost
   * - ``aggressive_burst_multiplier``
     - ``1.8``
     - Mutation-rate multiplier on stagnation
   * - ``max_mutation_probability``
     - ``0.65``
     - Cap on adaptive mutation probability
   * - ``batch_size``
     - ``None``
     - TorchSim batch size (when using a relaxer)
   * - ``relaxer``
     - ``None``
     - Optional TorchSim relaxer instance

**BH** (``optimizer_params["bh"]``):

.. list-table::
   :widths: 25 10 65

   * - ``temperature``
     - ``1.0`` eV
     - Metropolis energy scale for accepting uphill moves (eV; not a physical temperature)
   * - ``dr``
     - ``0.2``
     - Maximum step size (\ :math:`\AA`)
   * - ``move_fraction``
     - ``0.3``
     - Fraction of atoms to move
   * - ``deduplicate``
     - ``True``
     - BH end-of-run uniqueness pass (campaign filtering still runs)
   * - ``energy_tolerance``
     - ``0.02`` eV
     - Energy window for uniqueness (see :doc:`/uniqueness`)
   * - ``move_strategy``
     - ``"random"``
     - Atom move strategy
   * - ``comparator_tol``
     - ``0.015`` (``0.010`` for supported clusters)
     - Overall shape mismatch still counted as the same isomer
   * - ``comparator_pair_cor_max``
     - ``0.7`` Å (``0.45`` for supported clusters)
     - Largest allowed difference in any one interatomic distance
   * - ``comparator_n_top``
     - ``None``
     - Leave ``None``; uses the moving atoms for this system type
       (see :doc:`/uniqueness`)
   * - ``comparator_component_weights``
     - ``None``
     - Per-role block weights, e.g. ``{"mobile_slab": 0.2}`` (see :doc:`/uniqueness`)
   * - ``comparator_cross_weight``
     - ``1.0``
     - Weight of cross-block binding-geometry distance terms
   * - ``write_timing_json``
     - ``False``
     - Write ``{run_dir}/timing.json``; enables ``go_ts_timing.json`` rollup in ``run_go_ts``
   * - ``detailed_timing``
     - ``False``
     - Include per-iteration timing breakdown

TS Parameters
-------------

Passed as ``ts_params`` to ``run_ts_search``, ``run_ts_campaign``, ``run_go_ts``, etc. Sparse dicts are merged with :func:`~scgo.param_presets.get_ts_search_params` defaults at run time.

**Core:**

.. list-table::
   :widths: 25 10 60

   * - ``calculator``
     - ``"MACE"``
     - Calculator for TS search
   * - ``calculator_kwargs``
     - ``{}``
     - Calculator options
   * - ``max_pairs``
     - ``None``
     - Maximum endpoint pairs that run NEB (``None`` = all survivors). Soft
       ``pair_score_*`` ranking only matters when this caps the pool. Adsorbate
       searches may select more candidates first (see **Budget and
       oversampling** below); the runner always truncates to this value before
       NEB.
   * - ``energy_gap_threshold``
     - ``2.0`` / ``0.75`` (adsorbate)
     - Hard max energy gap between endpoints (eV); pairs above this are skipped
   * - ``use_torchsim``
     - ``True``
     - Use TorchSim for NEB
   * - ``dedupe_minima``
     - ``True``
     - Drop duplicate GO minima before pairing (GO uniqueness; see
       :doc:`/uniqueness`). For slab-search types the comparison window is
       the mobile partition ``[fixed | top layers | adsorbate]`` tail —
       matching GO-phase ``search_mobile_count`` semantics — so distinct
       top-layer registries survive and the frozen bottom slab cannot dilute
       the comparison. Other types compare the trailing mobile core +
       adsorbate atoms. The ``minima_energy_tolerance`` semantics are
       unchanged.
   * - ``tag_ts_in_db``
     - ``True``
     - Tag unique successful saddles in the minima databases (consumed by TS
       resume and downstream loading). Settable via ``ts_params``.
   * - ``connectivity_factor``
     - ``1.4``
     - Same connectivity spec as GO (float or per-element/pair dict); resolved
       with the same precedence for TS structural gates.
   * - ``similarity_tolerance``
     - ``0.015``
     - Overall shape mismatch for TS uniqueness (same role as GO
       ``comparator_tol``; see :doc:`/uniqueness`)
   * - ``similarity_pair_cor_max``
     - ``0.1`` Å
     - Largest allowed distance difference for TS uniqueness (tighter than GO
       ``0.7`` Å; see :doc:`/uniqueness`)
   * - ``pair_core_rms_max``
     - see **Pair selection** below
     - Hard max core RMS (Å) for adsorbate+core pairing. Gas cores are
       fingerprint-matched, Kabsch-aligned (including 1-atom translation),
       then spatially rematched so reflected fingerprint labelings cannot
       inflate RMS. Slab cores stay in the surface frame. Pair selection and
       NEB endpoint prep (including bare gas clusters) share this overlay.
   * - ``pair_score_*``
     - see **Pair selection** below
     - Soft ranking scales and weights (gap / distinct / mismatch / core)
   * - ``minima_energy_tolerance``
     - ``0.02`` eV
     - Energy window when dropping duplicate GO minima before pairing
       (see :doc:`/uniqueness`)
   * - ``dedupe_ts``
     - ``True``
     - Keep unique successful saddles in ``final_unique_ts/``
   * - ``ts_energy_tolerance``
     - ``0.02`` eV
     - Energy window for that TS uniqueness pass (geometry uses
       ``similarity_*``)
   * - ``write_timing_json``
     - ``False``
     - Write ``{ts_run_dir}/timing.json``; enables ``go_ts_timing.json`` rollup in ``run_go_ts``

Unknown ``ts_params`` keys are rejected up front with a
:class:`~scgo.exceptions.SCGOValidationError` listing the offending keys and
the expected set (mirroring the GO behavior) — a typo such as ``neb_fmx``
fails fast instead of being silently ignored.

Direct calls to :func:`~scgo.ts_search.run_transition_state_search` that omit
NEB knobs resolve them from the same per-system presets
(:func:`~scgo.param_presets.get_ts_defaults`) used by ``get_ts_search_params``:
adsorbate types get spring constant ``0.5``, steps ``4000`` and climb; bare
surfaces get steps ``2000``; MIC / cell remap / lattice rotation follow the
system policy.

.. note::

   With ``neb_steps="auto"`` on surface types, the step budget is derived from
   the **total expanded** atom count (slab included), not just the mobile
   region — budget accordingly when using large slabs.

**Pair selection**
(:func:`~scgo.ts_search.transition_state_io.select_structure_pairs`;
defaults from
:func:`~scgo.pair_selection_defaults.pair_selection_param_defaults`;
select budget from
:func:`~scgo.ts_search.transition_state_io.resolve_ts_pair_select_cap`):

Before any NEB force evaluation, SCGO:

1. Loads (and optionally deduplicates) GO minima.
2. Enumerates candidate endpoint pairs.
3. Applies **hard gates** (energy gap, similarity / mismatch / core RMS).
4. Soft-ranks survivors with ``pair_score_*``.
5. Truncates to the select cap from ``resolve_ts_pair_select_cap``.
6. For adsorbate types only (when TorchSim and ``max_endpoint_mismatch`` are set
   and more survivors remain than ``max_pairs``): re-ranks the oversampled pool
   by IDPP path quality and keeps the best ``max_pairs``.
7. Truncates to ``max_pairs`` again, then runs NEB.

Minima are laid out ``[slab | core | adsorbate]``. Pairing uses different hard
gates by regime:

- **Bare** (``gas_cluster`` / ``surface_cluster`` / ``surface``): fingerprint
  the full mobile region; skip similar pairs; optional
  ``max_endpoint_mismatch`` on fingerprint difference.
- **Adsorbate + metal core**: fingerprint the **core only**; do not skip
  similar cores (site hops look identical); hard-gate core difference with
  ``max_endpoint_mismatch`` and core RMS with ``pair_core_rms_max``. Soft rank
  prefers mid energy gap, similar cores, and some adsorbate site displacement.
- **Adsorbate-only slab** (no mobile core): do not skip similar; gate on
  adsorbate travel via ``max_endpoint_mismatch``.

Hard gates always apply. Soft ``pair_score_*`` terms only order candidates when
a finite select / ``max_pairs`` cap truncates the list.

**Budget and oversampling**

``max_pairs`` is the number of NEBs you pay for:

- **Bare** system types use ``max_pairs`` as the select cap. Soft scores pick
  the top N; those N bands run NEB. Setting ``max_endpoint_mismatch`` on bare
  surface presets (``3.0`` Å) enables pre-NEB path gates only. It does not
  turn on oversampling.
- **Adsorbate** system types with ``max_endpoint_mismatch`` set oversample the
  select pool to
  ``min(max_pairs * 10, max(max_pairs, 50))``
  (:func:`~scgo.ts_search.transition_state_io.adsorbate_pair_select_cap`), then
  re-rank by IDPP profile and keep ``max_pairs`` for NEB. Example:
  ``max_pairs=6`` selects up to 50 ranked pairs, then runs at most 6 NEBs. If
  TorchSim / IDPP screening is unavailable, the runner still truncates the
  oversampled list to ``max_pairs`` before NEB.

Per-system-type defaults (with a caller-set ``max_pairs=N``):

.. list-table::
   :widths: 28 18 18 36
   :header-rows: 1

   * - System type
     - Default ``max_endpoint_mismatch``
     - Oversample select?
     - Select cap → NEB count
   * - ``gas_cluster``
     - ``None``
     - No
     - ``N`` → ``N``
   * - ``surface_cluster``
     - ``2.5``
     - No
     - ``N`` → ``N``
   * - ``surface``
     - ``3.0``
     - No
     - ``N`` → ``N``
   * - ``gas_cluster_adsorbate``
     - ``1.25``
     - Yes
     - up to ``adsorbate_pair_select_cap(N)`` → ``N``
   * - ``surface_cluster_adsorbate``
     - ``1.5``
     - Yes
     - up to ``adsorbate_pair_select_cap(N)`` → ``N``
   * - ``surface_adsorbate``
     - ``3.0``
     - Yes
     - up to ``adsorbate_pair_select_cap(N)`` → ``N``

Per-system-type values for these knobs are returned by
:func:`~scgo.pair_selection_defaults.pair_selection_param_defaults`
(see its API entry in :doc:`/api/validation_and_constraints_api`).
Meaning of each soft term:

- ``pair_score_gap_*``: prefer energy gaps near ``gap_center`` (not too near
  degenerate, not near the hard ``energy_gap_threshold``).
- ``pair_score_cum_scale`` + ``w_distinct``: bare systems reward fingerprint
  distinctness; adsorbate systems reward max adsorbate atom displacement after
  core alignment (site hop).
- ``pair_score_mismatch_scale`` + ``w_mismatch``: bare systems tolerate some
  fingerprint difference; adsorbate systems prefer small core difference.
- ``pair_score_core_rms_scale`` + ``w_core``: adsorbate+core only; prefer small
  core RMS after matching. Gas cores use fingerprint correspondence then
  Kabsch (slab cores stay in the lab frame). The same overlay is applied
  before adsorbate hop scoring and before NEB interpolation.

Override any of these in ``ts_params``. If the adsorbate pair pool is empty,
logs include skip counts (energy gap, mismatch, core RMS, and so on).

**NEB:**

.. list-table::
   :widths: 25 15 50

   * - ``neb_n_images``
     - ``5`` / ``7`` (adsorbate)
     - Number of images
   * - ``neb_steps``
     - ``"auto"`` / ``2000`` (bare surface) / ``4000`` (adsorbate)
     - Maximum optimization steps
   * - ``neb_fmax``
     - ``0.20``
     - Force convergence (eV/\ :math:`\AA`); shared across all system types
   * - ``neb_spring_constant``
     - ``0.1`` / ``0.5`` (adsorbate)
     - Spring constant (eV/\ :math:`\AA`\ :sup:`2`)
   * - ``neb_climb``
     - ``False`` / ``True`` (adsorbate)
     - Use climbing image
   * - ``use_parallel_neb``
     - ``True``
     - Batch multiple NEB bands in one TorchSim force eval (all system types)
   * - ``parallel_neb_max_bands``
     - ``None`` / ``4`` (surface)
     - Explicit cap on concurrent bands in the parallel NEB runner. Surface
       defaults to ``4`` bands per force batch for OOM safety on large slab
       cells. When ``None``, bands are chunked by
       ``parallel_neb_max_batch_atoms`` instead
   * - ``parallel_neb_max_batch_atoms``
     - ``6000`` / ``4000`` (surface)
     - Atom budget (sum of ``n_images * n_atoms``) per fused parallel NEB force
       batch, used only when ``parallel_neb_max_bands`` is ``None``. Also sizes
       the TorchSim relaxer's ``expected_max_atoms`` / ``max_atoms_to_try``. A
       chunk that hits CUDA OOM is retried once at half the budget
   * - ``max_endpoint_mismatch``
     - ``None`` (bare gas) / ``1.25`` (gas adsorbate) / ``2.5`` (surface cluster) / ``1.5`` (surface cluster adsorbate) / ``3.0`` (bare surface and surface adsorbate)
     - Å geometric gate on comparator difference; when set, also enables the
       pre-NEB endpoint-displacement check. For adsorbate + metal-core systems,
       pair selection fingerprints the **core** and this gate means “cores too
       different”; adsorbate site hops with an identical core are kept. For
       adsorbate-only slabs the same threshold gates adsorbate travel (wider
       default so graphite hollow/bridge hops are not rejected wholesale). On
       adsorbate system types it also enables select oversampling (see **Budget
       and oversampling**); on bare surface it does not.
   * - ``neb_prescreen_clash_distance``
     - ``1.0`` (bare gas) / ``0.7`` (surface cluster + all adsorbate) / ``0.35`` (bare surface)
     - Interior NEB image min mobile pairwise distance (Å) below which the initial path is rejected.
   * - ``min_saddle_prominence``
     - ``0.10`` (bare gas) / ``0.40`` (surface + adsorbate)
     - Minimum interior-max prominence (eV) above both endpoints for a band to pass the pre-NEB energy profile gate.
   * - ``neb_max_spurious_barrier``
     - ``8.0`` / ``50.0`` (bare surface)
     - Maximum allowed IDPP barrier (eV) before a band is rejected as discontinuous.
   * - ``neb_align_endpoints``
     - ``True``
     - Align endpoints before interpolation
   * - ``neb_interpolation_mic``
     - ``False`` / ``True``
     - Use minimum image convention
   * - ``neb_perturb_sigma``
     - ``0.0``
     - Gaussian perturbation on band (Å)
   * - ``neb_interpolation_method``
     - ``"idpp"``
     - Interpolation method
   * - ``neb_tangent_method``
     - (default)
     - NEB tangent method
   * - ``neb_interpolation_bond_tolerance_a``
     - ``0.5``
     - Post-interpolation FixBondLengths stretch diagnostic (Å); warns, never
       raises. Applied on the serial, parallel, and IDPP-screen paths.
   * - ``layer_cluster_threshold_ang``
     - ``0.4``
     - Layer-clustering threshold (Å) used when resolving which slab layers
       count as distinct for NEB endpoint ``FixAtoms`` attachment
       (``n_relax_top_slab_layers`` / ``n_fix_bottom_slab_layers`` modes).
   * - ``binding_penetration_tolerance_a``
     - ``0.1``
     - Mobile-atoms-below-slab-top tolerance (Å) for the post-NEB surface
       geometry gate.
   * - ``torchsim_fmax``
     - ``0.20``
     - TorchSim force tolerance (mapped internally). Keep equal to ``neb_fmax`` unless you intentionally diverge them
   * - ``torchsim_max_steps``
     - ``"auto"`` / ``2000`` (bare surface) / ``4000`` (adsorbate)
     - TorchSim step budget (mapped internally)

**NEB pre-screen gates:**

Before any NEB optimization, ``validate_initial_neb_path`` runs for every
system type. ``validate_initial_neb_energy_profile`` runs only when
``max_endpoint_mismatch`` is set (bare ``gas_cluster`` leaves it ``None`` and
skips the energy-profile screen):

- Degenerate-path check (aligned endpoint mobile displacement must reach
  ``MIN_NEB_ENDPOINT_DISPLACEMENT_A`` = 0.3 Å; below it the two endpoints are
  the same minimum up to permutation/symmetry and no transition path exists)
  always runs. The interior-image clash check (minimum mobile pairwise distance
  vs ``neb_prescreen_clash_distance``) also always runs; an upper aligned
  endpoint-displacement gate applies when ``max_endpoint_mismatch`` is set.
- Energy-profile check (barrier cap ``neb_max_spurious_barrier``; endpoint-energy
  drift ``> 0.5`` eV and interior-max prominence below ``min_saddle_prominence``)
  runs only when ``max_endpoint_mismatch`` is set and endpoint energies are
  available. Bands with fewer than three images skip the prominence/drift check.

Per-system-type defaults for the three pre-screen knobs are listed under
:doc:`/validation_and_constraints` (bare gas is looser:
``neb_prescreen_clash_distance=1.0`` / ``min_saddle_prominence=0.10``;
surface clusters and adsorbates are tighter: ``0.7`` / ``0.40``; bare surface
is widest: ``0.35`` clash with a ``50.0`` eV barrier cap).

**Adsorbate NEB specifics** (beyond the gates above):

- Fragment-wise adsorbate matching and core-anchored alignment
- Pair selection / oversampling: see **Budget and oversampling** under
  **Pair selection** above (adsorbate-only; bare surface
  ``max_endpoint_mismatch`` does not oversample)
- Climbing NEB: two-stage only when the IDPP path has a clear interior maximum
  (barrier ``≥ 1.0`` eV); otherwise climb from step 0
- Finalize also rejects barriers ``> 8`` eV

**Surface NEB (differences from gas):**

- ``neb_interpolation_mic=True`` (forced)
- ``neb_surface_cell_remap=True``
- ``neb_surface_lattice_rotation=True`` for bare ``surface_cluster`` /
  ``surface``; ``False`` for ``surface_cluster_adsorbate`` /
  ``surface_adsorbate`` (keeps adsorbate-slab registry)
- ``neb_surface_max_lattice_shift=1``
- ``parallel_neb_max_bands=4`` (bands chunked four at a time for large slab
  cells)
- ``parallel_neb_max_batch_atoms=4000`` (atom budget used when the band cap is
  cleared to ``None``)

Surface Config
--------------

.. list-table::
   :widths: 25 10 65

   * - ``slab``
     - Required
     - ASE Atoms object
   * - ``name``
     - ``"slab"``
     - Path-key surface segment (filesystem-safe). Graphite preset uses
       ``"graphite"`` (e.g. ``Pt5_OH_OH_graphite_searches``).
   * - ``adsorption_height_min``
     - ``1.2`` (class) / ``2.0`` (``make_surface_config``)
     - Minimum height above slab (\ :math:`\AA`).
   * - ``adsorption_height_max``
     - ``3.0`` (class) / ``3.5`` (``make_surface_config``)
     - Maximum height above slab (\ :math:`\AA`).
   * - ``surface_normal_axis``
     - ``2``
     - Normal axis (0=x, 1=y, 2=z)
   * - ``fix_all_slab_atoms``
     - ``True``
     - Keep entire slab frozen
   * - ``n_relax_top_slab_layers``
     - ``None``
     - Top layers to relax
   * - ``n_fix_bottom_slab_layers``
     - ``None``
     - Bottom layers to freeze
   * - ``defect_bias_probability``
     - ``0.0`` (class) / ``0.5`` if ``monovacancy`` else ``0.0`` (preset)
     - Fraction (0.0 to 1.0) of placements biased onto a recorded slab vacancy;
       ignored when the slab has no vacancy (see :doc:`/surface_slab_guide`).
   * - ``comparator_use_mic``
     - ``True``
     - Use MIC in structure comparator on surfaces
   * - ``cluster_init_vacuum``
     - ``8.0``
     - Extra vacuum for cluster init on slab
   * - ``init_mode``
     - ``"smart"``
     - Surface cluster init mode: ``smart``, ``seed+growth``, ``random_spherical``,
       or ``template`` (see :doc:`/api/initialization`)
   * - ``max_placement_attempts``
     - ``200`` (class) / ``500`` (``make_surface_config``); presets use ``1000``
     - Max cluster placement attempts on slab
   * - ``structure_connectivity_factor``
     - ``1.4``
     - Fallback connectivity spec (float or dict; same forms as top-level
       ``connectivity_factor``, see :doc:`/validation_and_constraints`) when the
       GO/TS param is omitted. Used for slab-contact / supported-deposit checks,
       not only placement.

.. note::
   Use only one of the layer options, not both. See :doc:`/api/surface`.

.. note::
   The graphite preset functions override ``adsorption_height_min`` /
   ``adsorption_height_max``: ``make_graphite_surface_config`` and
   ``make_defected_graphite_surface_config`` use **0.5 / 1.0 Å**, while
   ``make_graphene_surface_config`` and ``make_n_doped_graphite_surface_config``
   use **0.5 / 1.5 Å**. The values above are the class and
   ``make_surface_config`` defaults, which apply only when you build a config
   directly rather than through a preset. See :doc:`/surface_slab_guide`.

.. note::
   Graphite and graphene presets default to the HOPG 5×5 × 3-layer footprint
   (~150 C atoms, ~12.3 Å hexagonal in-plane vectors, 30 Å total vacuum /
   ~36.7 Å cell height). Pass explicit ``slab_layers`` / ``slab_repeat_xy``
   (or ``nx`` / ``ny`` for graphene) to recover a smaller cell. Named helpers
   ``make_hopg_5x5_*`` pin the same geometry.

Adsorbate Config
----------------

.. list-table::
   :widths: 25 10 65

   * - ``height_min``
     - ``0.9``
     - Minimum placement height (\ :math:`\AA`).
   * - ``height_max``
     - ``2.2``
     - Maximum placement height (\ :math:`\AA`).
   * - ``max_placement_attempts``
     - ``80``
     - Maximum placement tries
   * - ``blmin_ratio``
     - ``0.7``
     - Clash threshold
   * - ``structure_connectivity_factor``
     - ``1.4``
     - Fallback connectivity spec (float or dict; same forms as top-level
       ``connectivity_factor``) when the GO/TS param is omitted.

See Also
----------

- :doc:`/quickstart` - How to use these parameters
- :doc:`/api/param_presets` - Preset functions and their defaults
- :doc:`/api/runner_api` - API function documentation
- :doc:`/validation_and_constraints` - How validation and constraints interact
