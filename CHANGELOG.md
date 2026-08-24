# Changelog

## 0.9.0

### Fixed

- TS pre-pair minima dedupe is now **partition-aware** for slab-search types
  (`surface`, `surface_adsorbate`): the comparison window is the mobile
  partition `[fixed | top layers | adsorbate]` tail, matching the GO-phase
  `search_mobile_count` contract. Distinct top-layer registries no longer
  collapse because the frozen slab dominated (or diluted) the fingerprint, and
  bare `surface` runs no longer compare the full frozen structure.
- `layer_cluster_threshold_ang` is now actually applied on the TS path: NEB
  endpoint `FixAtoms` attachment forwards the configured threshold instead of
  silently using the module constant.
- The IDPP priority screen honors the resolved per-system
  `min_saddle_prominence` / `neb_max_spurious_barrier` knobs (previously
  hardcoded screen defaults applied), and forwards
  `neb_interpolation_bond_tolerance_a` to interpolation on both the IDPP
  screen and the parallel NEB runner — matching the serial path.
- Direct calls to `run_transition_state_search` that omit NEB knobs now resolve
  them from the same per-system presets as `get_ts_search_params`
  (adsorbates: spring `0.5`, steps `4000`, climb; surfaces: steps `2000`;
  shared `neb_fmax=0.20`). High-level runner behavior is unchanged.
- Surface TS runs warn when `comparator_use_mic=False`: the knob affects GO
  comparators only; TS dedupe/pairing/NEB force MIC for surface types.
- `run_trials` logs which value wins when a `comparator_n_top` override differs
  from the resolved `search_mobile_count`.

### Added

- `tag_ts_in_db` is settable via `ts_params` (boolean, default `True`) and
  flows through to the TS runner.
- Strict `ts_params` key validation: unknown keys are rejected up front with an
  error listing the offending and expected keys (mirrors GO behavior).
- `DEFAULT_FMAX_THRESHOLD` constant replaces duplicated `0.05` literals;
  dead `DEFAULT_PAIR_COR_CUM_DIFF` constant removed.

### Changed

- Metal cores are rejected on slab-search adsorbate types
  (`validate_adsorbate_definition`): the slab top layers *are* the search core;
  pass adsorbates only.
- The `run_trials` final structural gate and the basin-hopping inline gates now
  pass the frozen bottom-layer prefix (`n_slab_deposit`) for slab-search types,
  so a detached/migrated search-mobile sheet is rejected exactly like at GA
  storage time.
- Shared default specifications: `default_energy_gap_threshold(has_adsorbate)`
  single-sources the `0.75`/`2.0` pair-gap rule, `TS_POSTPROCESS_DEFAULTS`
  single-sources the TS dedupe/tolerance defaults, `TS_NEB_FMAX` is the public
  shared NEB force tolerance, and one parameterized
  `_attach_torchsim_relaxer` builder replaces the three per-calculator
  relaxer builders.

### Removed

- `NebRunConfig.binding_penetration_tolerance_a` field: the value still flows
  via the explicit function argument into the post-NEB surface geometry gate;
  the never-read dataclass field is gone.

## 0.8.0

### Added

- Core-fingerprint adsorbate TS pair selection with tunable soft-rank gates
  (`pair_core_rms_max`, `pair_score_gap_*`, `pair_score_w_*`, …); gas/surface
  cores share the fingerprint + Kabsch overlay used for NEB endpoint prep.
- Grouped NEB pair outcome summaries at default verbosity (per-pair detail at
  verbosity ≥ 2).
- Run `verbosity` threaded end-to-end through GO/TS/logging helpers;
  `infer_verbosity` removed.
- TorchSim Model Memory Estimation / autobatcher probe prints collapsed to a
  one-line INFO scaler summary.
- GO-only ORR example: `examples/example_pt5_orr_defected_graphite.py` (bare
  Pt5 then O/OH/OOH on monovacancy graphite via `run_go`).
- `connectivity_factor` and `structure_connectivity_factor` accept a float or a
  dict of per-element and/or per-pair multipliers (`"Pt-C"` or `("Pt", "C")`;
  pair entries override element-derived thresholds; missing keys fall back to
  `1.4`). See :mod:`scgo.system_types.connectivity_factor`.
- Single top-level `n_jobs` CPU-parallelism knob. It defaults to `1`
  (sequential) in `get_default_params()` and, when set (e.g. `-2` = all but one
  CPU, `-1` = all CPUs, or a positive count), scales every CPU-bound stage at
  once: GA population initialization, GA offspring construction, and post-GO
  Hessian/force validation. The per-stage keys
  (`optimizer_params["ga"]["n_jobs_population_init"]`,
  `optimizer_params["ga"]["n_jobs_offspring"]`, and `validation_n_jobs`) remain
  valid overrides: `None` inherits the top-level `n_jobs`, an explicit value
  wins for that stage only. The shared worker-count helper
  :func:`scgo.utils.parallel_workers.resolve_n_jobs_for_tasks` now caps every
  pool at the number of tasks and floors at one.
- Graphene/graphite vacancy presets and defect-biased nanoparticle placement
  (`make_defected_graphite_*`, `make_n_doped_*`, `build_monovacancy_graphene_slab`,
  `defect_bias_probability`). `make_defected_graphite_surface_config` wires
  `defect_bias_probability` (default `0.5` when omitted).
- Named HOPG 5×5 × 3-layer graphite helpers:
  `make_hopg_5x5_graphite_surface_config`,
  `make_hopg_5x5_defected_graphite_surface_config`,
  `build_hopg_5x5_graphite_slab`, and `build_hopg_5x5_defected_graphite_slab`.
  Examples and the Kaggle GPU example matrix use these so slab geometry stays
  pinned to the 150-atom HOPG_5-5_3-layers footprint.
- Planar slab layers (graphene / graphite top) now expose hollow (`facet`)
  adsorption sites computed from the in-plane Voronoi diagram, in addition to
  the existing on-top (`vertex`) and bridge (`edge`) sites. A
  position/axis/PBC-keyed cache (`get_or_compute_planar_layer_site_candidates`,
  cleared by `clear_surface_site_cache`) keeps deposition at one Voronoi build
  per slab geometry.
- `run_go_campaign` now accepts `calculator_for_global_optimization`: pass a
  pre-warmed ASE/MLIP calculator to reuse it across every composition and avoid
  reloading the model per campaign. The campaign only builds its own calculator
  when the argument is omitted, and never tears down a caller-owned calculator.
  The new keyword sits immediately after `output_dir`, so purely positional calls
  that pass arguments past `output_dir` shift by one position (in-repo callers
  already use keyword arguments, so no change is required there).
- Adsorbate tag-partition validation at the minimum-structure gate for
  `*_adsorbate` system types.
- Configurable geometric thresholds exposed in TS defaults:
  `binding_penetration_tolerance_a`, `layer_cluster_threshold_ang`, and
  `neb_interpolation_bond_tolerance_a` (post-interpolation FixBondLengths
  diagnostic warns, never raises).
- Three user-facing transition-state pre-screen parameters, applied per system
  type via the TS presets and `run_transition_state_search` defaults:
  `neb_prescreen_clash_distance`, `min_saddle_prominence`, and
  `neb_max_spurious_barrier`. Bare gas uses the looser defaults `1.0` / `0.10` /
  `8.0`; surface and adsorbate use `0.7` / `0.40` / `8.0`.
- `parallel_neb_max_batch_atoms` TS parameter (`6000` gas / `4000` surface): when
  `parallel_neb_max_bands` is `None`, parallel NEB bands are greedily binned so
  each fused force batch stays within this atom budget.
- Low-effort UMA/UPET/TorchSim GA + TS presets for examples and Kaggle CI.
- `default_calculator_kwargs(calculator)` returns a fresh default
  `calculator_kwargs` dict for MACE / UMA / UPET (empty for EMT and unknown
  names).
- `DEFAULT_TS_PAIR_COR_MAX` (`0.1` Å) for TS pair near-dupe gating, distinct
  from GO uniqueness (`DEFAULT_PAIR_COR_MAX` = `0.7` Å).
- NEB resume: `run_transition_state_search(run_id=...)` skips pairs whose
  `neb_{pair_id}_metadata.json` already has `status="success"`,
  `neb_converged=true`, and a readable `ts_{pair_id}.xyz`
  (`load_completed_neb_result`). Run-dir `metadata.json` and NEB metadata are
  written via temp + `os.replace`.
- `torch_load_weights_only_false()` context manager scopes the MACE/e3nn
  `torch.load(..., weights_only=False)` shim (import + checkpoint load only).
- `default_params_top_level_keys()` returns a cached frozenset of default
  top-level GO keys so allowlist checks avoid a full `get_default_params()`
  deepcopy per run.

### Changed

- GA offspring job payloads can ship mobile-only atoms for slab runs and
  reconstruct full slab+mobile frames inside workers, reducing per-job
  serialization without changing pairing/mutation semantics.
- GA population uniqueness bookkeeping now tracks rediscovery counts during
  O(population)-time in-search checks (instead of rescanning full history),
  preserving `looks_like` penalty behavior while avoiding O(history) updates.
- GO uniqueness is energy **and** mobile geometry on every optimizer path; GA
  in-search no longer ORs ASE comparators. Shared knobs on ``simple`` / ``bh`` /
  ``ga``; see ``docs/source/uniqueness.rst``. TS pair gating and final
  unique-TS clustering share ``similarity_*`` and ``ts_energy_tolerance``.
- TorchInductor ``filelock`` acquire/release DEBUG spam is captured during
  ``configure_logging`` and collapsed into one INFO summary
  (``TorchInductor: N compile-cache lock event(s)``) when any events occurred;
  drained from TorchSim relax calls and at GA completion.
- GA post-relax ineligible INFO lines now include compact reason rollups
  (e.g. ``2 ineligible (disconnectedx2)``), matching NEB skip-summary style.
- Linear and planar clusters (for example Co₄ remnants after shrinking oversized
  magic-number templates) are classified with PCA before scipy's 3D ConvexHull.
  Degenerate Qhull failures no longer dump ``QH6154`` diagnostics; vertices,
  adsorbate sites, and hull-based growth fall back to endpoints / 2D hull /
  out-of-plane placement. Template shrink still requires a 3D hull; linear
  finished templates are discarded.
- `run_trials` final structural gate now honors top-level `connectivity_factor`
  (same precedence as algorithm and TS gates: explicit →
  `ClusterAdsorbateConfig.structure_connectivity_factor` →
  `SurfaceSystemConfig.structure_connectivity_factor` → `1.4`). Previously the
  dump-time gate resolved only from config fallbacks, so
  `go_params["connectivity_factor"]` did not affect it.
- Connectivity / clash validation hot paths specialize by factor kind (global /
  element / pair / mixed): vectorized threshold matrices, shared distance
  matrices for clash+connectivity diagnostics, nearest-neighbor-only slab
  contact thresholds, and cross-set adsorbate-fragment thresholds. The same
  resolved `connectivity_factor` (plus `cluster_adsorbate_config` fallback) is
  threaded through GA storage, BH/simple/TS entry points, and surface gates.
- Split `scgo.system_types` into a package (`policy`, `composition`,
  `validation`, `params`). Public import path unchanged.
- Run identity (`system_type`, `surface_config`, adsorbate context) is resolved
  once onto the run context / top-level `go_params` instead of being copied into
  every `optimizer_params` slot. Defaults and TorchSim / low-effort builders no
  longer embed `system_type` or `surface_config` in `simple` / `bh` / `ga` slots.
- `AdsorbateDefinition` is now a frozen dataclass (was a `TypedDict`). Call sites
  use attributes; boundary values may still be plain dicts and are coerced via
  `as_adsorbate_definition` / `AdsorbateDefinition.from_dict`.
- The NEB energy-profile pre-screen (`validate_initial_neb_energy_profile`) is
  gated on `max_endpoint_mismatch`: it runs only when `max_endpoint_mismatch` is
  set. Bare `gas_cluster` (which leaves `max_endpoint_mismatch=None`) skips the
  energy-profile screen, while the interior-image clash check
  (`validate_initial_neb_path`) remains universal across all system types.
- Surface TS presets enable `max_endpoint_mismatch=1.25` (was unset).
- TorchSim native autobatcher / OOM re-probe behavior hardened for GPU runs.
- MLIP extras require ``torch-sim-atomistic==0.6.1`` (MACE extra also pins
  ``nvalchemi-toolkit-ops[torch]==0.4.1``; UMA extra needs
  ``fairchem-core>=2.20.0``).
- Timing payloads require a `kind` discriminator
  (`timing_report.relax_seconds_from_timings`).
- `import scgo` no longer sets `PYTORCH_CUDA_ALLOC_CONF` at import time; call
  `scgo.configure()` (or any `run_*` entry point, which calls it) to apply the
  allocator setting.
- Consolidated duplicated template/initialization logic into the shared
  `scgo/initialization` helpers. Behavior-preserving; no public API or default
  changed.
- `_LOW_EFFORT_NEB_FLOOR_BARE` and `_LOW_EFFORT_NEB_FLOOR_ADSORBATE` collapsed
  into a single `_LOW_EFFORT_NEB_FLOOR` (both were `1000`).
- `_write_relaxed_candidate`'s constraint-index-derivation fallback now catches
  only expected validation/lookup errors; fatal errors such as `MemoryError` /
  `RecursionError` propagate.
- Untagged gas-cluster mirror mutation is omitted (isometry); surface mirror
  uses `surface_normal_axis` and re-anchors the mobile region to the slab.
- Per-pair parallel NEB `timings_s` keys renamed to `total_wall_avg_s` /
  `neb_optimization_avg_s` / `cpu_non_relax_avg_s` (no `neb_optimization_s`
  alias).
- Connectivity validation consolidated into `validate_connectivity_policy`; bare
  `gas_cluster` is now connectivity-checked.
- Parallel NEB atom-budget batching, force double-count fix, and cell+PBC image
  dedupe.
- CI marker contract tightened; PR matrix shrinks UMA/UPET to fast-only; Kaggle
  GPU supports `mode=smoke` (default/weekly) and `mode=full` (manual).
- Changing `calculator` vs GO defaults (or vs inherited GO settings in
  `initialize_ts_params`) replaces `calculator_kwargs` wholesale instead of
  nested-merging backend keys (e.g. UMA `task_name` no longer leaks into MACE).
- BH/GA slot `diversity_max_references` / `diversity_update_interval` default
  to `None` (inherit top-level). `get_diversity_params` stamps both top-level
  and slots.
- TorchSim GA attach uses `max_steps=None`; the GA assigns `relaxer.max_steps`
  from `niter_local_relaxation` at run time.
- Low-effort UMA/UPET builders attach the TorchSim relaxer after `model_name` /
  `version` / `task_name` and set top-level `n_jobs=1`.
- `resolve_run_id_from_db_path` returns `None` (callers skip the database)
  instead of falling back to the filename basename.
- Top-level `surface_config` in `go_params` / `ts_params` is enough when the
  run argument is omitted (must still agree when both are set).
- Parallel NEB persists each finished chunk immediately; energy-screen forces
  are reused at step 0 instead of a second `relax_batch`. Serial NEB runs the
  full-band single-point only when the energy-profile gate is enabled.
- Basin-hopping default `temperature` is now `1.0` eV (ASE-style Metropolis
  energy scale), not `500 * k_B`. `get_high_energy_params()` uses `2.0` eV.
  Optimizer startup logs print the eV scale only (no misleading Kelvin).
- GA mutation operators are stamped with the run-resolved `connectivity_factor`
  and mobile-connectivity gates honor it (was hard-coded module default).
- MACE `torch.load` patch is no longer process-wide; TorchSim MACE model load
  uses the same scoped context manager.
- TorchSim sticky-max_metric / CUDA-OOM retry paths share
  `_run_with_max_metric_retry` for single-point and relax.
- Sorted-distance fingerprint cache uses a cheap geometry dirty token before
  falling back to byte hashing; `filter_unique_minima` annotates `raw_score`
  only on unique survivors.
- `InitDiagnosticsCollector.emit_summary` clears accumulated records after
  copying; batch init always resets the collector when diagnostics are on.
- `attach_fix_bond_lengths` replaces any prior `FixBondLengths` instead of
  stacking duplicates; rigid adsorbate restore sets positions with
  `apply_constraint=False`.
- Seed clash checks and `steric_deficit` are vectorized (`cdist` /
  BLMIN matrix); slab-search config rebuild uses `dataclasses.replace`.
- Hierarchical surface deposition uses one inner placement attempt per outer
  retry (avoids nested attempt explosion).
- `ConnectivityFactorInput` accepts `numbers.Real` (including NumPy scalars).
- `run_go_ts_campaign` derives a reproducible per-composition sub-seed (same
  pattern as `run_go_campaign`).
- Seed-sampling “tried positions” tracking uses stable position hashes
  (`_get_positions_hash`) instead of `hash(...tobytes())`.
- Shared `emit_timing_data`: append to `timing_collector` when set; write
  `timing.json` when `write_timing_json` is true. BH collectors are populated
  independent of the write flag; GA writes `output_dir/timing.json` even when a
  collector is also set.
- Fragment reposition mutation uses a capped placement budget
  (`min(max(attempts * 3, 80), 400)`) and one pass over `(ca, relaxed)` instead
  of up to 16 nested retries.
- Graphite slab `vacuum` is total normal padding for all layer counts
  (`cell_z = (layers - 1) * 3.35 + vacuum`, stack centered).
- Graphite and graphene preset defaults now match the HOPG 5×5 × 3-layer slab
  (150 C atoms, ~12.3 Å hexagonal in-plane vectors, 30 Å total vacuum /
  ~36.7 Å cell height). Previous defaults were graphite 5 layers × 4×4 with
  12 Å vacuum and graphene 6×6 with an 18 Å cell. Call the generic
  `make_*_surface_config` builders with explicit `slab_layers` /
  `slab_repeat_xy` / `nx` / `ny` to recover a smaller cell.

### Fixed

- ``prepare_neb_endpoints`` passes full-slab ``n_slab`` plus
  ``n_slab_deposit=n_fixed`` for slab-search systems (NEB ``n_slab`` alone is the
  frozen prefix). Without that, empty-core ``surface_adsorbate`` NEBs failed with
  ``adsorbate_fragment_lengths must sum to mobile adsorbate length``.
- After loading minima, slab-search adsorbate runs infer ``n_core_mobile`` as the
  searchable top-layer atom count so NEB layout is ``[fixed | top | adsorbate]``.
- ``surface_cluster`` / bare ``surface`` TS presets allow temporary mobile
  fragmentation on NEB paths (``allow_cluster_fragmentation=True``), widen the
  endpoint-displacement gate for supported clusters (``2.5`` Å), and use a
  softer clash / higher spurious-barrier floor for bare-slab vacancy hops.
- ``_adsorbate_max_displacement`` accepts ``n_adsorbate`` so surface-adsorbate
  slab searches measure OH/site hops on the trailing adsorbate block only
  (not the whole searchable top layer). Without that, every N-doped OH pair
  failed the ``max_endpoint_mismatch`` hard gate.
- DB discovery no longer rejects slab+mobile TS loads when run-dir metadata
  stores a mobile-only formula (e.g. ``Pt5`` vs ``C150Pt5``); atom composition
  is checked instead of treating the metadata formula as an exclusive gate.
- ``surface_adsorbate`` ``max_endpoint_mismatch`` default is ``3.0`` Å (was
  shared ``1.5`` with cluster+adsorbate). Pair selection gates that type on
  adsorbate Cartesian hop, and graphite hollow/bridge site hops are ~2.5 Å, so
  the old gate rejected every candidate pair in the N-doped OH example.
- `run_go_ts_campaign` / `run_ts_campaign` / `run_go_campaign` resolve
  top-level ``params['surface_config']`` the same way as single-run APIs, and
  allow empty compositions when ``system_type`` is a slab search target.
- NEB resume only skips pairs with ``status=success``, ``neb_converged=true``,
  and a readable ``ts_*.xyz``; missing ``neb_converged`` is treated as not
  resume-ready. Surface geometry demotions rewrite pair metadata so stale
  success cannot be resumed. Parallel finalize clears ``neb_converged`` when
  status is failed.
- Surface crossover clash checks now use cached slab-image distance screening
  plus mobile-only no-copy clash helpers in `CutAndSplicePairing`, keeping
  acceptance logic equivalent to ASE-GA while reducing inner-loop overhead.
- `Population.update(new_cand=...)` now supports in-memory relaxed batches
  (including empty batches), filters by run-id and GA eligibility, syncs
  `already_returned`, and avoids redundant DB round-trips.
- Gas-phase TS pair selection and NEB endpoint prep share one core overlay:
  fingerprint correspondence, Kabsch (translation-only for a single core atom),
  spatial rematch in the overlaid frame, then re-Kabsch. That rematch recovers
  proper labels when fingerprint Hungarian assigns a reflected correspondence
  on near-symmetric cores (e.g. Pt5 TBP equatorials), so Cartesian RMS can
  fall below ``pair_core_rms_max``. The same overlay runs for bare
  ``gas_cluster`` NEB, not only adsorbate blocks. Adsorbate matching runs after
  the overlay. Slab cores stay in the surface lab frame.
- Surface PBC alignment is used only for a slab prefix or slab-like 2D
  periodicity. Gas vacuum boxes with ``pbc=True`` and ``n_slab == 0`` still
  3D-Kabsch.
- ``update_mutation_weights`` always maps operator names through
  ``_effective_operator_weight``, so partitioned ``in_plane_slide`` /
  ``_core`` / ``_ads`` variants share the table budget 70/15/15 instead of
  giving the unscoped slide the full mass on top of the scoped variants.
- ``FragmentRepositionMutation`` forwards the run-stamped ``connectivity_factor``
  into its mobile connectivity gate (e.g. ``Pt-C: 1.8`` on graphite).
- ``MirrorMutation`` steric-best rescue reanchors the mobile region to the slab
  before clash/connectivity checks, matching the main ranked-candidate loop.

- GA default mutation weights are keyed by ``system_type`` (not whole-composition
  element counts), so adsorbate runs no longer inflate rattle/overlap with unused
  ``permutation``/``shell_swap`` mass. ``fragment_reposition``, ``in_plane_slide``,
  and ``in_plane_rotate`` have first-class weights; surface adsorbate runs register
  ``fragment_reposition`` and whole-mobile orientation operators.
- Adsorbate crossover mixes tag-rigid fragments from either parent for
  ``*_cluster_adsorbate`` types; ``surface_adsorbate`` keeps the mobile slab on
  parent 0 while adsorbate fragments mix.

- Tagged ``flattening_ads`` / ``flattening_core`` keep the subset atom that
  contacts the leftover mobile atoms fixed, so flattening around the subset COM
  cannot pull a barely-connected adsorbate off the core.
- ``MirrorMutation`` no longer returns the steric-best core reflection when that
  geometry disconnects the adsorbate; clash, identity, and connectivity are
  checked across the ranked plane set before accepting a candidate.
- ``FragmentRepositionMutation`` no longer reanchors the whole mobile region to
  the slab: the core is not moved, and that reanchor lifted the core off the
  surface whenever the fragment became the lowest atom.
- GA batch writers reset eligible/ineligible counters at the start of each
  ``database_retry`` attempt so a mid-batch SQLite lock rollback cannot
  double-count outcomes relative to ``ga_eligible`` tags in the database.
- Database retry logging is unified: ``retry_transaction`` delegates to
  ``database_retry``; attempt/recovery lines use %-style messages at DEBUG
  (WARNING only for connection-open via ``retry_on_lock``); final failure
  remains ERROR.
- UMA/UPET CPU CI no longer errors collecting ``test_mace_torch_load_patch``:
  ``mace_helpers`` imports ``mace`` at module load, so the test is now marked
  ``requires_mace`` and imports the helper inside the test (``pytest -m`` still
  imports unmarked modules during collection).
- Bare TS system types no longer oversample pair selection when
  ``max_endpoint_mismatch`` is set; ``resolve_ts_pair_select_cap`` oversamples
  only for adsorbates (IDPP re-rank), and the runner always truncates to
  ``max_pairs`` before NEB (fixes Pt5/graphite selecting 50 pairs at
  ``max_pairs=6``).
- `run_go` honors the single `n_jobs` knob for GA population init, offspring, and
  validation.
- GA TorchSim relaxed DB rows and TS result JSON now persist `FixAtoms` /
  `FixBondLengths` (native ASE constraints plus index-list metadata tags).
- ASE `FixBondLengths` are honored during batched TorchSim relaxation via
  `scgo.calculators.torchsim_constraints.TorchSimFixBondLengths`.
- `TorchSimFixBondLengths.to(device)` no longer routes CUDA tensors through
  `np.asarray` (which raised during torch-sim `initialize_state` / state moves
  for adsorbate runs with frozen bond lengths). Pair indices are also packed on
  InFlight pop so remaining bonds stay valid after systems leave the batch.
- Low-level `scgo()` and TS campaign path keys coerce plain-dict
  `adsorbate_definition` values to `AdsorbateDefinition`; empty-core gas
  adsorbate GO again noops (`[]`).
- `run_trials` accepts `search_mobile_count` so slab-target dedupe uses the true
  trailing-mobile atom count instead of `len(composition)`.
- Kaggle runner falls back to the system Python interpreter when conda is absent
  and resolves the dataset mount path without a network fallback.
- `_prepare_atoms_for_metatomic_torchsim` zeroes only the lattice-vector *row*
  of each non-periodic cell direction (not the column), fixing partially-periodic
  slabs with `surface_normal_axis` 0/1.
- `is_true_minimum` derives vibration indices from `FixAtoms` and passes them to
  ASE `Vibrations(..., indices=...)` (ASE 3.22–3.28); all-fixed structures skip
  the Hessian check.
- Slab-search GA stays legal on layered crystals and frozen adsorbates.
- Silent TorchSim GPU degradation fixed: cached scaler reuse, real OOM retry,
  and stricter CI.
- Constrained TorchSim GPU GO no longer crashes when FIRE states still carry an
  autograd graph into the autobatcher split (``FixAtoms`` inplace on
  ``SplitWithSizes`` views). SCGO now requires ``torch-sim-atomistic==0.6.1``
  (upstream ``detach_state_graph`` in ``_chunked_apply``, #590) and detaches
  model outputs per TorchSim's ``ModelInterface`` contract so later InFlight
  pops stay safe. Constraint classes are unchanged.
- Parallel NEB no longer evaluates forces twice per step; `force_calls` is no
  longer double-counted; non-finite NEB forces mark the band failed.
- DB discovery no longer memoizes path lists (empty or non-empty); registry hits
  merge with a filesystem scan, and registration clears the process discovery
  singleton so same-process GO→TS reload sees the current ``ga_go.db``.
- e3nn `torch.load` `weights_only` protected at MACE import time (torch>=2.6).
- UMA/UPET CI no longer loads MACE; MLIP stacks detected by installed extra.
- SQLite connections closed on read paths (silences `ResourceWarning`).
- Corrected docstring of `validate_adsorbate_placement_feasibility` (raises
  `SCGOValidationError`, not `ValueError`).
- Low-effort NEB `neb_steps` floor raised to `1000` for bare system types so
  bands still converge for CI interior-TS assertions.
- GA generational-loop regression from the Phase 3 perf rewrite fixed.
- `setup_database(..., remove_existing=False)` no longer writes a second
  `simulation_cell=True` template row (ASE GA `assert len(rows) == 1`);
  a stored stoichiometry that disagrees with `atoms_template` raises.
- Chunked DB streaming loads each id batch with one `WHERE id IN (...)`
  (falls back to per-id `get_atoms` if ASE's private row decoder is missing).
- GO/TS param copies no longer `deepcopy` TorchSim relaxers when injecting
  `surface_config`.
- TorchSim GA `_read_candidate_batch` uses indexed `relaxed` / `queued`
  filters instead of a full-table `select()` scan; dense BLMIN prefilter
  thresholds are cached by atomic-number set.
- Permutation mutation samples distinct swap pairs without replacement.
- Chunked DB streaming no longer prefills unused ASE row slots with empty
  JSON/`null` before column remapping.
- Near-surface deposition tilt azimuth is drawn independently of in-plane spin.
- Adsorption height sampling is truncated-uniform on
  `[h_min, min(h_max, connectivity_threshold)]` (no biased nest of min/max).
- Empty-core deposits run `validate_supported_cluster_deposit` instead of
  returning early.
- GA `_BLMIN_THRESH_CACHE` is cleared each generation so recycled `id(blmin)`
  values and empty prefilter dicts cannot keep stale thresholds.
- `_write_relaxed_candidate` copies the relaxed cell with `scale_atoms=False`
  before setting positions (avoids unintended fractional rescaling).

### Removed

- Identity keys (`system_type`, `surface_config`, `adsorbate_definition`,
  `adsorbate_fragment_template`, `cluster_adsorbate_config`) are rejected inside
  `optimizer_params` slots. Use run arguments and/or top-level `go_params` keys.
- Removed `scgo.calculators.vasp_helpers` / `orca_helpers`
  (`write_vasp_inputs`, `write_orca_inputs`).
- Removed `MemoryScalerCache` / `get_global_memory_scaler_cache` from
  `scgo.calculators` public lazy exports.
- Narrowed `scgo.database` public `__all__` (registry/stamp helpers no longer
  re-exported from the package root).
- Removed dead code paths and unused internal aliases/helpers left over from the
  initialization refactor. No other public API was affected.
- Removed the redundant final `unique_minima.sort(...)` in `filter_unique_minima`.
- Deleted `build_connectivity_graph` and related helpers from `ts_network`.
- Removed the convex-hull site-capacity heuristic (and `_proxy_core_from_symbols`)
  from `validate_adsorbate_placement_feasibility`.
- Dropped the no-op `ClusterAdsorbateConfig` rebuild in slab-fragment deposition.

### Docs

- Documented TS pair-selection budget / adsorbate oversampling
  (``max_pairs`` vs ``resolve_ts_pair_select_cap``) under **Pair selection** in
  ``parameters.rst`` and ``validation_and_constraints.rst``.
- Documentation builds with zero Sphinx warnings under `nitpicky=True`.
- Documented `n_jobs`, vacancy / hollow sites, NEB energy-profile gating,
  `calculator_for_global_optimization`, and mirror-mutation omissions.
- Documented calculator-change kwargs replacement, TS vs GO pair-correlation
  caps, params-only `surface_config`, NEB pair resume, and database reuse.
- Documented BH `temperature` as a Metropolis energy scale in eV (default
  `1.0`; high-energy preset `2.0`), not a physical Kelvin temperature.
- Fixed documentation inaccuracies, normalized headings, and cleaned up
  cross-references.
- Plain-language glossary / mental-model cleanup (cluster/core, adsorbate,
  slab) across README and guides; surface guide notes defect-bias default.

## 0.7.0

### Added

- :mod:`scgo.metadata` package for structure tags, run-dir records, DB stamps,
  and output-JSON provenance (separate schemas under one namespace).

### Changed

- Structure tags live only in ASE ``key_value_pairs`` via ``set_tags`` /
  ``get_tag`` / ``get_tags`` / ``filter_by_tags``; remove the old
  ``atoms.info["metadata"]`` / ``provenance`` bags, ``add_metadata`` /
  ``get_metadata`` / ``update_metadata`` / ``filter_by_metadata``, and
  ``persist_provenance``.
- Move DB stamp helpers from ``scgo.database.schema`` to
  ``scgo.metadata.db_stamp`` (e.g. ``stamp_db``, ``is_scgo_db``,
  ``CURRENT_DB_SCHEMA_VERSION``); final-minima SQL tagging to
  ``scgo.metadata.persist``; run-dir tracking from ``utils.run_tracking`` to
  ``scgo.metadata.run_dir``; output-JSON provenance from ``utils.ts_provenance``
  to ``scgo.metadata.provenance``.
- ``simple`` optimizer writes ``simple_go.db`` (was incorrectly ``bh_go.db``);
  discovery still matches ``*.db``.
- Remove surface/cluster height cross-aliases (``height_*`` /
  ``adsorption_height_*``); each config keeps only its canonical names.
- Slim ``runner_api`` to the public run surface; tests patch definition modules.
- Per-shape ``generate_*`` template helpers are private; public API is
  ``generate_template_structure``.
- Import mutations from ``scgo.ase_ga_patches.mutations`` (``standardmutations``
  re-export removed).
- Candidate discovery prefers component path keys; packed ASE stems
  (e.g. ``H2O2Pt5_searches``) and pure-metal formulas still parse.
- Drop TorchSim ``max_steps=0`` warning/log filters (single-point uses
  ``ts.static``).
- Deduplicate UPET ``HAS_NVALCHEMIOPS`` disable into
  ``disable_metatomic_nvalchemiops``; extract UMA/UPET model-name infer helpers
  mirroring MACE; collapse ``retry_on_lock`` onto ``database_retry`` (drop
  unused ``PRESET_CONSERVATIVE``); DRY connection pragma apply; share
  ``_assign_penalty_energy`` in ``SCGODataConnection.add_relaxed_step``;
  simplify TorchSim step extraction; remove ASE Spacegroup deposition warmup.
- Drop package-level re-export of rigid adsorbate helpers (import from
  ``scgo.cluster_adsorbate.rigid``; avoids a ``system_types``/``surface`` cycle);
  drop duplicate GO ``rng`` validation and centralize seed fallback in
  ``_resolve_go_seed``; move TS defaults / ``SystemPolicy`` consistency checks
  into tests (no import-time assert); calculator registry uses
  ``functools.cache``.
- Kaggle GPU example suite shares stricter e2e artifact bars (run metadata,
  SCGO DB stamp) and requires TS candidates for cluster-bearing system types.

### Maintainer notes

- Temporary workarounds for upstream bugs (remove once fixed upstream):
  - TorchSim constraint device patch
    (``_patch_torchsim_constraint_device_mismatch``): 0.6.1 still does bare
    ``torch.isin`` with no CPU/CUDA ``atom_idx`` align in
    ``AtomConstraint.select_sub_constraint``.
  - TorchSim model-output detach (``_patch_torchsim_model_detach_outputs``):
    0.6.1 detaches in ``_chunked_apply`` but still ``pop``/splits InFlight
    states before detaching completed systems. Drop when upstream detaches
    before ``pop``.
  - ``HAS_NVALCHEMIOPS = False`` (``disable_metatomic_nvalchemiops``):
    metatomic-torchsim nvalchemiops CUDA NL still fails on non-cubic
    gas-phase cells; vesin path is the reliable fallback.
  - Kaggle/CI ``vesin==0.6.0`` force-install: ``metatomic-torchsim`` 0.1.4
    still declares ``vesin<0.6`` but needs ``NeighborList(skin=...)`` from
    0.6.0.
  - ``pytest.ini`` ``filterwarnings``: dependency noise (torch.jit / MACE /
    nvalchemiops alias / warp ``.grad`` / ASE EMT·NEB coincidences / e3nn);
    drop entries individually when upstream stops emitting them.
- Timing ``"trials"`` guard rejects legacy multi-trial ``timing.json`` shapes
  (intentional API protection, not an upstream workaround).
- ``select_structure_pairs(..., *, use_mic: bool)`` is required (no
  ``None``→``surface_aware`` default); pass ``use_mic=resolve_neb_mic(...)``.
- Low-level ``run_transition_state_search``: omitted ``energy_gap_threshold`` /
  ``neb_n_images`` / ``neb_climb`` resolve from TS presets (not the old
  hardcoded ``1.0`` / ``3`` / ``False``).
- ``scgo.runner_api`` no longer re-exports internals (``run_trials``,
  ``_run_go_trials``, etc.); import from definition modules.
- ``_run_go_trials`` / ``_run_go_campaign_compositions`` expect
  already-initialized params (``params_already_merged`` removed; merge lives
  in ``runner_params``).

## 0.6.5

### Added

- `build_torchsim_relaxer` factory for shared UMA → UPET → MACE TorchSim
  relaxer construction from a live calculator (GA). Presets still construct
  `TorchSimBatchRelaxer` directly when `model_kind` is already known.
- `validate_and_resolve_run_context` shared BH/GA preamble
  (policy, connectivity factor, fitness strategy).

### Changed

- Path-key resolution consolidated in `scgo.utils.path_keys.resolve_run_path_key`
  (importable from `runner_params`); GO/TS/minima search use the same helper.
- `run_go_campaign` result dict keys are always `path_key` (including failed
  compositions); gas-cluster keys still match the formula.
- GA `create_mutation_operators` uses a shared partitioned-mutation helper for
  flattening / breathing / in-plane slide core/_ads variants (names/weights unchanged).
- Drop unused aliases: `retry_with_backoff` (use `database_retry`),
  `assert_adsorption_height_in_bounds`, Kaggle `_install_scgo_mace`, and
  streaming `_relaxed_rows_where_clause`.
- QC polish: remove dead NEB helpers, deduplicate skip/endpoint/autobatcher paths,
  simplify GA early-stopping and CutAndSplice construction, trim trivial helper
  docstrings, document UPET model list in API docs.

### Fixed

- Bare `surface` / `surface_adsorbate` empty-core composition is accepted by
  `run_go` / `run_go_ts` / GA / TS (examples with `COMPOSITION=[]`).
- Slab-search TS uses fixed-bottom `n_slab` (not full slab length) so mobile
  top layers remain comparable.
- Adsorbate-only deposition on planar graphite: planar site fallback when the
  3D convex hull is empty, and skip whole-slab connectivity checks that reject
  van der Waals stacked layers.
- Penalty-energy path attaches a `SinglePointCalculator` so later energy/force
  reads do not hit a broken calculator.
- BH `_move_atoms`: single tag groups displace rigidly; empty movable sets log
  `Moved_atoms: none`; adsorbate-scaled moves no longer throttle core; single-
  atom descriptions are bracketed for ASE DB compatibility.
- `iter_databases_minima(max_structures=0)` yields nothing (`0` is not treated
  as unlimited).

### Maintainer notes

- Temporary workarounds for upstream bugs still in place at this release:
  TorchSim constraint device patch; `HAS_NVALCHEMIOPS = False`; Kaggle
  `vesin` force-install; `max_steps=0` warning filters;
  `standardmutations` re-export; `pytest.ini` filters. Timing `"trials"`
  guard is intentional API protection, not an upstream workaround.

## 0.6.4

### Added

- `surface` and `surface_adsorbate` system types: GA/BH search mobile
  top slab layers (bottom layers fixed), with optional adsorbates and no
  cluster core. Includes slab search partition helpers, defected/N-doped
  graphite presets, and examples.

### Changed

- On-disk path keys for searches, TS results, XYZ prefixes, and default
  campaign stems are component-aware: nanoparticle, each adsorbate fragment,
  then surface name (e.g. `Pt5_OH_OH_graphite`). Chemical composition
  matching still uses ASE-style formulas (`H2O2Pt5`).
- `SurfaceSystemConfig.name` (default `"slab"`) supplies the surface
  path-key segment; `make_graphite_surface_config` sets `name="graphite"`.

### Fixed

- Ruff import sorting / formatting leftovers from the surface-search merge
  so the GitHub Actions lint job passes on main.

## 0.6.3

### Fixed

- UPET/UMA TS preset tests expect `use_parallel_neb=True`, matching the
  0.6.2 default (fixes GitHub Actions UPET CI jobs).

## 0.6.2

### Changed

- NEB plumbing: serial and parallel runners share `NebRunConfig` and
  `prepare_neb_endpoints` (copy / FixAtoms / validate). Public
  `run_transition_state_search` kwargs are unchanged.
- BH surface post-relax framing is owned by `perform_local_relaxation`
  (`surface_mode` / `n_slab`); diversity scoring uses mobile composition.
  GA soft-fail storage validation wraps shared
  `canonicalize_and_validate_for_storage`.
- Structure MIC reads go through `resolve_structure_mic` /
  `resolve_neb_mic`. Pair selection takes explicit `use_mic` (scoring
  regime stays `surface_aware`). Empty-core adsorbate NEB dims use shared
  `resolve_neb_mobile_dims`.
- TS presets: `neb_fmax` / `torchsim_fmax` are shared at `0.20` for
  every system type; `use_parallel_neb=True` is the default everywhere
  (including the low-level `run_transition_state_search` signature).
  Surface types set `parallel_neb_max_bands=1` so large slab cells do not
  GPU-OOM; the parallel runner is still used, with bands chunked
  one-at-a-time (and CUDA cache cleared between chunks). Bare surface NEB
  step budget rises to `2000`. Supersedes the 0.6.1 surface-adsorbate
  `neb_fmax=0.25` / serial-NEB defaults. Removed unused
  `torchsim_batch_size` from TS presets (OOM safety is band concurrency).
- Adsorbate TS pair oversample is `min(max_pairs * 10, max(max_pairs, 50))`;
  pair selection reuses one structure comparator and avoids full Atoms
  slices for core-RMS.

### Fixed

- Parallel NEB refuses FIRE steps when band fmax is non-finite (e.g. ASE
  `improvedtangent` on a flat energy profile), marking the band failed
  instead of propagating NaN geometries.
- Unify structure MIC / surface-awareness across GO, GA, BH, and TS: Pure
  comparator honors `mic` literally under PBC; `SurfaceSystemConfig`
  defaults `comparator_use_mic=True` (GO/GA/BH via `resolve_structure_mic`).
  TS pair scoring uses `uses_surface` for the scoring regime and
  `resolve_neb_mic` (`neb_force_mic`) for geometry / minima dedupe — not
  the comparator knob. BH uses mobile `n_top` and skips COM recenter on
  surfaces; empty-core adsorbate enables blockwise NEB dims; core-RMS pair
  gate is permutation-invariant; GO final XYZ alignment forwards
  `n_core_mobile`.

## 0.6.1

### Fixed

- TorchSim NEB/TS single-point force evaluations now use `torch_sim.static`
  instead of `optimize(max_steps=0)`. The old path still took one FIRE step
  (wrong forces at displaced geometries) and spammed
  `All systems have reached the maximum number of steps: 0` via torch_sim's
  logger in production. Batched NEB spring/climb/tangent physics remain ASE's.
  Single-point calls default to `autobatcher=False` so TorchSim does not
  re-probe GPU memory on every NEB force evaluation.
- Parallel/serial TorchSim NEB finalize no longer fails with
  `The property "energy" is not available` after a final FIRE step: PES is
  refreshed at the final geometries, and energies are also cached in atoms
  metadata when attaching SinglePoint results.
- ASE `Atoms.copy()` shallow-shares nested `info` dicts; TorchSim
  single-points writing `raw_score` were corrupting minima reused by later
  NEB pairs (multi-eV bogus product energies). Endpoint/path copies now isolate
  nested metadata, and static result mapping uses `copy_atoms`.
- Surface-adsorbate NEBs no longer apply free in-plane Kabsch rotation (breaks
  adsorbate–slab registry). Cell remap / MIC remain on. Pre-NEB band checks
  reject aligned endpoint energy drift `> 0.5` eV vs canonical minima, and
  one-sided interior maxima with prominence `< 0.40` eV (slides that CI-NEB
  collapses to an endpoint).

### Changed

- Adsorbate TS presets (`gas_cluster_adsorbate` / `surface_cluster_adsorbate`)
  now use climbing NEB, spring `0.5`, `neb_steps=4000`, 7 images, a tighter
  `energy_gap_threshold` (`0.75` eV), and a hard `max_endpoint_mismatch`
  pair gate. Gas adsorbates use `neb_fmax=0.20` with `use_parallel_neb=True`;
  surface adsorbates use `neb_fmax=0.25` with serial NEB (avoids GPU OOM on
  large slab cells and matches attainable MLIP force convergence). Fragment-wise
  adsorbate matching, core-anchored endpoint alignment, and pre-NEB
  clash/discontinuity rejection improve path quality for multi-fragment
  adsorbates. Surface-adsorbate presets set
  `neb_surface_lattice_rotation=False`; pair selection also skips tiny
  adsorbate hops (`max_diff < 0.20` Å) that are usually barrierless slides.
- Parallel NEB no longer overwrites batch failures (e.g. CUDA OOM with
  `force_calls=0`) as `endpoint as TS` during finalize.
- Pre-NEB path/energy rejects in parallel NEB are recorded as `skipped`
  (consistent with structure-validation skips and the serial path), not
  `failed`.
- Provenance `scgo_version` now reads the in-tree version
  (`scgo._version`) so editable checkouts are not stuck on stale
  `dist-info` after a bump.
- Adsorbate NEBs reject IDPP bands with absurdly high barriers
  (`> 8` eV; likely discontinuous) before optimization, and use two-stage
  CI-NEB (relax without climb, then climb). Stage 2 always runs and keeps at
  least half the step budget.
- Parallel two-stage CI-NEB always runs the climb stage after no-climb
  relaxation when used, even if stage 1 already met `fmax`.
- Two-stage climb is skipped for endpoint-max / barrierless IDPP bands
  and for soft interior maxima (IDPP barrier `< 1.0` eV); climb starts
  immediately. A no-climb pre-relax was collapsing those MEPs so finalize
  reported `endpoint as TS` for adsorbate OH hops.
- Finalize rejects NEB results with barrier `> 8` eV (same discontinuous
  threshold as the pre-NEB IDPP energy gate).
- Adsorbate pair selection now prefers activated hops (moderate endpoint
  mismatch and core RMS) over near-isomer slides, oversamples candidates, and
  re-ranks by IDPP profile so NEB budgets favor robust interior maxima (and
  skip endpoint-max slides when any robust bands exist).
- TS minima deduplication for adsorbate systems uses core+adsorbate mobile
  count (matching GA `n_to_optimize`), not core-only length.

## 0.6.0

### Added

- UPET MLIP backend (`[upet]` extra) via metatomic-TorchSim, with CI matrix
  coverage alongside MACE and UMA, plus Kaggle GPU suites for MACE/UPET.
- Height aliases on surface and cluster-adsorbate configs: surface accepts
  `height_*` as aliases for `adsorption_height_*`; adsorbate configs accept
  `adsorption_height_*` as aliases for `height_*`. Conflicting values raise
  `SCGOValidationError`.
- Shared helpers: :mod:`scgo.calculators.torch_device`,
  :mod:`scgo.utils.config_aliases`, :mod:`scgo.utils.combine_atoms`.
- GO top-level parameter allowlist (including `validation_n_jobs`); unexpected
  keys raise `SCGOValidationError` with the expected set.

### Changed

- Split the large `runner_api` module into focused modules
  (`runner_composition`, `runner_params`, `runner_go`, `runner_ts`) while
  keeping the public `scgo.runner_api` / `scgo` import surface stable via
  re-exports (including names used by test monkeypatches).
- Split ASE GA `standardmutations` into
  :mod:`scgo.ase_ga_patches.mutations` (one module per family); the old import
  path remains a thin re-export.
- Unsupported Torch devices warn once and raise `SCGOValidationError` instead
  of silently falling back to CPU (MACE / UMA / UPET / TorchSim paths).
- `SCGOValidationError` no longer logs at ERROR on construction. Runner API
  entry points log validation failures at the prepare boundary; campaign and
  pair handlers catch `SCGOValidationError` and continue where appropriate.
- Top-level `surface_config` in `go_params` / `ts_params` is allowed and
  fanned into optimizer slots; only `system_type` remains rejected in params
  (use the run-function argument). Adsorbate placement knobs stay in
  `go_params`.
- Surface slab constraint attachment preserves non-`FixAtoms` constraints
  (e.g. `FixBondLength`). Multi-fragment hierarchical placement keeps sites
  on the original metal core.
- Parallel NEB skips re-evaluating endpoints after step 0 and uses a clearer
  max-atom force metric; force attachment requires forces.

### Fixed

- Restore auto GA scaling in the TorchSim preset.
- Align concurrent DB stress tests with production retry policy.
- Handle `SCGOValidationError` in growth, GA, and initialization fallbacks
  (and in GO campaign / TS pair error paths).

## 0.5.2

### Added

- Verbosity-level logging for GA runs with v1 phase headers and aggregated
  initialization/generation summaries, v2 per-individual detail. New
  :func:`~scgo.configure_logging` helper and
  :class:`~scgo.utils.phase_logging.InitDiagnosticsCollector` for batched
  initialization messages. Standardized %-style logging across runners and
  TS code paths.
- Typed parameter dicts: :class:`~scgo.system_types.GLOptimizerParams` and
  :class:`~scgo.system_types.TSParams` TypedDicts for GO and TS parameters,
  with :class:`~scgo.system_types.CalculatorKwargs` and
  :class:`~scgo.system_types.OptimizerSlotParams` for nested configuration.

### Changed

- Adsorbate/core partition reconciliation now routes through all runner paths
  via centralized `resolve_adsorbate_run_composition`, sharing the same
  core/adsorbate stripping logic across gas and surface runs, `run_go`,
  campaigns, GO+TS, and TS entry points.
- Simplified adsorbate/core reconciliation logic: use list-based stripping,
  drop redundant count checks, consolidate test coverage.
- Deduplicated candidate-discovery path filtering via shared path relevance
  helper, cleaning up parse/filter branches while preserving unparseable-path
  accounting.
- Hardened initialization fallback chains with coherent seed+growth behavior,
  magic-number tolerance for near templates, aligned radii usage in placement,
  and targeted logging/regression tests to prevent silent skips.
- Improved initialization logging: grouped seed-sampling failures into single
  INFO summaries with specific reasons; compact, consistently formatted placement
  error messages for large runs.
- Hardened database operations: production retries for reads, connection opens,
  structure extraction, and count queries via unified `retry_on_lock` /
  `database_retry` machinery; IMMEDIATE isolation for final-minima tagging;
  backoff on transient lock/I/O OperationalErrors; retry actual SQLite open
  during setup; log stamp failures instead of suppressing them.
- Aligned database retry logic: `database_retry` now only backs off on
  transient lock/I/O OperationalErrors, matching `retry_on_lock` and
  `retry_transaction`; shared retried `DataConnection` factory between
  `setup_database` and `get_connection`.
- Hardened composition parsing with explicit errors for empty and unknown
  symbols; expanded regression tests covering `HO2Ru9W2` adsorbate resolution
  and edge cases.
- Made compact formula parsing unambiguous: use ASE `Formula` with required
  chemical capitalization for multi-element strings; allow lowercase only for
  unambiguous single-element forms (`pt3`); reject ambiguous cases (`ho2`,
  `cu`, `pt3au`) with actionable errors; comma-separated symbols remain the
  fully unambiguous input format.
- Validation and configuration failures across SCGO now raise typed exceptions
  (`SCGOValidationError`, `SCGORuntimeError`, etc.) instead of bare
  `ValueError` / `RuntimeError`. Downstream code should catch
  `SCGOValidationError` (or `SCGOError`) rather than `ValueError`.
- `SCGOValidationError` is logged at ERROR when logging is configured
  (construct-time logging in 0.5.2; superseded in 0.6.0 by runner-boundary
  logging).
- Preset dicts are documented as `GLOptimizerParams` and `TSParams` TypedDicts;
  default GO params template is cached via `@cache`.

### Fixed

- MACE import on PyTorch 2.6+: patch `torch.load` before `mace`/e3nn import so
  checkpoint and constants loading no longer fails with `weights_only` unpickling errors.
- Fix lowercase compact formula parsing by normalizing all-lowercase strings
  (e.g., `pt3` → `Pt3`) before calling ASE `Formula`, preserving case-
  sensitive `HO2`-style formulas unchanged.
- Fix `parse_composition_arg` docstring for Sphinx `-W` builds by removing
  indented bullet continuation that docutils treated as invalid RST.
- Fix adsorbate/core partition reconciliation for oxide campaigns by deriving
  `core_symbols` from full mobile formulas when preset cores disagree,
  updating `adsorbate_definition` in place, and deep-copying preset definitions
  per campaign composition.
- SQLite connection handle leaks in database setup and configuration paths.

## 0.5.1

### Fixed

- ASE icosahedron/decahedron/octahedron templates for HCP elements by passing an
  explicit lattice constant (structures are still rescaled to covalent bond length).
- Compact formula parsing for hydrogen–oxide strings such as `HO2Ru9W2` (via ASE
  `Formula` instead of mis-reading `Ho` as holmium).
- Gas/surface adsorbate runs with a preset `adsorbate_definition`: reconcile
  campaign composition to `core_symbols + adsorbate_symbols` when counts match
  but symbol order differs, when only the core formula is supplied, or when the
  full mobile formula requires re-deriving `core_symbols` by stripping known
  `adsorbate_symbols` (oxide campaigns such as `HO2Ru9W2`). Applies to gas and
  surface adsorbate system types across all runner entry points.

### Changed

- Template discovery failures no longer emit per-attempt debug noise for expected
  ASE lattice-guess misses.

## 0.5.0

### Added

- Manual Kaggle GPU workflow for CUDA/MACE integration tests on T4 hardware.
- GPU example integration tests aligned with real example workloads.
- SQLite PRAGMA debug logging for easier HPC filesystem troubleshooting.

### Changed

- Refactored runner/database workflow to reduce repeated overhead and unify
  discovery, streaming, and candidate-loading paths.
- Fail-fast validation at API boundaries; reduced silent defensive fallbacks.
- Strengthened physics assertions, reproducibility checks, and CI strictness.
- Dual MACE/UMA CI matrix with marker-based test partitioning.
- Capped NumPy below 2.5 and aligned Kaggle GPU dependency installs with CI.
- Corrected algorithm selection docs: 3-atom adsorbate systems use GA, not BH.
- Docs version fallback now reads from `scgo.__version__` instead of a stale literal.

### Fixed

- SQLite connection handle leaks in `setup_database` and DB configuration paths.
- Concurrent SQLite write stress test stability in CI.
- Reference run provenance and streaming warning behavior.
- TorchSim warnings API usage and raw MACE model wrapping for `optimize()`.
- Kaggle runner resilience (conda detection, source tarball, log redaction, CUDA torch).
- Empty GA population crash and surface `run_go` e2e test stability.
- Cross-fragment adsorbate bonding rejection in integrity checks.
- Adsorption height checks and CI disk cleanup for UMA installs.

## 0.4.1

### Fixed

- Adsorbate partition overlap handling and `source_db_relpath` provenance fields.

### Documentation

- Minor documentation fixes following the 0.4.0 release.

## 0.4.0

### Changed

- Flattened GO runs to datetime-tagged `run_*` directories (removed `trial_*` layer).
- Run IDs and `metadata.json` timestamps now use UTC.
- Timing JSON (`timing.json`, `go_ts_timing.json`) includes structured provenance headers,
  `run_id`, and `timing_schema_version`.
- `go_ts_timing.json` links to per-run GO/TS timing files via `current_*_run_id` and
  `*_run_timing_relpath` fields.
- TS `results_summary.json` handles skipped pairs without KeyError.
- `get_provenance()` reads `provenance` and `key_value_pairs` in addition to `metadata`.
- Database discovery warns on unresolved `run_id` paths instead of silently skipping.

### Documentation

- Updated quickstart output layout, provenance fields, and timing schema.
- Corrected algorithm selection rules in `parameters.rst`.
