"""Basin Hopping global optimization implementation for atomic clusters.

This module implements the Basin Hopping algorithm, a global optimization method
that explores the potential energy surface through iterative random moves and
local minimizations, with Metropolis acceptance criteria.
"""

from __future__ import annotations

import os
from time import perf_counter
from typing import Any

import numpy as np
from ase import Atoms
from ase.optimize import LBFGS
from ase.optimize.optimize import Optimizer
from tqdm import tqdm

from scgo.algorithms.ga_common import (
    ga_run_metadata_extras,
    maybe_apply_mobile_core_ads_tags,
    setup_diversity_scorer,
)
from scgo.algorithms.run_context import validate_and_resolve_run_context
from scgo.cluster_adsorbate.config import ClusterAdsorbateConfig
from scgo.cluster_adsorbate.constraints import prepare_atoms_for_local_relax
from scgo.constants import (
    DEFAULT_COMPARATOR_TOL,
    DEFAULT_ENERGY_TOLERANCE,
    DEFAULT_FMAX_THRESHOLD,
    DEFAULT_PAIR_COR_MAX,
)
from scgo.database import HPC_DATABASE_EXCEPTIONS, setup_database
from scgo.database.sync import PRESET_HPC, database_retry
from scgo.exceptions import SCGOValidationError
from scgo.metadata.atoms import set_tags
from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types import (
    AdsorbateDefinition,
    AdsorbateFragmentInput,
    ConnectivityFactorInput,
    NormalizedConnectivityFactor,
    SystemType,
    resolve_structure_mic,
    validate_minimum_structure,
)
from scgo.utils.comparators import UniquenessSettings, create_geometry_comparator
from scgo.utils.fitness_strategies import (
    FitnessStrategy,
    calculate_fitness,
    get_fitness_from_atoms,
    set_fitness_in_atoms,
)
from scgo.utils.helpers import (
    _create_energy_bins,
    _find_unique_minima_with_binning,
    extract_minima_from_database,
    perform_local_relaxation,
)
from scgo.utils.logging import (
    get_logger,
    log_debug_v,
    log_info_v,
    should_show_progress,
)
from scgo.utils.timing_report import (
    build_timing_payload,
    emit_timing_data,
    log_timing_summary,
)
from scgo.utils.validation import (
    validate_atoms,
    validate_calculator_attached,
    validate_in_choices,
    validate_in_range,
    validate_integer,
    validate_positive,
)

logger = get_logger(__name__)


def _move_atoms(
    atoms: Atoms,
    dr: float,
    move_fraction: float = 0.3,
    move_strategy: str = "random",
    rng: np.random.Generator | None = None,
    movable_indices: list[int] | None = None,
    *,
    move_by_tag_groups: bool = False,
    recenter_com: bool = True,
    adsorbate_movable_indices: list[int] | None = None,
    adsorbate_dr: float | None = None,
    adsorbate_move_fraction: float | None = None,
) -> tuple[Atoms, str]:
    """Apply a random displacement to a subset of atoms.

    Args:
        atoms: The ASE Atoms object to apply displacement to.
        dr: The maximum displacement distance for each atom during the random
            move step (in Angstrom). Used for core / non-adsorbate atoms.
        move_fraction: The fraction of atoms to move during each perturbation step.
            Used for core / non-adsorbate atoms.
        move_strategy: The strategy for selecting atoms to move ('random',
            'highest_force', 'lowest_force').
        rng: Optional numpy random number generator for reproducibility.
        movable_indices: Indices that may be displaced.
        move_by_tag_groups: Move whole tag groups together when True.
        recenter_com: If True (default), restore the pre-move center of mass
            after the displacement. Set False for surface systems so the slab
            registry is preserved.
        adsorbate_movable_indices: Subset of movable indices treated as adsorbate
            (receive ``adsorbate_dr`` / ``adsorbate_move_fraction`` when set).
        adsorbate_dr: Displacement scale for adsorbate atoms/groups. When None,
            all movable atoms use ``dr``.
        adsorbate_move_fraction: Move fraction for adsorbate atoms/groups. When
            None, all movable atoms use ``move_fraction``.

    Returns:
        A tuple (Atoms, description) where description lists moved atoms
        in 1-indexed form.
    """
    atoms_new = atoms.copy()
    if rng is None:
        rng = np.random.default_rng()

    if movable_indices is None:
        movable_indices = list(range(len(atoms_new)))

    if not movable_indices:
        return atoms_new, "Moved_atoms: none"

    ads_set = (
        {int(i) for i in adsorbate_movable_indices}
        if adsorbate_movable_indices is not None
        else set()
    )
    use_split = (
        adsorbate_dr is not None
        and adsorbate_move_fraction is not None
        and bool(ads_set)
    )
    ads_dr_eff = float(adsorbate_dr) if use_split else 0.0
    ads_frac_eff = float(adsorbate_move_fraction) if use_split else 0.0
    core_indices = [i for i in movable_indices if int(i) not in ads_set]
    ads_indices = [i for i in movable_indices if int(i) in ads_set]

    def _select_atom_indices(
        pool: list[int],
        fraction: float,
    ) -> list[int]:
        n_atoms = len(pool)
        if n_atoms == 0:
            return []
        min_to_move = 1 if n_atoms == 1 else 2
        n_to_move_calculated = int(n_atoms * fraction)
        n_to_move = min(n_atoms, max(min_to_move, n_to_move_calculated))
        if move_strategy == "random":
            return list(rng.choice(pool, size=n_to_move, replace=False))
        if move_strategy in ["highest_force", "lowest_force"]:
            forces = atoms.get_forces()
            force_magnitudes = np.linalg.norm(forces[pool], axis=1)
            sorted_local = np.argsort(force_magnitudes)
            sorted_indices = np.asarray(pool, dtype=int)[sorted_local]
            if move_strategy == "highest_force":
                return list(sorted_indices[-n_to_move:])
            return list(sorted_indices[:n_to_move])
        raise SCGOValidationError(f"Unknown move_strategy: {move_strategy}")

    positions = atoms_new.get_positions()
    cm = atoms_new.get_center_of_mass()
    disp = np.zeros_like(positions)
    indices_to_move: list[int] = []

    groups: dict[int, list[int]] = {}

    def _choose_groups(pool: list[int], fraction: float) -> list[int]:
        n_groups = len(pool)
        if n_groups == 0:
            return []
        n_to_move_groups = min(n_groups, max(1, int(n_groups * fraction)))
        return list(rng.choice(pool, size=n_to_move_groups, replace=False))

    # Build (candidates, fraction, scale, is_group) pools. Tag-group pools move a
    # whole group rigidly (one shared displacement vector); atom pools move each
    # selected atom independently. Both cores and adsorbate share one apply loop.
    if move_by_tag_groups:
        tags = atoms_new.get_tags()
        for idx in movable_indices:
            groups.setdefault(int(tags[idx]), []).append(int(idx))
        group_ids = sorted(groups)
        if not group_ids:
            return atoms_new, "Moved_atoms: none"

        def _is_ads_group(group_id: int) -> bool:
            return use_split and any(int(i) in ads_set for i in groups[group_id])

        core_group_ids = [g for g in group_ids if not _is_ads_group(g)]
        ads_group_ids = [g for g in group_ids if _is_ads_group(g)]

        if use_split and (core_group_ids or ads_group_ids):
            pools = [
                (core_group_ids, move_fraction, dr, True),
                (ads_group_ids, ads_frac_eff, ads_dr_eff, True),
            ]
        else:
            pools = [(group_ids, move_fraction, dr, True)]
    elif use_split and (core_indices or ads_indices):
        pools = [
            (core_indices, move_fraction, dr, False),
            (ads_indices, ads_frac_eff, ads_dr_eff, False),
        ]
    else:
        pools = [(list(movable_indices), move_fraction, dr, False)]

    for candidates, fraction, scale, is_group in pools:
        if is_group:
            for group_id in _choose_groups(candidates, fraction):
                group_indices = groups[group_id]
                disp[group_indices, :] = scale * rng.uniform(-1.0, 1.0, 3)
                indices_to_move.extend(group_indices)
        else:
            chosen = _select_atom_indices(candidates, fraction)
            if chosen:
                disp[chosen, :] = rng.uniform(-1.0, 1.0, (len(chosen), 3)) * scale
                indices_to_move.extend(chosen)

    if not indices_to_move:
        return atoms_new, "Moved_atoms: none"

    atoms_new.set_positions(positions + disp)
    if recenter_com:
        atoms_new.translate(cm - atoms_new.get_center_of_mass())

    moved_indices_str = ",".join(str(i + 1) for i in sorted(indices_to_move))
    # Bracket the index list so ASE DB does not reject a bare numeric
    # ``description`` value (e.g. single-atom ``Moved_atoms: 1``).
    return atoms_new, f"Moved_atoms: [{moved_indices_str}]"


def bh_go(
    atoms: Atoms,
    output_dir: str,
    niter: int = 100,
    fmax: float = DEFAULT_FMAX_THRESHOLD,
    niter_local_relaxation: int = 250,
    optimizer: type[Optimizer] = LBFGS,
    dr: float = 0.5,
    move_fraction: float = 0.3,
    move_strategy: str = "random",
    temperature: float = 1.0,
    deduplicate: bool = True,
    energy_tolerance: float = DEFAULT_ENERGY_TOLERANCE,
    comparator_tol: float = DEFAULT_COMPARATOR_TOL,
    comparator_pair_cor_max: float = DEFAULT_PAIR_COR_MAX,
    comparator_n_top: int | None = None,
    verbosity: int = 1,
    run_id: str | None = None,
    clean: bool = False,
    fitness_strategy: str = "low_energy",
    diversity_reference_db: str | None = None,
    diversity_max_references: int = 100,
    diversity_update_interval: int = 5,
    surface_config: SurfaceSystemConfig | None = None,
    n_slab: int = 0,
    system_type: SystemType = "gas_cluster",
    adsorbate_definition: AdsorbateDefinition | None = None,
    write_timing_json: bool = False,
    detailed_timing: bool = False,
    timing_output_dir: str | None = None,
    timing_collector: list[dict[str, Any]] | None = None,
    cluster_adsorbate_config: ClusterAdsorbateConfig | None = None,
    connectivity_factor: ConnectivityFactorInput
    | NormalizedConnectivityFactor
    | None = None,
    allow_cluster_fragmentation: bool = False,
    allow_adsorbate_surface_detachment: bool = False,
    enforce_adsorbate_subgraph_integrity: bool = True,
    freeze_adsorbate_internal_geometry: bool = False,
    adsorbate_fragment_template: AdsorbateFragmentInput | None = None,
    db_enable_expression_indexes: bool = False,
    *,
    rng: np.random.Generator,
) -> list[tuple[float, Atoms]]:
    """Basin Hopping global optimization for a single trial.

    Args:
        atoms: Initial Atoms object representing the cluster. Calculator must be attached.
        output_dir: Directory where ASE database for the run will be stored.
        niter: Total number of Basin Hopping iterations.
        fmax: Maximum force criterion for convergence in local relaxations (eV/Å).
        niter_local_relaxation: Maximum steps allowed for each local relaxation.
        optimizer: ASE optimizer class (e.g., BFGS) for local relaxations.
        rng: Random number generator (numpy.random.Generator). Required.
        dr: Maximum displacement distance for each atom during random move step (Å).
        move_fraction: Fraction of atoms to move during each perturbation step.
        move_strategy: Strategy for selecting atoms to move ('random', 'highest_force', 'lowest_force').
        temperature: Metropolis energy scale (eV) for accepting uphill fitness
            moves. Default ``1.0`` matches ASE BasinHopping.
        write_timing_json: Optional ``timing.json`` (see ``timing_output_dir``).
            Set in ``optimizer_params['bh']`` inside ``params``/``go_params``.
        detailed_timing: Per-iteration split rows in JSON when ``write_timing_json``
            is set.
        timing_output_dir: Directory for ``timing.json`` (defaults to ``output_dir``
            when ``run_trials`` is not used).
        timing_collector: Optional list appended with the timing payload after the
            run (always, independent of ``write_timing_json``).
        deduplicate: If True (default), filter to structurally unique minima.
        energy_tolerance: Energy difference (eV) below which structures are considered duplicates.
        comparator_tol: Tolerance for interatomic distance comparator.
        comparator_pair_cor_max: Maximum pair correlation for comparator.
        comparator_n_top: Optional override of trailing mobile-atom count.
        verbosity: Verbosity level (0=quiet, 1=normal, 2=debug, 3=trace). Default 1.
        run_id: Optional run ID for tracking.
        clean: If True, remove an existing database in the output directory.
        fitness_strategy: Fitness strategy. One of: "low_energy", "high_energy", "diversity".
            Default "low_energy".
        diversity_reference_db: Glob pattern for reference structure databases.
            Required when fitness_strategy="diversity".
        diversity_max_references: Maximum number of reference structures to load.
        diversity_update_interval: Iterations between reference updates.

    Returns:
        List of (energy, Atoms) tuples for local minima found. If deduplicate=True (default),
        filtered to structurally unique minima, sorted by fitness (highest first) for
        non-low_energy strategies, or by energy (lowest first) for low_energy.

    Raises:
        SCGOValidationError: If atoms is not an ASE Atoms object, niter is not a
            positive integer, no calculator is attached, or any other parameter
            is invalid.
    """
    validate_atoms(atoms)
    validate_integer("niter", niter)
    validate_positive("niter", niter, strict=True)
    calculator = validate_calculator_attached(atoms, "basin hopping")
    validate_positive("fmax", fmax, strict=True)
    validate_positive("dr", dr, strict=True)
    validate_in_range("move_fraction", move_fraction, 0.0, 1.0)
    validate_in_choices(
        "move_strategy", move_strategy, ["random", "highest_force", "lowest_force"]
    )

    run_ctx = validate_and_resolve_run_context(
        system_type=system_type,
        surface_config=surface_config,
        connectivity_factor=connectivity_factor,
        cluster_adsorbate_config=cluster_adsorbate_config,
        fitness_strategy=fitness_strategy,
    )
    connectivity_factor = run_ctx.connectivity_factor
    policy = run_ctx.policy
    fitness_strategy = run_ctx.fitness_strategy
    surface_mode = policy.uses_surface
    mobile_composition = (
        list(atoms.get_chemical_symbols()[n_slab:])
        if surface_mode and n_slab > 0
        else list(atoms.get_chemical_symbols())
    )
    maybe_apply_mobile_core_ads_tags(
        atoms,
        n_slab if surface_mode else 0,
        mobile_composition,
        adsorbate_definition,
        system_type,
    )

    def _run_metadata_extras() -> dict[str, int | str]:
        return ga_run_metadata_extras(
            surface_config,
            n_slab,
            system_type,
            mobile_composition,
            adsorbate_definition=adsorbate_definition,
        )

    movable_indices = list(range(len(atoms)))
    # Match GA storage / run_trials backstop: for slab-search types the frozen
    # bottom prefix is the deposit boundary in structural gates.
    n_slab_deposit: int | None = None
    if surface_mode:
        if n_slab <= 0:
            if surface_config is None:
                raise SCGOValidationError(
                    "Surface system type requires n_slab > 0 or surface_config."
                )
            n_slab = len(surface_config.slab)
        if policy.slab_is_search_target:
            from scgo.surface.partition import resolve_slab_search_partition

            if surface_config is None:
                raise SCGOValidationError(
                    f"system_type={system_type!r} requires surface_config."
                )
            part = resolve_slab_search_partition(surface_config)
            movable_indices = list(range(part.n_fixed, len(atoms)))
            n_slab_deposit = int(part.n_fixed)
        else:
            movable_indices = list(range(n_slab, len(atoms)))
        if not movable_indices:
            raise SCGOValidationError(
                "Surface system has no movable atoms for basin hopping."
            )

    # Scale adsorbate moves only; keep full dr/fraction for core when mixed.
    tags = atoms.get_tags()
    ads_movable = [i for i in movable_indices if int(tags[i]) > 0]
    core_movable = [i for i in movable_indices if int(tags[i]) == 0]
    mixed_core_ads = (
        policy.constrain_adsorbate_moves and bool(ads_movable) and bool(core_movable)
    )
    if mixed_core_ads:
        move_dr = dr
        move_frac = move_fraction
        ads_dr: float | None = dr * policy.adsorbate_move_scale
        ads_frac: float | None = min(move_fraction, 0.25)
        ads_indices_arg: list[int] | None = ads_movable
    elif policy.constrain_adsorbate_moves and ads_movable:
        move_dr = dr * policy.adsorbate_move_scale
        move_frac = min(move_fraction, 0.25)
        ads_dr = None
        ads_frac = None
        ads_indices_arg = None
    else:
        move_dr = dr
        move_frac = move_fraction
        ads_dr = None
        ads_frac = None
        ads_indices_arg = None

    # Match GA/GO: mobile-only n_top and comparator_use_mic (not surface_mode alone).
    effective_n_top = (
        int(comparator_n_top) if comparator_n_top is not None else len(movable_indices)
    )
    comp_mic = resolve_structure_mic(system_type, surface_config)
    geometry = UniquenessSettings(
        comparator_tol=comparator_tol,
        comparator_pair_cor_max=comparator_pair_cor_max,
    )
    comparator = create_geometry_comparator(
        n_top=effective_n_top,
        mic=comp_mic,
        settings=geometry,
    )

    # Load reference structures and create DiversityScorer for diversity strategy
    diversity_scorer = setup_diversity_scorer(
        fitness_strategy=fitness_strategy,
        diversity_reference_db=diversity_reference_db,
        composition=mobile_composition,
        n_to_optimize=effective_n_top,
        diversity_max_references=diversity_max_references,
        logger=logger,
        base_dir=output_dir,
        mic=comp_mic,
        uniqueness=geometry,
    )

    # Detach calculator temporarily for DB setup to avoid pickling issues
    calc = atoms.calc
    atoms.calc = None
    da = setup_database(
        output_dir,
        "bh_go.db",
        atoms,
        initial_candidate=atoms,
        remove_existing=clean,
        enable_expression_indexes=db_enable_expression_indexes,
        run_id=run_id,
    )
    atoms.calc = calc

    try:
        profile_t0 = perf_counter()
        profile_timings: dict[str, float] = {}
        profile_counters: dict[str, int] = {
            "niter": int(niter),
            "accepted": 0,
            "rejected_invalid": 0,
        }
        per_iteration: list[dict[str, Any]] | None = [] if detailed_timing else None

        def _finish_bh_timing() -> None:
            total = perf_counter() - profile_t0
            profile_timings["total_wall_s"] = total
            profile_timings["kind"] = "bh"
            relax_sum = float(
                profile_timings.get("initial_local_relaxation_s", 0.0)
            ) + float(profile_timings.get("offspring_local_relaxation_s", 0.0))
            profile_timings["local_relaxation_s"] = relax_sum
            profile_timings["cpu_non_relax_s"] = max(0.0, total - relax_sum)
            log_timing_summary(
                logger, "basin_hopping", profile_timings, verbosity=verbosity
            )
            timing_dir = (
                timing_output_dir if timing_output_dir is not None else output_dir
            )
            run_id_for_timing = os.path.basename(str(timing_dir).rstrip(os.sep))
            extra: dict[str, Any] = {"counters": profile_counters}
            if per_iteration is not None:
                extra["per_iteration"] = per_iteration
            out = build_timing_payload(
                backend="basin_hopping",
                timings_s=profile_timings,
                run_id=run_id_for_timing,
                extra=extra,
            )
            emit_timing_data(
                out,
                write_timing_json=write_timing_json,
                output_dir=output_dir,
                timing_output_dir=timing_output_dir,
                timing_collector=timing_collector,
            )

        a_current = database_retry(
            da.get_an_unrelaxed_candidate,
            config=PRESET_HPC,
            exception_types=HPC_DATABASE_EXCEPTIONS,
        )
        a_current = prepare_atoms_for_local_relax(
            a_current,
            surface_mode=surface_mode,
            surface_config=surface_config,
            n_slab=(n_slab if surface_mode else 0),
            freeze_adsorbate_internal_geometry=freeze_adsorbate_internal_geometry,
            adsorbate_definition=adsorbate_definition,
            adsorbate_fragment_templates=adsorbate_fragment_template,
        )
        t_rel0 = perf_counter()
        e_current = perform_local_relaxation(
            a_current,
            calculator,
            optimizer,
            fmax,
            niter_local_relaxation,
            center_after_relax=not surface_mode,
            surface_mode=surface_mode,
            n_slab=n_slab,
        )
        profile_timings["initial_local_relaxation_s"] = perf_counter() - t_rel0
        try:
            validate_minimum_structure(
                a_current,
                system_type=system_type,
                surface_config=surface_config,
                n_slab=n_slab if surface_mode else None,
                adsorbate_definition=adsorbate_definition,
                connectivity_factor=connectivity_factor,
                allow_cluster_fragmentation=allow_cluster_fragmentation,
                allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
                enforce_adsorbate_subgraph_integrity=enforce_adsorbate_subgraph_integrity,
                n_slab_deposit=n_slab_deposit,
            )
        except SCGOValidationError as exc:
            # The initial seed must not crash the whole run: the trial gate
            # (below) and the run_trials final gate already treat an invalid
            # structure as rejectable/droppable. Proceed with the seed as the
            # starting point; subsequent moves and the final gate still enforce
            # connectivity, so disconnected minima are never reported downstream.
            logger.warning(
                "Initial relaxed seed fails structural gate (%s); proceeding "
                "with it as the starting structure.",
                exc,
            )
        set_tags(
            a_current,
            **_run_metadata_extras(),
        )

        if run_id is not None:
            set_tags(a_current, run_id=run_id)

        t_db0 = perf_counter()
        database_retry(
            lambda: da.add_relaxed_step(a_current),
            config=PRESET_HPC,
            exception_types=HPC_DATABASE_EXCEPTIONS,
        )
        profile_timings["initial_relaxed_write_s"] = perf_counter() - t_db0

        # Calculate and store initial fitness
        fitness_current = calculate_fitness(
            energy=e_current,
            atoms=a_current,
            strategy=fitness_strategy,
            diversity_scorer=diversity_scorer,
        )
        set_fitness_in_atoms(a_current, fitness_current, fitness_strategy)

        log_info_v(
            logger,
            "Starting Basin Hopping with fitness_strategy='%s' "
            "(initial energy: %.4f eV, fitness: %.4f)",
            fitness_strategy,
            e_current,
            fitness_current,
            verbosity=verbosity,
        )

        iteration_iterator = range(niter)
        if verbosity >= 1:
            iteration_iterator = tqdm(
                iteration_iterator,
                desc=f"  BH iterations for {len(atoms)} atoms",
                disable=not should_show_progress(verbosity),
            )

        for iteration in iteration_iterator:
            a_trial, desc = _move_atoms(
                a_current,
                move_dr,
                move_frac,
                move_strategy=move_strategy,
                rng=rng,
                movable_indices=movable_indices,
                move_by_tag_groups=freeze_adsorbate_internal_geometry,
                recenter_com=not surface_mode,
                adsorbate_movable_indices=ads_indices_arg,
                adsorbate_dr=ads_dr,
                adsorbate_move_fraction=ads_frac,
            )
            a_trial = prepare_atoms_for_local_relax(
                a_trial,
                surface_mode=surface_mode,
                surface_config=surface_config,
                n_slab=(n_slab if surface_mode else 0),
                freeze_adsorbate_internal_geometry=freeze_adsorbate_internal_geometry,
                adsorbate_definition=adsorbate_definition,
                adsorbate_fragment_templates=adsorbate_fragment_template,
            )
            if run_id is not None:
                set_tags(a_trial, run_id=run_id)

            t_ins0 = perf_counter()
            database_retry(
                lambda _t=a_trial, _d=desc: da.add_unrelaxed_candidate(
                    _t, description=_d
                ),
                config=PRESET_HPC,
                exception_types=HPC_DATABASE_EXCEPTIONS,
            )
            dt_ins = perf_counter() - t_ins0
            profile_timings["offspring_unrelaxed_insert_s"] = (
                profile_timings.get("offspring_unrelaxed_insert_s", 0.0) + dt_ins
            )

            t_rel0 = perf_counter()
            e_trial = perform_local_relaxation(
                a_trial,
                calculator,
                optimizer,
                fmax,
                niter_local_relaxation,
                center_after_relax=not surface_mode,
                surface_mode=surface_mode,
                n_slab=n_slab,
            )
            dt_rel = perf_counter() - t_rel0
            profile_timings["offspring_local_relaxation_s"] = (
                profile_timings.get("offspring_local_relaxation_s", 0.0) + dt_rel
            )
            try:
                validate_minimum_structure(
                    a_trial,
                    system_type=system_type,
                    surface_config=surface_config,
                    n_slab=n_slab if surface_mode else None,
                    adsorbate_definition=adsorbate_definition,
                    connectivity_factor=connectivity_factor,
                    allow_cluster_fragmentation=allow_cluster_fragmentation,
                    allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
                    enforce_adsorbate_subgraph_integrity=enforce_adsorbate_subgraph_integrity,
                    n_slab_deposit=n_slab_deposit,
                )
            except SCGOValidationError as exc:
                # A single invalid trial must not abort the whole run: count it as
                # rejected and continue with the next move.
                profile_counters["rejected_invalid"] += 1
                logger.warning(
                    "Iteration %d: rejecting invalid trial structure (%s)",
                    iteration,
                    exc,
                )
                continue
            set_tags(
                a_trial,
                **_run_metadata_extras(),
            )
            if run_id is not None:
                set_tags(a_trial, run_id=run_id)

            t_w0 = perf_counter()
            database_retry(
                lambda _t=a_trial: da.add_relaxed_step(_t),
                config=PRESET_HPC,
                exception_types=HPC_DATABASE_EXCEPTIONS,
            )
            dt_w = perf_counter() - t_w0
            profile_timings["offspring_relaxed_write_s"] = (
                profile_timings.get("offspring_relaxed_write_s", 0.0) + dt_w
            )

            # Calculate fitness for trial structure
            fitness_trial = calculate_fitness(
                energy=e_trial,
                atoms=a_trial,
                strategy=fitness_strategy,
                diversity_scorer=diversity_scorer,
            )
            set_fitness_in_atoms(a_trial, fitness_trial, fitness_strategy)

            # Fitness-based acceptance criterion
            accept = False
            if fitness_trial > fitness_current:
                # Better fitness - always accept
                accept = True
                log_debug_v(
                    logger,
                    "Iteration %d: Accepting (fitness improved: %.4f → %.4f)",
                    iteration,
                    fitness_current,
                    fitness_trial,
                    verbosity=verbosity,
                )
            elif temperature > 0.0:
                # Metropolis acceptance based on fitness difference
                fitness_diff = fitness_trial - fitness_current
                acceptance_prob = np.exp(fitness_diff / temperature)
                accept = rng.random() < acceptance_prob

                log_debug_v(
                    logger,
                    "Iteration %d: Metropolis test "
                    "(fitness_diff: %.4f, acceptance_prob: %.4f, accept: %s)",
                    iteration,
                    fitness_diff,
                    acceptance_prob,
                    accept,
                    verbosity=verbosity,
                )

            if accept:
                profile_counters["accepted"] += 1
                a_current = a_trial.copy()
                # Atoms.copy() drops ``calc``; re-attach so force-based moves and
                # the next relaxation keep working (MACE/UMA fail without it).
                a_current.calc = calculator
                e_current = e_trial
                fitness_current = fitness_trial

                # Periodic reference update for diversity strategy
                if (
                    fitness_strategy == FitnessStrategy.DIVERSITY
                    and diversity_scorer
                    and iteration % diversity_update_interval == 0
                ):
                    diversity_scorer.add_reference(a_trial)
                    log_debug_v(
                        logger,
                        "Updated reference structures (total: %d)",
                        len(diversity_scorer),
                        verbosity=verbosity,
                    )

            if per_iteration is not None:
                per_iteration.append(
                    {
                        "iteration": int(iteration),
                        "timings_s": {
                            "unrelaxed_insert_s": dt_ins,
                            "local_relaxation_s": dt_rel,
                            "relaxed_write_s": dt_w,
                        },
                    }
                )

        all_candidates = database_retry(
            da.get_all_relaxed_candidates,
            config=PRESET_HPC,
            exception_types=HPC_DATABASE_EXCEPTIONS,
        )
        all_minima = extract_minima_from_database(all_candidates)

        if not all_minima:
            _finish_bh_timing()
            return []

        if not deduplicate:
            _finish_bh_timing()
            return all_minima

        # Filter out non-finite energies
        valid_minima = [
            (energy, atoms) for energy, atoms in all_minima if np.isfinite(energy)
        ]

        if not valid_minima:
            _finish_bh_timing()
            return []

        # Reuse the comparator created before the run loop for deduplication
        # Sort by energy for binning (lowest first)
        sorted_minima = sorted(valid_minima, key=lambda x: x[0])

        # Set up energy binning for optimized duplicate detection
        get_bin_index, energy_bins = _create_energy_bins(
            energy_tolerance, sorted_minima[0]
        )

        # Find unique minima using energy binning optimization
        unique_minima = _find_unique_minima_with_binning(
            sorted_minima, comparator, energy_tolerance, get_bin_index, energy_bins
        )

        # Sort by fitness (highest first) for non-default strategies
        if fitness_strategy != FitnessStrategy.LOW_ENERGY:
            unique_minima.sort(
                key=lambda x: get_fitness_from_atoms(x[1], default=-float("inf")),
                reverse=True,  # Higher fitness first
            )
            logger.info(
                f"Sorted {len(unique_minima)} unique minima by {fitness_strategy} fitness"
            )

        _finish_bh_timing()
        return unique_minima

    finally:
        # Clean up database connection
        from scgo.database import close_data_connection

        close_data_connection(da, log_errors=False)
