"""Genetic Algorithm global optimization implementation for atomic clusters.

This module implements a Genetic Algorithm approach to cluster structure optimization,
using population-based evolution with crossover (cut-and-splice) and mutation operators
adapted for atomic clusters.
"""

from __future__ import annotations

import math
import os
from contextlib import suppress
from time import perf_counter
from typing import Any

import numpy as np
from ase import Atoms
from ase.optimize import LBFGS
from ase.optimize.optimize import Optimizer
from ase_ga.utilities import closest_distances_generator, get_all_atom_types
from tqdm import tqdm

from scgo.algorithms.ga_common import (
    ClusterStartGenerator,
    SurfaceClusterStartGenerator,
    create_ga_pairing,
    create_mutation_operators,
    create_structure_comparator,
    log_early_stopping_info,
    select_population_class,
    setup_diversity_scorer,
    slab_ga_metadata_extras,
    sort_minima_by_fitness,
    update_early_stopping_state_unified,
    update_mutation_weights,
    validate_ga_common_params,
)
from scgo.constants import DEFAULT_ENERGY_TOLERANCE
from scgo.database import HPC_DATABASE_EXCEPTIONS, close_data_connection, setup_database
from scgo.database.metadata import add_metadata, filter_by_metadata
from scgo.database.sync import retry_with_backoff
from scgo.initialization import compute_cell_side
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.constraints import attach_slab_constraints
from scgo.system_types import (
    SystemType,
    uses_surface,
    validate_structure_for_system_type,
    validate_system_type_settings,
)
from scgo.utils.fitness_strategies import FitnessStrategy, validate_fitness_strategy
from scgo.utils.helpers import (
    extract_minima_from_database,
    perform_local_relaxation,
)
from scgo.utils.logging import get_logger, should_show_progress
from scgo.utils.mutation_weights import get_adaptive_mutation_config
from scgo.utils.rng_helpers import ensure_rng_or_create
from scgo.utils.timing_report import log_timing_summary, write_timing_file
from scgo.utils.validation import validate_composition


def ga_go(
    composition: list[str],
    output_dir: str,
    calculator: Any,
    niter: int = 10,
    fmax: float = 0.05,
    niter_local_relaxation: int = 250,
    optimizer: type[Optimizer] = LBFGS,
    energy_tolerance: float = DEFAULT_ENERGY_TOLERANCE,
    mutation_probability: float = 0.4,
    population_size: int = 10,
    offspring_fraction: float = 0.5,
    n_jobs_population_init: int = -2,
    vacuum: float = 10.0,
    previous_search_glob: str = "**/*.db",
    use_adaptive_mutations: bool = True,
    stagnation_trigger: int = 4,
    stagnation_full_trigger: int = 8,
    recovery_window: int = 2,
    aggressive_burst_multiplier: float = 1.8,
    max_mutation_probability: float = 0.65,
    early_stopping_niter: int = 10,
    verbosity: int = 1,
    elite_fraction: float = 0.1,
    run_id: str | None = None,
    clean: bool = False,
    fitness_strategy: str = "low_energy",
    diversity_reference_db: str | None = None,
    diversity_max_references: int = 100,
    diversity_update_interval: int = 5,
    *,
    rng: np.random.Generator | None = None,
    surface_config: SurfaceSystemConfig | None = None,
    system_type: SystemType = "gas_cluster",
    write_timing_json: bool = False,
    detailed_timing: bool = False,
) -> list[tuple[float, Atoms]]:
    """Genetic Algorithm global optimization with adaptive mutations.

    The `rng` argument is optional and should be a `numpy.random.Generator`.

    Args:
        composition: List of element symbols defining the cluster composition.
        calculator: ASE calculator for energy/force evaluations.
        output_dir: Directory for calculation files and intermediate structures.
        niter: Number of generations to run.
        fmax: Maximum force criterion for convergence (eV/Å).
        niter_local_relaxation: Maximum steps for each local relaxation.
        optimizer: ASE optimizer class (e.g., LBFGS) for local relaxations.
        rng: Random number generator (numpy.random.Generator). Optional.
        energy_tolerance: Energy difference (eV) below which structures are considered duplicates.
        mutation_probability: Probability (0-1) of applying mutation to offspring.
            Only used if use_adaptive_mutations=False.
        population_size: Number of individuals in the population.
        n_jobs_population_init: Parallel workers for initial population generation.
            -1 = all CPUs, -2 = all except one. Default -2.
        vacuum: Vacuum to add around cluster to define simulation cell size.
        previous_search_glob: Glob pattern used to discover previous database
            files for seed-based initialization.
        use_adaptive_mutations: If True (default), use adaptive mutation operators
            that adjust based on composition, size, and generation.
        early_stopping_niter: Generations without improvement before stopping early.
            Uses fitness for non-low_energy strategies, energy for low_energy. If 0, disabled.
        verbosity: Verbosity level (0=quiet, 1=normal, 2=debug, 3=trace). Default 1.
        elite_fraction: Fraction of population to preserve as elite (top performers).
            Default 0.1.
        run_id: Optional run ID for tracking.
        clean: If True, start fresh (ignore previous databases).
        fitness_strategy: Fitness strategy. One of: "low_energy", "high_energy", "diversity".
            Default "low_energy".
        diversity_reference_db: Glob pattern for reference structure databases.
            Required when fitness_strategy="diversity".
        diversity_max_references: Maximum number of reference structures to load.
        diversity_update_interval: Generations between reference updates.
        surface_config: Optional :class:`~scgo.surface.config.SurfaceSystemConfig`
            for adsorbate-on-slab GA. When set, the slab defines the cell and the
            cluster is the trailing ``len(composition)`` atoms; ``vacuum`` is not
            used for the template cell.
        write_timing_json: Optional ``timing.json`` under ``output_dir``.
        detailed_timing: Per-generation split rows (more memory; included in JSON
            when ``write_timing_json`` is set).

    Returns:
        List of (energy, Atoms) tuples for unique local minima, sorted by fitness
        (highest first) for non-low_energy strategies, or by energy (lowest first) for low_energy.

    Raises:
        TypeError: If atoms is not an Atoms object or types are wrong.
        ValueError: If calculator is not attached or parameters are invalid.
    """
    logger = get_logger(__name__)
    profile_t0 = perf_counter()
    profile_timings: dict[str, float] = {}
    profile_counters: dict[str, int] = {"offspring_created": 0}
    per_generation: list[dict[str, Any]] | None = [] if detailed_timing else None

    if system_type == "gas_cluster" and surface_config is not None:
        system_type = "surface_cluster"
    validate_composition(composition, allow_empty=False, allow_tuple=False)
    validate_system_type_settings(
        system_type=system_type, surface_config=surface_config
    )
    validate_ga_common_params(
        niter=niter,
        population_size=population_size,
        n_jobs_population_init=n_jobs_population_init,
        calculator=calculator,
        mutation_probability=mutation_probability,
        offspring_fraction=offspring_fraction,
        vacuum=vacuum,
        fmax=fmax,
    )

    # Validate and normalize fitness strategy (coerce to Enum)
    validate_fitness_strategy(fitness_strategy)
    if isinstance(fitness_strategy, str):
        fitness_strategy = FitnessStrategy(fitness_strategy)

    # Normalize RNG early and enforce Generator-only policy
    rng = ensure_rng_or_create(rng)

    n_to_optimize = len(composition)

    surface_mode = uses_surface(system_type)
    if surface_mode:
        if not isinstance(surface_config, SurfaceSystemConfig):
            raise TypeError(
                "surface_config must be a SurfaceSystemConfig instance or None"
            )
        slab_ref = surface_config.slab.copy()
        n_slab = len(slab_ref)
        dummy_top = [[0.0, 0.0, 0.0] for _ in range(n_to_optimize)]
        atoms_template = Atoms(
            symbols=list(slab_ref.get_chemical_symbols()) + list(composition),
            positions=np.vstack([slab_ref.get_positions(), np.asarray(dummy_top)]),
            cell=slab_ref.get_cell(),
            pbc=slab_ref.get_pbc(),
        )
        atoms_template.calc = calculator
        center_after_relax = False
    else:
        n_slab = 0
        cell_side = compute_cell_side(composition, vacuum=vacuum)
        atoms_template = Atoms(
            symbols=composition,
            positions=[[0, 0, 0] for _ in range(n_to_optimize)],  # Dummy positions
            cell=[cell_side] * 3,
            pbc=False,
        )
        atoms_template.calc = calculator
        center_after_relax = True

    # Load reference structures and create DiversityScorer for diversity strategy
    diversity_scorer = setup_diversity_scorer(
        fitness_strategy=fitness_strategy,
        diversity_reference_db=diversity_reference_db,
        composition=composition,
        n_to_optimize=n_to_optimize,
        diversity_max_references=diversity_max_references,
        logger=logger,
    )

    slab_for_pairing = slab_ref if surface_mode else None
    pairing = create_ga_pairing(
        atoms_template,
        n_to_optimize,
        rng,
        slab_atoms=slab_for_pairing,
        system_type=system_type,
    )

    adaptive_config = get_adaptive_mutation_config(
        composition=composition,
        current_generation=0,
        total_generations=niter,
        use_adaptive=use_adaptive_mutations,
        generations_without_improvement=0,
        stagnation_trigger=stagnation_trigger,
        stagnation_full_trigger=stagnation_full_trigger,
        recovery_window=recovery_window,
        aggressive_burst_multiplier=aggressive_burst_multiplier,
        max_mutation_probability=max_mutation_probability,
    )

    idx_top = (
        range(n_slab, n_slab + n_to_optimize) if surface_mode else range(n_to_optimize)
    )
    top_z = list({int(atoms_template[i].number) for i in idx_top})
    all_atom_types = get_all_atom_types(atoms_template, top_z)
    blmin = closest_distances_generator(all_atom_types, ratio_of_covalent_radii=0.7)

    operators_list, name_map = create_mutation_operators(
        composition=composition,
        n_to_optimize=n_to_optimize,
        blmin=blmin,
        rng=rng,
        use_adaptive=use_adaptive_mutations,
        system_type=system_type,
        n_slab=n_slab,
        surface_normal_axis=(surface_config.surface_normal_axis if surface_mode else 2),
    )

    mutations = update_mutation_weights(
        operators_list=operators_list,
        name_map=name_map,
        adaptive_config=adaptive_config,
    )
    # Use user-provided mutation_probability when adaptive mutations are disabled
    current_mutation_probability = (
        mutation_probability
        if not use_adaptive_mutations
        else adaptive_config["mutation_probability"]
    )

    comp_mic = bool(surface_config.comparator_use_mic) if surface_mode else False
    comp = create_structure_comparator(n_to_optimize, energy_tolerance, mic=comp_mic)

    if surface_mode:
        start_generator = SurfaceClusterStartGenerator(
            composition,
            slab_ref,
            surface_config,
            blmin,
            rng=rng,
            calculator=None,
            population_size=population_size,
            previous_search_glob=previous_search_glob,
            n_jobs=n_jobs_population_init,
        )
    else:
        start_generator = ClusterStartGenerator(
            composition,
            vacuum,
            rng=rng,
            calculator=None,  # Do not attach calculator to initial population to avoid pickling issues
            population_size=population_size,
            mode="smart",
            previous_search_glob=previous_search_glob,
            n_jobs=n_jobs_population_init,
        )
    t0 = perf_counter()
    starting_population = [
        start_generator.get_new_candidate() for _ in range(population_size)
    ]
    profile_timings["initial_population_generation_s"] = perf_counter() - t0

    if verbosity >= 1:
        n_workers = (
            "all CPUs"
            if n_jobs_population_init == -1
            else "all but one CPU"
            if n_jobs_population_init == -2
            else f"{n_jobs_population_init} workers"
        )
        logger.info(
            f"Generated initial population of {population_size} candidates "
            f"(batched, parallel n_jobs={n_workers})"
        )

    # Do not pass initial_population to SetupDB (avoids formula keys in key_value_pairs).
    # Insert unrelaxed starters via the low-level write API, then relax and tag generation=0.
    da = setup_database(
        output_dir=output_dir,
        db_filename="ga_go.db",
        atoms_template=atoms_template,
        initial_population=None,
        remove_existing=clean,
        remove_aux_files=clean,
        run_id=run_id,
    )

    try:
        if verbosity >= 1:
            logger.info(
                f"Relaxing initial population of {population_size} candidates..."
            )

        _HPC_RETRY_EXCEPTIONS = HPC_DATABASE_EXCEPTIONS + (IOError,)

        def _insert_unrelaxed(cand):
            cand.info.setdefault("key_value_pairs", {})
            cand.info.setdefault("data", {})
            gaid = da.c.write(
                cand,
                origin="StartingCandidateUnrelaxed",
                relaxed=0,
                generation=0,
                extinct=0,
                description="initial",
            )
            da.c.update(gaid, gaid=gaid)
            cand.info["confid"] = gaid

        t0 = perf_counter()
        for cand in starting_population:
            validate_structure_for_system_type(
                cand,
                system_type=system_type,
                surface_config=surface_config,
                n_slab=n_slab,
            )
            retry_with_backoff(
                lambda _cand=cand: _insert_unrelaxed(_cand),
                max_retries=5,
                initial_delay=0.2,
                backoff_factor=2.0,
                exception_types=_HPC_RETRY_EXCEPTIONS,
            )
        profile_timings["initial_unrelaxed_insert_s"] = perf_counter() - t0

        initial_pop_count = 0
        t_relax = 0.0
        t_write = 0.0
        for cand in starting_population:
            if surface_mode:
                attach_slab_constraints(
                    cand,
                    n_slab,
                    fix_all_slab_atoms=surface_config.fix_all_slab_atoms,
                    n_fix_bottom_slab_layers=surface_config.n_fix_bottom_slab_layers,
                    n_relax_top_slab_layers=surface_config.n_relax_top_slab_layers,
                    surface_normal_axis=surface_config.surface_normal_axis,
                )
            t_start = perf_counter()
            perform_local_relaxation(
                cand,
                calculator,
                optimizer,
                fmax,
                niter_local_relaxation,
                center_after_relax=center_after_relax,
            )
            validate_structure_for_system_type(
                cand,
                system_type=system_type,
                surface_config=surface_config,
                n_slab=n_slab if surface_mode else None,
            )
            t_relax += perf_counter() - t_start
            add_metadata(
                cand,
                generation=0,
                run_id=run_id,
                **slab_ga_metadata_extras(surface_config, n_slab, system_type),
            )

            t_start = perf_counter()
            retry_with_backoff(
                lambda _a=cand: da.add_relaxed_step(_a),
                max_retries=5,
                initial_delay=0.2,
                backoff_factor=2.0,
                exception_types=_HPC_RETRY_EXCEPTIONS,
            )
            t_write += perf_counter() - t_start
            initial_pop_count += 1
        profile_timings["initial_local_relaxation_s"] = t_relax
        profile_timings["initial_relaxed_write_s"] = t_write

        if initial_pop_count > 0:
            logger.debug(
                "Tagged %s GA population members with generation=0",
                initial_pop_count,
            )

        log_file = os.path.join(output_dir, "population.log")

        with suppress(FileNotFoundError):
            os.remove(log_file)

        # Select appropriate Population class based on fitness strategy
        PopulationClass, population_kwargs = select_population_class(
            fitness_strategy=fitness_strategy,
            diversity_scorer=diversity_scorer,
            diversity_update_interval=diversity_update_interval,
            logger=logger,
        )

        population = PopulationClass(
            data_connection=da,
            population_size=population_size,
            comparator=comp,
            logfile=log_file,
            rng=rng,  # type: ignore[arg-type]
            elite_fraction=elite_fraction,
            run_id=run_id,
            **population_kwargs,
        )
        population._write_log()

        log_early_stopping_info(
            verbosity=verbosity,
            fitness_strategy=fitness_strategy,
            early_stopping_niter=early_stopping_niter,
            niter=niter,
            logger=logger,
        )

        # Track best value for early stopping (energy or fitness)
        best_value = None  # Energy for low_energy, fitness for others
        generations_without_improvement = 0

        for generation in tqdm(
            range(niter),
            desc=f"  GA generations for {len(composition)} atoms",
            disable=not should_show_progress(verbosity),
        ):
            if use_adaptive_mutations:
                adaptive_config = get_adaptive_mutation_config(
                    composition=composition,
                    current_generation=generation,
                    total_generations=niter,
                    use_adaptive=True,
                    generations_without_improvement=generations_without_improvement,
                    stagnation_trigger=stagnation_trigger,
                    stagnation_full_trigger=stagnation_full_trigger,
                    recovery_window=recovery_window,
                    aggressive_burst_multiplier=aggressive_burst_multiplier,
                    max_mutation_probability=max_mutation_probability,
                )
                mutations = update_mutation_weights(
                    operators_list=operators_list,
                    name_map=name_map,
                    adaptive_config=adaptive_config,
                )
                current_mutation_probability = adaptive_config["mutation_probability"]

            # Produce `n_offspring` children this generation (min 1).
            n_offspring = max(1, math.ceil(population_size * offspring_fraction))
            created = 0
            attempts = 0
            max_attempts = max(10, n_offspring * 10)

            t_loop = perf_counter()
            t_relax_gen = 0.0
            t_db_unrelaxed_gen = 0.0
            t_db_relaxed_gen = 0.0
            t_parent_select_gen = 0.0
            t_crossover_gen = 0.0
            t_mutation_gen = 0.0
            while created < n_offspring and attempts < max_attempts:
                attempts += 1
                t0 = perf_counter()
                candidates = population.get_two_candidates()
                t_parent_select_gen += perf_counter() - t0
                if candidates is None:
                    continue
                a1, a2 = candidates
                t0 = perf_counter()
                a3, desc = pairing.get_new_individual([a1, a2])
                t_crossover_gen += perf_counter() - t0
                if a3 is None:
                    continue

                if rng.random() < current_mutation_probability:
                    t0 = perf_counter()
                    a3_mutated = mutations.get_operator().mutate(a3)
                    t_mutation_gen += perf_counter() - t0
                    if a3_mutated is not None:
                        a3 = a3_mutated
                try:
                    validate_structure_for_system_type(
                        a3,
                        system_type=system_type,
                        surface_config=surface_config,
                        n_slab=n_slab,
                    )
                except ValueError as exc:
                    logger.debug(
                        "Offspring rejected by system_type validation: %s", exc
                    )
                    continue

                t_start = perf_counter()
                retry_with_backoff(
                    lambda _a3=a3, _desc=desc: da.add_unrelaxed_candidate(
                        _a3, description=_desc
                    ),
                    max_retries=5,
                    initial_delay=0.2,
                    backoff_factor=2.0,
                    exception_types=_HPC_RETRY_EXCEPTIONS,
                )
                t_db_unrelaxed_gen += perf_counter() - t_start

                if surface_mode:
                    attach_slab_constraints(
                        a3,
                        n_slab,
                        fix_all_slab_atoms=surface_config.fix_all_slab_atoms,
                        n_fix_bottom_slab_layers=surface_config.n_fix_bottom_slab_layers,
                        n_relax_top_slab_layers=surface_config.n_relax_top_slab_layers,
                        surface_normal_axis=surface_config.surface_normal_axis,
                    )
                t_start = perf_counter()
                perform_local_relaxation(
                    a3,
                    calculator,
                    optimizer,
                    fmax,
                    niter_local_relaxation,
                    center_after_relax=center_after_relax,
                )
                validate_structure_for_system_type(
                    a3,
                    system_type=system_type,
                    surface_config=surface_config,
                    n_slab=n_slab if surface_mode else None,
                )
                t_relax_gen += perf_counter() - t_start

                add_metadata(
                    a3,
                    generation=generation,
                    run_id=run_id,
                    **slab_ga_metadata_extras(surface_config, n_slab, system_type),
                )
                t_start = perf_counter()
                retry_with_backoff(
                    lambda _a3=a3: da.add_relaxed_step(_a3),
                    max_retries=5,
                    initial_delay=0.2,
                    backoff_factor=2.0,
                    exception_types=_HPC_RETRY_EXCEPTIONS,
                )
                t_db_relaxed_gen += perf_counter() - t_start

                created += 1
            profile_counters["offspring_created"] += created
            profile_timings["offspring_mutation_queue_s"] = profile_timings.get(
                "offspring_mutation_queue_s", 0.0
            ) + (perf_counter() - t_loop)
            profile_timings["offspring_parent_select_s"] = (
                profile_timings.get("offspring_parent_select_s", 0.0)
                + t_parent_select_gen
            )
            profile_timings["offspring_crossover_s"] = (
                profile_timings.get("offspring_crossover_s", 0.0) + t_crossover_gen
            )
            profile_timings["offspring_mutation_s"] = (
                profile_timings.get("offspring_mutation_s", 0.0) + t_mutation_gen
            )
            profile_timings["offspring_local_relaxation_s"] = (
                profile_timings.get("offspring_local_relaxation_s", 0.0) + t_relax_gen
            )
            profile_timings["offspring_unrelaxed_insert_s"] = (
                profile_timings.get("offspring_unrelaxed_insert_s", 0.0)
                + t_db_unrelaxed_gen
            )
            profile_timings["offspring_relaxed_write_s"] = (
                profile_timings.get("offspring_relaxed_write_s", 0.0) + t_db_relaxed_gen
            )
            profile_timings["offspring_db_io_s"] = profile_timings.get(
                "offspring_db_io_s", 0.0
            ) + (t_db_unrelaxed_gen + t_db_relaxed_gen)

            logger.debug(
                f"Generation {generation}: created and tagged {created} offspring"
            )

            # Update population once per generation (reflects all created offspring)
            t_start = perf_counter()
            population.update()
            pop_update_s = perf_counter() - t_start
            profile_timings["population_update_s"] = (
                profile_timings.get("population_update_s", 0.0) + pop_update_s
            )

            if per_generation is not None:
                per_generation.append(
                    {
                        "generation": int(generation),
                        "n_offspring_target": int(n_offspring),
                        "offspring_created": int(created),
                        "attempts": int(attempts),
                        "timings_s": {
                            "parent_select_s": t_parent_select_gen,
                            "crossover_s": t_crossover_gen,
                            "mutation_s": t_mutation_gen,
                            "db_unrelaxed_insert_s": t_db_unrelaxed_gen,
                            "relax_s": t_relax_gen,
                            "db_relaxed_write_s": t_db_relaxed_gen,
                            "population_update_s": pop_update_s,
                            "offspring_loop_wall_s": perf_counter() - t_loop,
                        },
                    }
                )

            if early_stopping_niter > 0:
                best_value, generations_without_improvement, should_stop = (
                    update_early_stopping_state_unified(
                        population=population,
                        fitness_strategy=fitness_strategy,
                        best_value=best_value,
                        generations_without_improvement=generations_without_improvement,
                        early_stopping_niter=early_stopping_niter,
                    )
                )
                if should_stop:
                    if verbosity >= 1:
                        stopping_metric = (
                            "fitness"
                            if fitness_strategy != FitnessStrategy.LOW_ENERGY
                            else "energy"
                        )
                        logger.info(
                            f"Early stopping triggered: no {stopping_metric} improvement for "
                            f"{generations_without_improvement} generations "
                            f"(best {stopping_metric}: {best_value:.6f})"
                        )
                    break

        all_candidates = retry_with_backoff(
            da.get_all_relaxed_candidates,
            max_retries=5,
            initial_delay=0.2,
            backoff_factor=2.0,
            exception_types=_HPC_RETRY_EXCEPTIONS,
        )
        if run_id is not None:
            all_candidates = filter_by_metadata(all_candidates, run_id=run_id)
        all_minima = extract_minima_from_database(all_candidates)

        if verbosity >= 1:
            logger.info(
                f"GA evolution complete. Found {len(all_minima)} unique minima."
            )

        # Sort by fitness (highest first) for non-default strategies
        sort_minima_by_fitness(
            all_minima=all_minima,
            fitness_strategy=fitness_strategy,
            logger=logger,
        )
        profile_timings["total_wall_s"] = perf_counter() - profile_t0
        profile_timings["cpu_non_relax_s"] = max(
            0.0,
            profile_timings["total_wall_s"]
            - profile_timings.get("initial_local_relaxation_s", 0.0)
            - profile_timings.get("offspring_local_relaxation_s", 0.0),
        )
        log_timing_summary(logger, "ase_ga", profile_timings, verbosity=verbosity)
        if write_timing_json:
            out_payload: dict[str, Any] = {
                "backend": "ase_ga",
                "timings_s": profile_timings,
                "counters": profile_counters,
            }
            if per_generation is not None:
                out_payload["per_generation"] = per_generation
            write_timing_file(output_dir, out_payload)

        return all_minima

    finally:
        close_data_connection(da, log_errors=False)
