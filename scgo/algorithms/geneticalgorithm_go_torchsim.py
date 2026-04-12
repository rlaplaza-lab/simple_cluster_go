"""TorchSim-enhanced Genetic Algorithm global optimization for clusters.

This module mirrors :mod:`scgo.geneticalgorithm_go` but swaps the sequential
ASE relaxation stage for a batched TorchSim relaxation helper. The database
interaction remains single-threaded to protect against SQLite locking issues.
"""

from __future__ import annotations

import math
import os
from contextlib import suppress
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize import FIRE
from ase.optimize.optimize import Optimizer
from ase_ga.data import DataConnection
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
from scgo.ase_ga_patches.population import Population
from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer
from scgo.constants import DEFAULT_ENERGY_TOLERANCE
from scgo.database import (
    HPC_DATABASE_EXCEPTIONS,
    close_data_connection,
    database_retry,
    setup_database,
)
from scgo.database.metadata import add_metadata, filter_by_metadata, update_metadata
from scgo.initialization import compute_cell_side
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.constraints import attach_slab_constraints
from scgo.utils.fitness_strategies import FitnessStrategy, validate_fitness_strategy
from scgo.utils.helpers import extract_minima_from_database
from scgo.utils.logging import get_logger, should_show_progress
from scgo.utils.mutation_weights import get_adaptive_mutation_config
from scgo.utils.rng_helpers import ensure_rng_or_create
from scgo.utils.validation import validate_composition


def _torchsim_prepare_relaxed_copy(
    cand: Atoms,
    surface_config: SurfaceSystemConfig | None,
    n_slab: int,
) -> Atoms:
    """Copy candidate and attach slab constraints before TorchSim relaxation."""
    c = cand.copy()
    if surface_config is not None and n_slab > 0:
        attach_slab_constraints(
            c,
            n_slab,
            fix_all_slab_atoms=surface_config.fix_all_slab_atoms,
            n_fix_bottom_slab_layers=surface_config.n_fix_bottom_slab_layers,
            n_relax_top_slab_layers=surface_config.n_relax_top_slab_layers,
            surface_normal_axis=surface_config.surface_normal_axis,
        )
    return c


def _relax_unrelaxed_candidates(
    da: DataConnection,
    relaxer: TorchSimBatchRelaxer,
    *,
    population: Population | None = None,
    max_batch: int | None = None,
    force: bool = False,
    generation: int | None = None,
    run_id: str | None = None,
    surface_config: SurfaceSystemConfig | None = None,
    n_slab: int = 0,
) -> int:
    """Relax unrelaxed candidates in batches and commit them to the database."""
    available = database_retry(
        da.get_number_of_unrelaxed_candidates,
        max_retries=5,
        operation_name="get_unrelaxed_candidates_count",
    )

    if available == 0:
        return 0
    if not force and max_batch is not None and available < max_batch:
        return 0

    to_take = available if force or max_batch is None else min(available, max_batch)

    # Batch read candidates under a single database connection
    def _read_batch_under_connection():
        """Read batch of candidates under a single connection."""
        batch: list[Atoms] = []
        with da.c:
            for _ in range(to_take):
                candidate = da.get_an_unrelaxed_candidate()
                if candidate is None:
                    break
                batch.append(candidate)
        return batch

    batch = database_retry(
        _read_batch_under_connection,
        max_retries=5,
        operation_name="read_candidate_batch",
    )

    if not batch:
        return 0

    relaxed_results = relaxer.relax_batch(
        [_torchsim_prepare_relaxed_copy(cand, surface_config, n_slab) for cand in batch]
    )
    if len(relaxed_results) != len(batch):
        raise RuntimeError("TorchSim relaxer returned mismatched batch size")

    # Batch write results under a single database connection
    def _write_batch_under_connection():
        """Write relaxed results under a single connection."""
        with da.c:
            for original, (energy, relaxed) in zip(batch, relaxed_results, strict=True):
                original.set_cell(relaxed.get_cell(), scale_atoms=True)
                original.set_pbc(relaxed.get_pbc())
                original.set_positions(relaxed.get_positions())

                # Copy forces if available (already converted to float64 by relaxer)
                if "forces" in relaxed.arrays:
                    original.arrays["forces"] = relaxed.arrays["forces"].copy()

                original.info.setdefault("key_value_pairs", {})
                update_metadata(
                    original,
                    **relaxed.info.get(
                        "key_value_pairs",
                        {"potential_energy": energy, "raw_score": -energy},
                    ),
                )
                extra = slab_ga_metadata_extras(surface_config, n_slab)
                if generation is not None:
                    add_metadata(
                        original,
                        generation=generation,
                        run_id=run_id,
                        **extra,
                    )
                elif run_id is not None:
                    add_metadata(original, run_id=run_id, **extra)

                original.calc = SinglePointCalculator(original, energy=energy)
                da.add_relaxed_step(original)

    database_retry(
        _write_batch_under_connection,
        max_retries=5,
        operation_name="write_relaxed_batch",
    )

    if population is not None:
        population.update()

    return len(batch)


def ga_go_torchsim(
    composition: list[str],
    output_dir: str,
    rng: np.random.Generator | None,
    calculator: Any,
    *,
    niter: int = 10,
    fmax: float = 0.05,
    niter_local_relaxation: int = 250,
    optimizer: type[Optimizer] = FIRE,
    energy_tolerance: float = DEFAULT_ENERGY_TOLERANCE,
    mutation_probability: float = 0.4,
    population_size: int = 10,
    offspring_fraction: float = 0.5,
    n_jobs_population_init: int = -2,
    vacuum: float = 10.0,
    use_adaptive_mutations: bool = True,
    stagnation_trigger: int = 4,
    stagnation_full_trigger: int = 8,
    recovery_window: int = 2,
    aggressive_burst_multiplier: float = 1.8,
    max_mutation_probability: float = 0.65,
    early_stopping_niter: int = 10,
    relaxer: TorchSimBatchRelaxer | None = None,
    batch_size: int | None = None,
    verbosity: int = 1,
    elite_fraction: float = 0.1,
    run_id: str | None = None,
    clean: bool = False,
    fitness_strategy: str = "low_energy",
    diversity_reference_db: str | None = None,
    diversity_max_references: int = 100,
    diversity_update_interval: int = 5,
    surface_config: SurfaceSystemConfig | None = None,
) -> list[tuple[float, Atoms]]:
    """Run the GA using TorchSim for batched relaxations.

    Parameters are mostly shared with :func:`scgo.geneticalgorithm_go.ga_go`.
    The ``relaxer`` argument controls TorchSim batching; when omitted the
    function instantiates a default :class:`TorchSimBatchRelaxer` using the
    provided ``fmax`` as a force tolerance.

    Args:
        composition: List of element symbols defining the cluster composition.
        calculator: ASE calculator for energy/force evaluations.
        early_stopping_niter: Number of consecutive generations with no improvement
                              before stopping early. Uses fitness for non-low_energy
                              strategies, energy for low_energy. If 0, no early stopping
                              is applied. Default 10.
        verbosity: Verbosity level (0=quiet, 1=normal, 2=debug, 3=trace). Defaults to 1.
        elite_fraction: Fraction of population to preserve as elite candidates
                         (top performers by fitness). Default 0.1 (top 10%).
        run_id: Optional run ID for tracking.
        clean: If True, start fresh (ignore previous databases).
        fitness_strategy: Fitness strategy to use. One of: "low_energy", "high_energy", "diversity".
            Defaults to "low_energy" (minimize energy).
        diversity_reference_db: Glob pattern for reference structure databases (for diversity strategy).
            Required when fitness_strategy="diversity", ignored otherwise.
        diversity_max_references: Maximum number of reference structures to load (for performance).
        diversity_update_interval: Number of generations between reference updates (for diversity strategy).
        surface_config: Same as :func:`scgo.geneticalgorithm_go.ga_go` (adsorbate-on-slab GA).
    """
    logger = get_logger(__name__)

    validate_composition(composition, allow_empty=False, allow_tuple=False)
    validate_ga_common_params(
        niter=niter,
        population_size=population_size,
        n_jobs_population_init=n_jobs_population_init,
        calculator=calculator,
        mutation_probability=mutation_probability,
        offspring_fraction=offspring_fraction,
        vacuum=vacuum,
    )

    # Validate and normalize fitness strategy (coerce to Enum)
    validate_fitness_strategy(fitness_strategy)
    if isinstance(fitness_strategy, str):
        fitness_strategy = FitnessStrategy(fitness_strategy)

    if batch_size is not None and batch_size <= 0:
        batch_size = None

    # Normalize RNG early and enforce Generator-only policy
    rng = ensure_rng_or_create(rng)

    if relaxer is None:
        relaxer = TorchSimBatchRelaxer(
            force_tol=fmax,
            mace_model_name="mace_matpes_0",
            max_steps=niter_local_relaxation,
        )
    elif getattr(relaxer, "max_steps", None) is None:
        relaxer.max_steps = niter_local_relaxation

    n_to_optimize = len(composition)

    if surface_config is not None:
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
    else:
        n_slab = 0
        slab_ref = None
        cell_side = compute_cell_side(composition, vacuum=vacuum)
        atoms_template = Atoms(
            symbols=composition,
            positions=[[0, 0, 0] for _ in range(n_to_optimize)],  # Dummy positions
            cell=[cell_side] * 3,
            pbc=False,
        )
    atoms_template.calc = calculator

    # Load reference structures and create DiversityScorer for diversity strategy
    diversity_scorer = setup_diversity_scorer(
        fitness_strategy=fitness_strategy,
        diversity_reference_db=diversity_reference_db,
        composition=composition,
        n_to_optimize=n_to_optimize,
        diversity_max_references=diversity_max_references,
        logger=logger,
    )

    slab_for_pairing = slab_ref
    pairing = create_ga_pairing(
        atoms_template, n_to_optimize, rng, slab_atoms=slab_for_pairing
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
        range(n_slab, n_slab + n_to_optimize)
        if surface_config is not None
        else range(n_to_optimize)
    )
    all_atom_types = get_all_atom_types(atoms_template, idx_top)
    blmin = closest_distances_generator(all_atom_types, ratio_of_covalent_radii=0.7)

    operators_list, name_map = create_mutation_operators(
        composition=composition,
        n_to_optimize=n_to_optimize,
        blmin=blmin,
        rng=rng,
        use_adaptive=use_adaptive_mutations,
        n_slab=n_slab,
        surface_normal_axis=(
            surface_config.surface_normal_axis if surface_config is not None else 2
        ),
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

    comp_mic = (
        bool(surface_config.comparator_use_mic) if surface_config is not None else False
    )
    comp = create_structure_comparator(n_to_optimize, energy_tolerance, mic=comp_mic)

    if surface_config is not None:
        assert slab_ref is not None
        start_generator = SurfaceClusterStartGenerator(
            composition,
            slab_ref,
            surface_config,
            blmin,
            rng=rng,
            calculator=None,
            population_size=population_size,
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
            n_jobs=n_jobs_population_init,
        )
    initial_population = [
        start_generator.get_new_candidate() for _ in range(population_size)
    ]

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
    # Insert unrelaxed starters via the low-level API, then batch-relax with TorchSim and tag generation=0.
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

        logger.debug(
            "Using GA database at %s",
            os.path.join(output_dir, "ga_go.db"),
        )

        initial_pop_count = 0

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

        for cand in initial_population:
            database_retry(
                lambda _cand=cand: _insert_unrelaxed(_cand),
                max_retries=5,
                operation_name="insert_unrelaxed_candidate",
            )

        # Helper to write a relaxed batch into the database under a single connection
        def _write_relaxed_batch(batch, relaxed_results):
            with da.c:
                for original, (energy, relaxed) in zip(
                    batch, relaxed_results, strict=True
                ):
                    original.set_cell(relaxed.get_cell(), scale_atoms=True)
                    original.set_pbc(relaxed.get_pbc())
                    original.set_positions(relaxed.get_positions())

                    # Copy forces if available
                    if "forces" in relaxed.arrays:
                        original.arrays["forces"] = relaxed.arrays["forces"].copy()

                    original.info.setdefault("key_value_pairs", {})
                    update_metadata(
                        original,
                        **relaxed.info.get(
                            "key_value_pairs",
                            {"potential_energy": energy, "raw_score": -energy},
                        ),
                    )
                    add_metadata(
                        original,
                        generation=0,
                        run_id=run_id,
                        **slab_ga_metadata_extras(surface_config, n_slab),
                    )
                    original.calc = SinglePointCalculator(original, energy=energy)
                    da.add_relaxed_step(original)

        # Process starting population in batches
        batch_size_internal = batch_size or len(initial_population)
        for i in range(0, len(initial_population), batch_size_internal):
            batch = initial_population[i : i + batch_size_internal]
            relaxed_results = relaxer.relax_batch(
                [
                    _torchsim_prepare_relaxed_copy(c, surface_config, n_slab)
                    for c in batch
                ]
            )
            if len(relaxed_results) != len(batch):
                raise RuntimeError("TorchSim relaxer returned mismatched batch size")

            database_retry(
                lambda _batch=batch, _results=relaxed_results: _write_relaxed_batch(
                    _batch, _results
                ),
                max_retries=5,
                operation_name="write_initial_relaxed_batch",
            )

            initial_pop_count += len(batch)

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
        logger.debug(
            "Initial Population created: size=%d, confids=%s",
            len(population.pop),
            [a.info.get("confid") for a in population.pop],
        )

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

            # Create up to `n_offspring` unrelaxed candidates for this generation;
            # TorchSim will handle batching/relaxation later.
            n_offspring = max(1, math.ceil(population_size * offspring_fraction))
            created = 0
            attempts = 0
            max_attempts = max(10, n_offspring * 10)

            while created < n_offspring and attempts < max_attempts:
                attempts += 1
                candidates = population.get_two_candidates()
                if candidates is None:
                    continue
                a1, a2 = candidates
                a3, desc = pairing.get_new_individual([a1, a2])
                if a3 is None:
                    continue

                if rng.random() < current_mutation_probability:
                    mutated = mutations.get_operator().mutate(a3)
                    if mutated is not None:
                        a3 = mutated

                database_retry(
                    lambda _a3=a3, _desc=desc: da.add_unrelaxed_candidate(
                        _a3, description=_desc
                    ),
                    max_retries=5,
                    operation_name="add_unrelaxed_offspring",
                )

                created += 1

            # Emit a concise per-generation summary at DEBUG level (one line)
            logger.debug(
                "Generation %s offspring loop: n_offspring=%d, created=%d, attempts=%d",
                generation,
                n_offspring,
                created,
                attempts,
            )

            # Ask TorchSim relaxer to process available unrelaxed candidates now.
            # Enforce a per-generation limit: when `batch_size` is None, treat the
            # per-call limit as the GA `n_offspring` so a single relax call does not
            # drain an unrelated backlog and make logs look cumulative.
            per_gen_max = batch_size if batch_size is not None else n_offspring
            offspring_count = _relax_unrelaxed_candidates(
                da,
                relaxer,
                population=population,
                max_batch=per_gen_max,
                generation=generation,
                run_id=run_id,
                surface_config=surface_config,
                n_slab=n_slab,
            )
            if offspring_count > 0:
                # Attempt to report a concise triple: created_this_gen / relaxed_this_call / total_relaxed
                try:
                    total_relaxed = database_retry(
                        da.get_all_relaxed_candidates,
                        max_retries=5,
                        operation_name="get_all_relaxed_candidates",
                    )
                    total_relaxed_cnt = len(total_relaxed)
                except HPC_DATABASE_EXCEPTIONS:
                    total_relaxed_cnt = None

                if total_relaxed_cnt is not None:
                    logger.debug(
                        "Generation %s: created=%d, relaxed_this_call=%d, total_relaxed=%d",
                        generation,
                        created,
                        offspring_count,
                        total_relaxed_cnt,
                    )
                else:
                    logger.debug(
                        "Generation %s: created=%d, relaxed_this_call=%d",
                        generation,
                        created,
                        offspring_count,
                    )

            population.update()

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

        _relax_unrelaxed_candidates(
            da,
            relaxer,
            population=population,
            max_batch=batch_size,
            force=True,
            run_id=run_id,
            surface_config=surface_config,
            n_slab=n_slab,
        )

        all_candidates = database_retry(
            da.get_all_relaxed_candidates,
            max_retries=5,
            operation_name="get_final_all_relaxed_candidates",
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

        return all_minima

    finally:
        close_data_connection(da, log_errors=False)
