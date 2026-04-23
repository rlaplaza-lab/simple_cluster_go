"""Basin Hopping global optimization implementation for atomic clusters.

This module implements the Basin Hopping algorithm, a global optimization method
that explores the potential energy surface through iterative random moves and
local minimizations, with Metropolis acceptance criteria.
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

import numpy as np
from ase import Atoms
from ase.optimize import LBFGS
from ase.optimize.optimize import Optimizer
from tqdm import tqdm

from scgo.constants import (
    BOLTZMANN_K_EV_PER_K,
    DEFAULT_COMPARATOR_TOL,
    DEFAULT_ENERGY_TOLERANCE,
    DEFAULT_PAIR_COR_MAX,
)
from scgo.database import HPC_DATABASE_EXCEPTIONS, SCGODatabaseManager, setup_database
from scgo.database.metadata import persist_provenance
from scgo.database.sync import retry_with_backoff
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.constraints import attach_slab_constraints_from_surface_config
from scgo.system_types import (
    SystemType,
    get_system_policy,
    validate_structure_for_system_type,
    validate_system_type_settings,
)
from scgo.utils.comparators import PureInteratomicDistanceComparator
from scgo.utils.diversity_scorer import DiversityScorer
from scgo.utils.fitness_strategies import (
    FitnessStrategy,
    calculate_fitness,
    get_fitness_from_atoms,
    set_fitness_in_atoms,
    validate_fitness_strategy,
)
from scgo.utils.helpers import (
    _create_energy_bins,
    _find_unique_minima_with_binning,
    extract_minima_from_database,
    perform_local_relaxation,
)
from scgo.utils.logging import get_logger, should_show_progress
from scgo.utils.timing_report import log_timing_summary, write_timing_file
from scgo.utils.validation import (
    validate_atoms,
    validate_calculator_attached,
    validate_in_choices,
    validate_in_range,
    validate_integer,
    validate_positive,
)


def _move_atoms(
    atoms: Atoms,
    dr: float,
    move_fraction: float = 0.3,
    move_strategy: str = "random",
    rng: np.random.Generator | None = None,
    movable_indices: list[int] | None = None,
) -> tuple[Atoms, str]:
    """Apply a random displacement to a subset of atoms.

    Args:
        atoms: The ASE Atoms object to apply displacement to.
        dr: The maximum displacement distance for each atom during the random
            move step (in Angstrom).
        move_fraction: The fraction of atoms to move during each perturbation step.
        move_strategy: The strategy for selecting atoms to move ('random',
            'highest_force', 'lowest_force').
        rng: Optional numpy random number generator for reproducibility.

    Returns:
        A tuple (Atoms, description) where description lists moved atoms
        in 1-indexed form.
    """
    atoms_new = atoms.copy()

    if movable_indices is None:
        movable_indices = list(range(len(atoms_new)))
    n_atoms = len(movable_indices)

    if n_atoms <= 1:
        moved = movable_indices[0] if movable_indices else 0
        return atoms_new, f"Moved_atoms: {moved + 1} {moved + 1}"

    n_to_move_calculated = int(n_atoms * move_fraction)
    n_to_move = min(n_atoms, max(2, n_to_move_calculated))

    if move_strategy == "random":
        indices_to_move = list(
            rng.choice(movable_indices, size=n_to_move, replace=False),
        )
    elif move_strategy in ["highest_force", "lowest_force"]:
        forces = atoms.get_forces()
        force_magnitudes = np.linalg.norm(forces[movable_indices], axis=1)
        sorted_local = np.argsort(force_magnitudes)
        sorted_indices = np.asarray(movable_indices, dtype=int)[sorted_local]
        if move_strategy == "highest_force":
            indices_to_move = sorted_indices[-n_to_move:]
        else:  # lowest_force
            indices_to_move = sorted_indices[:n_to_move]
    else:
        raise ValueError(f"Unknown move_strategy: {move_strategy}")

    positions = atoms_new.get_positions()
    cm = atoms_new.get_center_of_mass()

    disp = np.zeros_like(positions)
    disp[indices_to_move, :] = rng.uniform(-1.0, 1.0, (n_to_move, 3))
    positions_new = positions + dr * disp
    atoms_new.set_positions(positions_new)

    new_cm = atoms_new.get_center_of_mass()
    atoms_new.translate(cm - new_cm)

    moved_indices_str = " ".join(str(i + 1) for i in sorted(indices_to_move))
    desc = f"Moved_atoms: {moved_indices_str}"
    return atoms_new, desc


def bh_go(
    atoms: Atoms,
    output_dir: str,
    niter: int = 100,
    fmax: float = 0.05,
    niter_local_relaxation: int = 250,
    optimizer: type[Optimizer] = LBFGS,
    dr: float = 0.5,
    move_fraction: float = 0.3,
    move_strategy: str = "random",
    temperature: float = 500 * BOLTZMANN_K_EV_PER_K,
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
    write_timing_json: bool = False,
    detailed_timing: bool = False,
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
        temperature: Temperature for Metropolis criterion (eV), governing acceptance
            of structures based on fitness differences.
        write_timing_json: Optional ``timing.json`` under ``output_dir``.
        detailed_timing: Per-iteration split rows in JSON when ``write_timing_json`` is set.
        deduplicate: If True (default), filter to structurally unique minima.
        energy_tolerance: Energy difference (eV) below which structures are considered duplicates.
        comparator_tol: Tolerance for interatomic distance comparator.
        comparator_pair_cor_max: Maximum pair correlation for comparator.
        comparator_n_top: Number of top distances to use in comparator. If None, uses all.
        verbosity: Verbosity level (0=quiet, 1=normal, 2=debug, 3=trace). Default 1.
        run_id: Optional run ID for tracking.
        clean: If True, start fresh (ignore previous databases).
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
        TypeError: If atoms is not an Atoms object or niter is not an integer.
        ValueError: If calculator is not attached or parameters are invalid.
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
    validate_system_type_settings(
        system_type=system_type, surface_config=surface_config
    )
    policy = get_system_policy(system_type)
    surface_mode = policy.uses_surface
    effective_dr = dr * (
        policy.adsorbate_move_scale if policy.constrain_adsorbate_moves else 1.0
    )
    effective_move_fraction = (
        min(move_fraction, 0.25) if policy.constrain_adsorbate_moves else move_fraction
    )

    logger = get_logger(__name__)

    # Validate and setup fitness strategy
    validate_fitness_strategy(fitness_strategy)
    if isinstance(fitness_strategy, str):
        fitness_strategy = FitnessStrategy(fitness_strategy)

    # Create comparator for diversity calculations and deduplication
    comparator = PureInteratomicDistanceComparator(
        n_top=comparator_n_top if comparator_n_top is not None else None,
        tol=comparator_tol,
        pair_cor_max=comparator_pair_cor_max,
        mic=surface_mode,
    )
    movable_indices = list(range(len(atoms)))
    if surface_mode:
        if n_slab <= 0:
            if surface_config is None:
                raise ValueError(
                    "Surface system type requires n_slab > 0 or surface_config."
                )
            n_slab = len(surface_config.slab)
        movable_indices = list(range(n_slab, len(atoms)))
        if not movable_indices:
            raise ValueError("Surface system has no movable atoms above slab.")

    # Load reference structures and create DiversityScorer for diversity strategy
    diversity_scorer = None
    if fitness_strategy == FitnessStrategy.DIVERSITY:
        if diversity_reference_db is None:
            raise ValueError(
                "diversity_reference_db is required when fitness_strategy='diversity'. "
                "Provide a glob pattern (e.g., '**/*.db') to find reference databases."
            )

        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Loading reference structures from: {diversity_reference_db}")
        with SCGODatabaseManager(
            base_dir=output_dir, enable_caching=True
        ) as db_manager:
            reference_structures = db_manager.load_diversity_references(
                glob_pattern=diversity_reference_db,
                composition=atoms.get_chemical_symbols(),
                max_structures=diversity_max_references,
            )
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Loaded {len(reference_structures)} reference structures")

        if not reference_structures:
            logger.warning(
                "No reference structures found for diversity strategy. "
                "This may result in poor diversity optimization."
            )
        else:
            diversity_scorer = DiversityScorer(reference_structures, comparator)

    # Detach calculator temporarily for DB setup to avoid pickling issues
    calc = atoms.calc
    atoms.calc = None
    da = setup_database(
        output_dir,
        "bh_go.db",
        atoms,
        initial_candidate=atoms,
        remove_existing=clean,
        run_id=run_id,
    )
    atoms.calc = calc

    _HPC_RETRY_EXCEPTIONS = HPC_DATABASE_EXCEPTIONS + (IOError,)

    try:
        profile_t0 = perf_counter()
        profile_timings: dict[str, float] = {}
        profile_counters: dict[str, int] = {"niter": int(niter), "accepted": 0}
        per_iteration: list[dict[str, Any]] | None = [] if detailed_timing else None

        def _finish_bh_timing() -> None:
            total = perf_counter() - profile_t0
            profile_timings["total_wall_s"] = total
            relax_sum = float(
                profile_timings.get("initial_local_relaxation_s", 0.0)
            ) + float(profile_timings.get("offspring_local_relaxation_s", 0.0))
            profile_timings["local_relaxation_s"] = relax_sum
            profile_timings["cpu_non_relax_s"] = max(0.0, total - relax_sum)
            log_timing_summary(
                logger, "basin_hopping", profile_timings, verbosity=verbosity
            )
            if write_timing_json:
                out: dict[str, Any] = {
                    "backend": "basin_hopping",
                    "timings_s": profile_timings,
                    "counters": profile_counters,
                }
                if per_iteration is not None:
                    out["per_iteration"] = per_iteration
                write_timing_file(output_dir, out)

        a_current = retry_with_backoff(
            da.get_an_unrelaxed_candidate,
            max_retries=5,
            initial_delay=0.2,
            backoff_factor=2.0,
            exception_types=_HPC_RETRY_EXCEPTIONS,
        )
        if surface_mode and surface_config is not None:
            attach_slab_constraints_from_surface_config(a_current, surface_config)
        t_rel0 = perf_counter()
        e_current = perform_local_relaxation(
            a_current,
            calculator,
            optimizer,
            fmax,
            niter_local_relaxation,
        )
        profile_timings["initial_local_relaxation_s"] = perf_counter() - t_rel0
        validate_structure_for_system_type(
            a_current,
            system_type=system_type,
            surface_config=surface_config,
            n_slab=n_slab if surface_mode else None,
        )

        if run_id is not None and a_current is not None:
            persist_provenance(a_current, run_id=run_id)

        t_db0 = perf_counter()
        retry_with_backoff(
            lambda: da.add_relaxed_step(a_current),
            max_retries=5,
            initial_delay=0.2,
            backoff_factor=2.0,
            exception_types=_HPC_RETRY_EXCEPTIONS,
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

        if verbosity >= 1:
            logger.info(
                f"Starting Basin Hopping with fitness_strategy='{fitness_strategy}' "
                f"(initial energy: {e_current:.4f} eV, fitness: {fitness_current:.4f})"
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
                effective_dr,
                effective_move_fraction,
                move_strategy=move_strategy,
                rng=rng,
                movable_indices=movable_indices,
            )
            if surface_mode and surface_config is not None:
                attach_slab_constraints_from_surface_config(a_trial, surface_config)
            if run_id is not None:
                persist_provenance(a_trial, run_id=run_id)

            t_ins0 = perf_counter()
            retry_with_backoff(
                lambda _t=a_trial, _d=desc: da.add_unrelaxed_candidate(
                    _t, description=_d
                ),
                max_retries=5,
                initial_delay=0.2,
                backoff_factor=2.0,
                exception_types=_HPC_RETRY_EXCEPTIONS,
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
            )
            dt_rel = perf_counter() - t_rel0
            profile_timings["offspring_local_relaxation_s"] = (
                profile_timings.get("offspring_local_relaxation_s", 0.0) + dt_rel
            )
            validate_structure_for_system_type(
                a_trial,
                system_type=system_type,
                surface_config=surface_config,
                n_slab=n_slab if surface_mode else None,
            )
            if run_id is not None:
                persist_provenance(a_trial, run_id=run_id)

            t_w0 = perf_counter()
            retry_with_backoff(
                lambda _t=a_trial: da.add_relaxed_step(_t),
                max_retries=5,
                initial_delay=0.2,
                backoff_factor=2.0,
                exception_types=_HPC_RETRY_EXCEPTIONS,
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
                if verbosity >= 2:
                    logger.debug(
                        f"Iteration {iteration}: Accepting (fitness improved: "
                        f"{fitness_current:.4f} → {fitness_trial:.4f})"
                    )
            elif temperature > 0.0:
                # Metropolis acceptance based on fitness difference
                fitness_diff = fitness_trial - fitness_current
                acceptance_prob = np.exp(fitness_diff / temperature)
                accept = rng.random() < acceptance_prob

                if verbosity >= 2:
                    logger.debug(
                        f"Iteration {iteration}: Metropolis test "
                        f"(fitness_diff: {fitness_diff:.4f}, "
                        f"acceptance_prob: {acceptance_prob:.4f}, accept: {accept})"
                    )

            if accept:
                profile_counters["accepted"] += 1
                a_current = a_trial.copy()
                e_current = e_trial
                fitness_current = fitness_trial

                # Periodic reference update for diversity strategy
                if (
                    fitness_strategy == FitnessStrategy.DIVERSITY
                    and diversity_scorer
                    and iteration % diversity_update_interval == 0
                ):
                    diversity_scorer.add_reference(a_trial)
                    if verbosity >= 2:
                        logger.debug(
                            f"Updated reference structures (total: {len(diversity_scorer)})"
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

        all_candidates = retry_with_backoff(
            da.get_all_relaxed_candidates,
            max_retries=5,
            initial_delay=0.2,
            backoff_factor=2.0,
            exception_types=_HPC_RETRY_EXCEPTIONS,
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

        # Reuse the comparator created earlier (line 200) for deduplication
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
