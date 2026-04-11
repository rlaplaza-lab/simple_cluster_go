"""Shared components for Genetic Algorithm implementations.

This module contains code shared between the standard GA and TorchSim-enhanced GA
implementations to reduce duplication.
"""

from __future__ import annotations

import logging
import typing

import numpy as np
from numpy.random import Generator

if typing.TYPE_CHECKING:
    from scgo.utils.diversity_scorer import DiversityScorer
    from scgo.utils.fitness_strategies import FitnessStrategy
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase_ga.offspring_creator import OperationSelector
from ase_ga.standard_comparators import (
    InteratomicDistanceComparator,
    RawScoreComparator,
    SequentialComparator,
)
from ase_ga.startgenerator import StartGenerator
from ase_ga.utilities import closest_distances_generator, get_all_atom_types

from scgo.ase_ga_patches.cutandsplicepairing import CutAndSplicePairing
from scgo.ase_ga_patches.population import Population
from scgo.ase_ga_patches.standardmutations import (
    AnisotropicRattleMutation,
    CustomPermutationMutation,
    FlatteningMutation,
    MirrorMutation,
    RattleMutation,
    RotationalMutation,
)
from scgo.constants import (
    DEFAULT_COMPARATOR_TOL,
    DEFAULT_PAIR_COR_CUM_DIFF,
    DEFAULT_PAIR_COR_MAX,
)

# Prefer metadata reader for raw_score and other fields
from scgo.database.metadata import get_metadata
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.deposition import (
    create_deposited_cluster,
    create_deposited_cluster_batch,
)
from scgo.utils.rng_helpers import create_child_rng, ensure_rng_or_create
from scgo.utils.validation import (
    validate_in_range,
    validate_integer,
    validate_positive,
)


def slab_ga_metadata_extras(
    surface_config: SurfaceSystemConfig | None, n_slab: int
) -> dict[str, int | str]:
    """Extra metadata for slab+adsorbate GA (atom order: slab indices 0..n_slab-1)."""
    if surface_config is None or n_slab <= 0:
        return {}
    return {"n_slab_atoms": n_slab, "system_kind": "slab_adsorbate"}


def validate_ga_common_params(
    niter: int,
    population_size: int,
    n_jobs_population_init: int,
    calculator: typing.Any,
    mutation_probability: float,
    offspring_fraction: float,
    vacuum: float,
    fmax: float | None = None,
) -> None:
    """Validate parameters shared by GA and GA TorchSim implementations."""
    validate_integer("niter", niter)
    validate_positive("niter", niter, strict=True)
    validate_integer("population_size", population_size)
    validate_positive("population_size", population_size, strict=True)
    if n_jobs_population_init not in (-1, -2) and n_jobs_population_init < 1:
        raise ValueError(
            f"n_jobs_population_init must be -1, -2, or >= 1, got {n_jobs_population_init}"
        )
    if calculator is None:
        raise ValueError("calculator is required for genetic algorithm")
    if fmax is not None:
        validate_positive("fmax", fmax, strict=True)
    validate_in_range("mutation_probability", mutation_probability, 0.0, 1.0)
    validate_in_range("offspring_fraction", offspring_fraction, 0.0, 1.0)
    validate_positive("offspring_fraction", offspring_fraction, strict=True)
    validate_positive("vacuum", vacuum, strict=True)


class ClusterStartGenerator(StartGenerator):
    """StartGenerator creating initial clusters.

    Uses :func:`scgo.initialization.create_initial_cluster_batch` to produce
    starting candidates. When population_size is provided, pre-generates the
    entire population in one batch call.
    """

    def __init__(
        self,
        composition: list[str],
        vacuum: float,
        rng: np.random.Generator | None = None,
        calculator: Calculator | None = None,
        population_size: int | None = None,
        mode: str = "smart",
        n_jobs: int = 1,
    ) -> None:
        """Initialize ClusterStartGenerator.

        Args:
            composition: List of atomic symbols defining the cluster.
            vacuum: Amount of vacuum to add around the cluster.
            rng: Optional numpy random number generator for reproducibility.
            calculator: Optional calculator to assign to generated atoms.
            population_size: Optional total population size. If provided, pre-generates
                entire population in one batch call. If None, generates on demand.
            mode: Initialization mode. Default "smart".
            n_jobs: Number of parallel workers for batch initialization.
                Default 1 (sequential). Special values: -1 (all CPUs), -2 (all except one).
                Only used when population_size is provided.
        """
        # Normalize RNG if provided; allow None (falls back to default RNG later)
        self.rng: Generator | None = (
            ensure_rng_or_create(rng) if rng is not None else None
        )
        self.composition: list[str] = composition
        self.vacuum: float = vacuum
        self.calculator: Calculator | None = calculator
        self.population_size: int | None = population_size
        self.mode: str = mode
        self.n_jobs: int = n_jobs
        self._candidate_count = 0
        self._candidate_batch: list[Atoms] | None = None

        if population_size is not None and self.rng is not None:
            from scgo.initialization import create_initial_cluster_batch

            self._candidate_batch = create_initial_cluster_batch(
                composition=composition,
                n_structures=population_size,
                rng=self.rng,
                vacuum=vacuum,
                mode=mode,
                n_jobs=n_jobs,
            )

    def get_new_candidate(self, maxiter: typing.Any = None) -> Atoms:
        """Generate a single new, random cluster candidate.

        If population_size was provided, serves candidates from pre-generated batch.
        Otherwise, generates structures on-demand.
        """
        atoms = None
        if self._candidate_batch is not None and self._candidate_count < len(
            self._candidate_batch
        ):
            atoms = self._candidate_batch[self._candidate_count]
            self._candidate_count += 1

        if atoms is None:
            from scgo.initialization import create_initial_cluster

            atoms = create_initial_cluster(
                self.composition,
                vacuum=self.vacuum,
                rng=self.rng or np.random.default_rng(),
                mode=self.mode,
            )

        assert atoms is not None
        if self.calculator is not None:
            atoms.calc = self.calculator
        return atoms


class SurfaceClusterStartGenerator(StartGenerator):
    """StartGenerator for slab + adsorbate cluster using :mod:`scgo.surface.deposition`."""

    def __init__(
        self,
        composition: list[str],
        slab: Atoms,
        surface_config: SurfaceSystemConfig,
        blmin: dict,
        rng: np.random.Generator | None = None,
        calculator: Calculator | None = None,
        population_size: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        self.rng: Generator | None = (
            ensure_rng_or_create(rng) if rng is not None else None
        )
        self.composition = composition
        self.slab = slab.copy()
        self.surface_config = surface_config
        self.blmin = blmin
        self.calculator = calculator
        self.population_size = population_size
        self.n_jobs = n_jobs
        self._candidate_count = 0
        self._candidate_batch: list[Atoms] | None = None

        if population_size is not None and self.rng is not None:
            self._candidate_batch = create_deposited_cluster_batch(
                composition=composition,
                slab=self.slab,
                blmin=blmin,
                n_structures=population_size,
                rng=self.rng,
                config=surface_config,
                n_jobs=n_jobs,
            )

    def get_new_candidate(self, maxiter: typing.Any = None) -> Atoms:
        atoms = None
        if self._candidate_batch is not None and self._candidate_count < len(
            self._candidate_batch
        ):
            atoms = self._candidate_batch[self._candidate_count]
            self._candidate_count += 1

        if atoms is None:
            atoms = create_deposited_cluster(
                self.composition,
                self.slab,
                self.blmin,
                self.rng or np.random.default_rng(),
                self.surface_config,
            )
            if atoms is None:
                raise RuntimeError(
                    "SurfaceClusterStartGenerator could not place a valid structure; "
                    "increase max_placement_attempts or height range."
                )

        if self.calculator is not None:
            atoms.calc = self.calculator
        return atoms


def create_ga_pairing(
    atoms_template: Atoms,
    n_to_optimize: int,
    rng: np.random.Generator | None = None,
    slab_atoms: Atoms | None = None,
) -> CutAndSplicePairing:
    """Create a CutAndSplicePairing operator for GA evolution.

    Accepts an optional RNG; if provided it will be used as a parent RNG
    for creating child RNGs for internal operators.

    Args:
        atoms_template: Template Atoms object with cell and pbc settings.
        n_to_optimize: Number of atoms to optimize (trailing ``n_top`` atoms).
        rng: Random number generator.
        slab_atoms: If provided, real slab atoms for adsorbate GA (non-empty).
            If None, an empty slab with the template cell/pbc is used (gas-phase GA).

    Returns:
        Configured CutAndSplicePairing operator.
    """
    n_template = len(atoms_template)
    if slab_atoms is not None and len(slab_atoms) > 0:
        if n_template != len(slab_atoms) + n_to_optimize:
            raise ValueError(
                "atoms_template length must equal len(slab_atoms) + n_to_optimize "
                f"for surface GA, got {n_template}, slab={len(slab_atoms)}, "
                f"n_to_optimize={n_to_optimize}"
            )
        idx_top = range(len(slab_atoms), n_template)
    else:
        if n_template != n_to_optimize:
            raise ValueError(
                "atoms_template length must equal n_to_optimize for gas-phase GA"
            )
        idx_top = range(n_to_optimize)

    all_atom_types = get_all_atom_types(atoms_template, idx_top)
    blmin = closest_distances_generator(all_atom_types, ratio_of_covalent_radii=0.7)

    if slab_atoms is not None and len(slab_atoms) > 0:
        slab = slab_atoms.copy()
    else:
        slab = Atoms(cell=atoms_template.get_cell(), pbc=atoms_template.get_pbc())
    min_parent_fraction: float = min(0.5, max(0.3, 2.0 / max(1, n_to_optimize)))

    return CutAndSplicePairing(  # type: ignore[arg-type]
        slab,
        n_to_optimize,
        blmin,
        minfrac=min_parent_fraction,
        rng=create_child_rng(rng) if rng is not None else None,
    )


def create_mutation_operators(
    composition: list[str],
    n_to_optimize: int,
    blmin: dict,
    rng: np.random.Generator | None = None,
    use_adaptive: bool = True,
) -> tuple[list, dict[str, int]]:
    """Create mutation operators once at start of GA.

    Accepts an optional RNG (parent); child RNGs will be derived when needed.

    Args:
        composition: List of atomic symbols.
        n_to_optimize: Number of atoms to optimize.
        blmin: Bond length minimums dictionary.
        rng: Random number generator.
        use_adaptive: Whether to include adaptive mutation operators.

    Returns:
        Tuple of (operators_list, operator_name_to_index_map).
    """
    operators = []
    name_map = {}

    rattle: RattleMutation = RattleMutation(
        blmin,
        n_to_optimize,
        rattle_strength=0.8,
        rattle_prop=0.4,
        rng=create_child_rng(rng) if rng is not None else None,  # type: ignore[arg-type]
    )
    operators.append(rattle)
    name_map["rattle"] = 0

    if len(set(composition)) > 1:
        permutation: CustomPermutationMutation = CustomPermutationMutation(
            n_to_optimize, rng=create_child_rng(rng) if rng is not None else None
        )  # type: ignore[arg-type]
        operators.append(permutation)
        name_map["permutation"] = len(operators) - 1

    if use_adaptive:
        flattening: FlatteningMutation = FlatteningMutation(
            blmin,
            n_to_optimize,
            thickness_factor=0.5,
            rng=create_child_rng(rng) if rng is not None else None,  # type: ignore[arg-type]
        )
        operators.append(flattening)
        name_map["flattening"] = len(operators) - 1

        rotational: RotationalMutation = RotationalMutation(
            blmin,
            n_to_optimize,
            rng=create_child_rng(rng) if rng is not None else None,  # type: ignore[arg-type]
        )
        operators.append(rotational)
        name_map["rotational"] = len(operators) - 1

        mirror: MirrorMutation = MirrorMutation(
            blmin,
            n_to_optimize,
            rng=create_child_rng(rng) if rng is not None else None,  # type: ignore[arg-type]
        )
        operators.append(mirror)
        name_map["mirror"] = len(operators) - 1

        anisotropic: AnisotropicRattleMutation = AnisotropicRattleMutation(
            blmin,
            n_to_optimize,
            in_plane_strength=1.0,
            normal_strength=0.2,
            rattle_prop=0.5,
            rng=create_child_rng(rng) if rng is not None else None,  # type: ignore[arg-type]
        )
        operators.append(anisotropic)
        name_map["anisotropic_rattle"] = len(operators) - 1

    return operators, name_map


def update_mutation_weights(
    operators_list: list,
    name_map: dict[str, int],
    adaptive_config: dict,
) -> OperationSelector:
    """Update operator weights without recreating operators.

    Args:
        operators_list: List of operator instances.
        name_map: Mapping from operator names to list indices.
        adaptive_config: Config dict with operator_weights.

    Returns:
        Updated OperationSelector with new weights.
    """
    operator_weights = adaptive_config["operator_weights"]

    weights: list[float] = []
    for i in range(len(operators_list)):
        name: str | None = next((k for k, v in name_map.items() if v == i), None)
        if name and name in operator_weights:
            weights.append(operator_weights[name])
        else:
            weights.append(0.0)

    if "rattle" in name_map:
        rattle_idx: int = name_map["rattle"]
        rattle_op = operators_list[rattle_idx]
        rattle_op.rattle_strength = adaptive_config["rattle_strength"]
        rattle_op.rattle_prop = adaptive_config["rattle_prop"]

    if "anisotropic_rattle" in name_map:
        anisotropic_idx: int = name_map["anisotropic_rattle"]
        anisotropic_op = operators_list[anisotropic_idx]
        if "anisotropic_in_plane_strength" in adaptive_config:
            anisotropic_op.in_plane_strength = adaptive_config[
                "anisotropic_in_plane_strength"
            ]
        if "anisotropic_normal_strength" in adaptive_config:
            anisotropic_op.normal_strength = adaptive_config[
                "anisotropic_normal_strength"
            ]
        if "anisotropic_rattle_prop" in adaptive_config:
            anisotropic_op.rattle_prop = adaptive_config["anisotropic_rattle_prop"]

    return OperationSelector(weights, operators_list)


def create_structure_comparator(
    n_atoms: int,
    energy_tolerance: float,
    *,
    mic: bool = False,
) -> SequentialComparator:
    """Create a SequentialComparator for structure duplicate detection.

    This is the shared comparator factory used by both GA population management
    and run-level filtering to ensure consistent duplicate detection criteria.

    Args:
        n_atoms: Number of trailing (adsorbate) atoms to compare.
        energy_tolerance: Energy difference tolerance for duplicate detection (eV).
        mic: If True, use minimum-image convention for pairwise distances (slab PBC).

    Returns:
        Configured SequentialComparator.
    """
    return SequentialComparator(
        methods=[
            RawScoreComparator(dist=energy_tolerance),
            InteratomicDistanceComparator(
                n_top=n_atoms,
                mic=mic,
                dE=energy_tolerance,
                pair_cor_cum_diff=DEFAULT_PAIR_COR_CUM_DIFF,
            ),
        ],
    )


def update_early_stopping_state(
    population: Population,
    best_energy: float | None,
    generations_without_improvement: int,
    early_stopping_niter: int,
) -> tuple[float | None, int, bool]:
    """Update early stopping state and determine if stopping should occur.

    This helper function centralizes the early stopping logic used by both
    standard and TorchSim GA implementations to track the best energy found
    and count consecutive generations without improvement.

    Args:
         population: The GA Population object containing current candidates.
         best_energy: Current best energy found (None if not yet set).
         generations_without_improvement: Current count of generations without
             improvement.
        early_stopping_niter: Number of consecutive generations without improvement
                             required to trigger early stopping.

    Returns:
        Tuple of (updated_best_energy, updated_generations_without_improvement,
                  should_stop).
        should_stop is True if early stopping should be triggered.
    """
    if len(population.pop) == 0:
        return best_energy, generations_without_improvement, False

    current_best_energy = -float(
        get_metadata(population.pop[0], "raw_score", default=0.0)
    )

    if best_energy is None:
        return current_best_energy, 0, False

    if current_best_energy < best_energy:
        return current_best_energy, 0, False

    updated_generations: int = generations_without_improvement + 1
    should_stop: bool = updated_generations >= early_stopping_niter

    return best_energy, updated_generations, should_stop


def update_early_stopping_state_unified(
    population: Population,
    fitness_strategy: str | FitnessStrategy,
    best_value: float | None,
    generations_without_improvement: int,
    early_stopping_niter: int,
) -> tuple[float | None, int, bool]:
    """Update early stopping state for both energy-based and fitness-based strategies.

    This unified helper function centralizes early stopping logic used by both
    standard and TorchSim GA implementations. It handles both energy-based stopping
    (for low_energy strategy) and fitness-based stopping (for high_energy and diversity
    strategies).

    Args:
        population: The GA Population object containing current candidates.
        fitness_strategy: Fitness strategy name ("low_energy", "high_energy", "diversity").
        best_value: Current best value found (None if not yet set). For low_energy
            this is energy, for others it's fitness.
        generations_without_improvement: Current count of generations without
            improvement.
        early_stopping_niter: Number of consecutive generations without improvement
            required to trigger early stopping.

    Returns:
        Tuple of (updated_best_value, updated_generations_without_improvement,
                  should_stop).
        should_stop is True if early stopping should be triggered.
    """
    from scgo.utils.fitness_strategies import FitnessStrategy, get_fitness_from_atoms

    if isinstance(fitness_strategy, str):
        fitness_strategy = FitnessStrategy(fitness_strategy)

    if fitness_strategy != FitnessStrategy.LOW_ENERGY:
        # Fitness-based early stopping
        if len(population.pop) == 0:
            return best_value, generations_without_improvement, False

        current_best_fitness: float = max(
            (
                get_fitness_from_atoms(atoms_obj, default=-float("inf"))
                for atoms_obj in population.pop
            ),
        )

        if best_value is None or current_best_fitness > best_value:
            return current_best_fitness, 0, False

        updated_generations: int = generations_without_improvement + 1
        should_stop: bool = updated_generations >= early_stopping_niter
        return best_value, updated_generations, should_stop
    else:
        # Energy-based early stopping (delegate to existing function)
        return update_early_stopping_state(
            population=population,
            best_energy=best_value,
            generations_without_improvement=generations_without_improvement,
            early_stopping_niter=early_stopping_niter,
        )


def setup_diversity_scorer(
    fitness_strategy: str | FitnessStrategy,
    diversity_reference_db: str | None,
    composition: list[str],
    n_to_optimize: int,
    diversity_max_references: int,
    logger,
) -> DiversityScorer | None:
    """Setup DiversityScorer for diversity fitness strategy.

    Args:
        fitness_strategy: Fitness strategy name.
        diversity_reference_db: Glob pattern for reference structure databases.
        composition: List of atomic symbols.
        n_to_optimize: Number of atoms to optimize.
        diversity_max_references: Maximum number of reference structures to load.
        logger: Logger instance for logging messages.

    Returns:
        DiversityScorer instance if fitness_strategy is "diversity", None otherwise.

    Raises:
        ValueError: If diversity_reference_db is None when fitness_strategy is "diversity".
    """
    from scgo.utils.diversity_scorer import DiversityScorer
    from scgo.utils.fitness_strategies import FitnessStrategy

    if isinstance(fitness_strategy, str):
        fitness_strategy = FitnessStrategy(fitness_strategy)

    if fitness_strategy != FitnessStrategy.DIVERSITY:
        return None

    if diversity_reference_db is None:
        raise ValueError(
            "diversity_reference_db is required when fitness_strategy='diversity'"
        )

    from scgo.database import SCGODatabaseManager

    if logger.isEnabledFor(logging.INFO):
        logger.info(f"Loading reference structures from: {diversity_reference_db}")
    # Use current working directory as base_dir since this function doesn't have output_dir
    # The database manager will search relative to cwd for the glob pattern
    with SCGODatabaseManager(base_dir=".", enable_caching=True) as db_manager:
        reference_structures: list[Atoms] = db_manager.load_diversity_references(
            glob_pattern=diversity_reference_db,
            composition=composition,
            max_structures=diversity_max_references,
        )
    if logger.isEnabledFor(logging.INFO):
        logger.info(f"Loaded {len(reference_structures)} reference structures")

    if not reference_structures:
        logger.warning(
            "No reference structures found for diversity strategy. "
            "This may result in poor diversity optimization."
        )
        return None

    from scgo.utils.comparators import PureInteratomicDistanceComparator

    comparator_for_diversity = PureInteratomicDistanceComparator(
        n_top=n_to_optimize,
        tol=DEFAULT_COMPARATOR_TOL,
        pair_cor_max=DEFAULT_PAIR_COR_MAX,
        mic=False,
    )
    return DiversityScorer(reference_structures, comparator_for_diversity)


def select_population_class(
    fitness_strategy: str | FitnessStrategy,
    diversity_scorer: DiversityScorer | None,
    diversity_update_interval: int,
    logger,
) -> tuple[type, dict]:
    """Select appropriate Population class based on fitness strategy.

    Args:
        fitness_strategy: Fitness strategy name.
        diversity_scorer: DiversityScorer instance (if using diversity strategy).
        diversity_update_interval: Number of generations between reference updates.
        logger: Logger instance for logging messages.

    Returns:
        Tuple of (PopulationClass, population_kwargs).
    """
    from scgo.utils.fitness_strategies import FitnessStrategy

    if isinstance(fitness_strategy, str):
        fitness_strategy = FitnessStrategy(fitness_strategy)

    if fitness_strategy != FitnessStrategy.LOW_ENERGY:
        from scgo.ase_ga_patches.population import FitnessStrategyPopulation

        PopulationClass = FitnessStrategyPopulation
        population_kwargs = {
            "fitness_strategy": fitness_strategy,
            "diversity_scorer": diversity_scorer,
            "diversity_update_interval": diversity_update_interval,
        }
        logger.info(
            f"Using FitnessStrategyPopulation with fitness_strategy='{fitness_strategy}'"
        )
    else:
        PopulationClass = Population
        population_kwargs = {}

    return PopulationClass, population_kwargs


def log_early_stopping_info(
    verbosity: int,
    fitness_strategy: str | FitnessStrategy,
    early_stopping_niter: int,
    niter: int,
    logger,
) -> None:
    """Log early stopping configuration information.

    Args:
        verbosity: Verbosity level (0=quiet, 1=normal, 2=debug, 3=trace).
        fitness_strategy: Fitness strategy name.
        early_stopping_niter: Number of generations without improvement to trigger stopping.
        niter: Total number of generations to run.
        logger: Logger instance for logging messages.
    """
    if verbosity < 1:
        return

    from scgo.utils.fitness_strategies import FitnessStrategy

    if isinstance(fitness_strategy, str):
        fitness_strategy = FitnessStrategy(fitness_strategy)

    if logger.isEnabledFor(logging.INFO):
        logger.info(f"Starting GA evolution with {niter} generations...")
        logger.info(f"Using fitness_strategy='{fitness_strategy}'")
    if early_stopping_niter > 0:
        stopping_metric: str = (
            "fitness" if fitness_strategy != FitnessStrategy.LOW_ENERGY else "energy"
        )
        logger.info(
            f"Early stopping enabled: will stop after {early_stopping_niter} "
            f"generations with no {stopping_metric} improvement"
        )


def sort_minima_by_fitness(
    all_minima: list[tuple[float, Atoms]],
    fitness_strategy: str | FitnessStrategy,
    logger,
) -> None:
    """Sort minima by fitness for non-low_energy strategies.

    Mutates input list in place. For low_energy, list remains sorted by energy
    (lowest first). For other strategies, sorts by fitness (highest first).

    Args:
        all_minima: List of (energy, Atoms) tuples to sort.
        fitness_strategy: Fitness strategy name.
        logger: Logger instance.
    """
    from scgo.utils.fitness_strategies import FitnessStrategy, get_fitness_from_atoms

    if isinstance(fitness_strategy, str):
        fitness_strategy = FitnessStrategy(fitness_strategy)

    if fitness_strategy != FitnessStrategy.LOW_ENERGY:
        all_minima.sort(
            key=lambda x: get_fitness_from_atoms(x[1], default=-float("inf")),
            reverse=True,  # Higher fitness first
        )
        logger.info(
            f"Sorted {len(all_minima)} unique minima by {fitness_strategy} fitness"
        )
