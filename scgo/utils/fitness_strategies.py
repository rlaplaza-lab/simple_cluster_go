"""Fitness strategy implementations for global optimization.

This module provides different fitness calculation strategies for global
optimization, enabling optimization objectives beyond simple energy minimization.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from ase import Atoms

from scgo.utils.validation import validate_in_choices

if TYPE_CHECKING:
    from scgo.utils.diversity_scorer import DiversityScorer


class FitnessStrategy(StrEnum):
    """Available fitness strategies for global optimization.

    Attributes:
        LOW_ENERGY: Minimize energy (find stable structures)
        HIGH_ENERGY: Maximize energy (find metastable structures)
        DIVERSITY: Maximize structural dissimilarity (find diverse structures)
    """

    LOW_ENERGY = "low_energy"
    HIGH_ENERGY = "high_energy"
    DIVERSITY = "diversity"


def validate_fitness_strategy(strategy: str) -> None:
    """Validate fitness strategy parameter.

    Args:
        strategy: Fitness strategy name to validate.

    Raises:
        ValueError: If strategy is not a valid choice.
    """
    valid_strategies = [s.value for s in FitnessStrategy]
    validate_in_choices("fitness_strategy", strategy, valid_strategies)


def calculate_fitness(
    energy: float,
    atoms: Atoms,
    strategy: str | FitnessStrategy,
    diversity_scorer: DiversityScorer | None = None,
) -> float:
    """Return fitness for strategy: low_energy=>-energy, high_energy=>energy, diversity=>diversity_scorer.score(atoms) (requires diversity_scorer)."""
    if isinstance(strategy, str):
        validate_fitness_strategy(strategy)
        strategy = FitnessStrategy(strategy)

    if strategy == FitnessStrategy.LOW_ENERGY:
        return -energy

    elif strategy == FitnessStrategy.HIGH_ENERGY:
        return energy

    elif strategy == FitnessStrategy.DIVERSITY:
        # If no references have been provided, return a neutral diversity score
        # (0.0) rather than raising. This allows algorithms to continue running
        # with a warning when reference structures are unavailable.
        if diversity_scorer is None:
            from scgo.utils.logging import get_logger

            logger = get_logger(__name__)
            logger.warning(
                "No diversity_scorer provided; returning 0.0 fitness for diversity strategy"
            )
            return 0.0

        return float(diversity_scorer.score(atoms))

    else:
        raise ValueError(f"Unknown fitness strategy: {strategy}")


def get_fitness_from_atoms(atoms: Atoms, default: float = 0.0) -> float:
    """Extract fitness value from Atoms object info dict.

    Args:
        atoms: Atoms object with fitness stored in info dict.
        default: Default value if fitness not found.

    Returns:
        Fitness value or default if not found.
    """
    return getattr(atoms, "info", {}).get("fitness", default)


def set_fitness_in_atoms(atoms: Atoms, fitness: float, strategy: str) -> None:
    """Store fitness value and strategy in Atoms object info dict.

    Args:
        atoms: Atoms object to modify.
        fitness: Fitness value to store.
        strategy: Fitness strategy name.
    """
    atoms.info = getattr(atoms, "info", {})
    atoms.info["fitness"] = fitness
    atoms.info["fitness_strategy"] = strategy
