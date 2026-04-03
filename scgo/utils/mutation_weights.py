"""Adaptive mutation weight calculation for genetic algorithm optimization.

This module provides functions to dynamically adjust mutation operator weights
and parameters based on cluster properties (composition, size) and optimization
stage (generation number).
"""

from __future__ import annotations

import numpy as np

from scgo.utils.helpers import get_composition_counts


def _renormalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Return normalized non-negative operator weights."""
    non_negative = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(non_negative.values())
    if total <= 0.0:
        n = len(non_negative)
        if n == 0:
            return {}
        return dict.fromkeys(non_negative, 1.0 / n)
    return {k: v / total for k, v in non_negative.items()}


def _compute_stagnation_level(
    generations_without_improvement: int,
    stagnation_trigger: int,
    stagnation_full_trigger: int,
) -> float:
    """Map stagnation count to [0, 1] exploration boost level."""
    if generations_without_improvement < stagnation_trigger:
        return 0.0
    if stagnation_full_trigger <= stagnation_trigger:
        return 1.0
    progress = (generations_without_improvement - stagnation_trigger) / (
        stagnation_full_trigger - stagnation_trigger
    )
    return float(np.clip(progress, 0.0, 1.0))


def _apply_stagnation_boost(
    base_weights: dict[str, float],
    level: float,
    burst_multiplier: float,
) -> dict[str, float]:
    """Boost flat-landscape operators under stagnation and renormalize."""
    if level <= 0.0:
        return _renormalize_weights(base_weights)

    boosted = dict(base_weights)
    factor = 1.0 + level * max(0.0, burst_multiplier - 1.0)

    if "anisotropic_rattle" in boosted:
        boosted["anisotropic_rattle"] *= factor * 1.25
    if "flattening" in boosted:
        boosted["flattening"] *= factor * 1.15
    if "mirror" in boosted:
        boosted["mirror"] *= factor * 1.10

    return _renormalize_weights(boosted)


def calculate_composition_weights(
    composition: list[str],
) -> tuple[dict[str, float], bool]:
    """Calculate mutation operator weights based on cluster composition.

    Args:
        composition: List of element symbols in the cluster.

    Returns:
        A tuple containing:
        - Dictionary mapping operator names to their weights.
        - Boolean indicating if PermutationMutation should be included.
    """
    element_counts = get_composition_counts(composition)
    n_elements = len(element_counts)

    if n_elements == 1:
        # Pure metallic cluster - focus on structural exploration
        weights = _renormalize_weights(
            {
                "rattle": 0.30,
                "flattening": 0.24,
                "rotational": 0.18,
                "mirror": 0.08,
                "anisotropic_rattle": 0.20,
            }
        )
        use_permutation = False
    else:
        # Multi-element cluster - balance composition and structure
        # Calculate composition mixing ratio
        total_atoms = len(composition)
        sorted_counts = sorted(element_counts.values(), reverse=True)
        majority_fraction = sorted_counts[0] / total_atoms

        if majority_fraction > 0.7:
            # Core-shell biased (one element dominates)
            weights = _renormalize_weights(
                {
                    "rattle": 0.22,
                    "permutation": 0.24,
                    "flattening": 0.16,
                    "rotational": 0.12,
                    "mirror": 0.08,
                    "anisotropic_rattle": 0.18,
                }
            )
        else:
            # Well-mixed composition
            weights = _renormalize_weights(
                {
                    "rattle": 0.18,
                    "permutation": 0.27,
                    "flattening": 0.16,
                    "rotational": 0.12,
                    "mirror": 0.07,
                    "anisotropic_rattle": 0.20,
                }
            )
        use_permutation = True

    return weights, use_permutation


def calculate_size_adjustment(n_atoms: int) -> dict[str, float]:
    """Calculate size-dependent adjustments for mutation parameters.

    Args:
        n_atoms: Number of atoms in the cluster.

    Returns:
        Dictionary with adjusted parameters for mutation operators.
    """
    if n_atoms <= 4:
        # Small clusters - stronger mutations acceptable
        return {
            "rattle_strength": 1.0,
            "mutation_probability": 0.30,
            "rattle_prop": 0.4,
        }
    if n_atoms <= 10:
        # Medium clusters - balanced exploration
        return {
            "rattle_strength": 0.8,
            "mutation_probability": 0.25,
            "rattle_prop": 0.3,
        }
    # Large clusters - gentler mutations
    return {
        "rattle_strength": 0.6,
        "mutation_probability": 0.20,
        "rattle_prop": 0.25,
    }


def calculate_generation_adjustment(
    current_generation: int,
    total_generations: int,
) -> dict[str, float]:
    """Calculate generation-dependent adjustments for mutation parameters.

    Strategy:
    - Early generations: Aggressive exploration
    - Mid generations: Balanced exploration
    - Late generations: Fine-tuning

    Args:
        current_generation: Current GA generation number (0-indexed).
        total_generations: Total number of generations to run.

    Returns:
        Dictionary with adjusted parameters for current generation.
    """
    progress = current_generation / max(total_generations, 1)

    if progress < 0.3:
        # Early stage - aggressive exploration
        return {
            "mutation_probability_multiplier": 1.5,
            "rattle_strength_multiplier": 1.2,
        }
    if progress < 0.7:
        # Mid stage - balanced
        return {
            "mutation_probability_multiplier": 1.0,
            "rattle_strength_multiplier": 1.0,
        }
    # Late stage - fine-tuning
    return {
        "mutation_probability_multiplier": 0.7,
        "rattle_strength_multiplier": 0.8,
    }


def get_adaptive_mutation_config(
    composition: list[str],
    current_generation: int = 0,
    total_generations: int = 10,
    use_adaptive: bool = True,
    generations_without_improvement: int = 0,
    stagnation_trigger: int = 4,
    stagnation_full_trigger: int = 8,
    recovery_window: int = 2,
    aggressive_burst_multiplier: float = 1.8,
    max_mutation_probability: float = 0.65,
) -> dict:
    """Get complete adaptive mutation configuration.

    This is the main entry point for adaptive mutation configuration.
    It combines composition-based, size-based, and generation-based
    adjustments into a single configuration dictionary.

    Args:
        composition: List of element symbols in the cluster.
        current_generation: Current GA generation number.
        total_generations: Total number of generations.
        use_adaptive: If False, returns default/static configuration.

    Returns:
        Dictionary containing:
        - 'operator_weights': Dict mapping operator names to probabilities.
        - 'use_permutation': Boolean for whether to include permutation.
        - 'mutation_probability': Overall mutation probability.
        - 'rattle_strength': Rattle mutation strength parameter.
        - 'rattle_prop': Rattle mutation proportion parameter.
    """
    if not use_adaptive:
        # Return default static configuration
        n_elements = len(set(composition))
        if n_elements == 1:
            return {
                "operator_weights": _renormalize_weights(
                    {"rattle": 0.6, "flattening": 0.15, "anisotropic_rattle": 0.25}
                ),
                "use_permutation": False,
                "mutation_probability": 0.2,
                "rattle_strength": 0.8,
                "rattle_prop": 0.3,
                "anisotropic_in_plane_strength": 1.0,
                "anisotropic_normal_strength": 0.2,
                "anisotropic_rattle_prop": 0.5,
            }
        return {
            "operator_weights": _renormalize_weights(
                {
                    "rattle": 0.3,
                    "permutation": 0.35,
                    "flattening": 0.1,
                    "anisotropic_rattle": 0.25,
                }
            ),
            "use_permutation": True,
            "mutation_probability": 0.2,
            "rattle_strength": 0.8,
            "rattle_prop": 0.3,
            "anisotropic_in_plane_strength": 1.0,
            "anisotropic_normal_strength": 0.2,
            "anisotropic_rattle_prop": 0.5,
        }

    # Adaptive configuration
    n_atoms = len(composition)

    # Get base weights from composition
    operator_weights, use_permutation = calculate_composition_weights(composition)

    # Get size-based adjustments
    size_params = calculate_size_adjustment(n_atoms)

    # Get generation-based adjustments
    gen_adjustments = calculate_generation_adjustment(
        current_generation,
        total_generations,
    )

    # Apply generation adjustments to size-based parameters
    final_mutation_prob = (
        size_params["mutation_probability"]
        * gen_adjustments["mutation_probability_multiplier"]
    )
    final_rattle_strength = (
        size_params["rattle_strength"] * gen_adjustments["rattle_strength_multiplier"]
    )
    final_rattle_prop = size_params["rattle_prop"]
    anisotropic_in_plane_strength = np.clip(final_rattle_strength * 1.25, 0.4, 1.8)
    anisotropic_normal_strength = np.clip(final_rattle_strength * 0.35, 0.1, 0.6)
    anisotropic_rattle_prop = np.clip(final_rattle_prop * 1.10, 0.1, 0.9)

    # Ensure values stay in reasonable bounds.
    final_mutation_prob = np.clip(final_mutation_prob, 0.1, max_mutation_probability)
    final_rattle_strength = np.clip(final_rattle_strength, 0.3, 1.2)
    final_rattle_prop = np.clip(final_rattle_prop, 0.1, 0.9)

    # Stagnation-aware aggressive boost for flat-landscape exploration.
    level = _compute_stagnation_level(
        generations_without_improvement,
        stagnation_trigger=stagnation_trigger,
        stagnation_full_trigger=stagnation_full_trigger,
    )
    if recovery_window > 0 and generations_without_improvement < stagnation_trigger:
        recovery_factor = min(1.0, generations_without_improvement / recovery_window)
        level *= recovery_factor

    operator_weights = _apply_stagnation_boost(
        operator_weights,
        level=level,
        burst_multiplier=aggressive_burst_multiplier,
    )
    if level > 0.0:
        burst_scale = 1.0 + level * max(0.0, aggressive_burst_multiplier - 1.0)
        final_mutation_prob = np.clip(
            final_mutation_prob * burst_scale,
            0.1,
            max_mutation_probability,
        )
        final_rattle_strength = np.clip(final_rattle_strength * burst_scale, 0.3, 1.2)
        final_rattle_prop = np.clip(final_rattle_prop * burst_scale, 0.1, 0.9)
        anisotropic_in_plane_strength = np.clip(
            anisotropic_in_plane_strength * burst_scale,
            0.4,
            2.0,
        )
        anisotropic_normal_strength = np.clip(
            anisotropic_normal_strength * (1.0 + 0.5 * level),
            0.1,
            0.8,
        )
        anisotropic_rattle_prop = np.clip(
            anisotropic_rattle_prop * burst_scale,
            0.1,
            0.95,
        )

    return {
        "operator_weights": operator_weights,
        "use_permutation": use_permutation,
        "mutation_probability": final_mutation_prob,
        "rattle_strength": final_rattle_strength,
        "rattle_prop": final_rattle_prop,
        "anisotropic_in_plane_strength": anisotropic_in_plane_strength,
        "anisotropic_normal_strength": anisotropic_normal_strength,
        "anisotropic_rattle_prop": anisotropic_rattle_prop,
    }
