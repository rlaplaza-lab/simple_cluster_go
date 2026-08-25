"""Adaptive mutation weight calculation for genetic algorithm optimization.

This module provides functions to dynamically adjust mutation operator weights
and parameters based on system type, cluster properties (composition, size), and
optimization stage (generation number).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scgo.utils.helpers import get_composition_counts

if TYPE_CHECKING:
    from scgo.system_types import AdsorbateDefinition, SystemType


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
    if "rotational" in boosted:
        boosted["rotational"] *= factor * 1.10
    if "mirror" in boosted:
        boosted["mirror"] *= factor * 1.10
    if "permutation" in boosted:
        # Order-disorder swaps are prime stagnation escapers for alloys.
        boosted["permutation"] *= factor * 1.10
    if "breathing" in boosted:
        boosted["breathing"] *= factor * 1.08
    if "overlap_relief" in boosted:
        boosted["overlap_relief"] *= factor * 1.12
    if "shell_swap" in boosted:
        boosted["shell_swap"] *= factor * 1.08
    if "fragment_reposition" in boosted:
        boosted["fragment_reposition"] *= factor * 1.15
    if "in_plane_slide" in boosted:
        boosted["in_plane_slide"] *= factor * 1.12
    if "in_plane_rotate" in boosted:
        boosted["in_plane_rotate"] *= factor * 1.12

    return _renormalize_weights(boosted)


def _core_composition_for_weights(
    composition: list[str],
    system_type: SystemType,
    adsorbate_definition: AdsorbateDefinition | None,
) -> list[str]:
    """Composition slice used for mono/multi-element alloy weight tables."""
    from scgo.system_types import get_system_policy

    policy = get_system_policy(system_type)
    if policy.has_adsorbate and adsorbate_definition is not None:
        return [str(s) for s in adsorbate_definition.core_symbols]
    return list(composition)


def _alloy_weights_from_core(
    core_composition: list[str],
) -> tuple[dict[str, float], bool]:
    """Return alloy-biased structural weights when the core is multi-element.

    Only operators registered for every alloy-capable system type are listed;
    surface-only keys (``in_plane_slide``) and surface-registered keys
    (``mirror``) are applied as floors by the per-type branches in
    :func:`calculate_system_type_weights`.
    """
    element_counts = get_composition_counts(core_composition)
    total_atoms = len(core_composition)
    sorted_counts = sorted(element_counts.values(), reverse=True)
    majority_fraction = sorted_counts[0] / max(total_atoms, 1)

    if majority_fraction > 0.7:
        weights = _renormalize_weights(
            {
                "rattle": 0.14,
                "overlap_relief": 0.14,
                "permutation": 0.18,
                "shell_swap": 0.12,
                "flattening": 0.11,
                "rotational": 0.07,
                "anisotropic_rattle": 0.11,
                "breathing": 0.05,
            }
        )
    else:
        weights = _renormalize_weights(
            {
                "rattle": 0.13,
                "overlap_relief": 0.12,
                "permutation": 0.18,
                "shell_swap": 0.14,
                "flattening": 0.10,
                "rotational": 0.07,
                "anisotropic_rattle": 0.10,
                "breathing": 0.04,
            }
        )
    return weights, True


def _pure_structural_weights(*, include_surface_slide: bool) -> dict[str, float]:
    """Base structural weights for a single-element cluster or core."""
    weights = {
        "rattle": 0.24,
        "overlap_relief": 0.18,
        "flattening": 0.18,
        "rotational": 0.14,
        "anisotropic_rattle": 0.14,
        "breathing": 0.05,
    }
    if include_surface_slide:
        weights["in_plane_slide"] = 0.03
        weights["in_plane_rotate"] = 0.03
    return _renormalize_weights(weights)


def calculate_system_type_weights(
    system_type: SystemType,
    composition: list[str],
    adsorbate_definition: AdsorbateDefinition | None = None,
) -> tuple[dict[str, float], bool]:
    """Calculate operator weights from system type and core composition."""
    from scgo.system_types import get_system_policy

    policy = get_system_policy(system_type)
    core_composition = _core_composition_for_weights(
        composition, system_type, adsorbate_definition
    )
    n_core_elements = len(set(core_composition)) if core_composition else 1
    use_permutation = n_core_elements > 1 and policy.allow_composition_permutations

    if system_type == "gas_cluster":
        if use_permutation:
            return _alloy_weights_from_core(core_composition)
        weights = _pure_structural_weights(include_surface_slide=False)
        return weights, False

    if system_type == "surface_cluster":
        if use_permutation:
            weights, _ = _alloy_weights_from_core(core_composition)
            weights = dict(weights)
            weights.update(
                {
                    "in_plane_slide": max(weights.get("in_plane_slide", 0.03), 0.10),
                    "in_plane_rotate": 0.11,
                    "rotational": max(weights.get("rotational", 0.07), 0.10),
                    "mirror": max(weights.get("mirror", 0.05), 0.08),
                }
            )
            return _renormalize_weights(weights), True
        return (
            _renormalize_weights(
                {
                    "rattle": 0.16,
                    "overlap_relief": 0.14,
                    "flattening": 0.12,
                    "rotational": 0.12,
                    "mirror": 0.08,
                    "anisotropic_rattle": 0.12,
                    "breathing": 0.05,
                    "in_plane_slide": 0.10,
                    "in_plane_rotate": 0.11,
                }
            ),
            False,
        )

    if system_type == "gas_cluster_adsorbate":
        return (
            _renormalize_weights(
                {
                    "rattle": 0.14,
                    "overlap_relief": 0.12,
                    "fragment_reposition": 0.25,
                    "flattening": 0.10,
                    "rotational": 0.08,
                    "mirror": 0.05,
                    "anisotropic_rattle": 0.12,
                    "breathing": 0.04,
                }
            ),
            False,
        )

    if system_type == "surface_cluster_adsorbate":
        return (
            _renormalize_weights(
                {
                    "rattle": 0.12,
                    "overlap_relief": 0.10,
                    "fragment_reposition": 0.20,
                    "flattening": 0.08,
                    "rotational": 0.06,
                    "mirror": 0.05,
                    "anisotropic_rattle": 0.10,
                    "breathing": 0.04,
                    "in_plane_slide": 0.12,
                    "in_plane_rotate": 0.08,
                }
            ),
            False,
        )

    if system_type == "surface":
        if use_permutation:
            weights, perm_flag = _alloy_weights_from_core(core_composition)
            weights = dict(weights)
            weights["in_plane_slide"] = max(weights.get("in_plane_slide", 0.08), 0.28)
            weights["rattle"] = max(weights.get("rattle", 0.13), 0.22)
            weights["overlap_relief"] = max(weights.get("overlap_relief", 0.12), 0.20)
            for key in (
                "flattening",
                "rotational",
                "mirror",
                "anisotropic_rattle",
                "breathing",
                "shell_swap",
            ):
                weights.pop(key, None)
            return _renormalize_weights(weights), perm_flag
        return (
            _renormalize_weights(
                {
                    "rattle": 0.35,
                    "overlap_relief": 0.25,
                    "in_plane_slide": 0.40,
                }
            ),
            False,
        )

    if system_type == "surface_adsorbate":
        return (
            _renormalize_weights(
                {
                    "rattle": 0.12,
                    "overlap_relief": 0.12,
                    "fragment_reposition": 0.38,
                    "in_plane_slide": 0.28,
                    "in_plane_rotate": 0.10,
                }
            ),
            False,
        )

    # Fallback for unknown future types: legacy composition-only behavior.
    return calculate_composition_weights(composition)


def calculate_composition_weights(
    composition: list[str],
) -> tuple[dict[str, float], bool]:
    """Calculate mutation operator weights based on cluster composition.

    Deprecated for GA runs: prefer :func:`calculate_system_type_weights`.
    """
    element_counts = get_composition_counts(composition)
    n_elements = len(element_counts)

    if n_elements == 1:
        weights = _pure_structural_weights(include_surface_slide=True)
        use_permutation = False
    else:
        weights, use_permutation = _alloy_weights_from_core(composition)

    return weights, use_permutation


def _static_weights_for_system_type(
    system_type: SystemType,
    composition: list[str],
    adsorbate_definition: AdsorbateDefinition | None,
) -> tuple[dict[str, float], bool]:
    """Non-adaptive operator weight tables keyed by system type."""
    from scgo.system_types import get_system_policy

    policy = get_system_policy(system_type)
    core_composition = _core_composition_for_weights(
        composition, system_type, adsorbate_definition
    )
    use_permutation = (
        len(set(core_composition)) > 1 and policy.allow_composition_permutations
    )

    if system_type == "gas_cluster":
        if use_permutation:
            return (
                _renormalize_weights(
                    {
                        "rattle": 0.20,
                        "overlap_relief": 0.17,
                        "permutation": 0.23,
                        "shell_swap": 0.17,
                        "flattening": 0.08,
                        "anisotropic_rattle": 0.10,
                        "breathing": 0.03,
                    }
                ),
                True,
            )
        return (
            _renormalize_weights(
                {
                    "rattle": 0.42,
                    "overlap_relief": 0.24,
                    "flattening": 0.10,
                    "anisotropic_rattle": 0.14,
                    "breathing": 0.07,
                }
            ),
            False,
        )

    if system_type == "surface_cluster":
        if use_permutation:
            return (
                _renormalize_weights(
                    {
                        "rattle": 0.16,
                        "overlap_relief": 0.14,
                        "permutation": 0.18,
                        "shell_swap": 0.12,
                        "flattening": 0.08,
                        "anisotropic_rattle": 0.10,
                        "in_plane_slide": 0.10,
                        "in_plane_rotate": 0.08,
                    }
                ),
                True,
            )
        return (
            _renormalize_weights(
                {
                    "rattle": 0.30,
                    "overlap_relief": 0.20,
                    "flattening": 0.10,
                    "anisotropic_rattle": 0.12,
                    "breathing": 0.06,
                    "in_plane_slide": 0.12,
                    "in_plane_rotate": 0.10,
                }
            ),
            False,
        )

    if system_type == "gas_cluster_adsorbate":
        return (
            _renormalize_weights(
                {
                    "rattle": 0.18,
                    "overlap_relief": 0.15,
                    "fragment_reposition": 0.28,
                    "flattening": 0.10,
                    "anisotropic_rattle": 0.14,
                    "breathing": 0.05,
                }
            ),
            False,
        )

    if system_type == "surface_cluster_adsorbate":
        return (
            _renormalize_weights(
                {
                    "rattle": 0.14,
                    "overlap_relief": 0.12,
                    "fragment_reposition": 0.22,
                    "flattening": 0.08,
                    "anisotropic_rattle": 0.12,
                    "in_plane_slide": 0.14,
                    "in_plane_rotate": 0.10,
                }
            ),
            False,
        )

    if system_type == "surface":
        if use_permutation:
            return (
                _renormalize_weights(
                    {
                        "rattle": 0.25,
                        "overlap_relief": 0.20,
                        "permutation": 0.15,
                        "in_plane_slide": 0.40,
                    }
                ),
                True,
            )
        return (
            _renormalize_weights(
                {
                    "rattle": 0.35,
                    "overlap_relief": 0.25,
                    "in_plane_slide": 0.40,
                }
            ),
            False,
        )

    if system_type == "surface_adsorbate":
        return (
            _renormalize_weights(
                {
                    "rattle": 0.12,
                    "overlap_relief": 0.10,
                    "fragment_reposition": 0.38,
                    "in_plane_slide": 0.28,
                    "in_plane_rotate": 0.12,
                }
            ),
            False,
        )

    return (
        _renormalize_weights(
            {
                "rattle": 0.42,
                "overlap_relief": 0.24,
                "flattening": 0.10,
                "anisotropic_rattle": 0.14,
                "breathing": 0.07,
            }
        ),
        False,
    )


def calculate_size_adjustment(n_atoms: int) -> dict[str, float]:
    """Calculate size-dependent adjustments for mutation parameters."""
    if n_atoms <= 4:
        return {
            "rattle_strength": 1.0,
            "mutation_probability": 0.30,
            "rattle_prop": 0.4,
        }
    if n_atoms <= 10:
        return {
            "rattle_strength": 0.8,
            "mutation_probability": 0.25,
            "rattle_prop": 0.3,
        }
    return {
        "rattle_strength": 0.6,
        "mutation_probability": 0.20,
        "rattle_prop": 0.25,
    }


def calculate_generation_adjustment(
    current_generation: int,
    total_generations: int,
) -> dict[str, float]:
    """Calculate generation-dependent adjustments for mutation parameters."""
    progress = current_generation / max(total_generations, 1)

    if progress < 0.3:
        return {
            "mutation_probability_multiplier": 1.5,
            "rattle_strength_multiplier": 1.2,
        }
    if progress < 0.7:
        return {
            "mutation_probability_multiplier": 1.0,
            "rattle_strength_multiplier": 1.0,
        }
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
    system_type: SystemType = "gas_cluster",
    adsorbate_definition: AdsorbateDefinition | None = None,
) -> dict:
    """Get complete adaptive mutation configuration."""
    if not use_adaptive:
        operator_weights, use_permutation = _static_weights_for_system_type(
            system_type, composition, adsorbate_definition
        )
        return {
            "operator_weights": operator_weights,
            "use_permutation": use_permutation,
            "mutation_probability": 0.2,
            "rattle_strength": 0.8,
            "rattle_prop": 0.3,
            "anisotropic_in_plane_strength": 1.0,
            "anisotropic_normal_strength": 0.2,
            "anisotropic_rattle_prop": 0.5,
        }

    n_atoms = len(composition)

    operator_weights, use_permutation = calculate_system_type_weights(
        system_type,
        composition,
        adsorbate_definition=adsorbate_definition,
    )

    size_params = calculate_size_adjustment(n_atoms)
    gen_adjustments = calculate_generation_adjustment(
        current_generation,
        total_generations,
    )

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

    final_mutation_prob = np.clip(final_mutation_prob, 0.1, max_mutation_probability)
    final_rattle_strength = np.clip(final_rattle_strength, 0.3, 1.2)
    final_rattle_prop = np.clip(final_rattle_prop, 0.1, 0.9)

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
