"""Initial cluster structure generation with seed-based growth strategies.

This module provides functions for creating initial cluster structures for global
optimization, including intelligent seed selection from previous runs and adaptive
growth strategies based on available candidates.
"""

from __future__ import annotations

import itertools
import threading
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from ase import Atoms

from scgo.exceptions import (
    SCGORuntimeError,
    SCGOValidationError,
)
from scgo.metadata.atoms import set_tags
from scgo.system_types.connectivity_factor import (
    ConnectivityFactorInput,
    NormalizedConnectivityFactor,
)
from scgo.utils.helpers import (
    get_cluster_formula,
    get_composition_counts,
)
from scgo.utils.logging import get_logger
from scgo.utils.parallel_workers import resolve_n_jobs_for_tasks
from scgo.utils.phase_logging import InitDiagnosticsCollector
from scgo.utils.validation import validate_composition

from .atomic_radii import get_vdw_radius
from .candidate_discovery import (
    _discover_all_candidates,
    _find_smaller_candidates,
    get_structure_signature,
    is_composition_subset,
)
from .geometry_helpers import (
    _classify_seed_geometry,
    _get_positions_hash,
    _set_cubic_cell_and_center,
    _validate_cluster_defaults,
    reorder_cluster_to_composition,
    validate_cluster,
)
from .initialization_config import (
    BOLTZMANN_TEMPERATURE_MAX,
    BOLTZMANN_TEMPERATURE_MIN,
    CONNECTIVITY_FACTOR,
    ENERGY_SPREAD_DIVISOR,
    ENERGY_SPREAD_TOLERANCE,
    MAX_REASONABLE_CELL_SIDE,
    MIN_DISTANCE_FACTOR_DEFAULT,
    MULTI_ELEMENT_TEMPLATE_PENALTY,
    PACKING_EFFICIENCY_FCC_HCP,
    PLACEMENT_RADIUS_SCALING_DEFAULT,
    SEED_COMBINATION_STRATEGY_COUNT,
    TEMPLATE_BASE_WEIGHTS,
    TEMPLATE_DIVERSITY_BOOST_FACTOR,
    VACUUM_DEFAULT,
)
from .random_spherical import grow_from_seed, random_spherical
from .seed_combiners import combine_and_grow
from .strategy_allocation import _allocate_initialization_strategies
from .templates import (
    _get_or_create_rotated_variants,
    generate_template_matches,
)

logger = get_logger(__name__)

__all__: list[str] = []


class _SeedSamplingLogCollector:
    """Thread-safe accumulator for seed-sampling failures across batch workers."""

    _lock = threading.Lock()
    _records: list[tuple[str, str]] = []

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._records.clear()

    @classmethod
    def record(cls, formula: str, reason: str) -> None:
        with cls._lock:
            cls._records.append((formula, reason))

    @classmethod
    def emit_summary_if_any(cls) -> None:
        with cls._lock:
            if not cls._records:
                return
            records = list(cls._records)
            cls._records.clear()

        formula_reasons: dict[str, Counter[str]] = defaultdict(Counter)
        for formula, reason in records:
            formula_reasons[formula][reason] += 1

        parts: list[str] = []
        for formula in sorted(
            formula_reasons,
            key=lambda f: (-sum(formula_reasons[f].values()), f),
        ):
            counter = formula_reasons[formula]
            total = sum(counter.values())
            if len(counter) == 1:
                (reason,) = counter
                parts.append(f"{formula}x{total} [{reason}]")
            else:
                reason_detail = ", ".join(
                    f"{reason}x{count}" for reason, count in counter.most_common()
                )
                parts.append(f"{formula}x{total} [{reason_detail}]")

        logger.info(
            "Seed+growth: no suitable seed (%d failures): %s",
            len(records),
            ", ".join(parts),
        )


def compute_cell_side(composition: list[str], vacuum: float = VACUUM_DEFAULT) -> float:
    """Estimate a cubic cell side from atomic van-der-Waals volumes.

    The estimate computes atomic volumes using ASE's van-der-Waals radii,
    converts that to an effective spherical radius and returns a cubic
    side that contains the cluster plus the requested ``vacuum`` padding.

    For elements where ASE's vdw_radii is NaN (e.g., Co, Fe, Ru), uses
    interpolated values from neighboring elements (cached per element).

    Args:
        composition: Sequence of element symbols (e.g. ["Pt", "Pt"])
        vacuum: Extra padding (Å) to add to the estimated diameter.

    Returns:
        Cubic cell side length in Å. Returns 0.0 for an empty composition.

    """
    if not composition:
        return 0.0

    vdw_radii_list = [get_vdw_radius(s) for s in composition]
    total_atomic_volume = sum(4.0 / 3.0 * np.pi * r**3 for r in vdw_radii_list)
    # Apply packing efficiency factor (~0.74 for FCC/HCP)
    packed_volume = total_atomic_volume / PACKING_EFFICIENCY_FCC_HCP
    effective_cluster_radius = (3.0 * packed_volume / (4.0 * np.pi)) ** (1.0 / 3.0)
    cell_side = 2 * effective_cluster_radius + vacuum

    # Warn if computed cell side is unreasonably large
    if cell_side > MAX_REASONABLE_CELL_SIDE:
        logger.warning(
            "Computed cell_side (%.1f Å) exceeds reasonable threshold "
            "(%s Å) for %d atoms; this may indicate a very large composition or vacuum value",
            cell_side,
            MAX_REASONABLE_CELL_SIDE,
            len(composition),
        )

    return cell_side


def _boltzmann_sample(
    candidates: list[tuple[float, Atoms]],
    rng: np.random.Generator,
    temperature: float | None = None,
) -> tuple[float, Atoms] | None:
    """Sample a candidate using Boltzmann weights built from energies.

    IMPORTANT: This function assumes all candidates have the same chemical composition.
    Boltzmann weighting is only meaningful when comparing energies of clusters with
    the same composition, as energies scale with cluster size and composition.

    Args:
        candidates: List of (energy, atoms) tuples for candidates of the same composition
        rng: Random number generator for sampling
        temperature: Optional temperature for Boltzmann weighting. If None, an adaptive
                    temperature is inferred from the energy span.

    Returns:
        A randomly sampled (energy, atoms) tuple, or None if no candidates provided

    Raises:
        SCGOValidationError: If the candidates do not all share the same element
            counts, or if an explicit ``temperature`` is not positive.

    """
    if not candidates:
        return None

    # Fast-path for single candidate
    if len(candidates) == 1:
        energy, atoms = candidates[0]
        return energy, atoms.copy()

    # Verify all candidates have the same composition (counts, not atom order)
    first_counts = get_composition_counts(candidates[0][1].get_chemical_symbols())
    for _energy, atoms in candidates[1:]:
        if get_composition_counts(atoms.get_chemical_symbols()) != first_counts:
            raise SCGOValidationError(
                "All candidates must have the same composition for Boltzmann sampling. "
                f"Found {first_counts} vs "
                f"{get_composition_counts(atoms.get_chemical_symbols())}"
            )

    energies = np.array([e for e, _ in candidates])
    min_energy = np.min(energies)
    max_energy = np.max(energies)

    # Adaptive temperature if not provided
    if temperature is None:
        energy_spread = max_energy - min_energy
        if energy_spread < ENERGY_SPREAD_TOLERANCE:
            # All energies are essentially the same - use uniform sampling
            selected_idx = rng.integers(0, len(candidates))
            energy, atoms = candidates[selected_idx]
            return energy, atoms.copy()

        # Use adaptive temperature based on energy spread
        # Clamp to reasonable range to avoid extreme weights
        temperature = np.clip(
            energy_spread / ENERGY_SPREAD_DIVISOR,
            BOLTZMANN_TEMPERATURE_MIN,
            BOLTZMANN_TEMPERATURE_MAX,
        )

    # At this point, temperature is guaranteed to be a float, but mypy can't narrow the type
    if temperature is None:
        raise TypeError("temperature must be a float after adaptive clamping")

    # Validate temperature
    if temperature <= 0:
        raise SCGOValidationError(f"Temperature must be positive, got {temperature}")

    # Compute Boltzmann weights: exp(-E/kT)
    # Shift energies to avoid overflow (subtract min_energy)
    shifted_energies = energies - min_energy
    weights = np.exp(-shifted_energies / temperature)
    probabilities = weights / np.sum(weights)

    # Sample according to probabilities
    selected_idx: int = int(rng.choice(len(candidates), p=probabilities))
    energy, atoms = candidates[selected_idx]
    return energy, atoms.copy()


def _calculate_template_weight(
    template_type: str,
    n_unique_elements: int,
    template_type_counts: dict[str, int],
    total_candidates: int,
) -> float:
    """Calculate weight for a template type based on quality and diversity.

    Weights favor:
    - High-quality template types (from TEMPLATE_BASE_WEIGHTS)
    - Less common template types (diversity boost)
    - Templates suitable for the composition (penalty for multi-element)

    Args:
        template_type: Type of template (e.g., "icosahedron")
        n_unique_elements: Number of unique elements in composition
        template_type_counts: Dictionary counting occurrences of each template type
        total_candidates: Total number of template candidates

    Returns:
        Weight for this template type; always non-negative so the weights can be
        normalized into a probability vector for ``rng.choice``.
    """
    # Extract base weight from TEMPLATE_BASE_WEIGHTS
    base_weight = TEMPLATE_BASE_WEIGHTS.get(template_type, 1.0)

    # Boost less common template types for diversity
    type_count = template_type_counts.get(template_type, 0)
    diversity_boost = (
        TEMPLATE_DIVERSITY_BOOST_FACTOR * (1.0 - type_count / total_candidates)
        if total_candidates > 0
        else 0.0
    )

    # Penalty for multi-element compositions (templates work better for single-element)
    multi_element_penalty = (
        MULTI_ELEMENT_TEMPLATE_PENALTY if n_unique_elements > 1 else 0.0
    )

    # Clamp at zero: the penalty can exceed the base weight of low-ranked template
    # types (e.g. cube/tetrahedron at 0.8 vs a 0.9 penalty), and negative weights
    # would produce negative probabilities in rng.choice(p=...).
    return max(0.0, base_weight + diversity_boost - multi_element_penalty)


def _get_template_type(atoms: Atoms) -> str:
    """Extract template type from atoms info, defaulting to "unknown" if not present.

    Args:
        atoms: The Atoms object to extract template type from

    Returns:
        Template type string, or "unknown" if not available
    """
    info = getattr(atoms, "info", None)
    return info.get("template_type", "unknown") if info else "unknown"


def _template_sort_key(atoms: Atoms) -> tuple:
    """Deterministic sort key for template candidates.

    Sorts by template type, then by rounded center-of-mass coordinates, then by
    atom count so candidate ordering (and thus template indices) is stable and
    reproducible across discovery and generation.

    Args:
        atoms: Template Atoms object to derive a sort key from.

    Returns:
        Tuple usable as a ``list.sort`` key.
    """
    template_type = _get_template_type(atoms)
    com = atoms.get_center_of_mass()
    return (
        template_type,
        round(com[0], 8),
        round(com[1], 8),
        round(com[2], 8),
        len(atoms),
    )


def _prepare_template_candidates(
    composition: list[str],
    n_atoms: int,
    rng: np.random.Generator,
    cell_side: float,
    placement_radius_scaling: float,
    min_distance_factor: float,
    connectivity_factor: ConnectivityFactorInput | NormalizedConnectivityFactor,
) -> list[Atoms]:
    """Build the deduplicated, deterministically sorted template candidate list.

    Generates exact and near-match template candidates via
    :func:`generate_template_matches`, removes duplicates shared between the two
    match kinds, and sorts them deterministically for reproducible indexing.

    Args:
        composition: Target composition list.
        n_atoms: Number of atoms.
        rng: Random number generator.
        cell_side: Cubic cell side length.
        placement_radius_scaling: Scaling for placement radius.
        min_distance_factor: Factor for minimum distance checks.
        connectivity_factor: Factor for connectivity threshold.

    Returns:
        Sorted, deduplicated list of template candidate ``Atoms`` (empty if none
        were generated).
    """
    template_candidates = generate_template_matches(
        composition=composition,
        n_atoms=n_atoms,
        rng=rng,
        cell_side=cell_side,
        placement_radius_scaling=placement_radius_scaling,
        min_distance_factor=min_distance_factor,
        connectivity_factor=connectivity_factor,
        include_exact=True,
        include_near=True,
    )
    if not template_candidates:
        return []

    original_count = len(template_candidates)
    template_candidates = _deduplicate_template_structures(template_candidates)
    if len(template_candidates) < original_count:
        logger.debug(
            "Deduplicated templates: %d -> %d unique structures "
            "(duplicates between exact and near matches removed)",
            original_count,
            len(template_candidates),
        )

    template_candidates.sort(key=_template_sort_key)
    return template_candidates


def _deduplicate_template_structures(
    template_candidates: list[Atoms],
) -> list[Atoms]:
    """Remove templates that produce identical structures.

    Templates with the same structure signature (sorted interatomic distances)
    are considered duplicates. When duplicates are found, we keep one representative
    per unique structure, preferring to preserve different template types when possible.

    Args:
        template_candidates: List of template Atoms objects

    Returns:
        Deduplicated list of template Atoms objects
    """
    if len(template_candidates) <= 1:
        return template_candidates

    # Group templates by structure signature
    signature_groups: dict[tuple, list[tuple[str, Atoms]]] = {}
    for atoms in template_candidates:
        signature = get_structure_signature(atoms)
        template_type = _get_template_type(atoms)
        if signature not in signature_groups:
            signature_groups[signature] = []
        signature_groups[signature].append((template_type, atoms))

    # For each signature group, keep one representative
    # Prefer keeping templates with different types
    deduplicated = []
    seen_types = set()

    for group in signature_groups.values():
        # Find preferred template: prefer unseen types, otherwise use first
        preferred = None
        for template_type, atoms in group:
            if template_type not in seen_types:
                preferred = atoms
                seen_types.add(template_type)
                break

        # If all types already seen, use first template
        if preferred is None:
            preferred = group[0][1]
            seen_types.add(group[0][0])

        deduplicated.append(preferred)

    return deduplicated


def _apply_template_rotation_and_validate(
    selected: Atoms,
    cell_side: float,
    rng: np.random.Generator,
    min_distance_factor: float,
    connectivity_factor: ConnectivityFactorInput | NormalizedConnectivityFactor,
    composition: list[str] | None = None,
) -> Atoms | None:
    """Apply rotation diversity, set cell, center, and validate a template structure.

    Used both when generating from template candidates and when reusing
    discovery templates in smart-mode batch generation.

    Args:
        selected: Template Atoms to process (will be copied before rotation).
        cell_side: Cubic cell side length.
        rng: Random number generator for rotation.
        min_distance_factor: Factor for minimum distance checks.
        connectivity_factor: Factor for connectivity threshold.
        composition: Optional target composition for atom reordering before the
            validation-complete flag is set.

    Returns:
        Validated Atoms with rotation applied, or None if validation fails.
    """
    selected = selected.copy()
    _set_cubic_cell_and_center(selected, cell_side)

    rotation_candidates = _get_or_create_rotated_variants(selected, cell_side)
    selected = rotation_candidates[rng.integers(0, len(rotation_candidates))].copy()

    is_valid, error_message = _validate_cluster_defaults(
        selected, min_distance_factor, connectivity_factor
    )
    if not is_valid:
        logger.warning(
            "Template structure validation failed: %s; discarding this template candidate",
            error_message,
        )
        return None
    if composition is not None:
        selected = reorder_cluster_to_composition(selected, composition)
    selected.info["scgo_validation_complete"] = True
    return selected


def _try_template_generation(
    composition: list[str],
    n_atoms: int,
    cell_side: float,
    rng: np.random.Generator,
    placement_radius_scaling: float,
    min_distance_factor: float,
    connectivity_factor: ConnectivityFactorInput | NormalizedConnectivityFactor,
    template_index: int | None = None,
    discovery_templates: list[Atoms] | None = None,
) -> Atoms | None:
    """Try to generate a cluster using template structures.

    This helper function encapsulates template generation logic for the smart mode.
    It generates diverse template candidates (exact and near matches) and selects
    one with enhanced diversity mechanisms.

    Args:
        composition: Target composition list
        n_atoms: Number of atoms
        cell_side: Cell side length
        rng: Random number generator
        placement_radius_scaling: Scaling for placement radius
        min_distance_factor: Factor for minimum distance checks
        connectivity_factor: Factor for connectivity threshold
        template_index: Optional index to select a specific template from candidates.
                       If None, uses weighted random selection.
        discovery_templates: Optional list of pre-discovered templates from batch
                            generation. When provided with a valid template_index,
                            uses the template directly to maintain index alignment.

    Returns:
        Atoms object if successful, None otherwise
    """
    # If discovery_templates is provided and template_index is valid, use it directly
    # to maintain index alignment in batch generation
    if (
        discovery_templates is not None
        and template_index is not None
        and 0 <= template_index < len(discovery_templates)
    ):
        selected = discovery_templates[template_index].copy()
        result = _apply_template_rotation_and_validate(
            selected,
            cell_side,
            rng,
            min_distance_factor,
            connectivity_factor,
            composition=list(composition),
        )
        if result is not None:
            return result
        # Validation failed; fall through to normal generation below
        # IMPORTANT: Reset template_index because it refers to discovery_templates list,
        # which may not align with the generated template_candidates list below.
        template_index = None

    # Get all template candidates (exact and near matches) using unified function
    template_candidates = _prepare_template_candidates(
        composition=composition,
        n_atoms=n_atoms,
        rng=rng,
        cell_side=cell_side,
        placement_radius_scaling=placement_radius_scaling,
        min_distance_factor=min_distance_factor,
        connectivity_factor=connectivity_factor,
    )

    if not template_candidates:
        logger.debug(
            "Template: no usable templates for %d atoms (composition=%s)",
            n_atoms,
            composition,
        )
        return None

    # Enhanced diversity: create weighted pool of ALL candidates across all types
    n_unique_elements = len(set(composition))
    template_types = [_get_template_type(c) for c in template_candidates]
    template_type_counts = Counter(template_types)
    template_type_weights = {
        t: _calculate_template_weight(
            t, n_unique_elements, template_type_counts, len(template_candidates)
        )
        for t in template_type_counts
    }
    weighted_candidates = [
        (template_type_weights[t], c, t)
        for c, t in zip(template_candidates, template_types, strict=True)
    ]

    # Select from weighted pool
    if template_index is not None:
        # Use specific template index if provided
        if template_index < 0 or template_index >= len(weighted_candidates):
            logger.warning(
                "Invalid template_index %d, must be in range [0, %d); using random selection",
                template_index,
                len(weighted_candidates),
            )
            selected_idx = int(rng.integers(0, len(weighted_candidates)))
        else:
            selected_idx = template_index
    else:
        # Use weighted random selection
        weights = [w for w, _, _ in weighted_candidates]
        total_weight = sum(weights)
        if total_weight > 0:
            probabilities = [w / total_weight for w in weights]
            selected_idx = int(rng.choice(len(weighted_candidates), p=probabilities))
        else:
            selected_idx = int(rng.integers(0, len(weighted_candidates)))

    selected = weighted_candidates[selected_idx][1].copy()
    selected_type = weighted_candidates[selected_idx][2]
    n_unique_template_types = len({t for _, _, t in weighted_candidates})

    result = _apply_template_rotation_and_validate(
        selected,
        cell_side,
        rng,
        min_distance_factor,
        connectivity_factor,
        composition=list(composition),
    )
    if result is None:
        return None
    logger.debug(
        "Smart mode: using template %s (%d unique type(s) available, %d total candidates)",
        selected_type,
        n_unique_template_types,
        len(template_candidates),
    )
    return result


def _filter_candidates_by_geometry(
    candidates_by_formula: dict[str, list[tuple[float, Atoms]]],
) -> dict[str, list[tuple[float, Atoms]]]:
    """Filter seed candidates to only include those with suitable geometries.

    Removes linear and 1D candidates, keeping only planar and 3D structures.
    This ensures seeds have reasonable geometry for combination and growth.

    Args:
        candidates_by_formula: Dictionary mapping formulas to candidate lists

    Returns:
        Filtered dictionary with only suitable geometries
    """
    filtered = {}
    for formula, candidates in candidates_by_formula.items():
        suitable = []
        for energy, atoms in candidates:
            geometry = _classify_seed_geometry(atoms)
            if geometry in ["planar", "3d"]:
                suitable.append((energy, atoms))
        if suitable:
            filtered[formula] = suitable
    return filtered


def _sample_seed_with_strategy(
    candidates: list[tuple[float, Atoms]],
    strategy: int,
    rng: np.random.Generator,
) -> tuple[float, Atoms] | None:
    """Sample a seed from candidates using a specified strategy.

    Strategies provide different ways to select seeds for diversity:
    0: Boltzmann sampling (energy-weighted)
    1: Low-energy sampling (prefer lowest energy)
    2: High-energy sampling (prefer highest energy, for diversity)
    3: Mid-energy sampling (prefer middle energies)
    4: Random sampling (uniform)

    Args:
        candidates: List of (energy, atoms) tuples
        strategy: Strategy index (0-4)
        rng: Random number generator

    Returns:
        Selected (energy, atoms) tuple, or None if ``candidates`` is empty

    Raises:
        SCGOValidationError: If ``strategy`` is not one of the indices 0-4
    """
    if not candidates:
        return None

    # Strategy dispatch using dictionary for cleaner code
    strategy_handlers = {
        0: lambda: _boltzmann_sample(candidates, rng),
        1: lambda: candidates[0],  # Already sorted by energy
        2: lambda: candidates[-1],
        3: lambda: candidates[len(candidates) // 2],
        4: lambda: candidates[rng.integers(0, len(candidates))],
    }

    handler = strategy_handlers.get(strategy)
    if handler is None:
        raise SCGOValidationError(
            f"Invalid seed sampling strategy: {strategy!r} (expected 0-4)"
        )

    return handler()


def _grow_from_random_seed(
    composition: list[str],
    cell_side: float,
    rng: np.random.Generator,
    placement_radius_scaling: float,
    min_distance_factor: float,
    connectivity_factor: ConnectivityFactorInput | NormalizedConnectivityFactor,
) -> Atoms | None:
    """Generate a small random seed and grow it to the target composition.

    This function is used when no usable database seeds are available, or when
    every database seed combination failed. It creates a small random cluster
    (about 1/4 of the target size, clamped to 3-15 atoms, and never exceeding
    the per-element counts of the target) and grows it to the target
    composition using convex-hull-based placement.

    The growth approach provides different structural characteristics than pure
    random_spherical, as the initial seed geometry influences the final structure.

    Args:
        composition: Target composition list
        cell_side: Cell side length
        rng: Random number generator
        placement_radius_scaling: Scaling for placement radius
        min_distance_factor: Factor for minimum distance checks
        connectivity_factor: Factor for connectivity threshold

    Returns:
        Atoms object if successful, None otherwise (empty composition, seed
        generation failure, or growth failure)
    """
    n_atoms = len(composition)

    # Determine seed size: about 1/4 of target, minimum 3 atoms, maximum 15
    seed_size = max(3, min(15, n_atoms // 4))

    # Create seed composition by sampling from target composition without
    # exceeding per-element counts (so growth can reach the target).
    if not composition:
        # Empty composition - return None early
        return None
    remaining = get_composition_counts(composition)
    seed_composition: list[str] = []
    for _ in range(seed_size):
        available = [elem for elem, count in remaining.items() if count > 0]
        if not available:
            break
        elem = str(rng.choice(available))
        seed_composition.append(elem)
        remaining[elem] -= 1
    if not seed_composition:
        return None

    # Generate small random seed cluster
    seed_cell_side = compute_cell_side(seed_composition)
    try:
        seed_atoms = random_spherical(
            composition=seed_composition,
            cell_side=seed_cell_side,
            rng=rng,
            placement_radius_scaling=placement_radius_scaling,
            min_distance_factor=min_distance_factor,
            connectivity_factor=connectivity_factor,
        )
    except (ValueError, SCGOValidationError):
        logger.debug(
            "Seed cluster generation failed for composition %s",
            seed_composition,
        )
        return None

    try:
        result = grow_from_seed(
            seed_atoms=seed_atoms,
            target_composition=composition,
            placement_radius_scaling=placement_radius_scaling,
            cell_side=cell_side,
            rng=rng,
            min_distance_factor=min_distance_factor,
            connectivity_factor=connectivity_factor,
        )
        return result
    except (ValueError, SCGOValidationError):
        logger.debug(
            "Growth from random seed failed for composition %s",
            composition,
        )
        return None


def _find_valid_seed_combinations(
    candidates_by_formula: dict[str, list[tuple[float, Atoms]]],
    target_counts: dict[str, int],
) -> list[tuple[str, ...]]:
    """Find all valid seed formula combinations that are sub-compositions of target.

    Combinations contain between one and three distinct formulas (each formula
    is used at most once) and are kept only when their summed element counts
    fit within ``target_counts``.

    Args:
        candidates_by_formula: Dictionary mapping formulas to candidate lists
        target_counts: Target composition counts

    Returns:
        List of valid formula combinations (as tuples)
    """
    seed_compositions = {
        formula: get_composition_counts(candidates[0][1].get_chemical_symbols())
        for formula, candidates in candidates_by_formula.items()
    }

    valid_combinations = []
    for n_seeds in range(1, min(len(candidates_by_formula) + 1, 4)):
        for combo in itertools.combinations(candidates_by_formula.keys(), n_seeds):
            combo_counts: Counter[str] = Counter()
            for formula in combo:
                combo_counts.update(seed_compositions[formula])

            if is_composition_subset(combo_counts, target_counts):
                valid_combinations.append(combo)

    return valid_combinations


def _sample_suitable_seed(
    candidates: list[tuple[float, Atoms]],
    strategy: int,
    tried_positions: set[str],
    existing_geometries: list[str],
    rng: np.random.Generator,
    max_attempts: int = 10,
) -> tuple[Atoms | None, str | None]:
    """Sample a suitable seed from candidates with geometry diversity preference.

    Args:
        candidates: List of (energy, atoms) tuples
        strategy: Sampling strategy index
        tried_positions: Set of position hashes already tried
        existing_geometries: List of geometries of already-selected seeds
        rng: Random number generator
        max_attempts: Maximum attempts to find suitable seed

    Returns:
        Tuple of (suitable seed Atoms object or None, failure reason if None)
    """
    # Precompute position hashes once; seeds are not mutated during sampling.
    precomputed: list[tuple[float, Atoms, str]] = [
        (e, a, _get_positions_hash(a.get_positions())) for e, a in candidates
    ]
    available_candidates = [
        (e, a, pos_hash)
        for e, a, pos_hash in precomputed
        if pos_hash not in tried_positions
    ]

    if not available_candidates:
        return None, "all candidates already tried"

    rejection_counts: Counter[str] = Counter()

    for attempt in range(max_attempts):
        sampled = _sample_seed_with_strategy(
            [(e, a) for e, a, _ in available_candidates],
            strategy=(strategy + attempt) % 5,
            rng=rng,
        )

        if sampled is None:
            return None, "candidate sampling failed"

        _, sampled_seed = sampled
        geometry = _classify_seed_geometry(sampled_seed)

        # Same positions as the precomputed entry even if the sampler returned a copy.
        pos_hash = _get_positions_hash(sampled_seed.get_positions())
        tried_positions.add(pos_hash)

        available_candidates = [
            (e, a, h) for e, a, h in available_candidates if h != pos_hash
        ]

        # Accept if suitable geometry
        if geometry not in ["planar", "3d"]:
            rejection_counts[f"unsuitable {geometry} geometry"] += 1
            if not available_candidates:
                break
            continue

        # Prefer geometry diversity: if all existing are same, prefer different
        if existing_geometries:
            all_same = all(g == existing_geometries[0] for g in existing_geometries)
            if all_same and geometry == existing_geometries[0]:
                rejection_counts["need mixed seed geometries"] += 1
                if not available_candidates:
                    break
                continue  # Prefer different geometry

        return sampled_seed.copy(), None

    if rejection_counts:
        reason, _ = rejection_counts.most_common(1)[0]
        return None, reason
    return None, "candidates exhausted after sampling"


def _try_seed_growth(
    composition: list[str],
    cell_side: float,
    rng: np.random.Generator,
    placement_radius_scaling: float,
    min_distance_factor: float,
    connectivity_factor: ConnectivityFactorInput | NormalizedConnectivityFactor,
    candidates_by_formula: dict[str, list[tuple[float, Atoms]]],
    valid_combinations: list[tuple[str, ...]],
) -> Atoms | None:
    """Try to generate a cluster using seed+growth strategy.

    This helper function encapsulates seed+growth logic for the smart mode.
    It combines seeds found in previous runs and grows them to the target
    composition. When no database seeds or no valid seed combinations are
    available, and also when every combination strategy fails, it falls back
    to growing from a freshly generated random seed.

    Args:
        composition: Target composition list
        cell_side: Cell side length
        rng: Random number generator
        placement_radius_scaling: Scaling for placement radius
        min_distance_factor: Factor for minimum distance checks
        connectivity_factor: Factor for connectivity threshold
        candidates_by_formula: Precomputed seed candidates by formula
        valid_combinations: Precomputed valid seed formula combinations

    Returns:
        Atoms object if successful, None otherwise (including compositions
        with two atoms or fewer, which are left to other strategies)
    """
    if len(composition) <= 2:
        return None
    random_seed_kwargs = {
        "composition": composition,
        "cell_side": cell_side,
        "rng": rng,
        "placement_radius_scaling": placement_radius_scaling,
        "min_distance_factor": min_distance_factor,
        "connectivity_factor": connectivity_factor,
    }
    if not candidates_by_formula:
        logger.info("Seed+growth: no database seeds found; using random seed growth")
        return _grow_from_random_seed(**random_seed_kwargs)

    if not valid_combinations:
        logger.info(
            "Seed+growth: no valid DB seed combinations; using random seed growth"
        )
        return _grow_from_random_seed(**random_seed_kwargs)

    tried_positions: set[str] = set()

    # Try multiple strategies
    for strategy_idx in range(SEED_COMBINATION_STRATEGY_COUNT):
        combo = valid_combinations[rng.integers(0, len(valid_combinations))]
        seeds_to_combine: list[Atoms] = []
        existing_geometries: list[str] = []

        # Sample seeds for each formula in the combination
        for formula in combo:
            candidates = candidates_by_formula[formula]
            seed, failure_reason = _sample_suitable_seed(
                candidates,
                strategy_idx,
                tried_positions,
                existing_geometries,
                rng,
            )

            if seed is None:
                reason = failure_reason or "unknown"
                _SeedSamplingLogCollector.record(formula, reason)
                logger.trace(
                    "Seed+growth: no suitable seed for %s (%s)",
                    formula,
                    reason,
                )
                seeds_to_combine = []
                break

            seeds_to_combine.append(seed)
            existing_geometries.append(_classify_seed_geometry(seed))

        # Require a complete combo; do not grow from a partial seed set.
        if len(seeds_to_combine) != len(combo):
            continue

        out = combine_and_grow(
            seeds=seeds_to_combine,
            target_composition=composition,
            cell_side=cell_side,
            rng=rng,
            vdw_scaling=placement_radius_scaling,
            min_distance_factor=min_distance_factor,
            connectivity_factor=connectivity_factor,
        )

        if out is not None:
            return out

    # All DB combination strategies failed; still try random-seed growth before
    # yielding None (outer chain may then fall back to random_spherical).
    logger.info(
        "Seed+growth: all %d combination strategies failed; "
        "DB combinations exhausted; trying random-seed growth",
        SEED_COMBINATION_STRATEGY_COUNT,
    )
    return _grow_from_random_seed(**random_seed_kwargs)


def _discover_available_strategies(
    composition: list[str],
    n_atoms: int,
    cell_side: float,
    rng: np.random.Generator,
    placement_radius_scaling: float,
    min_distance_factor: float,
    connectivity_factor: ConnectivityFactorInput | NormalizedConnectivityFactor,
    candidates_by_formula: dict[str, list[tuple[float, Atoms]]],
    valid_combinations: list[tuple[str, ...]],
    n_exact: int = 0,
) -> dict[str, Any]:
    """Discover available templates and seeds for Metropolis allocation.

    Args:
        composition: Target composition list
        n_atoms: Number of atoms
        cell_side: Cell side length
        rng: Random number generator
        placement_radius_scaling: Scaling for placement radius
        min_distance_factor: Factor for minimum distance checks
        connectivity_factor: Factor for connectivity threshold
        candidates_by_formula: Precomputed seed candidates by formula
        valid_combinations: Precomputed valid seed formula combinations

    Returns:
        Dict with:
        - 'templates': list of unique template Atoms objects (for tracking which ones used)
        - 'n_templates': count of unique templates
        - 'n_seed_formulas': number of seed formula types available
        - 'n_seed_combinations': number of valid seed combinations
    """
    # Discover templates using unified function
    templates = _prepare_template_candidates(
        composition=composition,
        n_atoms=n_atoms,
        rng=rng,
        cell_side=cell_side,
        placement_radius_scaling=placement_radius_scaling,
        min_distance_factor=min_distance_factor,
        connectivity_factor=connectivity_factor,
    )

    # Discover seeds
    n_seed_formulas = len(candidates_by_formula)
    n_seed_combinations = len(valid_combinations)

    # Note: Discovery info logging moved to _allocate_initialization_strategies
    # to avoid duplicate messages when create_initial_cluster is called
    # before batch generation (e.g., for creating a template structure)

    return {
        "templates": templates,
        "n_templates": len(templates),
        "n_seed_formulas": n_seed_formulas,
        "n_seed_combinations": n_seed_combinations,
        "n_exact": n_exact,
    }


def _try_strategies_in_order(
    strategies: list[tuple[str, Callable[..., Atoms]]],
    composition: list[str],
    connectivity_factor: ConnectivityFactorInput | NormalizedConnectivityFactor,
    min_distance_factor: float = MIN_DISTANCE_FACTOR_DEFAULT,
    return_strategy: bool = False,
) -> Atoms | tuple[Atoms, str, str | None]:
    """Try initialization strategies in order until one succeeds.

    This provides a clean way to implement fallback chains. The last strategy
    in the list is considered the "final fallback" and its exceptions are
    propagated rather than caught, ensuring errors are not silently swallowed.

    Args:
        strategies: List of (name, function) tuples. Functions should accept
                   no arguments and return Atoms | None. The last strategy
                   should be a guaranteed fallback (e.g., random_spherical).
        composition: Target composition (for validation)
        connectivity_factor: Factor for connectivity threshold (for validation)
        min_distance_factor: Factor for minimum distance checks (for validation)
        return_strategy: When True, also return the strategy that produced the
            result and the primary strategy it fell back from (``None`` when
            the primary strategy succeeded).

    Returns:
        Atoms object if successful. When ``return_strategy=True``, returns
        a tuple of (Atoms, used_strategy, fallback_from).

    Raises:
        SCGOValidationError: If ``strategies`` is empty, or propagated from the
            final fallback strategy or its validation.
        SCGORuntimeError: If every strategy returned None.
        ValueError: Propagated from the final fallback strategy.
        RuntimeError: Propagated from the final fallback strategy.
    """
    if not strategies:
        raise SCGOValidationError("No strategies provided to _try_strategies_in_order")

    primary_strategy = strategies[0][0]

    for idx, (strategy_name, strategy_func) in enumerate(strategies):
        is_last_strategy = idx == len(strategies) - 1

        try:
            result = strategy_func()
            if result is not None:
                if result.info.get("scgo_validation_complete"):
                    validated_atoms = result
                else:
                    validated_atoms, _, _ = validate_cluster(
                        result,
                        composition=composition,
                        min_distance_factor=min_distance_factor,
                        connectivity_factor=connectivity_factor,
                        sort_atoms=True,
                        raise_on_failure=True,
                        source=strategy_name,
                    )
                if return_strategy:
                    fallback_from = (
                        primary_strategy if strategy_name != primary_strategy else None
                    )
                    return validated_atoms, strategy_name, fallback_from
                return validated_atoms
            else:
                if not is_last_strategy:
                    next_strategy = strategies[idx + 1][0]
                    logger.debug(
                        "%s strategy returned None, falling back to %s",
                        strategy_name,
                        next_strategy,
                    )
        except (ValueError, RuntimeError, SCGOValidationError) as e:
            if is_last_strategy:
                raise
            next_strategy = strategies[idx + 1][0]
            logger.debug(
                "%s strategy failed (%s): %s; falling back to %s",
                strategy_name,
                type(e).__name__,
                e,
                next_strategy,
            )
            continue

    strategy_names = [name for name, _ in strategies]
    logger.warning(
        "All initialization strategies returned None: composition=%s, "
        "n_atoms=%s, strategies=%s",
        composition,
        len(composition),
        strategy_names,
    )
    raise SCGORuntimeError(
        f"All initialization strategies returned None: composition={composition}, "
        f"n_atoms={len(composition)}, strategies={strategy_names}"
    )


def create_initial_cluster(
    composition: list[str],
    rng: np.random.Generator,
    placement_radius_scaling: float = PLACEMENT_RADIUS_SCALING_DEFAULT,
    min_distance_factor: float = MIN_DISTANCE_FACTOR_DEFAULT,
    vacuum: float = VACUUM_DEFAULT,
    previous_search_glob: str = "**/*.db",
    mode: str = "smart",
    connectivity_factor: ConnectivityFactorInput
    | NormalizedConnectivityFactor = CONNECTIVITY_FACTOR,
    *,
    plan: BatchInitPlan | None = None,
    allocation: tuple[str, int | None] | None = None,
    emit_diagnostics: bool = True,
    reuse_exact_matches: bool = True,
    verbosity: int = 1,
) -> Atoms:
    """Create an initial cluster using several strategies.

    This function provides the single entry point for building starting
    structures for global optimization. It is implemented as a wrapper around
    :func:`create_initial_cluster_batch` with ``n_structures=1`` to ensure
    consistent behavior. For "smart" mode, uses probabilistic strategy selection
    for single calls (deterministic allocation for batch calls).

    Independent of the creation mode, successful returns obey the same basic
    invariants:

    - no hard clashes according to ``min_distance_factor`` and covalent radii
    - the cluster is connected under ``connectivity_factor``
    - positions are reproducible for a given ``rng`` seed

    Args:
        composition: target list of element symbols.
        rng: numpy ``Generator`` providing all randomness for this call.
        placement_radius_scaling: scale factor for radii in random placement.
        min_distance_factor: scale factor for minimum distance
            checks; the placement loop relaxes it slightly if repeated
            attempts fail.
        vacuum: extra padding for the generated simulation cell.
        previous_search_glob: glob pattern to find database files.
        mode: Initialization strategy: ``smart`` (default Metropolis mix of
            templates, seed+growth, and random_spherical), ``seed+growth``,
            ``random_spherical``, or ``template``.
        connectivity_factor: Factor to multiply sum of covalent radii for
            connectivity threshold. Defaults to ``CONNECTIVITY_FACTOR`` (1.4).
        plan: A pre-computed :class:`BatchInitPlan` (discovery + allocation).
            When provided, this call reuses the already-resolved templates,
            seeds, and strategy allocation instead of re-running discovery. Use
            :func:`plan_batch_initialization` to build one plan per batch so
            discovery and allocation run a single time per batch.
        allocation: Override the ``(strategy, template_index)`` allocation for
            this single structure. Only meaningful together with ``plan``;
            without it the plan's first allocation is used.
        emit_diagnostics: When ``False``, suppress the per-call diagnostic
            summary logging (the batch owner emits the aggregate summary).
        verbosity: Verbosity for initialization diagnostic summaries (0-3).

    Returns:
        An :class:`ase.Atoms` instance with the initial cluster. When
        ``composition`` is empty, returns an empty ``Atoms`` object.

    Raises:
        SCGOValidationError: If ``composition`` is ``None``, is not a
            list/tuple of element symbols, if numeric parameters are invalid,
            if ``mode`` is unsupported, or if a valid cluster satisfying the
            distance/connectivity constraints cannot be constructed.
        SCGORuntimeError: If every initialization strategy returned ``None``.

    Note:
        This function is implemented as a wrapper around
        :func:`create_initial_cluster_batch` to ensure consistent behavior.
        For generating multiple structures, use :func:`create_initial_cluster_batch`
        directly for better performance and deterministic strategy allocation.

    """
    results = create_initial_cluster_batch(
        composition=composition,
        n_structures=1,
        rng=rng,
        placement_radius_scaling=placement_radius_scaling,
        min_distance_factor=min_distance_factor,
        vacuum=vacuum,
        previous_search_glob=previous_search_glob,
        mode=mode,
        connectivity_factor=connectivity_factor,
        n_jobs=1,  # Single structure, no parallelization needed
        plan=plan,
        allocation=allocation,
        emit_diagnostics=emit_diagnostics,
        reuse_exact_matches=reuse_exact_matches,
        verbosity=verbosity,
    )
    return results[0]


def _try_exact_match(
    composition: list[str],
    exact_candidates: dict[str, list[tuple[float, Atoms]]],
    rng: np.random.Generator,
) -> Atoms | None:
    """Reuse a previous exact-composition minimum as an initial seed.

    Selects a diverse candidate for the formula matching ``composition`` (cycling
    through the energy-sorted, geometry-deduplicated list via ``rng``), strips the
    ``final_unique_minimum`` / ``raw_score`` tags so it does not leak into the new
    run's database, stamps it as a reused seed, reorders it to the target
    composition, and validates it like any freshly built cluster.

    Args:
        composition: Target composition list.
        exact_candidates: Mapping of formula -> ``(energy, atoms)`` candidates as
            produced by :func:`_find_exact_candidates` / :func:`_discover_all_candidates`.
        rng: Random number generator for diverse candidate selection.

    Returns:
        A validated ``Atoms`` object, or ``None`` if no usable exact candidate
        exists or the chosen one fails validation (caller should fall back).
    """
    if not composition:
        return None
    formula = get_cluster_formula(composition)
    candidates = exact_candidates.get(formula)
    if not candidates:
        return None
    if len(composition) <= 2:
        return None

    # Diverse selection across calls: cycle through the (energy-sorted,
    # geometry-deduplicated) list using rng, so a batch of "exact" allocations
    # does not all reuse the very same global minimum.
    idx = int(rng.integers(0, len(candidates))) if len(candidates) > 1 else 0
    _energy, atoms = candidates[idx]

    atoms = atoms.copy()
    # Atoms.copy() shares the nested key_value_pairs bag with the cached source
    # atoms, so build a fresh bag before mutating it; otherwise stripping the
    # tags would corrupt the global candidate cache reused across the batch.
    bag = dict(atoms.info.get("key_value_pairs", {}))
    atoms.info["key_value_pairs"] = bag
    bag.pop("final_unique_minimum", None)
    bag.pop("raw_score", None)
    source_db = bag.get("scgo_source_db")
    source_run = bag.get("scgo_source_run_id")
    set_tags(
        atoms,
        scgo_reused_from_previous_search=True,
        scgo_source_db=source_db,
        scgo_source_run_id=source_run,
    )

    try:
        ordered = reorder_cluster_to_composition(atoms, composition)
        validated_atoms, _, _ = validate_cluster(
            ordered,
            composition=composition,
            min_distance_factor=MIN_DISTANCE_FACTOR_DEFAULT,
            connectivity_factor=CONNECTIVITY_FACTOR,
            sort_atoms=True,
            raise_on_failure=True,
            source="exact",
        )
    except (ValueError, RuntimeError, SCGOValidationError):
        logger.debug(
            "Exact-match reuse candidate for %s failed validation; skipping",
            formula,
        )
        return None
    return validated_atoms


def _generate_single_structure_internal(
    composition: list[str],
    strategy: str,
    structure_rng: np.random.Generator,
    vacuum: float = VACUUM_DEFAULT,
    placement_radius_scaling: float = PLACEMENT_RADIUS_SCALING_DEFAULT,
    min_distance_factor: float = MIN_DISTANCE_FACTOR_DEFAULT,
    connectivity_factor: ConnectivityFactorInput
    | NormalizedConnectivityFactor = CONNECTIVITY_FACTOR,
    template_index: int | None = None,
    discovery_templates: list[Atoms] | None = None,
    precomputed_candidates_by_formula: dict[str, list[tuple[float, Atoms]]]
    | None = None,
    valid_seed_combinations: list[tuple[str, ...]] | None = None,
    exact_candidates: dict[str, list[tuple[float, Atoms]]] | None = None,
) -> tuple[Atoms, str, str | None]:
    """Internal helper to generate a single structure using a specific strategy."""
    cell_side = compute_cell_side(composition, vacuum=vacuum)
    n_atoms = len(composition)

    def _run_template_strategy() -> Atoms | None:
        if n_atoms == 2:
            return None
        return _try_template_generation(
            composition=composition,
            n_atoms=n_atoms,
            cell_side=cell_side,
            rng=structure_rng,
            placement_radius_scaling=placement_radius_scaling,
            min_distance_factor=min_distance_factor,
            connectivity_factor=connectivity_factor,
            template_index=template_index,
            discovery_templates=discovery_templates,
        )

    def _run_seed_growth_strategy() -> Atoms | None:
        if n_atoms <= 2:
            return None
        return _try_seed_growth(
            composition=composition,
            cell_side=cell_side,
            rng=structure_rng,
            placement_radius_scaling=placement_radius_scaling,
            min_distance_factor=min_distance_factor,
            connectivity_factor=connectivity_factor,
            candidates_by_formula=precomputed_candidates_by_formula or {},
            valid_combinations=valid_seed_combinations or [],
        )

    def _run_exact_strategy() -> Atoms | None:
        if n_atoms <= 2:
            return None
        return _try_exact_match(
            composition=composition,
            exact_candidates=exact_candidates or {},
            rng=structure_rng,
        )

    def _run_random_spherical_strategy() -> Atoms:
        return random_spherical(
            composition=composition,
            cell_side=cell_side,
            rng=structure_rng,
            placement_radius_scaling=placement_radius_scaling,
            min_distance_factor=min_distance_factor,
            connectivity_factor=connectivity_factor,
        )

    strategies = {
        "template": _run_template_strategy,
        "seed+growth": _run_seed_growth_strategy,
        "exact": _run_exact_strategy,
        "random_spherical": _run_random_spherical_strategy,
    }

    if strategy == "random_spherical":
        atoms = _run_random_spherical_strategy()
        validated_atoms, _, _ = validate_cluster(
            atoms,
            composition=composition,
            min_distance_factor=min_distance_factor,
            connectivity_factor=connectivity_factor,
            sort_atoms=True,
            raise_on_failure=True,
            source="random_spherical",
        )
        return validated_atoms, "random_spherical", None

    sequence = [strategy, "random_spherical"]
    strategy_functions = [(name, strategies[name]) for name in sequence]

    return _try_strategies_in_order(
        strategies=strategy_functions,
        composition=composition,
        connectivity_factor=connectivity_factor,
        min_distance_factor=min_distance_factor,
        return_strategy=True,
    )


def _generate_structure_batch_item(
    assignment: tuple[int, str, int | None, int],
    composition: list[str],
    vacuum: float = VACUUM_DEFAULT,
    placement_radius_scaling: float = PLACEMENT_RADIUS_SCALING_DEFAULT,
    min_distance_factor: float = MIN_DISTANCE_FACTOR_DEFAULT,
    connectivity_factor: ConnectivityFactorInput
    | NormalizedConnectivityFactor = CONNECTIVITY_FACTOR,
    discovery_templates: list[Atoms] | None = None,
    precomputed_candidates_by_formula: dict[str, list[tuple[float, Atoms]]]
    | None = None,
    valid_seed_combinations: list[tuple[str, ...]] | None = None,
    exact_candidates: dict[str, list[tuple[float, Atoms]]] | None = None,
) -> tuple[int, Atoms, str, str | None]:
    """Helper for batch processing an individual structure assignment."""
    idx, strategy, template_index, structure_seed = assignment
    structure_rng = np.random.default_rng(structure_seed)
    atoms, used_strategy, fallback_from = _generate_single_structure_internal(
        composition=composition,
        strategy=strategy,
        structure_rng=structure_rng,
        vacuum=vacuum,
        placement_radius_scaling=placement_radius_scaling,
        min_distance_factor=min_distance_factor,
        connectivity_factor=connectivity_factor,
        template_index=template_index,
        discovery_templates=discovery_templates,
        precomputed_candidates_by_formula=precomputed_candidates_by_formula,
        valid_seed_combinations=valid_seed_combinations,
        exact_candidates=exact_candidates,
    )
    return idx, atoms, used_strategy, fallback_from


def reset_init_diagnostics() -> None:
    """Clear the per-batch initialization diagnostic collectors.

    The owner of a population batch calls this before generating, passes
    ``emit_diagnostics=False`` to the inner single-structure calls, and finishes
    with :func:`emit_init_diagnostics`, so a run logs one aggregate summary
    instead of one per candidate.
    """
    _SeedSamplingLogCollector.reset()
    InitDiagnosticsCollector.reset()


def emit_init_diagnostics(
    n_structures: int,
    *,
    verbosity: int = 1,
    extra: str = "",
) -> None:
    """Emit the aggregated initialization summaries collected for one batch."""
    _SeedSamplingLogCollector.emit_summary_if_any()
    InitDiagnosticsCollector.emit_summary(
        logger,
        verbosity=verbosity,
        n_structures=n_structures,
        extra=extra,
    )


@dataclass
class BatchInitPlan:
    """Resolved, reusable outcome of discovery + strategy allocation for a batch.

    Computing this is the expensive (and noisy) part of initialization: it scans
    previous-search databases and decides how many structures use templates,
    seed+growth, and random placement. It is produced exactly once per batch and
    then reused across every structure, so discovery and allocation run a single
    time instead of once per generated candidate.
    """

    allocations: list[tuple[str, int | None]]
    discovery_templates: list[Atoms] | None
    precomputed_candidates_by_formula: dict[str, list[tuple[float, Atoms]]]
    valid_seed_combinations: list[tuple[str, ...]]
    exact_candidates: dict[str, list[tuple[float, Atoms]]] = field(default_factory=dict)

    def allocation_for(self, index: int) -> tuple[str, int | None]:
        """Return the ``(strategy, template_index)`` allocation for ``index``.

        Indices cycle through the plan, so single-structure calls and retries
        that reuse a batch plan keep sampling the planned strategy mix instead
        of always repeating the first allocation.
        """
        return self.allocations[index % len(self.allocations)]


def plan_batch_initialization(
    composition: list[str],
    n_structures: int,
    rng: np.random.Generator,
    *,
    vacuum: float = VACUUM_DEFAULT,
    previous_search_glob: str = "**/*.db",
    mode: str = "smart",
    placement_radius_scaling: float = PLACEMENT_RADIUS_SCALING_DEFAULT,
    min_distance_factor: float = MIN_DISTANCE_FACTOR_DEFAULT,
    connectivity_factor: ConnectivityFactorInput
    | NormalizedConnectivityFactor = CONNECTIVITY_FACTOR,
    reuse_exact_matches: bool = True,
) -> BatchInitPlan:
    """Run discovery + strategy allocation once for a whole batch.

    Returns a :class:`BatchInitPlan` that downstream batched generators reuse,
    so the (noisy, DB-scanning) discovery step and the strategy allocation
    happen a single time per batch rather than once per generated structure.

    The discovery/allocation INFO logs ("Candidate discovery:", "Initialization
    for N-atom clusters:", "Strategy allocation (...)") are emitted exactly once
    here, when the plan is built.

    Raises:
        SCGOValidationError: If ``n_structures`` is below 1, if ``composition``
            is invalid, or if ``mode`` is not one of ``smart``, ``template``,
            ``seed+growth``, or ``random_spherical``.
    """
    if n_structures < 1:
        raise SCGOValidationError(f"n_structures must be >= 1, got {n_structures}")
    validate_composition(composition, allow_empty=True, allow_tuple=True)
    if not composition:
        return BatchInitPlan(
            allocations=[("random_spherical", None)] * n_structures,
            discovery_templates=None,
            precomputed_candidates_by_formula={},
            valid_seed_combinations=[],
        )

    n_atoms = len(composition)
    cell_side = compute_cell_side(composition, vacuum=vacuum)

    precomputed_candidates_by_formula: dict[str, list[tuple[float, Atoms]]] = {}
    valid_seed_combinations: list[tuple[str, ...]] = []
    exact_candidates: dict[str, list[tuple[float, Atoms]]] = {}
    if mode in ("smart", "seed+growth") and n_atoms > 2:
        # In smart mode, the exact-match tier is discovered together with the
        # sub-composition (seed) tier in a single DB scan. Other modes never
        # allocate exact matches, so we keep using the cheaper sub-only scan.
        if mode == "smart" and n_atoms > 2 and reuse_exact_matches:
            sub, exact_candidates = _discover_all_candidates(
                composition, previous_search_glob
            )
        else:
            sub = _find_smaller_candidates(composition, previous_search_glob)
            exact_candidates = {}
        precomputed_candidates_by_formula = _filter_candidates_by_geometry(sub)
        if precomputed_candidates_by_formula:
            target_counts = get_composition_counts(composition)
            valid_seed_combinations = _find_valid_seed_combinations(
                precomputed_candidates_by_formula, target_counts
            )
        exact_candidates = _filter_candidates_by_geometry(exact_candidates)

    discovery_templates = None
    n_exact = sum(len(cands) for cands in exact_candidates.values())
    if mode == "smart":
        discovery = _discover_available_strategies(
            composition=composition,
            n_atoms=n_atoms,
            cell_side=cell_side,
            rng=rng,
            placement_radius_scaling=placement_radius_scaling,
            min_distance_factor=min_distance_factor,
            connectivity_factor=connectivity_factor,
            candidates_by_formula=precomputed_candidates_by_formula,
            valid_combinations=valid_seed_combinations,
            n_exact=n_exact,
        )
        allocations = _allocate_initialization_strategies(
            n_structures=n_structures,
            templates=discovery["templates"],
            n_seed_formulas=discovery["n_seed_formulas"],
            n_seed_combinations=discovery["n_seed_combinations"],
            rng=rng,
            n_atoms=n_atoms,
            n_exact=n_exact,
        )
        discovery_templates = discovery["templates"]
    elif mode in ("template", "seed+growth", "random_spherical"):
        allocations = [(mode, None)] * n_structures
    else:
        raise SCGOValidationError(f'Unsupported mode: "{mode}"')

    return BatchInitPlan(
        allocations=allocations,
        discovery_templates=discovery_templates,
        precomputed_candidates_by_formula=precomputed_candidates_by_formula,
        valid_seed_combinations=valid_seed_combinations,
        exact_candidates=exact_candidates,
    )


def create_initial_cluster_batch(
    composition: list[str],
    n_structures: int,
    rng: np.random.Generator,
    placement_radius_scaling: float = PLACEMENT_RADIUS_SCALING_DEFAULT,
    min_distance_factor: float = MIN_DISTANCE_FACTOR_DEFAULT,
    vacuum: float = VACUUM_DEFAULT,
    previous_search_glob: str = "**/*.db",
    mode: str = "smart",
    connectivity_factor: ConnectivityFactorInput
    | NormalizedConnectivityFactor = CONNECTIVITY_FACTOR,
    n_jobs: int | None = None,
    *,
    plan: BatchInitPlan | None = None,
    allocation: tuple[str, int | None] | None = None,
    emit_diagnostics: bool = True,
    reuse_exact_matches: bool = True,
    verbosity: int = 1,
) -> list[Atoms]:
    """Create multiple initial clusters with deterministic per-structure RNG.

    For ``smart`` mode, uses Metropolis allocation across templates,
    seed+growth, and random_spherical. Each structure receives an independent
    seed derived from ``rng`` (``batch_base_seed + index * 7919``), so batch
    results are reproducible and identical for ``n_jobs=1`` vs parallel workers
    when the parent ``rng`` state matches.

    Validated structures are reordered to match ``composition`` for GA pairing.

    Discovery (previous-search DB scan) and strategy allocation run exactly once
    for the batch. Pass a :class:`BatchInitPlan` produced by
    :func:`plan_batch_initialization` via ``plan=`` to reuse an already-resolved
    plan (the recommended path for all batched initialization, so discovery and
    allocation happen a single time per batch rather than once per structure).

    Args:
        n_jobs: Parallelism for structure generation; ``None`` uses the
            project default (``DEFAULT_N_JOBS`` from
            :mod:`scgo.utils.parallel_workers`, single worker; opt in with -1/-2
            for parallelism).
        plan: Pre-computed :class:`BatchInitPlan`; when given, discovery and
            allocation are NOT re-run and their INFO logs are emitted only when
            the plan itself is built.
        allocation: Override the ``(strategy, template_index)`` allocation for
            every structure in this call. Only meaningful together with ``plan``
            (used to steer a single retry without rebuilding the plan).
        emit_diagnostics: When ``False``, suppress the per-batch diagnostic
            summary (the batch owner emits the aggregate summary instead).
        verbosity: Verbosity for initialization diagnostic summaries (0-3).

    Returns:
        List of ``n_structures`` :class:`ase.Atoms` objects. When
        ``composition`` is empty, the list contains empty ``Atoms`` objects.

    Raises:
        SCGOValidationError: If ``n_structures`` is below 1, if ``composition``
            is invalid, if numeric parameters are invalid, or if ``mode`` is
            not one of ``smart``, ``template``, ``seed+growth``, or
            ``random_spherical``.
    """
    if n_structures < 1:
        raise SCGOValidationError(f"n_structures must be >= 1, got {n_structures}")

    validate_composition(composition, allow_empty=True, allow_tuple=True)

    if not composition:
        return [Atoms() for _ in range(n_structures)]

    if placement_radius_scaling <= 0:
        raise SCGOValidationError(
            f"placement_radius_scaling must be positive, got {placement_radius_scaling}"
        )

    if min_distance_factor < 0:
        raise SCGOValidationError(
            f"min_distance_factor must be non-negative, got {min_distance_factor}"
        )

    if vacuum < 0:
        raise SCGOValidationError(f"vacuum must be non-negative, got {vacuum}")

    if plan is None:
        plan = plan_batch_initialization(
            composition,
            n_structures,
            rng,
            vacuum=vacuum,
            previous_search_glob=previous_search_glob,
            mode=mode,
            placement_radius_scaling=placement_radius_scaling,
            min_distance_factor=min_distance_factor,
            connectivity_factor=connectivity_factor,
            reuse_exact_matches=reuse_exact_matches,
        )

    if allocation is not None:
        allocations = [allocation] * n_structures
    else:
        allocations = [plan.allocation_for(i) for i in range(n_structures)]

    discovery_templates = plan.discovery_templates
    precomputed_candidates_by_formula = plan.precomputed_candidates_by_formula
    valid_seed_combinations = plan.valid_seed_combinations
    exact_candidates = plan.exact_candidates

    batch_base_seed = rng.integers(0, 2**31)
    structure_assignments = []
    for i, (strategy, template_index) in enumerate(allocations):
        structure_seed = (batch_base_seed + i * 7919) % (2**31)
        structure_assignments.append((i, strategy, template_index, structure_seed))

    if emit_diagnostics:
        _SeedSamplingLogCollector.reset()
        InitDiagnosticsCollector.reset()

    def _worker_wrapper(assignment):
        return _generate_structure_batch_item(
            assignment=assignment,
            composition=composition,
            vacuum=vacuum,
            placement_radius_scaling=placement_radius_scaling,
            min_distance_factor=min_distance_factor,
            connectivity_factor=connectivity_factor,
            discovery_templates=discovery_templates,
            precomputed_candidates_by_formula=precomputed_candidates_by_formula,
            valid_seed_combinations=valid_seed_combinations,
            exact_candidates=exact_candidates,
        )

    max_workers = resolve_n_jobs_for_tasks(n_jobs, n_structures)
    results: list[Atoms | None] = [None] * n_structures
    fallback_info: dict[int, tuple[str, str | None]] = {}

    if max_workers == 1:
        for assignment in structure_assignments:
            idx, atoms, used_strat, fallback = _worker_wrapper(assignment)
            results[idx] = atoms
            fallback_info[idx] = (used_strat, fallback)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_worker_wrapper, a) for a in structure_assignments
            ]
            for future in as_completed(futures):
                idx, atoms, used_strat, fallback = future.result()
                results[idx] = atoms
                fallback_info[idx] = (used_strat, fallback)

    for used_strat, fallback in fallback_info.values():
        if fallback is not None:
            InitDiagnosticsCollector.record_fallback(used_strat, fallback)

    if emit_diagnostics:
        _SeedSamplingLogCollector.emit_summary_if_any()

        if n_structures > 1:
            InitDiagnosticsCollector.emit_summary(
                logger,
                verbosity=verbosity,
                n_structures=n_structures,
            )

    return results  # type: ignore[return-value]
