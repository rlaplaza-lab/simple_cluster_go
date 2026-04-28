"""Initial cluster structure generation with seed-based growth strategies.

This module provides functions for creating initial cluster structures for global
optimization, including intelligent seed selection from previous runs and adaptive
growth strategies based on available candidates.
"""

from __future__ import annotations

import itertools
import logging
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, vdw_radii

from scgo.database.cache import get_global_cache
from scgo.utils.helpers import (
    get_composition_counts,
)
from scgo.utils.logging import get_logger
from scgo.utils.parallel_workers import resolve_n_jobs_to_workers
from scgo.utils.validation import validate_composition

from .candidate_discovery import (
    _find_smaller_candidates,
    deduplicate_seed_candidates,
    get_structure_signature,
    is_composition_subset,
)
from .geometry_helpers import (
    _classify_seed_geometry,
    _generate_rotation_matrix,
    _should_check_connectivity,
    get_covalent_radius,
    validate_cluster,
    validate_cluster_structure,
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
    TEMPLATE_ROTATION_CANDIDATES,
    VACUUM_DEFAULT,
)
from .random_spherical import grow_from_seed, random_spherical
from .seed_combiners import combine_and_grow
from .strategy_allocation import _allocate_strategies_metropolis
from .templates import generate_template_matches

# Re-export internal names for test compatibility
_get_structure_signature = get_structure_signature
_deduplicate_seed_candidates = deduplicate_seed_candidates

# Re-export cache namespaces for tests
_TEMPLATE_ROTATIONS_CACHE_NS = "template_rotations"
_DB_CANDIDATES_CACHE_NS = "db_candidates"

logger = get_logger(__name__)


class InitStrategy(Enum):
    """Initialization strategies used by allocation and generation logic."""

    TEMPLATE = "template"
    SEED_GROWTH = "seed+growth"
    RANDOM_SPHERICAL = "random_spherical"


def _get_effective_vdw_radius(symbol: str) -> float:
    """Get van-der-Waals radius, falling back to scaled covalent radius if VdW is NaN.

    ASE's vdw_radii table has NaN for many transition metals (e.g., Co, Fe, Ru).
    For such elements, we use covalent_radius * 1.3 as a reasonable estimate
    (typical VdW/covalent ratio for metals is ~1.2-1.4).

    Args:
        symbol: Element symbol (e.g., "Pt", "Co")

    Returns:
        Effective VdW radius in Angstroms
    """
    try:
        r = float(vdw_radii[atomic_numbers[symbol]])
    except KeyError as exc:
        raise ValueError(
            f"Unknown element symbol: {symbol}. Could not find VdW radius."
        ) from exc
    if not np.isfinite(r) or r <= 0:
        r = get_covalent_radius(symbol) * 1.3
        logger.debug(
            f"VdW radius for {symbol} is missing/NaN in ASE; using covalent*1.3 = {r:.3f} Å"
        )
    return r


def compute_cell_side(composition: list[str], vacuum: float = VACUUM_DEFAULT) -> float:
    """Estimate a cubic cell side from atomic van-der-Waals volumes.

    The estimate computes atomic volumes using ASE's van-der-Waals radii,
    converts that to an effective spherical radius and returns a cubic
    side that contains the cluster plus the requested ``vacuum`` padding.

    For elements where ASE's vdw_radii is NaN (e.g., Co, Fe, Ru), falls back
    to scaled covalent radius to support multi-element clusters like Pt4Co1.

    Args:
        composition: Sequence of element symbols (e.g. ["Pt", "Pt"])
        vacuum: Extra padding (Å) to add to the estimated diameter.

    Returns:
        Cubic cell side length in Å. Returns 0.0 for an empty composition.

    """
    if not composition:
        return 0.0

    vdw_radii_list = [_get_effective_vdw_radius(s) for s in composition]
    total_atomic_volume = sum(4.0 / 3.0 * np.pi * r**3 for r in vdw_radii_list)
    # Apply packing efficiency factor (~0.74 for FCC/HCP)
    packed_volume = total_atomic_volume / PACKING_EFFICIENCY_FCC_HCP
    effective_cluster_radius = (3.0 * packed_volume / (4.0 * np.pi)) ** (1.0 / 3.0)
    cell_side = 2 * effective_cluster_radius + vacuum

    # Warn if computed cell side is unreasonably large
    if cell_side > MAX_REASONABLE_CELL_SIDE:
        logger.warning(
            f"Computed cell_side ({cell_side:.1f} Å) exceeds reasonable threshold "
            f"({MAX_REASONABLE_CELL_SIDE} Å) for {len(composition)} atoms. "
            f"This may indicate very large composition or vacuum value."
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

    """
    if not candidates:
        return None

    # Fast-path for single candidate
    if len(candidates) == 1:
        energy, atoms = candidates[0]
        return energy, atoms.copy()

    # Verify all candidates have the same composition
    first_symbols = tuple(candidates[0][1].get_chemical_symbols())
    for _energy, atoms in candidates[1:]:
        if tuple(atoms.get_chemical_symbols()) != first_symbols:
            raise ValueError(
                "All candidates must have the same composition for Boltzmann sampling. "
                f"Found {first_symbols} vs {tuple(atoms.get_chemical_symbols())}"
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
    assert temperature is not None, (
        "Temperature should never be None after the above block"
    )

    # Validate temperature
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

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
    n_atoms: int,
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
        n_atoms: Number of atoms
        n_unique_elements: Number of unique elements in composition
        template_type_counts: Dictionary counting occurrences of each template type
        total_candidates: Total number of template candidates

    Returns:
        Weight for this template type
    """
    # Extract base weight from TEMPLATE_BASE_WEIGHTS dict
    # TEMPLATE_BASE_WEIGHTS is dict[str, dict[str, float]], so we need to get the "base" value
    weight_config = TEMPLATE_BASE_WEIGHTS.get(template_type, {})
    if isinstance(weight_config, dict):
        base_weight = weight_config.get("base", 1.0)
    else:
        base_weight = 1.0

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

    return base_weight + diversity_boost - multi_element_penalty


def _get_template_type(atoms: Atoms) -> str:
    """Extract template type from atoms info, defaulting to "unknown" if not present.

    Args:
        atoms: The Atoms object to extract template type from

    Returns:
        Template type string, or "unknown" if not available
    """
    info = getattr(atoms, "info", None)
    return info.get("template_type", "unknown") if info else "unknown"


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
        signature = _get_structure_signature(atoms)
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
    connectivity_factor: float,
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

    Returns:
        Validated Atoms with rotation applied, or None if validation fails.
    """
    selected = selected.copy()
    selected.set_cell([cell_side, cell_side, cell_side])
    selected.center()

    center = selected.get_center_of_mass()

    # Check if we have pre-computed rotations for this template signature
    template_signature = _get_structure_signature(selected)
    rotation_cache_key = (template_signature, cell_side)
    rotation_candidates = get_global_cache().get(
        _TEMPLATE_ROTATIONS_CACHE_NS, rotation_cache_key
    )

    if rotation_candidates is None:
        # Generate and cache rotation candidates using a deterministic seed
        # derived from the template signature to ensure reproducibility
        # even across different RNG instances
        signature_seed = abs(hash(template_signature)) % (2**31)
        rotation_rng = np.random.default_rng(signature_seed)

        rotation_candidates = []
        for _ in range(TEMPLATE_ROTATION_CANDIDATES):
            axis = rotation_rng.standard_normal(3)
            axis /= np.linalg.norm(axis)
            angle = rotation_rng.uniform(0, 2 * np.pi)
            R = _generate_rotation_matrix(axis, angle)

            rotated = selected.copy()
            positions = rotated.get_positions()
            rotated.set_positions(center + (positions - center) @ R.T)
            rotation_candidates.append(rotated)

        # Store in cache for future use
        get_global_cache().set(
            _TEMPLATE_ROTATIONS_CACHE_NS, rotation_cache_key, rotation_candidates
        )

    selected = rotation_candidates[rng.integers(0, len(rotation_candidates))].copy()

    is_valid, error_message = validate_cluster_structure(
        selected,
        min_distance_factor,
        connectivity_factor,
        check_clashes=True,
        check_connectivity=_should_check_connectivity(selected),
    )
    if not is_valid:
        logger.warning(
            f"Template structure validation failed: {error_message}. Falling back to random_spherical."
        )
        return None
    return selected


def _try_template_generation(
    composition: list[str],
    n_atoms: int,
    cell_side: float,
    rng: np.random.Generator,
    placement_radius_scaling: float,
    min_distance_factor: float,
    connectivity_factor: float,
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
        )
        if result is not None:
            return result
        # Validation failed; fall through to normal generation below
        # IMPORTANT: Reset template_index because it refers to discovery_templates list,
        # which may not align with the generated template_candidates list below.
        template_index = None

    # Get all template candidates (exact and near matches) using unified function
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
        return None

    # Deduplicate templates to remove duplicates between exact and near-match templates
    # (e.g., cube, tetrahedron, and octahedron all producing octahedral structures,
    # or near-match templates producing structures identical to exact matches)
    original_count = len(template_candidates)
    template_candidates = _deduplicate_template_structures(template_candidates)
    if len(template_candidates) < original_count:
        logger.debug(
            f"Deduplicated templates: {original_count} -> {len(template_candidates)} "
            f"unique structures (removed duplicates between exact and near-match templates)"
        )

    # Enhanced diversity: create weighted pool of ALL candidates across all types
    # Sort candidates deterministically for reproducibility
    def get_sort_key(atoms):
        template_type = _get_template_type(atoms)
        com = atoms.get_center_of_mass()
        return (
            template_type,
            round(com[0], 8),
            round(com[1], 8),
            round(com[2], 8),
            len(atoms),
        )

    template_candidates.sort(key=get_sort_key)

    # Calculate weights for all candidates based on template type quality
    n_unique_elements = len(set(composition))
    weighted_candidates = []

    # Count template types in candidates for systematic diversity (compute once)
    template_type_counts = {}
    for candidate in template_candidates:
        template_type = _get_template_type(candidate)
        template_type_counts[template_type] = (
            template_type_counts.get(template_type, 0) + 1
        )

    # Pre-compute weights per template type to avoid redundant calculations
    template_type_weights = {}
    for template_type in template_type_counts:
        template_type_weights[template_type] = _calculate_template_weight(
            template_type,
            n_atoms,
            n_unique_elements,
            template_type_counts,
            len(template_candidates),
        )

    # Apply pre-computed weights to candidates
    for candidate in template_candidates:
        template_type = _get_template_type(candidate)
        weight = template_type_weights.get(template_type, 1.0)
        weighted_candidates.append((weight, candidate, template_type))

    # Select from weighted pool
    if template_index is not None:
        # Use specific template index if provided
        if template_index < 0 or template_index >= len(weighted_candidates):
            logger.warning(
                f"Invalid template_index {template_index}, "
                f"must be in range [0, {len(weighted_candidates)}). Using random selection."
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
    )
    if result is None:
        return None
    logger.debug(
        f"Smart mode: using template {selected_type} ({n_unique_template_types} unique types available, {len(template_candidates)} total candidates)"
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
        Selected (energy, atoms) tuple, or None if no suitable candidate found
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
        raise ValueError(f"Invalid seed sampling strategy: {strategy!r} (expected 0-4)")

    return handler()


def _grow_from_random_seed(
    composition: list[str],
    cell_side: float,
    rng: np.random.Generator,
    placement_radius_scaling: float,
    min_distance_factor: float,
    connectivity_factor: float,
) -> Atoms | None:
    """Generate a small random seed and grow it to the target composition.

    This function is used when no external seeds from previous runs are available.
    It creates a small random cluster (about 1/4 of target size, minimum 3 atoms)
    and grows it to the target composition using convex-hull-based placement.

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
        Atoms object if successful, None otherwise
    """
    n_atoms = len(composition)

    # Determine seed size: about 1/4 of target, minimum 3 atoms, maximum 15
    seed_size = max(3, min(15, n_atoms // 4))

    # Create seed composition by sampling from target composition
    # Sample to preserve composition ratios (vs. taking first N elements).
    if not composition:
        # Empty composition - return None early
        return None
    seed_composition = list(rng.choice(composition, size=seed_size, replace=True))

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
    except ValueError:
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
    except ValueError:
        return None


def _find_valid_seed_combinations(
    candidates_by_formula: dict[str, list[tuple[float, Atoms]]],
    target_counts: dict[str, int],
) -> list[tuple[str, ...]]:
    """Find all valid seed formula combinations that are sub-compositions of target.

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
    tried_positions: set[int],
    existing_geometries: list[str],
    rng: np.random.Generator,
    max_attempts: int = 10,
) -> Atoms | None:
    """Sample a suitable seed from candidates with geometry diversity preference.

    Args:
        candidates: List of (energy, atoms) tuples
        strategy: Sampling strategy index
        tried_positions: Set of position hashes already tried
        existing_geometries: List of geometries of already-selected seeds
        rng: Random number generator
        max_attempts: Maximum attempts to find suitable seed

    Returns:
        Suitable seed Atoms object, or None if not found
    """
    # Pre-filter candidates to remove already-tried positions
    available_candidates = [
        (e, a)
        for e, a in candidates
        if hash(a.get_positions().tobytes()) not in tried_positions
    ]

    if not available_candidates:
        return None

    for attempt in range(max_attempts):
        sampled = _sample_seed_with_strategy(
            available_candidates,
            strategy=(strategy + attempt) % 5,
            rng=rng,
        )

        if sampled is None:
            return None

        _, sampled_seed = sampled
        geometry = _classify_seed_geometry(sampled_seed)

        # Mark this position as tried
        pos_hash = hash(sampled_seed.get_positions().tobytes())
        tried_positions.add(pos_hash)

        # Remove from available candidates to avoid sampling again
        available_candidates = [
            (e, a)
            for e, a in available_candidates
            if hash(a.get_positions().tobytes()) != pos_hash
        ]

        # If no more candidates available, stop trying
        if not available_candidates:
            break

        # Accept if suitable geometry
        if geometry not in ["planar", "3d"]:
            continue

        # Prefer geometry diversity: if all existing are same, prefer different
        if existing_geometries:
            all_same = all(g == existing_geometries[0] for g in existing_geometries)
            if all_same and geometry == existing_geometries[0]:
                continue  # Prefer different geometry

        return sampled_seed.copy()

    return None


def _try_seed_growth(
    composition: list[str],
    cell_side: float,
    rng: np.random.Generator,
    placement_radius_scaling: float,
    min_distance_factor: float,
    connectivity_factor: float,
    candidates_by_formula: dict[str, list[tuple[float, Atoms]]],
    valid_combinations: list[tuple[str, ...]],
) -> Atoms | None:
    """Try to generate a cluster using seed+growth strategy.

    This helper function encapsulates seed+growth logic for the smart mode.
    It finds suitable seeds from previous runs and grows them to the target composition.

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
        Atoms object if successful, None otherwise
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
        logger.info("seed+growth: no database seeds found; using random seed growth")
        return _grow_from_random_seed(**random_seed_kwargs)

    if not valid_combinations:
        logger.info(
            "seed+growth: no valid DB seed combinations; using random seed growth"
        )
        return _grow_from_random_seed(**random_seed_kwargs)

    tried_positions: set[int] = set()

    # Try multiple strategies
    for strategy_idx in range(SEED_COMBINATION_STRATEGY_COUNT):
        combo = valid_combinations[rng.integers(0, len(valid_combinations))]
        seeds_to_combine: list[Atoms] = []
        existing_geometries: list[str] = []

        # Sample seeds for each formula in the combination
        for formula in combo:
            candidates = candidates_by_formula[formula]
            seed = _sample_suitable_seed(
                candidates,
                strategy_idx,
                tried_positions,
                existing_geometries,
                rng,
            )

            if seed is None:
                logger.debug(f"No suitable seed found for {formula} after attempts")
                break

            seeds_to_combine.append(seed)
            existing_geometries.append(_classify_seed_geometry(seed))

        if not seeds_to_combine:
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

    # All seed combination strategies failed
    logger.info(
        "seed+growth: all %d combination strategies failed; trying next strategy",
        SEED_COMBINATION_STRATEGY_COUNT,
    )
    return None


def _discover_available_strategies(
    composition: list[str],
    n_atoms: int,
    cell_side: float,
    rng: np.random.Generator,
    placement_radius_scaling: float,
    min_distance_factor: float,
    connectivity_factor: float,
    candidates_by_formula: dict[str, list[tuple[float, Atoms]]],
    valid_combinations: list[tuple[str, ...]],
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
    all_templates = generate_template_matches(
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

    # Deduplicate templates
    templates = _deduplicate_template_structures(all_templates)

    # Sort templates to ensure consistent indexing with _try_template_generation
    def get_sort_key(atoms):
        template_type = _get_template_type(atoms)
        com = atoms.get_center_of_mass()
        return (
            template_type,
            round(com[0], 8),
            round(com[1], 8),
            round(com[2], 8),
            len(atoms),
        )

    templates.sort(key=get_sort_key)

    # Discover seeds
    n_seed_formulas = len(candidates_by_formula)
    n_seed_combinations = len(valid_combinations)

    # Note: Discovery info logging moved to _allocate_strategies_metropolis
    # to avoid duplicate messages when create_initial_cluster is called
    # before batch generation (e.g., for creating a template structure)

    return {
        "templates": templates,
        "n_templates": len(templates),
        "n_seed_formulas": n_seed_formulas,
        "n_seed_combinations": n_seed_combinations,
    }




def _try_strategies_in_order(
    strategies: list[tuple[str, Callable[..., Atoms]]],
    composition: list[str],
    connectivity_factor: float,
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

    Returns:
        Atoms object if successful. When ``return_strategy=True``, returns
        a tuple of (Atoms, used_strategy, fallback_from).

    Raises:
        ValueError: If the final fallback strategy fails
        RuntimeError: If the final fallback strategy fails
    """
    if not strategies:
        raise ValueError("No strategies provided to _try_strategies_in_order")

    primary_strategy = strategies[0][0]

    for idx, (strategy_name, strategy_func) in enumerate(strategies):
        is_last_strategy = idx == len(strategies) - 1

        try:
            result = strategy_func()
            if result is not None:
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
        except (ValueError, RuntimeError) as e:
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
    raise RuntimeError(
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
    connectivity_factor: float = CONNECTIVITY_FACTOR,
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
        placement_radius_scaling: scale factor for radii in random placement.
        min_distance_factor: scale factor for minimum distance
            checks; the placement loop relaxes it slightly if repeated
            attempts fail.
        vacuum: extra padding for the generated simulation cell.
        previous_search_glob: glob pattern to find database files.
        mode: how to create the initial cluster. Can be one of:
            - "smart": (default) uses Metropolis allocation to distribute structures
              across templates, seed+growth, and random_spherical based on availability.
              Ensures all templates are sampled while maintaining diversity.
            - "seed+growth": grow from a smaller, low-energy candidate from
              previous searches using convex-hull-based placement. Falls back to
              ``random_spherical`` if no suitable seed is found or if all growth
              attempts fail.
            - "random_spherical": places atoms randomly within a sphere using
              the same distance/connectivity parameters and validation logic.
            - "template": force use of template structures (icosahedral,
              decahedral, octahedral) if available.
        connectivity_factor: Factor to multiply sum of covalent radii for
            connectivity threshold. Defaults to ``CONNECTIVITY_FACTOR`` (1.4).
        rng: numpy ``Generator`` providing all randomness for this call.

    Returns:
        An :class:`ase.Atoms` instance with the initial cluster. When
        ``composition`` is empty, returns an empty ``Atoms`` object.

    Raises:
        TypeError: If ``composition`` is ``None`` or not a list/tuple of
            strings.
        ValueError: If numeric parameters are invalid or a valid cluster
            satisfying the distance/connectivity constraints cannot be
            constructed.

    Note:
        This function is implemented as a wrapper around
        :func:`create_initial_cluster_batch` to ensure consistent behavior.
        For generating multiple structures, use :func:`create_initial_cluster_batch`
        directly for better performance and deterministic strategy allocation.

    """
    # Validate composition type
    validate_composition(composition, allow_empty=True, allow_tuple=True)

    # Handle empty composition
    if not composition:
        return Atoms()

    if placement_radius_scaling <= 0:
        raise ValueError(
            f"placement_radius_scaling must be positive, got {placement_radius_scaling}"
        )

    if min_distance_factor < 0:
        raise ValueError(
            f"min_distance_factor must be non-negative, got {min_distance_factor}"
        )

    if vacuum < 0:
        raise ValueError(f"vacuum must be non-negative, got {vacuum}")

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
    )
    return results[0]


def _generate_single_structure_internal(
    composition: list[str],
    strategy: str,
    structure_rng: np.random.Generator,
    vacuum: float = VACUUM_DEFAULT,
    placement_radius_scaling: float = PLACEMENT_RADIUS_SCALING_DEFAULT,
    min_distance_factor: float = MIN_DISTANCE_FACTOR_DEFAULT,
    connectivity_factor: float = CONNECTIVITY_FACTOR,
    template_index: int | None = None,
    discovery_templates: list[Atoms] | None = None,
    precomputed_candidates_by_formula: dict[str, list[tuple[float, Atoms]]] | None = None,
    valid_seed_combinations: list[tuple[str, ...]] | None = None,
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
        "random_spherical": _run_random_spherical_strategy,
    }

    if strategy == "random_spherical":
        atoms = _run_random_spherical_strategy()
        validated_atoms, _, _ = validate_cluster(
            atoms,
            composition=composition,
            min_distance_factor=min_distance_factor,
            connectivity_factor=connectivity_factor,
            sort_atoms=False,
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
    connectivity_factor: float = CONNECTIVITY_FACTOR,
    discovery_templates: list[Atoms] | None = None,
    precomputed_candidates_by_formula: dict[str, list[tuple[float, Atoms]]] | None = None,
    valid_seed_combinations: list[tuple[str, ...]] | None = None,
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
    )
    return idx, atoms, used_strategy, fallback_from


def create_initial_cluster_batch(
    composition: list[str],
    n_structures: int,
    rng: np.random.Generator,
    placement_radius_scaling: float = PLACEMENT_RADIUS_SCALING_DEFAULT,
    min_distance_factor: float = MIN_DISTANCE_FACTOR_DEFAULT,
    vacuum: float = VACUUM_DEFAULT,
    previous_search_glob: str = "**/*.db",
    mode: str = "smart",
    connectivity_factor: float = CONNECTIVITY_FACTOR,
    n_jobs: int = 1,
) -> list[Atoms]:
    """Create multiple initial clusters with logarithmic scaling strategy allocation."""
    if n_structures < 1:
        raise ValueError(f"n_structures must be >= 1, got {n_structures}")

    validate_composition(composition, allow_empty=True, allow_tuple=True)

    if not composition:
        return [Atoms() for _ in range(n_structures)]

    n_atoms = len(composition)
    cell_side = compute_cell_side(composition, vacuum=vacuum)

    precomputed_candidates_by_formula: dict[str, list[tuple[float, Atoms]]] = {}
    valid_seed_combinations: list[tuple[str, ...]] = []
    if mode in ("smart", "seed+growth") and n_atoms > 2:
        precomputed_candidates_by_formula = _find_smaller_candidates(
            composition,
            previous_search_glob,
        )
        precomputed_candidates_by_formula = _filter_candidates_by_geometry(
            precomputed_candidates_by_formula
        )
        if precomputed_candidates_by_formula:
            target_counts = get_composition_counts(composition)
            valid_seed_combinations = _find_valid_seed_combinations(
                precomputed_candidates_by_formula, target_counts
            )

    discovery_templates = None
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
        )

        allocations = _allocate_strategies_metropolis(
            n_structures=n_structures,
            templates=discovery["templates"],
            n_seed_formulas=discovery["n_seed_formulas"],
            n_seed_combinations=discovery["n_seed_combinations"],
            rng=rng,
            n_atoms=n_atoms,
        )
        discovery_templates = discovery["templates"]
    elif mode in ("template", "seed+growth", "random_spherical"):
        allocations = [(mode, None)] * n_structures
    else:
        raise ValueError(f'Unsupported mode: "{mode}"')

    batch_base_seed = rng.integers(0, 2**31)
    structure_assignments = []
    for i, (strategy, template_index) in enumerate(allocations):
        structure_seed = (batch_base_seed + i * 7919) % (2**31)
        structure_assignments.append((i, strategy, template_index, structure_seed))

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
        )

    max_workers = min(resolve_n_jobs_to_workers(n_jobs), n_structures)
    results: list[Atoms | None] = [None] * n_structures
    fallback_info: dict[int, tuple[str, str | None]] = {}

    if max_workers == 1:
        for assignment in structure_assignments:
            idx, atoms, used_strat, fallback = _worker_wrapper(assignment)
            results[idx] = atoms
            fallback_info[idx] = (used_strat, fallback)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_worker_wrapper, a) for a in structure_assignments]
            for future in as_completed(futures):
                idx, atoms, used_strat, fallback = future.result()
                results[idx] = atoms
                fallback_info[idx] = (used_strat, fallback)

    if logger.isEnabledFor(logging.DEBUG):
        template_to_random = sum(1 for u, f in fallback_info.values() if u == "random_spherical" and f == "template")
        seed_to_random = sum(1 for u, f in fallback_info.values() if u == "random_spherical" and f == "seed+growth")
        logger.debug("Fallbacks: template->random=%d, seed->random=%d", template_to_random, seed_to_random)

    return results  # type: ignore[return-value]


def generate_initial_population(
    composition: list[str],
    n_structures: int,
    rng: np.random.Generator,
    mode: str = "smart",
    n_jobs: int = 1,
    vacuum: float = VACUUM_DEFAULT,
    previous_search_glob: str = "**/*.db",
) -> list[Atoms]:
    """Generate an initial population of cluster structures.

    Simplified interface for batch initialization with sensible defaults.
    This function wraps :func:`create_initial_cluster_batch` with a cleaner
    API that exposes only the most commonly used parameters.

    All other parameters (placement_radius_scaling, min_distance_factor,
    connectivity_factor) use module defaults, which are suitable for most
    production use cases.

    Args:
        composition: Target list of element symbols.
        n_structures: Number of structures to generate.
        rng: Numpy Generator for randomness.
        mode: Initialization strategy. Options:
            - "smart": (default) Intelligently selects strategy based on cluster
              size. For magic numbers, uses mixed strategies (templates, seed+growth,
              random). For other sizes, uses seed+growth and random.
            - "template": Force use of template structures (icosahedral, decahedral,
              octahedral) if available.
            - "seed+growth": Grow from smaller, low-energy candidates from previous
              searches.
            - "random_spherical": Place atoms randomly within a sphere.
        n_jobs: Number of parallel workers. Default: 1 (sequential execution).
            Use >1 for parallel execution to speed up batch generation. Special values:
            - -1: Use all available CPU cores
            - -2: Use all available CPU cores except one
            Parallel execution maintains reproducibility through deterministic seeds.
        vacuum: Extra padding for the generated simulation cell. Default: 10.0 Å.
        previous_search_glob: Glob pattern for finding database files to use as
            seeds for seed+growth strategy. Default: "**/*.db".

    Returns:
        List of Atoms objects, one for each requested structure.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> population = generate_initial_population(
        ...     composition=["Pt"] * 55,
        ...     n_structures=100,
        ...     rng=rng,
        ...     mode="smart",
        ...     n_jobs=4,  # Use 4 parallel workers
        ... )
        >>> len(population)
        100
    """
    return create_initial_cluster_batch(
        composition=composition,
        n_structures=n_structures,
        rng=rng,
        mode=mode,
        vacuum=vacuum,
        previous_search_glob=previous_search_glob,
        n_jobs=n_jobs,
    )
