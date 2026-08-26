"""Seed combination and growth strategies for cluster generation."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from scipy.spatial.distance import cdist

from scgo.system_types.connectivity_factor import (
    ConnectivityFactorInput,
    NormalizedConnectivityFactor,
    format_connectivity_factor,
    normalize_connectivity_factor,
)
from scgo.utils.logging import get_logger

from .geometry_helpers import (
    _check_composition_feasibility,
    _generate_rotation_matrix,
    _set_cubic_cell_and_center,
    analyze_disconnection,
    compute_bond_distance_params,
    get_covalent_radius,
    get_largest_facets,
    is_cluster_connected,
    place_multi_atom_seed_on_facet,
    validate_cluster,
)
from .initialization_config import (
    CLASH_TOLERANCE,
    CONNECTIVITY_FACTOR,
    SEED_CLASH_FACTOR,
)
from .random_spherical import grow_from_seed

logger = get_logger(__name__)


def _apply_random_rotation(atoms: Atoms, rng: np.random.Generator) -> Atoms:
    """Apply a random rotation to atoms around their center of mass.

    Args:
        atoms: The Atoms object to rotate
        rng: Random number generator for reproducible randomness

    Returns:
        New Atoms object with rotated positions
    """
    rotated_atoms = atoms.copy()
    rotated_atoms.center()
    center = rotated_atoms.get_center_of_mass()
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(0, 2 * np.pi)
    R = _generate_rotation_matrix(axis, angle)
    positions = rotated_atoms.get_positions()
    rotated_positions = center + (positions - center) @ R.T
    rotated_atoms.set_positions(rotated_positions)
    return rotated_atoms


def _is_valid_placement(
    seed_to_add: Atoms,
    combined_atoms: Atoms,
    connectivity_factor: ConnectivityFactorInput | NormalizedConnectivityFactor,
    min_distance_factor: float = SEED_CLASH_FACTOR,
) -> bool:
    """Check if a seed placement is valid (no clashes and maintains connectivity)."""
    seed_positions = seed_to_add.get_positions()
    existing_positions = combined_atoms.get_positions()
    existing_symbols = combined_atoms.get_chemical_symbols()
    seed_symbols = seed_to_add.get_chemical_symbols()

    if len(seed_positions) and len(existing_positions):
        distances = cdist(seed_positions, existing_positions)
        r_seed = np.array([get_covalent_radius(s) for s in seed_symbols], dtype=float)
        r_exist = np.array(
            [get_covalent_radius(s) for s in existing_symbols], dtype=float
        )
        min_allowed = (r_seed[:, None] + r_exist[None, :]) * min_distance_factor
        if np.any(distances < min_allowed - CLASH_TOLERANCE):
            return False

    temp_combined = combined_atoms.copy()
    temp_combined.extend(seed_to_add)
    return is_cluster_connected(temp_combined, connectivity_factor, use_mic=False)


def combine_seeds(
    seeds: list[Atoms],
    cell_side: float,
    rng: np.random.Generator,
    separation_scaling: float = 1.0,
    connectivity_factor: ConnectivityFactorInput
    | NormalizedConnectivityFactor = CONNECTIVITY_FACTOR,
    min_distance_factor: float = SEED_CLASH_FACTOR,
) -> Atoms | None:
    """Combine seed clusters into one structure using facet-to-facet placement.

    The first seed is used as the base; each further seed is randomly rotated
    and placed against one of the three largest facets of the growing cluster,
    with a random-direction offset as a last resort.

    Args:
        seeds: Seed clusters to combine.
        cell_side: Cubic cell side length applied to the combined structure.
        rng: Random number generator for rotations and fallback directions.
        separation_scaling: Placement radius scaling used for seed separation.
        connectivity_factor: Factor for the connectivity threshold.
        min_distance_factor: Factor for minimum-distance (clash) checks.

    Returns:
        The combined :class:`ase.Atoms` object, or ``None`` if a seed could
        not be placed without clashes or if a placement left the cluster
        disconnected. For an empty ``seeds`` list, returns an empty ``Atoms``
        object with the requested cubic cell.

    Raises:
        SCGOValidationError: If the combined cluster fails final validation.
    """
    if not seeds:
        return Atoms(cell=[cell_side, cell_side, cell_side])

    # Apply random rotation to first seed for diversity (reproducible with same seed)
    combined_atoms = _apply_random_rotation(seeds[0], rng)

    for i in range(1, len(seeds)):
        # Apply random rotation to seed before placement for diversity
        base_seed_to_add = _apply_random_rotation(seeds[i], rng)

        seed_symbols = base_seed_to_add.get_chemical_symbols()
        existing_symbols = combined_atoms.get_chemical_symbols()

        seed_avg_radius = np.mean([get_covalent_radius(sym) for sym in seed_symbols])
        existing_avg_radius = np.mean(
            [get_covalent_radius(sym) for sym in existing_symbols]
        )
        # Use shared bond distance calculation for consistency
        bond_distance, _, _ = compute_bond_distance_params(
            max_existing_radius=existing_avg_radius,
            avg_new_radius=seed_avg_radius,
            connectivity_factor=connectivity_factor,
            min_distance_factor=min_distance_factor,
            placement_radius_scaling=separation_scaling,
        )

        placement_success = False
        seed_to_add = None
        existing_facets = get_largest_facets(combined_atoms, n_facets=3)

        for facet_centroid, facet_normal, _ in existing_facets:
            candidate = place_multi_atom_seed_on_facet(
                base_seed_to_add,
                facet_centroid,
                facet_normal,
                bond_distance,
            )

            if _is_valid_placement(
                candidate, combined_atoms, connectivity_factor, min_distance_factor
            ):
                seed_to_add = candidate
                placement_success = True
                break

        if not placement_success:
            candidate = base_seed_to_add.copy()
            center = combined_atoms.get_center_of_mass()
            direction = rng.standard_normal(3)
            direction /= np.linalg.norm(direction)
            candidate.translate(
                center + direction * bond_distance - candidate.get_center_of_mass()
            )
            if _is_valid_placement(
                candidate, combined_atoms, connectivity_factor, min_distance_factor
            ):
                seed_to_add = candidate
                placement_success = True

        if not placement_success:
            logger.warning(
                "Failed to place seed %d on the largest facets or at a random "
                "offset; discarding this seed combination",
                i + 1,
            )
            return None

        combined_atoms.extend(seed_to_add)
        combined_atoms.center()

        if not is_cluster_connected(combined_atoms, connectivity_factor, use_mic=False):
            suggested_factor, analysis_msg = analyze_disconnection(
                combined_atoms, connectivity_factor, use_mic=False
            )
            logger.warning(
                "Seed %d placement created disconnected cluster. Current "
                "connectivity_factor=%s. Analysis: %s. Suggested connectivity_factor: %.2f",
                i + 1,
                format_connectivity_factor(
                    normalize_connectivity_factor(connectivity_factor)
                ),
                analysis_msg,
                suggested_factor,
            )
            return None

    _set_cubic_cell_and_center(combined_atoms, cell_side)

    validated_atoms, _, _ = validate_cluster(
        combined_atoms,
        composition=None,
        min_distance_factor=min_distance_factor,
        connectivity_factor=connectivity_factor,
        sort_atoms=False,
        raise_on_failure=True,
        source="combine_seeds",
    )

    return validated_atoms


def combine_and_grow(
    seeds: list[Atoms],
    target_composition: list[str],
    cell_side: float,
    rng: np.random.Generator,
    vdw_scaling: float = 1.0,
    min_distance_factor: float = SEED_CLASH_FACTOR,
    connectivity_factor: ConnectivityFactorInput
    | NormalizedConnectivityFactor = CONNECTIVITY_FACTOR,
) -> Atoms | None:
    """Combine seeds and grow the result to the target composition.

    Args:
        seeds: Seed clusters to combine.
        target_composition: Target composition as a list of element symbols.
        cell_side: Cubic cell side length.
        rng: Random number generator.
        vdw_scaling: Placement radius scaling for combination and growth.
        min_distance_factor: Factor for minimum-distance (clash) checks.
        connectivity_factor: Factor for the connectivity threshold.

    Returns:
        Grown :class:`ase.Atoms` with the target composition, or ``None`` if
        the seed combination, the feasibility check, or the growth step failed.
    """
    combined_seed = combine_seeds(
        seeds=seeds,
        cell_side=cell_side,
        rng=rng,
        separation_scaling=vdw_scaling,
        connectivity_factor=connectivity_factor,
        min_distance_factor=min_distance_factor,
    )

    if combined_seed is None:
        logger.warning("Initial seed combination failed")
        return None

    # Check composition feasibility before attempting growth
    base_composition = combined_seed.get_chemical_symbols()
    is_feasible, error_message = _check_composition_feasibility(
        base_composition, target_composition, operation="grow"
    )
    if not is_feasible:
        logger.warning(
            "Composition growth not feasible after seed combination: %s. "
            "Combined seed: %s, Target: %s",
            error_message,
            base_composition,
            target_composition,
        )
        return None

    return grow_from_seed(
        seed_atoms=combined_seed,
        target_composition=target_composition,
        placement_radius_scaling=vdw_scaling,
        cell_side=cell_side,
        rng=rng,
        min_distance_factor=min_distance_factor,
        connectivity_factor=connectivity_factor,
    )
