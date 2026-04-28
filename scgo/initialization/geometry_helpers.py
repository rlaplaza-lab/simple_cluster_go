"""Geometric utility functions for cluster structure generation.

This module provides helper functions for placing atoms based on cluster geometry,
particularly using convex hull analysis to guide growth strategies.
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from scipy.spatial import (  # Changed from ase.geometry.analysis
    ConvexHull,
    KDTree,
    QhullError,
)

from scgo.database.cache import get_global_cache
from scgo.utils.helpers import get_composition_counts

from .initialization_config import (
    CLASH_TOLERANCE,
    CONNECTIVITY_FACTOR,
    CONNECTIVITY_SUGGESTION_BUFFER,
    CONVEX_HULL_PERTURBATION_SCALE,
    CONVEX_HULL_VOLUME_TOLERANCE,
    LINEAR_GEOMETRY_TOLERANCE,
    ROTATION_AXIS_TOLERANCE,
    SMART_FILTERING_PERTURBATION_SCALE,
)

# =============================================================================
# CORE UTILITIES
# =============================================================================

def get_covalent_radius(symbol: str) -> float:
    """Get the covalent radius for an element symbol.

    Args:
        symbol: Element symbol (e.g., "Pt", "Au")

    Returns:
        Covalent radius in Angstroms

    Raises:
        KeyError: If element symbol is not found in atomic_numbers
    """
    return covalent_radii[atomic_numbers[symbol]]


def format_placement_error_message(
    context: str,
    composition: list[str] | None,
    n_atoms: int | None,
    placement_radius_scaling: float,
    min_distance_factor: float,
    connectivity_factor: float,
    cell_side: float | None = None,
    diagnostics: StructureDiagnostics | None = None,
    additional_info: str = "",
) -> str:
    """Format a consistent error message for placement failures.

    This helper creates standardized error messages with parameter values,
    diagnostics, and suggestions for common placement failures.

    Args:
        context: Context description (e.g., "random_spherical", "seed+growth")
        composition: Composition list (for display)
        n_atoms: Number of atoms (for display)
        placement_radius_scaling: Current placement radius scaling
        min_distance_factor: Current minimum distance factor
        connectivity_factor: Current connectivity factor
        cell_side: Optional cell side length
        diagnostics: Optional structure diagnostics
        additional_info: Additional context information

    Returns:
        Formatted error message string
    """
    parts = [f"Could not {context}"]

    if composition:
        parts.append(f"for composition {composition}")
    if n_atoms:
        parts.append(f"({n_atoms} atoms)")

    parts.append(".\n")

    # Parameter values
    param_parts = [
        f"placement_radius_scaling={placement_radius_scaling:.2f}",
        f"min_distance_factor={min_distance_factor:.2f}",
        f"connectivity_factor={connectivity_factor:.2f}",
    ]
    if cell_side is not None:
        param_parts.append(f"cell_side={cell_side:.2f} Å")

    parts.append(f"Parameters: {', '.join(param_parts)}")

    # Diagnostics if provided
    if diagnostics:
        parts.append(f"\nDiagnostics: {diagnostics.summary}")

    # Additional info
    if additional_info:
        parts.append(f"\n{additional_info}")

    # Suggestions
    parts.append("\nSuggestions:")
    parts.append(
        f"  - Increase placement_radius_scaling to {placement_radius_scaling * 1.5:.2f}"
    )
    parts.append(
        f"  - Decrease min_distance_factor to {max(min_distance_factor * 0.8, 0.3):.2f}"
    )
    parts.append(f"  - Increase connectivity_factor to {connectivity_factor * 1.2:.2f}")
    if cell_side is not None:
        parts.append(f"  - Increase cell_side to {cell_side * 1.5:.2f} Å")

    return "\n".join(parts)


# =============================================================================
# CONVEX HULL & CACHING
# =============================================================================

# Cache namespace for convex hull computations
_CONVEX_HULL_CACHE_NS = "convex_hull"


def _get_positions_hash(positions: np.ndarray) -> str:
    """Generate a collision-resistant hash for positions array.

    Uses SHA256 to avoid hash collisions that could return wrong cached results.
    For small arrays (<100 points), also stores positions bytes for collision detection.

    Args:
        positions: Array of atomic positions

    Returns:
        SHA256 hash string
    """
    positions_bytes = positions.tobytes()
    return hashlib.sha256(positions_bytes).hexdigest()


def _get_cached_hull(positions: np.ndarray) -> ConvexHull:
    """Get convex hull from cache or compute and cache it.

    Uses LRU eviction policy to maintain cache size limit.
    Uses SHA256 hashing to avoid collisions and verifies cached results match positions.

    Args:
        positions: Array of atomic positions

    Returns:
        ConvexHull object for the given positions

    Raises:
        ValueError: If positions array has fewer than 4 points (insufficient for 3D hull)
    """
    if len(positions) < 4:
        raise ValueError(
            f"Convex hull requires at least 4 points in 3D, got {len(positions)}"
        )

    positions_hash = _get_positions_hash(positions)
    positions_bytes = positions.tobytes()

    def compute_hull() -> ConvexHull:
        try:
            return ConvexHull(positions)
        except (QhullError, ValueError) as e:
            # Handle degenerate cases (collinear/coplanar points)
            raise ValueError(
                f"Convex hull computation failed for {len(positions)} points: {e}"
            ) from e

    # Use cache key with validation bytes to detect collisions
    cache_key = (positions_hash, positions_bytes)
    return get_global_cache().get_or_compute(
        _CONVEX_HULL_CACHE_NS, cache_key, compute_hull
    )


def get_convex_hull_vertex_indices(atoms: Atoms) -> np.ndarray:
    """Return atom indices that are vertices of the cluster's convex hull.

    Uses scipy's ConvexHull; vertices are the extreme points on the hull.

    Args:
        atoms: The Atoms object representing the cluster.

    Returns:
        1D array of atom indices (vertex indices). Empty array if hull cannot
        be computed (e.g. fewer than 4 atoms, degenerate geometry).
    """
    if len(atoms) < 4:
        return np.array([], dtype=np.intp)
    try:
        positions = atoms.get_positions()
        hull = _get_cached_hull(positions)
        return np.asarray(hull.vertices, dtype=np.intp)
    except (ValueError, QhullError):
        return np.array([], dtype=np.intp)


def _adjust_bond_distance_for_facet_geometry(
    bond_distance: float,
    min_centroid_dist: float,
    max_centroid_dist: float,
    min_connectivity_dist: float | None,
    max_connectivity_dist: float | None,
) -> float:
    """Adjust bond distance based on facet geometry constraints.

    This function adjusts the bond distance to ensure connectivity constraints
    are satisfied when placing atoms on convex hull facets. For small facets,
    it uses strict constraints to ensure connectivity. For large facets, it
    gradually relaxes the constraint while still relying on per-candidate
    connectivity checks to maintain cluster connectivity.

    Args:
        bond_distance: Initial bond distance for placement
        min_centroid_dist: Minimum distance from centroid to facet vertices
        max_centroid_dist: Maximum distance from centroid to facet vertices
        min_connectivity_dist: Minimum connectivity distance threshold
        max_connectivity_dist: Maximum connectivity distance threshold

    Returns:
        Adjusted bond distance that satisfies geometry constraints
    """
    adjusted_bond_distance = bond_distance
    if (
        min_connectivity_dist is not None
        and max_connectivity_dist is not None
        and min_centroid_dist < max_connectivity_dist
    ):
        # Calculate maximum bond_distance that ensures connectivity
        # Use strict constraint: no relaxation of connectivity threshold
        max_bond_from_geometry = np.sqrt(
            max_connectivity_dist**2 - min_centroid_dist**2
        )
        if max_bond_from_geometry > 0:
            adjusted_bond_distance = min(bond_distance, max_bond_from_geometry)
            # Also ensure it's at least the minimum if possible
            if min_centroid_dist < min_connectivity_dist:
                diff_sq = min_connectivity_dist**2 - max_centroid_dist**2
                if diff_sq > 0:
                    min_bond_from_geometry = np.sqrt(diff_sq)
                    adjusted_bond_distance = max(
                        adjusted_bond_distance, min_bond_from_geometry
                    )
    return adjusted_bond_distance


def compute_bond_distance_params(
    max_existing_radius: float,
    avg_new_radius: float,
    connectivity_factor: float,
    min_distance_factor: float,
    placement_radius_scaling: float,
    *,
    effective_min_distance: float | None = None,
    effective_scaling: float | None = None,
) -> tuple[float, float, float]:
    """Compute bond distance and connectivity bounds for facet-based placement.

    Shared logic used by both template growth (grow_template_via_facets) and
    batch placement (_add_atoms_batch_mode).

    Args:
        max_existing_radius: Max covalent radius among existing atoms.
        avg_new_radius: Mean covalent radius of atoms to add.
        connectivity_factor: Connectivity threshold factor.
        min_distance_factor: Factor for minimum separation.
        placement_radius_scaling: Scaling for placement distance.
        effective_min_distance: If set, use instead of min_distance_factor for min_dist.
        effective_scaling: If set, use instead of placement_radius_scaling for target_dist.

    Returns:
        (bond_distance, min_connectivity_dist, max_connectivity_dist)
    """
    base = max_existing_radius + avg_new_radius
    max_conn = float(base * connectivity_factor)
    min_dist = base * (
        effective_min_distance
        if effective_min_distance is not None
        else min_distance_factor
    )
    scale = (
        effective_scaling if effective_scaling is not None else placement_radius_scaling
    )
    target_dist = base * min(scale, connectivity_factor)
    bond_distance = float(np.clip(target_dist, min_dist, max_conn))
    return (bond_distance, float(min_dist), max_conn)


def _compute_facet_properties(
    hull: ConvexHull, atoms: Atoms
) -> list[tuple[np.ndarray, np.ndarray, float, tuple[float, float]]]:
    """Compute properties for all facets of a convex hull.

    Args:
        hull: ConvexHull object from scipy
        atoms: Atoms object to compute facet properties for

    Returns:
        List of tuples (centroid, normal, area, (min_centroid_dist, max_centroid_dist))
        for each facet
    """
    positions = atoms.get_positions()
    center_of_mass = atoms.get_center_of_mass()
    facet_properties = []

    for simplex_indices in hull.simplices:
        facet_positions = positions[simplex_indices]

        # Calculate facet area (assuming triangular facets for 3D convex hull)
        v1 = facet_positions[1] - facet_positions[0]
        v2 = facet_positions[2] - facet_positions[0]
        area = 0.5 * np.linalg.norm(np.cross(v1, v2))

        # Calculate facet centroid
        centroid = np.mean(facet_positions, axis=0)

        # Calculate distances from centroid to each vertex
        centroid_to_vertices = [
            np.linalg.norm(vertex_pos - centroid) for vertex_pos in facet_positions
        ]
        min_centroid_dist = min(centroid_to_vertices)
        max_centroid_dist = max(centroid_to_vertices)

        # Calculate facet normal (outward pointing)
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)

        # Ensure normal points outwards from the center of mass
        if np.dot(normal, centroid - center_of_mass) < 0:
            normal *= -1

        facet_properties.append(
            (centroid, normal, area, (min_centroid_dist, max_centroid_dist))
        )

    return facet_properties


def _filter_safe_facets_for_placement(
    atoms: Atoms,
    facet_properties: list[tuple[np.ndarray, np.ndarray, float, tuple[float, float]]],
    bond_distance: float,
    min_connectivity_dist: float | None,
    max_connectivity_dist: float | None,
    min_distance_factor: float | None,
    new_atom_radius: float | None,
    positions: np.ndarray,
    symbols_list: list[str],
    connectivity_factor: float = CONNECTIVITY_FACTOR,
) -> list[int]:
    """Filter facets that can safely accommodate atom placement.

    Analyzes facet geometry to determine which facets can safely have an atom
    placed on them without clashes or connectivity issues. This avoids
    trial-and-error approaches.

    Args:
        atoms: The Atoms object representing the current cluster structure.
        facet_properties: List of (centroid, normal, area, (min_dist, max_dist))
            for each facet.
        bond_distance: Target bond distance for placement.
        min_connectivity_dist: Minimum connectivity distance threshold.
        max_connectivity_dist: Maximum connectivity distance threshold.
        min_distance_factor: Factor for minimum distance checks.
        new_atom_radius: Covalent radius of the atom to be placed.
        positions: Array of existing atom positions.
        symbols_list: List of element symbols for existing atoms.
        connectivity_factor: Factor to multiply sum of covalent radii for
            connectivity threshold. Defaults to CONNECTIVITY_FACTOR.

    Returns:
        List of facet indices that are safe for placement.
    """
    safe_facet_indices = []

    for idx, (
        centroid,
        normal,
        _area,
        (min_centroid_dist, max_centroid_dist),
    ) in enumerate(facet_properties):
        # Skip facets where centroid is significantly beyond connectivity threshold
        # Use scaled threshold to allow large facets with relaxed constraints
        # For very large facets (min_centroid_dist > 1.5 * max_connectivity_dist),
        # skip them as they're unlikely to yield valid placements even with relaxation
        if (
            max_connectivity_dist is not None
            and min_centroid_dist > max_connectivity_dist * 1.5
        ):
            continue

        # Adjust bond_distance based on facet geometry
        adjusted_bond_distance = _adjust_bond_distance_for_facet_geometry(
            bond_distance,
            min_centroid_dist,
            max_centroid_dist,
            min_connectivity_dist,
            max_connectivity_dist,
        )

        # Calculate expected placement position (without perturbation for analysis)
        expected_pos = centroid + normal * adjusted_bond_distance

        is_safe = True

        if min_distance_factor is not None and new_atom_radius is not None:
            for i, existing_pos in enumerate(positions):
                dist = np.linalg.norm(expected_pos - existing_pos)
                r_existing = get_covalent_radius(symbols_list[i])
                min_allowed = (new_atom_radius + r_existing) * min_distance_factor
                if dist < min_allowed:
                    is_safe = False
                    break

            if (
                is_safe
                and max_connectivity_dist is not None
                and new_atom_radius is not None
            ):
                is_connected = False
                for i, existing_pos in enumerate(positions):
                    dist = np.linalg.norm(expected_pos - existing_pos)
                    r_existing = get_covalent_radius(symbols_list[i])
                    # Connectivity threshold for this pair
                    connectivity_threshold = (
                        new_atom_radius + r_existing
                    ) * connectivity_factor
                    if dist <= connectivity_threshold:
                        is_connected = True
                        break
                if not is_connected:
                    is_safe = False

        if is_safe:
            safe_facet_indices.append(idx)

    return safe_facet_indices


def _generate_batch_positions_on_convex_hull(
    atoms: Atoms,
    n_candidates: int,
    bond_distance: float,
    rng: np.random.Generator,
    min_connectivity_dist: float | None = None,
    max_connectivity_dist: float | None = None,
    use_all_facets: bool = False,
    min_distance_factor: float | None = None,
    new_atom_symbol: str | None = None,
    smart_facet_filtering: bool = False,
    connectivity_factor: float = CONNECTIVITY_FACTOR,
) -> list[np.ndarray]:
    """Generate multiple candidate atom positions on convex hull facets.

    This function computes the convex hull once and generates candidate positions
    for multiple atoms (one per facet, up to n_candidates). This is more efficient
    than computing the hull multiple times.

    Args:
        atoms: The Atoms object representing the current cluster structure.
        n_candidates: Maximum number of candidate positions to generate. Ignored if
            use_all_facets is True.
        bond_distance: Distance (in Angstroms) from the surface at which to place
                      the new atoms. This represents the bond separation distance.
        rng: Numpy random number generator for reproducible randomness.
        min_connectivity_dist: Optional minimum distance constraint. If provided along
                              with max_connectivity_dist, bond_distance will be adjusted
                              to ensure connectivity with at least one facet vertex.
        max_connectivity_dist: Optional maximum distance constraint for connectivity.
                               If provided along with min_connectivity_dist, bond_distance
                               will be adjusted based on facet geometry.
        use_all_facets: If True, generate one position per facet (all facets) in
            deterministic order (area descending); n_candidates is ignored.
        min_distance_factor: Optional factor for minimum distance checks. If provided
                             along with new_atom_symbol, candidates will be validated
                             for clashes before being returned.
        new_atom_symbol: Optional symbol of the atom to be placed. Used with
                        min_distance_factor for clash validation.
        smart_facet_filtering: If True, pre-filter facets based on geometry analysis
                               to avoid trial-and-error. Only generates candidates
                               for facets that are known to be safe.
        connectivity_factor: Factor to multiply sum of covalent radii for
                             connectivity threshold. Defaults to CONNECTIVITY_FACTOR.

    Returns:
        List of 3D numpy arrays representing candidate positions for new atoms.
        The list may have fewer than n_candidates if there are fewer facets available
        or if some candidates fail validation. When use_all_facets is True, returns
        one position per facet (after validation).

    Note:
        For clusters with <4 atoms, returns empty list (caller should handle fallback).
    """
    if len(atoms) < 4:
        # Cannot compute convex hull for <4 atoms
        return []

    # Get convex hull from cache or compute it
    positions = atoms.get_positions()
    try:
        hull = _get_cached_hull(positions)
    except ValueError:
        # Convex hull computation failed (degenerate geometry)
        return []

    # Compute facet properties using shared helper
    facet_properties = _compute_facet_properties(hull, atoms)
    facet_areas = [prop[2] for prop in facet_properties]
    facet_centroids = [prop[0] for prop in facet_properties]
    facet_normals = [prop[1] for prop in facet_properties]
    facet_vertex_distances = [prop[3] for prop in facet_properties]

    total_area = sum(facet_areas)
    if total_area == 0:  # Handle degenerate cases
        return []

    # Sort facets by area (descending), then centroid coordinates for reproducibility
    facet_indices = list(range(len(facet_areas)))
    facet_indices.sort(
        key=lambda i: (
            -facet_areas[i],  # Negative for descending order
            round(facet_centroids[i][0], 8),
            round(facet_centroids[i][1], 8),
            round(facet_centroids[i][2], 8),
        )
    )

    # Reorder facets according to deterministic sort
    sorted_facet_centroids = [facet_centroids[i] for i in facet_indices]
    sorted_facet_normals = [facet_normals[i] for i in facet_indices]
    sorted_facet_vertex_distances = [facet_vertex_distances[i] for i in facet_indices]

    # Get symbols for validation if enabled
    symbols_list = atoms.get_chemical_symbols()
    new_atom_radius = None
    if new_atom_symbol is not None:
        new_atom_radius = get_covalent_radius(new_atom_symbol)

    n_facets = len(sorted_facet_centroids)

    # Pre-filter facets if smart filtering is enabled
    if (
        smart_facet_filtering
        and min_distance_factor is not None
        and new_atom_radius is not None
    ):
        # Create sorted facet properties list (using sorted indices)
        sorted_facet_properties = [
            (
                sorted_facet_centroids[i],
                sorted_facet_normals[i],
                facet_areas[facet_indices[i]],
                sorted_facet_vertex_distances[i],
            )
            for i in range(n_facets)
        ]
        # Filter to safe facets (returns indices in sorted_facet_properties order)
        safe_sorted_indices = _filter_safe_facets_for_placement(
            atoms,
            sorted_facet_properties,
            bond_distance,
            min_connectivity_dist,
            max_connectivity_dist,
            min_distance_factor,
            new_atom_radius,
            positions,
            symbols_list,
            connectivity_factor=connectivity_factor,
        )
        n_safe_facets = len(safe_sorted_indices)
    else:
        safe_sorted_indices = list(range(n_facets))
        n_safe_facets = n_facets

    if use_all_facets:
        # Use all safe facets (or all facets if not filtering)
        selected_indices = safe_sorted_indices
        n_to_select = len(selected_indices)
    else:
        n_to_select = min(n_candidates, n_safe_facets)
        if n_to_select == 0:
            return []
        if smart_facet_filtering and len(safe_sorted_indices) > 0:
            # Select from safe facets based on area
            sorted_safe_areas = [
                facet_areas[facet_indices[i]] for i in safe_sorted_indices
            ]
            safe_total_area = sum(sorted_safe_areas)
            if safe_total_area > 0:
                probabilities = np.array(sorted_safe_areas) / safe_total_area
                selected_safe = rng.choice(
                    len(safe_sorted_indices),
                    size=n_to_select,
                    replace=False,
                    p=probabilities,
                )
                selected_indices = [safe_sorted_indices[i] for i in selected_safe]
            else:
                selected_indices = safe_sorted_indices[:n_to_select]
        else:
            sorted_facet_areas = [
                facet_areas[facet_indices[i]] for i in range(n_facets)
            ]
            probabilities = np.array(sorted_facet_areas) / total_area
            selected_indices = rng.choice(
                n_facets, size=n_to_select, replace=False, p=probabilities
            )

    # Generate candidate positions for selected facets
    candidates = []
    perturbation_scale = CONVEX_HULL_PERTURBATION_SCALE

    for idx in selected_indices:
        chosen_centroid = sorted_facet_centroids[idx]
        chosen_normal = sorted_facet_normals[idx]
        min_centroid_dist, max_centroid_dist = sorted_facet_vertex_distances[idx]

        # Skip facets where centroid is significantly beyond connectivity threshold
        # For very large facets (min_centroid_dist > 1.5 * max_connectivity_dist),
        # skip them as they're unlikely to yield valid placements
        if (
            max_connectivity_dist is not None
            and min_centroid_dist > max_connectivity_dist * 1.5
        ):
            continue

        # For large facets, place closer to vertices instead of centroid
        # This ensures atoms are within connectivity distance of existing atoms
        facet_size_ratio = (
            min_centroid_dist / max_connectivity_dist
            if max_connectivity_dist is not None and max_connectivity_dist > 0
            else 0.0
        )
        if facet_size_ratio > 0.6:
            # Large facet: interpolate between centroid and nearest vertex
            original_facet_idx = facet_indices[idx]
            if original_facet_idx < len(hull.simplices):
                facet_vertex_indices = hull.simplices[original_facet_idx]
                facet_vertex_positions = positions[facet_vertex_indices]

                # Find nearest vertex to centroid
                vertex_dists = [
                    np.linalg.norm(vpos - chosen_centroid)
                    for vpos in facet_vertex_positions
                ]
                nearest_vertex_idx = np.argmin(vertex_dists)
                nearest_vertex = facet_vertex_positions[nearest_vertex_idx]

                # Interpolate: for ratio > 0.6, move from centroid toward nearest vertex
                # At ratio=0.6: use centroid (0% interpolation)
                # At ratio=1.0: use 50% toward vertex
                # At ratio=1.5: use 100% at vertex
                if facet_size_ratio <= 1.0:
                    # 0 to 0.5 interpolation
                    interpolation_factor = (facet_size_ratio - 0.6) / 0.4 * 0.5
                else:
                    # 0.5 to 1.0 interpolation
                    interpolation_factor = 0.5 + (facet_size_ratio - 1.0) / 0.5 * 0.5

                placement_base = (
                    chosen_centroid * (1 - interpolation_factor)
                    + nearest_vertex * interpolation_factor
                )
            else:
                placement_base = chosen_centroid
        else:
            # Small/medium facets: use centroid
            placement_base = chosen_centroid

        # Adjust bond_distance based on facet geometry if constraints are provided
        adjusted_bond_distance = _adjust_bond_distance_for_facet_geometry(
            bond_distance,
            min_centroid_dist,
            max_centroid_dist,
            min_connectivity_dist,
            max_connectivity_dist,
        )

        # Use more conservative perturbation when close to connectivity limit
        effective_perturbation_scale = perturbation_scale
        if (
            max_connectivity_dist is not None
            and min_centroid_dist is not None
            and min_centroid_dist > max_connectivity_dist * 0.8
        ):
            # Reduce perturbation when close to connectivity limit
            effective_perturbation_scale = perturbation_scale * 0.5

        if smart_facet_filtering:
            perturbation = rng.standard_normal(3) * (
                effective_perturbation_scale * SMART_FILTERING_PERTURBATION_SCALE
            )
        else:
            perturbation = rng.standard_normal(3) * effective_perturbation_scale

        candidate_pos = (
            placement_base + chosen_normal * adjusted_bond_distance + perturbation
        )

        # Always validate if constraints are provided, even with smart filtering,
        # because perturbation or interpolation might shift the point into a clash.
        is_valid = True
        if min_distance_factor is not None and new_atom_radius is not None:
            for i, existing_pos in enumerate(positions):
                dist = np.linalg.norm(candidate_pos - existing_pos)
                r_existing = get_covalent_radius(symbols_list[i])
                min_allowed = (new_atom_radius + r_existing) * min_distance_factor
                if dist < min_allowed:
                    is_valid = False
                    break

            if (
                is_valid
                and max_connectivity_dist is not None
                and new_atom_radius is not None
            ):
                is_connected = False
                for i, existing_pos in enumerate(positions):
                    dist = np.linalg.norm(candidate_pos - existing_pos)
                    r_existing = get_covalent_radius(symbols_list[i])
                    # Connectivity threshold for this pair
                    connectivity_threshold = (
                        new_atom_radius + r_existing
                    ) * connectivity_factor
                    if dist <= connectivity_threshold:
                        is_connected = True
                        break
                if not is_connected:
                    is_valid = False

        if not is_valid:
            # Skip this candidate if validation failed
            # Debug logging for large facets
            if (
                max_connectivity_dist is not None
                and min_centroid_dist > max_connectivity_dist * 0.7
            ):
                debug_logger = logging.getLogger("scgo.initialization.templates")
                # Check if it was a clash or connectivity issue
                if min_distance_factor is not None and new_atom_radius is not None:
                    min_dist_to_existing = min(
                        np.linalg.norm(candidate_pos - existing_pos)
                        for existing_pos in positions
                    )
                    debug_logger.debug(
                        f"candidate rejected: min_dist={min_dist_to_existing:.3f}, "
                        f"min_centroid_dist={min_centroid_dist:.3f}, "
                        f"max_conn={max_connectivity_dist:.3f}"
                    )
            continue

        # Add candidate (either pre-validated or passed validation)
        candidates.append(candidate_pos)

    return candidates


def get_largest_facets(
    atoms: Atoms, n_facets: int = 3
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Get the largest facets of a cluster's convex hull.

    Args:
        atoms: The Atoms object representing the cluster
        n_facets: Number of largest facets to return

    Returns:
        List of tuples (centroid, normal, area) for the largest facets

    """
    if len(atoms) < 4:
        center = atoms.get_center_of_mass()
        return [(center, np.array([1.0, 0.0, 0.0]), 1.0)]

    try:
        hull = _get_cached_hull(atoms.get_positions())
    except (ValueError, RuntimeError, QhullError):
        # Convex hull computation failed (degenerate geometry, collinear points, etc.)
        center = atoms.get_center_of_mass()
        geometry = _classify_seed_geometry(atoms)

        if geometry in ["linear", "planar"]:
            positions = atoms.get_positions()
            centered_positions = positions - center
            if len(positions) > 1:
                cov_matrix = np.cov(centered_positions.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                normal = eigenvectors[:, -1] / np.linalg.norm(eigenvectors[:, -1])
            else:
                normal = np.array([1.0, 0.0, 0.0])
        else:
            normal = np.array([1.0, 0.0, 0.0])
        return [(center, normal, 1.0)]

    # Compute facet properties using shared helper
    facet_properties = _compute_facet_properties(hull, atoms)
    facets = [(prop[0], prop[1], prop[2]) for prop in facet_properties]

    # Sort by area and return the largest ones
    facets.sort(key=lambda x: x[2], reverse=True)
    return facets[:n_facets]


def _classify_seed_geometry(atoms: Atoms) -> str:
    """Classify the geometric structure of a seed cluster.

    Args:
        atoms: The Atoms object to classify

    Returns:
        Geometry classification: "single", "linear", "planar", or "3d"
    """
    n_atoms = len(atoms)

    if n_atoms == 1:
        return "single"

    if n_atoms == 2:
        return "linear"

    if n_atoms >= 3:
        positions = atoms.get_positions()
        center = np.mean(positions, axis=0)
        centered_positions = positions - center

        cov_matrix = np.cov(centered_positions.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Check if structure is linear (1D) vs planar (2D)
        # Linear: λ1 ≈ 0 AND λ2 ≈ 0 (only one dimension has variation)
        # Planar: λ1 ≈ 0 BUT λ2 > 0 (two dimensions have variation)
        # Use tolerance to allow for structures that are almost linear but not perfectly linear
        linear_tolerance = LINEAR_GEOMETRY_TOLERANCE
        if (
            eigenvalues[0] < linear_tolerance * eigenvalues[2]
            and eigenvalues[1] < linear_tolerance * eigenvalues[2]
        ):
            return "linear"  # Both λ1 ≈ 0 and λ2 ≈ 0 → 1D linear (or almost linear)
        # λ1 ≈ 0 but λ2 > tolerance → 2D planar (fall through to convex hull check)

    try:
        if len(positions) >= 4:
            hull = _get_cached_hull(positions)
            if hull.volume < CONVEX_HULL_VOLUME_TOLERANCE:
                return "planar"
            return "3d"
        else:
            return "planar"

    except (QhullError, ValueError, RuntimeError):
        # Convex hull computation failed (degenerate geometry, collinear points, etc.)
        return "planar"


def _generate_rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Generate a 3D rotation matrix using Rodrigues' rotation formula.

    Args:
        axis: Rotation axis vector (will be normalized)
        angle: Rotation angle in radians

    Returns:
        3x3 rotation matrix
    """
    axis = np.asarray(axis)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < ROTATION_AXIS_TOLERANCE:
        # Degenerate case: return identity matrix
        return np.eye(3)
    axis = axis / axis_norm

    # Skew-symmetric matrix for cross product
    K = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    # Rodrigues' rotation formula
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R


def place_multi_atom_seed_on_facet(
    seed_atoms: Atoms,
    target_facet_centroid: np.ndarray,
    target_facet_normal: np.ndarray,
    bond_distance: float,
    rng: np.random.Generator,
) -> Atoms:
    """Place a multi-atom seed so that its largest facet contacts the target facet.

    Args:
        seed_atoms: The seed to place
        target_facet_centroid: Centroid of the target facet
        target_facet_normal: Normal vector of the target facet
        bond_distance: Desired bond distance between facets
        rng: Random number generator for rotation

    Returns:
        The seed atoms with new positions

    """
    # Get the largest facet of the seed
    seed_facets = get_largest_facets(seed_atoms, n_facets=1)
    if not seed_facets:
        # Fallback: use center of mass
        seed_normal = np.array([1.0, 0.0, 0.0])
    else:
        seed_facet_centroid, seed_facet_normal, _ = seed_facets[0]
        seed_normal = seed_facet_normal

    # Create a copy to work with
    placed_seed = seed_atoms.copy()

    # Step 1: Rotate the seed so its facet normal aligns with the target normal
    # We want the seed normal to point towards the target (opposite direction)
    target_direction = -target_facet_normal

    # Calculate rotation axis and angle
    rotation_axis = np.cross(seed_normal, target_direction)
    rotation_axis_norm = np.linalg.norm(rotation_axis)

    if rotation_axis_norm > 1e-6:  # Not parallel
        rotation_axis /= rotation_axis_norm
        cos_angle = np.dot(seed_normal, target_direction)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
        rotation_angle = np.arccos(cos_angle)

        # Apply rotation using Rodrigues' formula
        R = _generate_rotation_matrix(rotation_axis, rotation_angle)

        # Apply rotation around the center of mass
        center = placed_seed.get_center_of_mass()
        positions = placed_seed.get_positions()
        rotated_positions = center + (positions - center) @ R.T
        placed_seed.set_positions(rotated_positions)

    # Step 2: Translate the seed so its facet contacts the target facet
    # Get the new facet position after rotation
    new_facets = get_largest_facets(placed_seed, n_facets=1)
    if new_facets:
        new_facet_centroid, _, _ = new_facets[0]
    else:
        new_facet_centroid = placed_seed.get_center_of_mass()

    # Calculate translation vector
    # The facet centroids may be inside the clusters, so we need separation
    target_position = target_facet_centroid + target_facet_normal * bond_distance
    translation = target_position - new_facet_centroid

    # Apply translation
    placed_seed.translate(translation)

    return placed_seed


def _find_connected_components(
    atoms: Atoms, connectivity_factor: float
) -> tuple[dict[int, list[int]], list[int]]:
    """Find connected components using Union-Find algorithm.

    Args:
        atoms: The Atoms object to check
        connectivity_factor: Factor to multiply sum of covalent radii for connectivity threshold

    Returns:
        Tuple of (components dict mapping root to atom indices, parent array for Union-Find)
    """
    if len(atoms) <= 1:
        return {0: [0] if len(atoms) == 1 else []}, list(range(len(atoms)))

    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    n_atoms = len(atoms)

    parent = list(range(n_atoms))

    def find(x: int) -> int:
        """Find root of x with path compression."""
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> bool:
        """Union two components. Returns True if union was performed."""
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False

    if n_atoms < 50:
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                distance = np.linalg.norm(positions[i] - positions[j])
                r_i = get_covalent_radius(symbols[i])
                r_j = get_covalent_radius(symbols[j])
                threshold = (r_i + r_j) * connectivity_factor
                if distance <= threshold:
                    union(i, j)
    else:
        tree = KDTree(positions)
        unique_radii = {get_covalent_radius(s) for s in symbols}
        max_radius = max(unique_radii)
        query_radius = 2 * max_radius * connectivity_factor

        for i in range(n_atoms):
            neighbor_indices = tree.query_ball_point(positions[i], query_radius)
            r_i = get_covalent_radius(symbols[i])

            for j in neighbor_indices:
                if j <= i:
                    continue

                distance = np.linalg.norm(positions[i] - positions[j])
                r_j = get_covalent_radius(symbols[j])
                threshold = (r_i + r_j) * connectivity_factor

                if distance <= threshold:
                    union(i, j)

    components: dict[int, list[int]] = {}
    for i in range(n_atoms):
        root = find(i)
        if root not in components:
            components[root] = []
        components[root].append(i)

    return components, parent


def is_cluster_connected(
    atoms: Atoms, connectivity_factor: float = CONNECTIVITY_FACTOR
) -> bool:
    """Check if all atoms in a cluster are connected within the specified distance threshold.

    Uses a Union-Find algorithm with KDTree spatial indexing to efficiently determine if all
    atoms form a single connected component where edges exist between atoms within
    (r_i + r_j) * connectivity_factor.

    This optimized version uses scipy.spatial.KDTree for efficient neighbor queries,
    providing O(n log n) performance instead of O(n²) for large clusters.

    Args:
        atoms: The Atoms object to check
        connectivity_factor: Factor to multiply sum of covalent radii for connectivity threshold.
                           Defaults to CONNECTIVITY_FACTOR (1.4).

    Returns:
        True if all atoms are in one connected component, False otherwise.

    """
    components, _ = _find_connected_components(atoms, connectivity_factor)
    return len(components) <= 1


def analyze_disconnection(
    atoms: Atoms, connectivity_factor: float = CONNECTIVITY_FACTOR
) -> tuple[float, float, str]:
    """Analyze disconnection in a cluster and suggest appropriate connectivity factor.

    Args:
        atoms: The Atoms object to analyze
        connectivity_factor: Current connectivity factor used

    Returns:
        Tuple of (max_disconnection_distance, suggested_connectivity_factor, analysis_message)
    """
    if len(atoms) <= 1:
        return 0.0, connectivity_factor, "Single atom or empty cluster"

    components, _ = _find_connected_components(atoms, connectivity_factor)

    if len(components) <= 1:
        return 0.0, connectivity_factor, "Cluster is connected"

    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    min_inter_component_distance = float("inf")
    closest_atoms = None

    component_list = list(components.values())
    for i in range(len(component_list)):
        for j in range(i + 1, len(component_list)):
            comp1, comp2 = component_list[i], component_list[j]
            for atom1 in comp1:
                for atom2 in comp2:
                    distance = np.linalg.norm(positions[atom1] - positions[atom2])
                    if distance < min_inter_component_distance:
                        min_inter_component_distance = distance
                        closest_atoms = (atom1, atom2, symbols[atom1], symbols[atom2])

    if closest_atoms is None:
        return float("inf"), connectivity_factor, "Unable to analyze disconnection"

    atom1_idx, atom2_idx, sym1, sym2 = closest_atoms
    r1 = get_covalent_radius(sym1)
    r2 = get_covalent_radius(sym2)

    # Calculate what connectivity factor would be needed to connect these atoms
    suggested_factor = min_inter_component_distance / (r1 + r2)

    # Add a small buffer to ensure connectivity
    suggested_factor *= CONNECTIVITY_SUGGESTION_BUFFER

    analysis_msg = (
        f"Cluster has {len(components)} disconnected components. "
        f"Closest atoms are {sym1}({atom1_idx}) and {sym2}({atom2_idx}) "
        f"at distance {min_inter_component_distance:.3f}Å. "
        f"Suggested connectivity_factor: {suggested_factor:.2f}"
    )

    return min_inter_component_distance, suggested_factor, analysis_msg


def _identify_safe_removal_candidates(
    cluster: Atoms,
    candidate_indices: list[int],
    connectivity_factor: float,
    max_to_check: int = 10,
) -> list[int]:
    """Identify which candidates can be safely removed without disconnecting.

    Pre-checks connectivity impact before actual removal by testing each
    candidate atom's removal and verifying the cluster remains connected.

    Args:
        cluster: The cluster to analyze
        candidate_indices: List of atom indices to test for safe removal
        connectivity_factor: Factor for connectivity threshold
        max_to_check: Maximum number of candidates to check (for performance)

    Returns:
        List of atom indices that can be safely removed without disconnecting
        the cluster. Empty list if none are safe or if cluster is too small.
    """
    if len(cluster) <= 2:
        # Removing from 1-2 atom clusters would leave disconnected or empty
        return []

    if not candidate_indices:
        return []

    # Limit checks for performance
    candidates_to_check = candidate_indices[:max_to_check]
    safe_candidates = []

    for idx in candidates_to_check:
        if idx >= len(cluster) or idx < 0:
            continue

        # Create test cluster without this atom
        test_cluster = cluster.copy()
        del test_cluster[idx]

        # Check if cluster remains connected after removal
        if len(test_cluster) > 1:
            if is_cluster_connected(test_cluster, connectivity_factor):
                safe_candidates.append(idx)
        else:
            # Single atom left - always connected
            safe_candidates.append(idx)

    return safe_candidates


class StructureDiagnostics:
    """Container for comprehensive structure diagnostics.

    Attributes:
        is_valid: True if structure has no clashes and is connected
        has_clashes: True if atomic clashes were detected
        is_disconnected: True if cluster has multiple disconnected components
        clash_details: List of clash description strings
        n_components: Number of disconnected components (1 if connected)
        closest_inter_component_distance: Distance between closest atoms in different components
        suggested_connectivity_factor: Connectivity factor needed to connect all components
        summary: Human-readable summary of all issues
    """

    def __init__(
        self,
        is_valid: bool,
        has_clashes: bool,
        is_disconnected: bool,
        clash_details: list[str],
        n_components: int,
        closest_inter_component_distance: float,
        suggested_connectivity_factor: float,
        summary: str,
    ):
        self.is_valid = is_valid
        self.has_clashes = has_clashes
        self.is_disconnected = is_disconnected
        self.clash_details = clash_details
        self.n_components = n_components
        self.closest_inter_component_distance = closest_inter_component_distance
        self.suggested_connectivity_factor = suggested_connectivity_factor
        self.summary = summary


def get_structure_diagnostics(
    atoms: Atoms,
    min_distance_factor: float,
    connectivity_factor: float,
) -> StructureDiagnostics:
    """Get comprehensive diagnostics for a cluster structure.

    This function analyzes both clashes and connectivity issues and returns
    detailed diagnostic information useful for debugging initialization failures.

    Args:
        atoms: The Atoms object to analyze
        min_distance_factor: Factor to scale covalent radii for minimum distance checks
        connectivity_factor: Factor to multiply sum of covalent radii for connectivity threshold

    Returns:
        StructureDiagnostics object containing detailed analysis results
    """
    if len(atoms) == 0:
        return StructureDiagnostics(
            is_valid=True,
            has_clashes=False,
            is_disconnected=False,
            clash_details=[],
            n_components=0,
            closest_inter_component_distance=0.0,
            suggested_connectivity_factor=connectivity_factor,
            summary="Empty cluster",
        )

    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    n_atoms = len(atoms)

    # Analyze clashes
    # Pre-compute covalent radii to avoid repeated lookups
    radii = {symbol: get_covalent_radius(symbol) for symbol in set(symbols)}

    clash_details = []
    for i in range(n_atoms):
        if len(clash_details) >= 10:
            break
        for j in range(i + 1, n_atoms):
            distance = np.linalg.norm(positions[i] - positions[j])
            r_i = radii[symbols[i]]
            r_j = radii[symbols[j]]
            min_allowed = (r_i + r_j) * min_distance_factor

            if distance < min_allowed - CLASH_TOLERANCE:
                clash_details.append(
                    f"{symbols[i]}({i})-{symbols[j]}({j}): "
                    f"{distance:.3f}Å < {min_allowed:.3f}Å (gap: {min_allowed - distance:.3f}Å)"
                )
            if len(clash_details) >= 10:
                break

    has_clashes = bool(clash_details)

    # Analyze connectivity
    components, _ = _find_connected_components(atoms, connectivity_factor)
    n_components = len(components)
    is_disconnected = n_components > 1

    closest_inter_component_distance = 0.0
    suggested_connectivity_factor = connectivity_factor

    if is_disconnected:
        # Find closest inter-component distance
        min_dist = float("inf")
        closest_pair = None
        component_list = list(components.values())

        for ci in range(len(component_list)):
            for cj in range(ci + 1, len(component_list)):
                for atom1 in component_list[ci]:
                    for atom2 in component_list[cj]:
                        dist = np.linalg.norm(positions[atom1] - positions[atom2])
                        if dist < min_dist:
                            min_dist = dist
                            closest_pair = (atom1, atom2)

        closest_inter_component_distance = min_dist

        if closest_pair is not None:
            i1, i2 = closest_pair
            r1 = get_covalent_radius(symbols[i1])
            r2 = get_covalent_radius(symbols[i2])
            suggested_connectivity_factor = (
                min_dist / (r1 + r2)
            ) * CONNECTIVITY_SUGGESTION_BUFFER

    # Build summary
    summary_parts = []
    if has_clashes:
        summary_parts.append(
            f"Clashes ({len(clash_details)}): {'; '.join(clash_details[:3])}"
            + (f" (+{len(clash_details) - 3} more)" if len(clash_details) > 3 else "")
        )
    if is_disconnected:
        summary_parts.append(
            f"Disconnected: {n_components} components, "
            f"gap={closest_inter_component_distance:.3f}Å, "
            f"suggested factor={suggested_connectivity_factor:.2f}"
        )

    if summary_parts:
        summary = "; ".join(summary_parts)
    else:
        summary = "Structure is valid (no clashes, connected)"

    return StructureDiagnostics(
        is_valid=not has_clashes and not is_disconnected,
        has_clashes=has_clashes,
        is_disconnected=is_disconnected,
        clash_details=clash_details,
        n_components=n_components,
        closest_inter_component_distance=closest_inter_component_distance,
        suggested_connectivity_factor=suggested_connectivity_factor,
        summary=summary,
    )


def validate_cluster_structure(
    atoms: Atoms,
    min_distance_factor: float,
    connectivity_factor: float,
    check_clashes: bool = True,
    check_connectivity: bool = True,
) -> tuple[bool, str]:
    """Validate a cluster structure for clashes and connectivity.

    This function provides a centralized validation that ensures all returned
    cluster structures meet the specified constraints. It checks for atomic
    clashes and connectivity using the same logic as the placement algorithms.

    Args:
        atoms: The Atoms object to validate
        min_distance_factor: Factor to scale covalent radii for minimum distance checks
        connectivity_factor: Factor to multiply sum of covalent radii for connectivity threshold
        check_clashes: Whether to check for atomic clashes (default: True)
        check_connectivity: Whether to check connectivity (default: True)

    Returns:
        Tuple of (is_valid, error_message). If is_valid is True, error_message is empty.
        If is_valid is False, error_message contains diagnostic information.

    """
    # Early exit if no checks requested
    if not check_clashes and not check_connectivity:
        return True, ""

    # Use get_structure_diagnostics for the actual analysis
    diagnostics = get_structure_diagnostics(
        atoms, min_distance_factor, connectivity_factor
    )

    # Filter based on what checks are requested
    has_issues = False
    error_parts = []

    if check_clashes and diagnostics.has_clashes:
        has_issues = True
        error_parts.append(
            f"Atomic clashes detected with min_distance_factor={min_distance_factor}:\n"
            f"  " + "\n  ".join(diagnostics.clash_details[:5])
        )

    if check_connectivity and diagnostics.is_disconnected:
        has_issues = True
        error_parts.append(
            f"Cluster is not connected with connectivity_factor={connectivity_factor}. "
            f"Atoms are not within bonding distance of each other.",
        )

    if has_issues:
        composition = atoms.get_chemical_formula()
        error_message = (
            f"Validation failed for {composition} cluster ({len(atoms)} atoms):\n"
            + "\n".join(error_parts)
        )
        return False, error_message

    return True, ""


def validate_cluster(
    atoms: Atoms,
    composition: list[str] | None = None,
    min_distance_factor: float | None = None,
    connectivity_factor: float = CONNECTIVITY_FACTOR,
    check_clashes: bool = True,
    check_connectivity: bool | None = None,
    sort_atoms: bool = True,
    raise_on_failure: bool = False,
    source: str = "",
) -> tuple[Atoms, bool, str]:
    """Unified cluster validation with comprehensive checks.

    This function consolidates all validation logic used across the initialization
    module. It can check composition, clashes, connectivity, and optionally sort
    atoms by element.

    Args:
        atoms: The Atoms object to validate
        composition: Optional expected composition to verify exact match
        min_distance_factor: Factor for minimum distance checks. If None, uses
                           MIN_DISTANCE_FACTOR_DEFAULT when check_clashes is True
        connectivity_factor: Factor for connectivity threshold
        check_clashes: Whether to check for atomic clashes (default: True)
        check_connectivity: Whether to check connectivity. If None, auto-detects
                          based on atom count (>2 atoms)
        sort_atoms: Whether to sort atoms by element symbol (default: True)
        raise_on_failure: Whether to raise ValueError on validation failure
        source: Context string for error messages (e.g., "template", "seed+growth")

    Returns:
        Tuple of (validated_atoms, is_valid, error_message). If is_valid is True,
        error_message is empty. validated_atoms may be sorted if sort_atoms=True.

    Raises:
        ValueError: If raise_on_failure=True and validation fails

    """
    from scgo.initialization.initialization_config import MIN_DISTANCE_FACTOR_DEFAULT
    from scgo.utils.helpers import get_composition_counts

    # Auto-detect if we should check connectivity
    if check_connectivity is None:
        check_connectivity = _should_check_connectivity(atoms)

    # Use default min_distance_factor if not provided and checks are requested
    if min_distance_factor is None and (check_clashes or check_connectivity):
        min_distance_factor = MIN_DISTANCE_FACTOR_DEFAULT

    # Verify exact composition if provided
    if composition is not None and not _verify_exact_composition(atoms, composition):
        expected_counts = get_composition_counts(composition)
        actual_counts = get_composition_counts(atoms.get_chemical_symbols())
        error_msg = (
            f"{'[' + source + '] ' if source else ''}Composition mismatch. "
            f"Expected {composition} (counts: {expected_counts}), "
            f"got {atoms.get_chemical_symbols()} (counts: {actual_counts})"
        )
        if raise_on_failure:
            raise ValueError(error_msg)
        return atoms, False, error_msg

    # Sort atoms by element if requested
    if sort_atoms:
        atoms = _sort_atoms_by_element(atoms)

    # Validate structure (clashes and connectivity)
    if check_clashes or check_connectivity:
        is_valid, error_message = validate_cluster_structure(
            atoms,
            min_distance_factor,
            connectivity_factor,
            check_clashes=check_clashes,
            check_connectivity=check_connectivity,
        )

        if not is_valid:
            full_error = f"{'[' + source + '] ' if source else ''}{error_message}"
            if raise_on_failure:
                raise ValueError(full_error)
            return atoms, False, full_error

    return atoms, True, ""


def _sort_atoms_by_element(atoms: Atoms) -> Atoms:
    """Sort atoms by element symbol to ensure consistent ordering.

    This ensures that clusters with the same composition always have
    the same atom ordering, which is required for GA pairing operations.

    Args:
        atoms: The Atoms object to sort

    Returns:
        A new Atoms object with atoms sorted by element symbol
    """
    if len(atoms) <= 1:
        return atoms  # No sorting needed for single atoms or empty clusters

    # Get element symbols
    symbols = atoms.get_chemical_symbols()

    # Early exit: check if already sorted
    is_sorted = all(symbols[i] <= symbols[i + 1] for i in range(len(symbols) - 1))
    if is_sorted:
        # Already sorted, return copy to maintain API contract
        return atoms.copy()

    positions = atoms.get_positions()

    # Create tuples of (element_symbol, original_index) for stable sorting
    indexed_symbols = [(symbol, i) for i, symbol in enumerate(symbols)]

    # Sort by element symbol (alphabetically), then by original index for stability
    indexed_symbols.sort(key=lambda x: (x[0], x[1]))

    # Extract sorted indices
    sorted_indices = [idx for _, idx in indexed_symbols]

    # Create new Atoms object with sorted order
    sorted_symbols = [symbols[i] for i in sorted_indices]
    sorted_positions = positions[sorted_indices]

    sorted_atoms = Atoms(
        symbols=sorted_symbols,
        positions=sorted_positions,
        cell=atoms.get_cell(),
        pbc=atoms.get_pbc(),
    )

    # Copy calculator and info if present
    if atoms.calc is not None:
        sorted_atoms.calc = atoms.calc
    if hasattr(atoms, "info") and atoms.info:
        sorted_atoms.info = atoms.info.copy()

    return sorted_atoms


def _should_check_connectivity(atoms: Atoms) -> bool:
    """Determine if connectivity check should be performed for a cluster.

    Connectivity checks are only meaningful for clusters with more than 2 atoms.
    For very small clusters (<= 2 atoms), the notion of connectivity is ambiguous.

    Args:
        atoms: The Atoms object to check

    Returns:
        True if connectivity should be checked, False otherwise
    """
    return len(atoms) > 2


def _verify_exact_composition(atoms: Atoms, composition: list[str]) -> bool:
    """Verify that atoms object has exactly the composition specified.

    Args:
        atoms: The Atoms object to verify
        composition: Target composition list

    Returns:
        True if composition matches exactly, False otherwise
    """
    if len(atoms) != len(composition):
        return False

    atoms_symbols = atoms.get_chemical_symbols()
    atoms_counts = get_composition_counts(atoms_symbols)
    comp_counts = get_composition_counts(composition)

    return atoms_counts == comp_counts


def _cycle_composition_to_length(
    composition: list[str], target_length: int
) -> list[str]:
    """Cycle a composition list to match a target length, producing exact element counts.

    This function repeats the composition list as many times as needed to reach
    the target length, then truncates to exactly match the target length. This
    produces exact element counts that match what cycling the pattern would create.

    Args:
        composition: List of element symbols to cycle
        target_length: Target length for the resulting composition list

    Returns:
        List of element symbols with length equal to target_length

    Example:
        >>> _cycle_composition_to_length(["Pt", "Au"], 5)
        ["Pt", "Au", "Pt", "Au", "Pt"]
    """
    if not composition:
        raise ValueError("Cannot cycle empty composition to target length")

    if target_length <= 0:
        return []

    n_cycles = (target_length // len(composition)) + (
        1 if target_length % len(composition) > 0 else 0
    )
    return (composition * n_cycles)[:target_length]


def _assign_exact_composition(
    cluster: Atoms,
    composition: list[str],
    n_atoms: int | None = None,
    rng: np.random.Generator | None = None,
) -> Atoms:
    """Assign exact composition to cluster, ensuring atom count matches.

    This function ensures the final composition matches the target composition
    exactly by cycling through the composition list to match n_atoms, producing
    exact element counts.

    Args:
        cluster: The cluster to assign composition to
        composition: Target composition list (will be cycled to match n_atoms)
        n_atoms: Expected number of atoms. If None, uses len(cluster)
        rng: Optional RNG to shuffle the assigned composition (prevents patterns)

    Returns:
        Atoms object with exact composition assigned

    Raises:
        ValueError: If cluster atom count doesn't match n_atoms (when provided) or if
                   composition assignment fails
    """
    if n_atoms is None:
        n_atoms = len(cluster)

    if len(cluster) != n_atoms:
        raise ValueError(
            f"Cannot assign composition: cluster has {len(cluster)} atoms "
            f"but target is {n_atoms} atoms"
        )

    # Assign exact composition
    if not composition:
        raise ValueError(
            f"Cannot assign empty composition to cluster with {n_atoms} atoms"
        )
    elif len(composition) == n_atoms:
        # Exact match - use composition directly
        cluster.set_chemical_symbols(composition)
    else:
        # Create extended composition list by cycling
        extended_composition = _cycle_composition_to_length(composition, n_atoms)

        if rng is not None:
            rng.shuffle(extended_composition)

        # Verify exact counts match what cycling produced
        expected_counts = get_composition_counts(extended_composition)

        # Assign the extended composition
        cluster.set_chemical_symbols(extended_composition)

        # Verify the assignment produced exact counts
        actual_counts = get_composition_counts(cluster.get_chemical_symbols())
        if actual_counts != expected_counts:
            raise ValueError(
                f"Composition assignment failed: expected counts {expected_counts}, "
                f"got {actual_counts} after assignment"
            )

    return cluster


def _compute_composition_delta(
    base_counts: dict[str, int],
    target_counts: dict[str, int],
) -> tuple[list[str], dict[str, int], dict[str, int]]:
    """Compute atoms to add and remove to match target composition.

    Args:
        base_counts: Current composition counts as dict mapping element to count
        target_counts: Target composition counts as dict mapping element to count

    Returns:
        Tuple of (atoms_to_add, atoms_to_remove_dict, excess_elements_dict):
        - atoms_to_add: List of element symbols to add
        - atoms_to_remove: Dict mapping element to count to remove
        - excess_elements: Dict of elements that would need to be removed
          but aren't present in sufficient quantity (for error reporting)
    """
    atoms_to_add = []
    atoms_to_remove = {}
    excess_elements = {}

    all_elements = set(base_counts.keys()) | set(target_counts.keys())

    for elem in all_elements:
        base_count = base_counts.get(elem, 0)
        target_count = target_counts.get(elem, 0)
        diff = target_count - base_count

        if diff > 0:
            # Need to add this element
            atoms_to_add.extend([elem] * diff)
        elif diff < 0:
            # Need to remove this element
            removal_count = -diff
            if base_count >= removal_count:
                atoms_to_remove[elem] = removal_count
            else:
                # Can't remove enough - this is an excess element
                excess_elements[elem] = removal_count - base_count

    return atoms_to_add, atoms_to_remove, excess_elements


def _check_composition_feasibility(
    base_composition: list[str],
    target_composition: list[str],
    operation: str = "grow",
) -> tuple[bool, str]:
    """Check if composition change is feasible.

    Validates whether it's possible to transform base_composition into
    target_composition through growth or reduction operations.

    Args:
        base_composition: Current composition as list of element symbols
        target_composition: Target composition as list of element symbols
        operation: Type of operation - "grow" or "reduce" (affects error messages)

    Returns:
        Tuple of (is_feasible, error_message):
        - is_feasible: True if operation is possible, False otherwise
        - error_message: Empty string if feasible, detailed error message if not
    """
    # Handle empty composition cases
    if not base_composition:
        if not target_composition or operation == "grow":
            return True, ""
        return False, (
            f"Cannot {operation} from empty composition to non-empty target "
            f"{target_composition}. Use growth operation instead."
        )

    if not target_composition:
        if operation == "reduce":
            return True, ""
        return False, (
            f"Cannot {operation} from non-empty composition {base_composition} "
            f"to empty target. Use reduction operation instead."
        )

    base_counts = get_composition_counts(base_composition)
    target_counts = get_composition_counts(target_composition)

    _, atoms_to_remove, excess_elements = _compute_composition_delta(
        base_counts, target_counts
    )

    # Check if we can achieve target composition
    if excess_elements:
        # Some elements need to be removed but aren't present in sufficient quantity
        excess_details = ", ".join(
            f"{elem} (need to remove {count} more than available)"
            for elem, count in excess_elements.items()
        )
        return False, (
            f"Cannot achieve target composition {target_composition} from "
            f"base {base_composition}: insufficient quantity of elements to remove. "
            f"Excess elements: {excess_details}"
        )

    base_total = sum(base_counts.values())
    target_total = sum(target_counts.values())

    if operation == "grow" and target_total < base_total:
        return False, (
            f"Cannot grow from {base_total} atoms to {target_total} atoms. "
            f"Target has fewer atoms than base. Use reduction operation instead."
        )

    if operation == "reduce" and target_total > base_total:
        return False, (
            f"Cannot reduce from {base_total} atoms to {target_total} atoms. "
            f"Target has more atoms than base. Use growth operation instead."
        )

    return True, ""


def clear_convex_hull_cache() -> None:
    """Clear the convex hull computation cache.

    This is useful for testing or when memory usage becomes a concern.
    """
    get_global_cache().clear_namespace(_CONVEX_HULL_CACHE_NS)
