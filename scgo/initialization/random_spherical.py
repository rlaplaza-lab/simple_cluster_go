"""Random spherical cluster generation and growth algorithms.

This module provides functions for generating initial atomic cluster structures
through random placement in spherical volumes and convex-hull-guided growth
strategies that operate on the current cluster geometry.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from ase import Atom, Atoms
from ase.data import covalent_radii
from scipy.spatial import KDTree

from scgo.utils.helpers import get_composition_counts
from scgo.utils.logging import get_logger

from .geometry_helpers import (
    _check_composition_feasibility,
    _generate_batch_positions_on_convex_hull,
    _verify_exact_composition,
    analyze_disconnection,
    compute_bond_distance_params,
    format_placement_error_message,
    get_covalent_radius,
    get_structure_diagnostics,
    is_cluster_connected,
    validate_cluster,
)
from .initialization_config import (
    CONNECTIVITY_FACTOR,
    GROWTH_ORDER_STRATEGY_COUNT,
    KDTREE_THRESHOLD,
    MAX_CONNECTIVITY_RETRIES,
    MAX_CONSECUTIVE_FAILURES,
    MAX_PLACEMENT_ATTEMPTS_PER_ATOM,
    MIN_DISTANCE_FACTOR_DEFAULT,
    MIN_DISTANCE_THRESHOLD_HIGH,
    MIN_DISTANCE_THRESHOLD_LOW,
    PLACEMENT_RADIUS_SCALING_DEFAULT,
    PLACEMENT_RELAXATION_FACTOR,
)

logger = get_logger(__name__)


# Growth order strategy implementations
def _group_atoms_by_element(atoms_to_add: list[str]) -> dict[str, list[str]]:
    """Group atoms by element symbol.

    Args:
        atoms_to_add: List of element symbols to group

    Returns:
        Dictionary mapping element symbol to list of occurrences
    """
    element_groups: dict[str, list[str]] = {}
    for atom in atoms_to_add:
        if atom not in element_groups:
            element_groups[atom] = []
        element_groups[atom].append(atom)
    return element_groups


def _growth_order_random(
    atoms_to_add: list[str], rng: np.random.Generator
) -> list[str]:
    """Strategy 0: Keep random order (already shuffled).

    Args:
        atoms_to_add: List of element symbols to add
        rng: Random number generator (unused, kept for API consistency)

    Returns:
        List of element symbols in original random order
    """
    return atoms_to_add


def _growth_order_by_element(
    atoms_to_add: list[str], rng: np.random.Generator
) -> list[str]:
    """Strategy 1: Group by element, add all of one element first.

    Args:
        atoms_to_add: List of element symbols to add
        rng: Random number generator for shuffling element order

    Returns:
        List of element symbols grouped by element type
    """
    element_groups = _group_atoms_by_element(atoms_to_add)
    element_order = list(element_groups.keys())
    rng.shuffle(element_order)
    return [atom for element in element_order for atom in element_groups[element]]


def _growth_order_alternating(
    atoms_to_add: list[str], rng: np.random.Generator
) -> list[str]:
    """Strategy 2: Interleave elements to promote mixing.

    Args:
        atoms_to_add: List of element symbols to add
        rng: Random number generator (unused, kept for API consistency)

    Returns:
        List of element symbols with interleaved element types
    """
    element_groups = _group_atoms_by_element(atoms_to_add)
    max_len = max(len(group) for group in element_groups.values())
    result = [
        element_groups[element][i]
        for i in range(max_len)
        for element in element_groups
        if i < len(element_groups[element])
    ]
    return result


def _growth_order_by_size(
    atoms_to_add: list[str], rng: np.random.Generator
) -> list[str]:
    """Strategy 3: Sort by atomic size (covalent radius).

    Args:
        atoms_to_add: List of element symbols to add
        rng: Random number generator for selecting sort direction

    Returns:
        List of element symbols sorted by covalent radius
    """
    size_direction = rng.integers(0, 2)  # 0=larger first, 1=smaller first
    atoms_with_sizes = [(atom, get_covalent_radius(atom)) for atom in atoms_to_add]
    atoms_with_sizes.sort(key=lambda x: x[1], reverse=(size_direction == 0))
    return [atom for atom, _ in atoms_with_sizes]


def _growth_order_element_clustering(
    atoms_to_add: list[str], rng: np.random.Generator
) -> list[str]:
    """Strategy 4: Maximize or minimize element clustering.

    Args:
        atoms_to_add: List of element symbols to add
        rng: Random number generator for selecting clustering preference

    Returns:
        List of element symbols ordered to maximize or minimize clustering
    """
    element_counts = get_composition_counts(atoms_to_add)

    if len(element_counts) <= 1:
        # Single element - keep random order
        return atoms_to_add

    cluster_preference = rng.integers(0, 2)  # 0=maximize clustering, 1=minimize
    element_groups = _group_atoms_by_element(atoms_to_add)

    if cluster_preference == 0:
        # Maximize clustering: add all of one element, then all of next
        element_order = sorted(
            element_groups.keys(),
            key=lambda e: len(element_groups[e]),
            reverse=True,
        )
        return [atom for element in element_order for atom in element_groups[element]]
    else:
        # Minimize clustering: maximize interleaving
        max_len = max(len(group) for group in element_groups.values())
        result: list[str] = []
        for i in range(max_len):
            # Shuffle element order each iteration for more dispersion
            element_order = list(element_groups.keys())
            rng.shuffle(element_order)
            result.extend(
                element_groups[element][i]
                for element in element_order
                if i < len(element_groups[element])
            )
        return result


def _growth_order_by_composition_balance(
    atoms_to_add: list[str],
    base_composition: list[str],
    target_composition: list[str],
    rng: np.random.Generator,
) -> list[str]:
    """Strategy 5: Order growth to maintain target composition ratios.

    Calculates composition deficit for each element and prioritizes
    adding atoms that bring composition closer to target ratios.
    This helps maintain balanced composition throughout growth.

    Args:
        atoms_to_add: List of element symbols to add
        base_composition: Current composition of the seed/cluster
        target_composition: Target composition to achieve
        rng: Random number generator

    Returns:
        Reordered list of atoms to add, prioritizing elements with largest deficits
    """
    if not base_composition or not target_composition:
        # Fallback to random if compositions are missing
        return atoms_to_add

    base_counts = get_composition_counts(base_composition)
    target_counts = get_composition_counts(target_composition)

    # Calculate current composition (base + what we're adding)
    current_counts = base_counts.copy()
    for atom in atoms_to_add:
        current_counts[atom] = current_counts.get(atom, 0) + 1

    # Calculate deficit for each element (how far from target ratio)
    deficits = {}
    all_elements = set(target_composition) | set(atoms_to_add)

    for elem in all_elements:
        target_count = target_counts.get(elem, 0)
        current_count = current_counts.get(elem, 0)
        deficit = target_count - current_count
        deficits[elem] = deficit

    atoms_with_deficit = [(atom, deficits.get(atom, 0)) for atom in atoms_to_add]
    atoms_with_deficit.sort(key=lambda x: x[1], reverse=True)

    # Group atoms by deficit and shuffle within each group for diversity
    deficit_groups = defaultdict(list)
    for atom, deficit in atoms_with_deficit:
        deficit_groups[deficit].append(atom)

    # Sort by deficit (descending) and shuffle within each group
    result = []
    for deficit in sorted(deficit_groups.keys(), reverse=True):
        group = deficit_groups[deficit]
        rng.shuffle(group)
        result.extend(group)

    return result


# Growth order strategy dispatch table
_GROWTH_ORDER_STRATEGIES = {
    0: _growth_order_random,
    1: _growth_order_by_element,
    2: _growth_order_alternating,
    3: _growth_order_by_size,
    4: _growth_order_element_clustering,
    5: _growth_order_by_composition_balance,
}


def _compute_effective_placement_params(
    attempt_ratio: float,
    min_distance_factor: float,
    placement_radius_scaling: float,
) -> tuple[float, float]:
    """Compute effective placement parameters with progressive relaxation.

    Args:
        attempt_ratio: Ratio of current attempt to max attempts (0.0 to 1.0)
        min_distance_factor: Base minimum distance factor
        placement_radius_scaling: Base placement radius scaling

    Returns:
        Tuple of (effective_scaling, effective_min_distance)
    """
    scale_growth = min(1.0, min_distance_factor)
    if placement_radius_scaling < PLACEMENT_RELAXATION_FACTOR:
        effective_scaling = placement_radius_scaling
    else:
        effective_scaling = placement_radius_scaling * (
            1.0 + scale_growth * attempt_ratio
        )

    relaxed_factor = min_distance_factor * (
        1.0 - PLACEMENT_RELAXATION_FACTOR * attempt_ratio
    )

    if min_distance_factor >= MIN_DISTANCE_THRESHOLD_HIGH:
        effective_min_distance = min_distance_factor
    elif min_distance_factor >= MIN_DISTANCE_THRESHOLD_LOW:
        effective_min_distance = max(MIN_DISTANCE_THRESHOLD_LOW, relaxed_factor)
    else:
        effective_min_distance = max(0.0, relaxed_factor)

    return effective_scaling, effective_min_distance


def _apply_growth_order_strategy(
    atoms_to_add: list[str],
    strategy: int,
    rng: np.random.Generator,
    base_composition: list[str] | None = None,
    target_composition: list[str] | None = None,
) -> list[str]:
    """Apply a growth order strategy to atoms_to_add list.

    Args:
        atoms_to_add: List of element symbols to add
        strategy: Strategy index (0-5)
        rng: Random number generator
        base_composition: Current composition (required for strategy 5)
        target_composition: Target composition (required for strategy 5)

    Returns:
        Reordered list of atoms to add
    """
    strategy_func = _GROWTH_ORDER_STRATEGIES.get(strategy, _growth_order_random)

    # Strategy 5 requires base and target composition
    if strategy == 5:
        if base_composition is None or target_composition is None:
            # Fallback to random if compositions not provided
            return _growth_order_random(atoms_to_add, rng)
        return strategy_func(atoms_to_add, base_composition, target_composition, rng)

    # Other strategies only need atoms_to_add and rng
    return strategy_func(atoms_to_add, rng)


def random_spherical(
    composition: list[str],
    cell_side: float,
    rng: np.random.Generator,
    placement_radius_scaling: float = PLACEMENT_RADIUS_SCALING_DEFAULT,
    min_distance_factor: float = MIN_DISTANCE_FACTOR_DEFAULT,
    connectivity_factor: float = CONNECTIVITY_FACTOR,
    max_connectivity_retries: int = MAX_CONNECTIVITY_RETRIES,
) -> Atoms:
    """Place atoms randomly within a compact sphere, ensuring minimum distances.

    Atoms are added iteratively with covalent-radii-based clash checks and
    connectivity enforcement. For each retry attempt the algorithm slightly
    relaxes the effective placement radius and distance thresholds within the
    user-specified bounds to improve the chance of finding a valid connected
    configuration. A final validation step (when enabled via
    :mod:`initialization_config`) reuses the same logic to guard against
    clashes and disconnected clusters.

    Args:
        composition: List of element symbols for the atoms.
        cell_side: The side length of the cubic cell for the returned Atoms object.
        placement_radius_scaling: A scaling factor used to determine the initial
            spherical volume for atom placement. Larger values result in a larger
            initial volume.
        min_distance_factor: Factor to scale the sum of covalent radii for
            minimum allowed distance between atoms. A value of 1.0 means no
            overlap, while < 1.0 allows some overlap.
        connectivity_factor: Factor to multiply sum of covalent radii for
            connectivity threshold.
        max_connectivity_retries: Maximum number of retries if connectivity
            validation fails.
        rng: Optional numpy ``Generator`` for reproducible randomness.

    Returns:
        An :class:`ase.Atoms` instance with the randomly placed cluster.

    Raises:
        ValueError: If all atoms cannot be placed within the given constraints
        after a maximum number of attempts, or if connectivity validation
        fails after all retries.

    """
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

    if cell_side <= 0:
        raise ValueError(f"cell_side must be positive, got {cell_side}")

    n_atoms = len(composition)
    if n_atoms == 0:
        return Atoms()

    # Retry logic for connectivity validation
    for retry_attempt in range(max_connectivity_retries):
        new_atoms = Atoms()
        new_atoms.set_cell([cell_side, cell_side, cell_side])

        final_atoms = _add_atoms_to_cluster_iteratively(
            base_atoms=new_atoms,
            atoms_to_add=composition,
            min_distance_factor=min_distance_factor,
            placement_radius_scaling=placement_radius_scaling,
            rng=rng,
            connectivity_factor=connectivity_factor,
        )

        if final_atoms is None:
            if retry_attempt == max_connectivity_retries - 1:
                error_msg = format_placement_error_message(
                    context=f"place all {n_atoms} atoms after {max_connectivity_retries} attempts",
                    composition=composition,
                    n_atoms=n_atoms,
                    placement_radius_scaling=placement_radius_scaling,
                    min_distance_factor=min_distance_factor,
                    connectivity_factor=connectivity_factor,
                    cell_side=cell_side,
                    additional_info=(
                        f"This may indicate:\n"
                        f"  - Cell too small: cell_side={cell_side:.2f} Å may be insufficient for {n_atoms} atoms\n"
                        f"  - Placement volume too small\n"
                        f"  - Distance constraints too strict\n"
                        f"  - Connectivity factor too strict"
                    ),
                )
                raise ValueError(error_msg)
            continue  # Try again with different random placement

        if len(final_atoms) > 2 and not is_cluster_connected(
            final_atoms, connectivity_factor
        ):
            if retry_attempt == max_connectivity_retries - 1:
                diagnostics = get_structure_diagnostics(
                    final_atoms, min_distance_factor, connectivity_factor
                )
                error_msg = format_placement_error_message(
                    context=f"create connected cluster after {max_connectivity_retries} attempts",
                    composition=composition,
                    n_atoms=n_atoms,
                    placement_radius_scaling=placement_radius_scaling,
                    min_distance_factor=min_distance_factor,
                    connectivity_factor=connectivity_factor,
                    diagnostics=diagnostics,
                    additional_info="The cluster is disconnected.",
                )
                raise ValueError(error_msg)
            continue  # Try again with different random placement

        final_atoms.center()

        validated_atoms, _, _ = validate_cluster(
            final_atoms,
            composition=None,
            min_distance_factor=min_distance_factor,
            connectivity_factor=connectivity_factor,
            sort_atoms=False,
            raise_on_failure=True,
            source="random_spherical",
        )

        return validated_atoms

    # Should never reach here; raise to indicate placement failure
    raise ValueError(
        "Failed to place atoms into a connected cluster within allowed attempts"
    )


def grow_from_seed(
    seed_atoms: Atoms,
    target_composition: list[str],
    placement_radius_scaling: float,
    cell_side: float,
    rng: np.random.Generator,
    min_distance_factor: float = MIN_DISTANCE_FACTOR_DEFAULT,
    connectivity_factor: float = CONNECTIVITY_FACTOR,
) -> Atoms | None:
    """Try to grow a smaller candidate :class:`ase.Atoms` to the target composition.

    Growth is performed by repeatedly adding atoms to the existing seed using
    convex-hull-based placement (via :func:`_add_atoms_to_cluster_iteratively`),
    with covalent-radii-based clash checks and connectivity enforcement.

    Args:
        seed_atoms: The seed :class:`ase.Atoms` object to grow from.
        target_composition: The target composition as a list of element symbols.
        placement_radius_scaling: A scaling factor to determine the placement shell
            radius.
        min_distance_factor: Factor to scale covalent radii for minimum distance
            checks.
        cell_side: The side length of the cubic cell for the new :class:`ase.Atoms`
            object.
        connectivity_factor: Factor to multiply sum of covalent radii for
            connectivity threshold.
        rng: Optional numpy random number generator.

    Returns:
        A new :class:`ase.Atoms` object of the target composition on success,
        or ``None`` on failure.

    """
    base_atoms = seed_atoms.copy()
    base_composition = base_atoms.get_chemical_symbols()

    # Handle empty target composition - return seed with proper cell/centering
    if not target_composition:
        base_atoms.set_cell([cell_side, cell_side, cell_side])
        base_atoms.center()
        return base_atoms

    # Check composition feasibility before attempting growth
    is_feasible, error_message = _check_composition_feasibility(
        base_composition, target_composition, operation="grow"
    )
    if not is_feasible:
        logger.warning(
            f"Composition growth not feasible: {error_message}. "
            f"Base: {base_composition}, Target: {target_composition}"
        )
        return None

    base_counts = get_composition_counts(base_composition)
    target_counts = get_composition_counts(target_composition)

    atoms_to_add = list((target_counts - base_counts).elements())

    if not atoms_to_add:
        base_atoms.set_cell([cell_side, cell_side, cell_side])
        base_atoms.center()
        return base_atoms

    growth_order_strategy = rng.integers(0, GROWTH_ORDER_STRATEGY_COUNT)
    atoms_to_add = _apply_growth_order_strategy(
        atoms_to_add,
        growth_order_strategy,
        rng,
        base_composition=base_composition,
        target_composition=target_composition,
    )

    final_atoms = _add_atoms_to_cluster_iteratively(
        base_atoms=base_atoms,
        atoms_to_add=atoms_to_add,
        min_distance_factor=min_distance_factor,
        placement_radius_scaling=placement_radius_scaling,
        rng=rng,
        connectivity_factor=connectivity_factor,
    )

    if final_atoms:
        final_atoms.set_cell([cell_side, cell_side, cell_side])
        final_atoms.center()

        # Verify exact composition match after growth
        if not _verify_exact_composition(final_atoms, target_composition):
            expected_counts = get_composition_counts(target_composition)
            actual_counts = get_composition_counts(final_atoms.get_chemical_symbols())
            raise ValueError(
                f"Growth operation produced incorrect composition. "
                f"Expected {target_composition} (counts: {expected_counts}), "
                f"got {final_atoms.get_chemical_symbols()} (counts: {actual_counts})"
            )

        validated_atoms, _, _ = validate_cluster(
            final_atoms,
            composition=None,
            min_distance_factor=min_distance_factor,
            connectivity_factor=connectivity_factor,
            sort_atoms=False,
            raise_on_failure=True,
            source="grow_from_seed",
        )

        return validated_atoms

    # If growth could not complete above, explicitly signal failure
    return None


def _add_atoms_to_cluster_iteratively(
    base_atoms: Atoms,
    atoms_to_add: list[str],
    min_distance_factor: float,
    placement_radius_scaling: float,
    rng: np.random.Generator,
    connectivity_factor: float = CONNECTIVITY_FACTOR,
) -> Atoms | None:
    """Iteratively adds atoms to a base Atoms object within a spherical volume.

    This function places new atoms in batches when possible (for clusters with ≥4 atoms),
    using convex hull facets to place multiple atoms per hull computation. For smaller
    clusters or edge cases, it falls back to single-atom placement. The placement volume
    is dynamically adjusted. Connectivity is checked during growth to ensure the cluster
    remains connected.

    Args:
        base_atoms: The starting Atoms object to add atoms to
        atoms_to_add: List of element symbols to add.
        min_distance_factor: Factor for minimum distance validation.
        placement_radius_scaling: Scaling for placement radius.
        rng: Random number generator.
        connectivity_factor: Factor for connectivity threshold.

    Returns:
        A new Atoms object with all atoms added, or None if addition failed
    """
    if not atoms_to_add:
        return base_atoms.copy()

    atoms = base_atoms.copy()
    atoms_to_add = list(atoms_to_add)
    rng.shuffle(atoms_to_add)

    new_atoms = atoms.copy()

    radii_to_add = {s: get_covalent_radius(s) for s in set(atoms_to_add)}

    # Determine if we're creating a 2-atom cluster
    total_atoms = len(base_atoms) + len(atoms_to_add)
    is_two_atom_cluster = total_atoms == 2

    # Handle first atom placement (always at origin)
    if not new_atoms:
        if not atoms_to_add:
            return new_atoms
        first_symbol = atoms_to_add.pop(0)
        new_atoms.append(Atom(first_symbol, np.array([0.0, 0.0, 0.0])))

    # Use batch placement for clusters with ≥4 atoms, single-atom for smaller clusters
    max_attempts_per_atom = MAX_PLACEMENT_ATTEMPTS_PER_ATOM

    # Single-atom placement mode for clusters with <4 atoms
    if len(new_atoms) < 4:
        return _add_atoms_single_mode(
            new_atoms,
            atoms_to_add,
            radii_to_add,
            min_distance_factor,
            placement_radius_scaling,
            rng,
            connectivity_factor,
            is_two_atom_cluster,
            max_attempts_per_atom,
            logger,
            base_atoms,
        )

    # Batch placement mode for clusters with ≥4 atoms
    return _add_atoms_batch_mode(
        new_atoms,
        atoms_to_add,
        radii_to_add,
        min_distance_factor,
        placement_radius_scaling,
        rng,
        connectivity_factor,
        max_attempts_per_atom,
        logger,
        base_atoms,
    )


def _add_atoms_single_mode(
    new_atoms: Atoms,
    atoms_to_add: list[str],
    radii_to_add: dict[str, float],
    min_distance_factor: float,
    placement_radius_scaling: float,
    rng: np.random.Generator,
    connectivity_factor: float,
    is_two_atom_cluster: bool,
    max_attempts_per_atom: int,
    logger,
    base_atoms: Atoms,
) -> Atoms | None:
    """Single-atom placement mode for clusters with <4 atoms."""
    for atom_idx, atom_symbol in enumerate(atoms_to_add):
        atom_radius = radii_to_add[atom_symbol]
        new_pos = None
        consecutive_failures = 0

        # Create KDTree once per atom if cluster is large enough
        use_kdtree = len(new_atoms) >= KDTREE_THRESHOLD
        tree = None
        if use_kdtree and len(new_atoms) > 0:
            tree = KDTree(new_atoms.get_positions())

        for attempt in range(max_attempts_per_atom):
            attempt_ratio = (attempt + 1) / max_attempts_per_atom
            effective_scaling, effective_min_distance = (
                _compute_effective_placement_params(
                    attempt_ratio, min_distance_factor, placement_radius_scaling
                )
            )
            current_positions = new_atoms.get_positions()
            n_current = len(new_atoms)

            if n_current > 0:
                center = new_atoms.get_center_of_mass()

                # Special handling for 2-atom clusters
                if is_two_atom_cluster and n_current == 1:
                    existing_radius = covalent_radii[new_atoms.numbers[0]]
                    bond_distance = existing_radius + atom_radius
                    direction = rng.standard_normal(3)
                    direction /= np.linalg.norm(direction)
                    candidate_pos = center + direction * bond_distance
                elif n_current == 1:
                    existing_radius = covalent_radii[new_atoms.numbers[0]]
                    base_extent = existing_radius
                    placement_distance = (
                        existing_radius + atom_radius * effective_scaling
                    )
                else:
                    max_dist_from_center = np.max(
                        np.linalg.norm(current_positions - center, axis=1),
                    )
                    base_extent = max_dist_from_center
                    placement_distance = (
                        max_dist_from_center + atom_radius * effective_scaling
                    )

                if not (is_two_atom_cluster and n_current == 1):
                    available_shell = placement_distance - base_extent
                    if available_shell < atom_radius * effective_min_distance:
                        continue

                    current_radii = np.array(
                        [covalent_radii[n] for n in new_atoms.numbers]
                    )
                    max_existing_radius = (
                        np.max(current_radii) if len(current_radii) > 0 else atom_radius
                    )
                    bond_distance, min_dist, max_connectivity_dist = (
                        compute_bond_distance_params(
                            max_existing_radius,
                            atom_radius,
                            connectivity_factor,
                            min_distance_factor,
                            placement_radius_scaling,
                            effective_min_distance=effective_min_distance,
                            effective_scaling=effective_scaling,
                        )
                    )

                    candidates = _generate_batch_positions_on_convex_hull(
                        new_atoms,
                        n_candidates=1,
                        bond_distance=bond_distance,
                        rng=rng,
                        min_connectivity_dist=min_dist,
                        max_connectivity_dist=max_connectivity_dist,
                        use_all_facets=False,
                        connectivity_factor=connectivity_factor,
                    )
                    if not candidates:
                        # Fallback to spherical if batch generation failed
                        center = new_atoms.get_center_of_mass()
                        direction = rng.standard_normal(3)
                        direction /= np.linalg.norm(direction)
                        candidate_pos = center + direction * bond_distance
                    else:
                        candidate_pos = candidates[0]

            if len(current_positions) > 0:
                current_radii = np.array([covalent_radii[n] for n in new_atoms.numbers])

                # Use KDTree for large clusters to optimize distance checks
                if use_kdtree and tree is not None:
                    dists, _ = tree.query(candidate_pos, k=len(current_positions))
                    dists = np.asarray(dists).flatten()
                else:
                    dists = np.linalg.norm(current_positions - candidate_pos, axis=1)

                min_allowed_dists = (
                    current_radii + atom_radius
                ) * effective_min_distance

                if np.any(dists < min_allowed_dists):
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        break  # Early termination
                    continue

                max_connectivity_dists = (
                    current_radii + atom_radius
                ) * connectivity_factor

                if not np.any(dists <= max_connectivity_dists):
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        break  # Early termination
                    continue

            new_pos = candidate_pos
            consecutive_failures = 0  # Reset on success
            break

        if new_pos is None:
            current_count = len(new_atoms)
            total_target = len(base_atoms) + len(atoms_to_add)
            attempts_made = min(attempt + 1, max_attempts_per_atom)
            early_term = consecutive_failures >= MAX_CONSECUTIVE_FAILURES
            reason = "early termination" if early_term else f"{attempts_made} attempts"

            # Get current composition state
            current_composition = new_atoms.get_chemical_symbols()
            current_counts = get_composition_counts(current_composition)
            remaining_atoms = atoms_to_add[atom_idx:]
            remaining_counts = get_composition_counts(remaining_atoms)

            diagnostics = get_structure_diagnostics(
                new_atoms, min_distance_factor, connectivity_factor
            )

            additional_info = (
                f"Failed to place atom {atom_symbol} ({current_count + 1}/{total_target}) after {reason}.\n"
                f"Current state: {current_count} atoms placed, composition counts: {current_counts}\n"
                f"Remaining atoms to place: {len(remaining_atoms)} atoms, composition counts: {remaining_counts}"
            )

            error_msg = format_placement_error_message(
                context="place atom",
                composition=None,
                n_atoms=None,
                placement_radius_scaling=placement_radius_scaling,
                min_distance_factor=min_distance_factor,
                connectivity_factor=connectivity_factor,
                diagnostics=diagnostics,
                additional_info=additional_info,
            )
            logger.debug(error_msg)
            return None

        new_atoms.append(Atom(atom_symbol, new_pos))

        if len(new_atoms) >= 2 and not is_cluster_connected(
            new_atoms, connectivity_factor
        ):
            disconnection_distance, suggested_factor, analysis_msg = (
                analyze_disconnection(new_atoms, connectivity_factor)
            )
            current_composition = new_atoms.get_chemical_symbols()
            current_counts = get_composition_counts(current_composition)
            remaining_atoms = atoms_to_add[atom_idx + 1 :]
            remaining_counts = (
                get_composition_counts(remaining_atoms) if remaining_atoms else {}
            )
            diagnostics = get_structure_diagnostics(
                new_atoms, min_distance_factor, connectivity_factor
            )

            additional_info = (
                f"Cluster became disconnected after placing atom {atom_symbol} "
                f"({len(new_atoms)}/{len(base_atoms) + len(atoms_to_add)} atoms placed).\n"
                f"Current state: {len(new_atoms)} atoms, composition counts: {current_counts}\n"
                f"Remaining atoms: {len(remaining_atoms)} atoms, composition counts: {remaining_counts}\n"
                f"Analysis: {analysis_msg}\n"
                f"Suggested connectivity_factor: {suggested_factor:.2f}"
            )

            error_msg = format_placement_error_message(
                context="maintain connectivity",
                composition=None,
                n_atoms=None,
                placement_radius_scaling=placement_radius_scaling,
                min_distance_factor=min_distance_factor,
                connectivity_factor=connectivity_factor,
                diagnostics=diagnostics,
                additional_info=additional_info,
            )
            logger.debug(error_msg)
            return None

    return new_atoms


def _add_atoms_batch_mode(
    new_atoms: Atoms,
    atoms_to_add: list[str],
    radii_to_add: dict[str, float],
    min_distance_factor: float,
    placement_radius_scaling: float,
    rng: np.random.Generator,
    connectivity_factor: float,
    max_attempts_per_atom: int,
    logger,
    base_atoms: Atoms,
) -> Atoms | None:
    """Batch placement mode for clusters with ≥4 atoms."""
    total_target = len(base_atoms) + len(atoms_to_add)
    batch_attempt = 0
    max_batch_attempts = max_attempts_per_atom  # Limit total batch attempts

    while atoms_to_add:
        batch_attempt += 1
        if batch_attempt > max_batch_attempts:
            current_count = len(new_atoms)
            current_composition = new_atoms.get_chemical_symbols()
            current_counts = get_composition_counts(current_composition)
            remaining_counts = get_composition_counts(atoms_to_add)

            diagnostics = get_structure_diagnostics(
                new_atoms, min_distance_factor, connectivity_factor
            )

            additional_info = (
                f"Failed to place remaining {len(atoms_to_add)} atoms "
                f"({current_count}/{total_target} placed) after {max_batch_attempts} batch attempts.\n"
                f"Current state: {current_count} atoms placed, composition counts: {current_counts}\n"
                f"Remaining atoms: {len(atoms_to_add)} atoms, composition counts: {remaining_counts}"
            )

            error_msg = format_placement_error_message(
                context="place remaining atoms in batch mode",
                composition=None,
                n_atoms=None,
                placement_radius_scaling=placement_radius_scaling,
                min_distance_factor=min_distance_factor,
                connectivity_factor=connectivity_factor,
                diagnostics=diagnostics,
                additional_info=additional_info,
            )
            logger.debug(error_msg)
            return None

        # Progressive relaxation for batch attempts
        attempt_ratio = batch_attempt / max_batch_attempts
        effective_scaling, effective_min_distance = _compute_effective_placement_params(
            attempt_ratio, min_distance_factor, placement_radius_scaling
        )

        current_positions = new_atoms.get_positions()

        # Calculate placement parameters
        current_radii = np.array([covalent_radii[n] for n in new_atoms.numbers])
        max_existing_radius = float(
            np.max(current_radii) if len(current_radii) > 0 else 0.0
        )
        avg_atom_radius = float(np.mean([radii_to_add[s] for s in atoms_to_add]))

        bond_distance, min_dist, max_connectivity_dist = compute_bond_distance_params(
            max_existing_radius,
            avg_atom_radius,
            connectivity_factor,
            min_distance_factor,
            placement_radius_scaling,
            effective_min_distance=effective_min_distance,
            effective_scaling=effective_scaling,
        )

        # Generate batch of candidate positions
        batch_size = min(len(atoms_to_add), 100)  # Limit batch size for efficiency
        candidates = _generate_batch_positions_on_convex_hull(
            new_atoms,
            batch_size,
            bond_distance,
            rng,
            min_connectivity_dist=min_dist,
            max_connectivity_dist=max_connectivity_dist,
            connectivity_factor=connectivity_factor,
        )

        if not candidates:
            # Fall back to single-atom mode if batch generation fails
            return _add_atoms_single_mode(
                new_atoms,
                atoms_to_add,
                radii_to_add,
                min_distance_factor,
                placement_radius_scaling,
                rng,
                connectivity_factor,
                False,
                max_attempts_per_atom,
                logger,
                base_atoms,
            )

        # Validate and place candidates
        atoms_to_place = atoms_to_add[: len(candidates)]
        valid_placements = []

        # Use KDTree for large clusters to optimize distance checks
        use_kdtree = len(new_atoms) >= KDTREE_THRESHOLD
        if use_kdtree and len(current_positions) > 0:
            tree = KDTree(current_positions)

        for i, (atom_symbol, candidate_pos) in enumerate(
            zip(atoms_to_place, candidates, strict=True)
        ):
            atom_radius = radii_to_add[atom_symbol]

            # Check distance to existing atoms
            if len(current_positions) > 0:
                if use_kdtree:
                    dists_to_existing, _ = tree.query(
                        candidate_pos, k=len(current_positions)
                    )
                    dists_to_existing = np.asarray(dists_to_existing).flatten()
                else:
                    dists_to_existing = np.linalg.norm(
                        current_positions - candidate_pos, axis=1
                    )
                min_allowed_dists = (
                    current_radii + atom_radius
                ) * effective_min_distance

                if np.any(dists_to_existing < min_allowed_dists):
                    continue  # Clash with existing atoms

                # Check connectivity
                max_connectivity_dists = (
                    current_radii + atom_radius
                ) * connectivity_factor

                if not np.any(dists_to_existing <= max_connectivity_dists):
                    continue  # Too far for connectivity

            # Check distance to other candidates in batch
            valid = True
            for j, (other_symbol, other_pos) in enumerate(
                zip(atoms_to_place, candidates, strict=True)
            ):
                if i == j:
                    continue
                other_radius = radii_to_add[other_symbol]
                dist = np.linalg.norm(candidate_pos - other_pos)
                min_allowed = (atom_radius + other_radius) * effective_min_distance
                if dist < min_allowed:
                    valid = False
                    break

            if valid:
                valid_placements.append((atom_symbol, candidate_pos))

        # Place all valid candidates
        if not valid_placements:
            # No valid placements in this batch, try again
            continue

        # Remove placed atoms from atoms_to_add (one per placement, not all of each symbol)
        placed_counts = Counter(sym for sym, _ in valid_placements)
        remaining = []
        for atom in atoms_to_add:
            if placed_counts.get(atom, 0) > 0:
                placed_counts[atom] -= 1
            else:
                remaining.append(atom)
        atoms_to_add = remaining

        for atom_symbol, pos in valid_placements:
            new_atoms.append(Atom(atom_symbol, pos))

        if len(new_atoms) >= 2 and not is_cluster_connected(
            new_atoms, connectivity_factor
        ):
            disconnection_distance, suggested_factor, analysis_msg = (
                analyze_disconnection(new_atoms, connectivity_factor)
            )
            current_composition = new_atoms.get_chemical_symbols()
            current_counts = get_composition_counts(current_composition)
            remaining_counts = get_composition_counts(atoms_to_add)
            diagnostics = get_structure_diagnostics(
                new_atoms, min_distance_factor, connectivity_factor
            )

            additional_info = (
                f"Cluster became disconnected after batch placement "
                f"({len(new_atoms)}/{total_target} atoms placed).\n"
                f"Current state: {len(new_atoms)} atoms, composition counts: {current_counts}\n"
                f"Remaining atoms: {len(atoms_to_add)} atoms, composition counts: {remaining_counts}\n"
                f"Analysis: {analysis_msg}\n"
                f"Suggested connectivity_factor: {suggested_factor:.2f}"
            )

            error_msg = format_placement_error_message(
                context="maintain connectivity after batch placement",
                composition=None,
                n_atoms=None,
                placement_radius_scaling=placement_radius_scaling,
                min_distance_factor=min_distance_factor,
                connectivity_factor=connectivity_factor,
                diagnostics=diagnostics,
                additional_info=additional_info,
            )
            logger.warning(error_msg)
            return None

        # Reset batch attempt counter on successful placement
        if valid_placements:
            batch_attempt = 0

    return new_atoms
