"""Template structure generators for high-symmetry cluster motifs.

This module provides functions to generate regular polyhedral structures
(icosahedra, decahedra, octahedra) using ASE's cluster module. These templates
are used in the smart initialization mode to ensure exploration of important
high-symmetry basins in the potential energy surface.

Based on the Doye group's comprehensive study of Morse/LJ clusters:
https://doye.chem.ox.ac.uk/jon/structures/Morse/paper/node5.html

Polyhedra scaling:
------------------
All template structures are scaled so the **longest vertex–vertex distance**
(or longest edge) equals **(r_i + r_j) × connectivity_factor**, with
a = 2 × avg covalent radius for the composition.

- Custom templates (Tetrahedron, Cube, Cuboctahedron, Truncated Octahedron):
  Apply this rule directly: geometric parameters (edge lengths, vertex
  positions) are derived from a and connectivity_factor so the longest
  edge ≤ a × connectivity_factor.

- ASE-based templates (Icosahedron, Decahedron, Octahedron):
  ASE scales using atomic radii. We rescale after generation via
  _rescale_cluster_to_bond_length so nn distances match covalent-based
  bond length, keeping structures within the connectivity threshold.

- Validation:
  All templates are validated with validate_cluster() (connectivity
  (r_i + r_j) × connectivity_factor, minimum distances × min_distance_factor).
  No magic numbers; all thresholds derive from covalent radii.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from logging import Logger
from typing import Any, cast

import numpy as np
from ase import Atom, Atoms
from ase.cluster import Decahedron, Icosahedron, Octahedron
from ase.cluster.cluster import Cluster
from ase.symbols import Symbols
from numpy.random import Generator

from scgo.utils.helpers import get_composition_counts
from scgo.utils.logging import get_logger
from scgo.utils.rng_helpers import ensure_rng_or_create

from .geometry_helpers import (
    _assign_exact_composition,
    _check_composition_feasibility,
    _cycle_composition_to_length,
    _generate_batch_positions_on_convex_hull,
    _identify_safe_removal_candidates,
    _should_check_connectivity,
    _verify_exact_composition,
    clear_convex_hull_cache,
    compute_bond_distance_params,
    get_convex_hull_vertex_indices,
    get_covalent_radius,
    validate_cluster,
    validate_cluster_structure,
)
from .initialization_config import (
    BOND_DISTANCE_MULTIPLIER_2ATOM,
    BOND_DISTANCE_MULTIPLIER_3ATOM,
    CONNECTIVITY_FACTOR,
    MAGIC_NUMBER_TOLERANCE,
    MAGIC_NUMBERS,
    MIN_DISTANCE_FACTOR_DEFAULT,
    PLACEMENT_RADIUS_SCALING_DEFAULT,
    POSITION_COMPARISON_TOLERANCE_FACTOR,
    VACUUM_DEFAULT,
)
from .random_spherical import grow_from_seed

logger: Logger = get_logger(__name__)

# =============================================================================
# CONSTANTS & REGISTRY
# =============================================================================

ICOSAHEDRON_SHELL_TO_ATOMS: dict[int, int] = {1: 1, 2: 13, 3: 55, 4: 147, 5: 309}

_TEMPLATE_REGISTRY = {}

# =============================================================================
# CORE HELPERS
# =============================================================================

def _get_base_element(composition: list[str]) -> str:
    """Get the base element from composition.

    Args:
        composition: List of element symbols

    Returns:
        First element symbol

    Raises:
        ValueError: If composition is empty
    """
    if not composition:
        raise ValueError("Cannot get base element from empty composition")
    return composition[0]


def _get_typical_bond_length(composition: list[str]) -> float:
    """Calculate typical bond length from composition using covalent radii.

    Computes the average covalent radius for all unique elements in the
    composition and returns twice that value as the typical bond length.
    This provides element-specific scaling for template structures.

    Args:
        composition: List of element symbols

    Returns:
        Typical bond length in Angstroms (2 × average covalent radius)

    Raises:
        ValueError: If composition is empty
    """
    if not composition:
        raise ValueError("Cannot calculate bond length from empty composition")

    unique_elements: set[str] = set(composition)
    radii: list[float] = [get_covalent_radius(elem) for elem in unique_elements]
    avg_radius: float = sum(radii) / len(radii)
    return 2.0 * avg_radius


def get_nearest_magic_number(n_atoms: int) -> int | None:
    """Find the nearest magic number to the given atom count.

    Args:
        n_atoms: Number of atoms in the cluster

    Returns:
        The nearest magic number, or None if no magic numbers are defined
    """
    if not MAGIC_NUMBERS:
        return None

    nearest: int = min(MAGIC_NUMBERS, key=lambda x: abs(x - n_atoms))
    return nearest


def is_near_magic_number(n_atoms: int, tolerance: int = MAGIC_NUMBER_TOLERANCE) -> bool:
    """Check if the atom count is near a magic number.

    Args:
        n_atoms: Number of atoms in the cluster
        tolerance: Maximum difference from magic number to be considered "near"

    Returns:
        True if n_atoms is within tolerance of any magic number
    """
    nearest: int | None = get_nearest_magic_number(n_atoms)
    if nearest is None:
        return False
    return abs(n_atoms - nearest) <= tolerance


def _find_icosahedron_shells(n_atoms: int) -> int | None:
    """Find the noshells parameter for an icosahedron closest to n_atoms.

    Args:
        n_atoms: Target number of atoms

    Returns:
        noshells parameter, or None if no suitable match
    """
    for noshells, count in ICOSAHEDRON_SHELL_TO_ATOMS.items():
        if count == n_atoms:
            return noshells

    closest_shell = None
    min_diff = float("inf")
    for noshells, count in ICOSAHEDRON_SHELL_TO_ATOMS.items():
        diff: int = abs(count - n_atoms)
        if diff < min_diff:
            min_diff = diff
            closest_shell = noshells

    return closest_shell


def _find_decahedron_params(n_atoms: int) -> tuple[int, int, int] | None:
    """Find decahedron parameters (p, q, r) closest to n_atoms.

    Note: Uses "Pt" as a placeholder element symbol for parameter search only.
    The actual cluster will be created with the correct element later.

    Args:
        n_atoms: Target number of atoms

    Returns:
        Tuple of (p, q, r) parameters, or None if no suitable match
    """
    best_params = None
    min_diff = float("inf")

    for p in range(1, 6):
        for q in range(1, 6):
            for r in [0, 1]:
                try:
                    cluster: Atoms = Decahedron(symbol="Pt", p=p, q=q, r=r)
                    count: int = len(cluster)
                    diff: int = abs(count - n_atoms)
                    if diff < min_diff:
                        min_diff = diff
                        best_params = (p, q, r)
                except (ValueError, RuntimeError, TypeError):
                    continue

    return best_params


def _find_octahedron_params(n_atoms: int) -> tuple[int, int] | None:
    """Find octahedron parameters (length, cutoff) closest to n_atoms.

    Note: Uses "Pt" as a placeholder element symbol for parameter search only.
    The actual cluster will be created with the correct element later.

    Args:
        n_atoms: Target number of atoms

    Returns:
        Tuple of (length, cutoff) parameters, or None if no suitable match
    """
    best_params = None
    min_diff = float("inf")

    for length in range(1, 8):
        max_cutoff: int = (length - 1) // 2
        for cutoff in range(max_cutoff + 1):
            try:
                cluster: Cluster = Octahedron(symbol="Pt", length=length, cutoff=cutoff)
                count: int = len(cluster)
                diff: int = abs(count - n_atoms)
                if diff < min_diff:
                    min_diff = diff
                    best_params = (length, cutoff)
            except (ValueError, RuntimeError, TypeError):
                continue

    return best_params


def remove_atoms_from_vertices(
    cluster: Atoms,
    n_remove: int,
    target_composition: list[str] | None = None,
    connectivity_factor: float = CONNECTIVITY_FACTOR,
    min_distance_factor: float = MIN_DISTANCE_FACTOR_DEFAULT,
    rng: np.random.Generator | None = None,
) -> Atoms | None:
    """Remove atoms from convex-hull vertices in bulk.

    Uses hull.vertices as the only candidate set. Orders by distance from COM
    (descending), respects target_composition when given, and supports
    multi-round removal when n_remove exceeds the number of vertices.
    Single validation per round (no per-atom connectivity loop).

    Args:
        cluster: The cluster to remove atoms from.
        n_remove: Number of atoms to remove.
        target_composition: Optional; preserves exact element counts in result.
        connectivity_factor: Factor for connectivity threshold when validating.
        min_distance_factor: Factor for minimum distance checks when validating.
        rng: Optional RNG.

    Returns:
        New Atoms with atoms removed, or None if removal fails (e.g. <4 atoms,
        cannot satisfy composition from vertices, or validation fails).
    """
    rng = ensure_rng_or_create(rng)
    if n_remove <= 0:
        return cluster.copy()
    if n_remove >= len(cluster):
        raise ValueError(
            f"Cannot remove {n_remove} atoms from cluster with {len(cluster)} atoms"
        )
    if len(cluster) < 4:
        return None

    initial_len: int = len(cluster)
    final_total: int = initial_len - n_remove
    base_composition = cluster.get_chemical_symbols()

    final_counts: Counter[str] | None
    if target_composition is not None:
        final_composition: list[str] = _cycle_composition_to_length(
            target_composition, final_total
        )
        final_counts = get_composition_counts(final_composition)

        # Check composition feasibility before attempting removal
        is_feasible, _ = _check_composition_feasibility(
            base_composition, final_composition, operation="reduce"
        )
        if not is_feasible:
            return None
    else:
        final_counts = None

    current: Atoms = cluster.copy()
    total_removed = 0

    while total_removed < n_remove:
        vertices: np.ndarray[tuple[Any, ...], np.dtype[Any]] = (
            get_convex_hull_vertex_indices(current)
        )
        if len(vertices) == 0:
            return None

        positions: Any | np.ndarray[tuple[Any, ...], np.dtype[Any]] = (
            current.get_positions()
        )
        symbols = current.get_chemical_symbols()
        center: np.ndarray[tuple[Any, ...], np.dtype[Any]] | Any = (
            current.get_center_of_mass()
        )
        distances = np.linalg.norm(positions - center, axis=1)
        remaining_to_remove: int = n_remove - total_removed

        coordination: np.ndarray = np.zeros(len(current), dtype=np.int_)
        if len(current) > 1:
            for i in range(len(current)):
                r_i: float = get_covalent_radius(symbols[i])
                for j in range(len(current)):
                    if i == j:
                        continue
                    r_j: float = get_covalent_radius(symbols[j])
                    d: np.floating[Any] = np.linalg.norm(positions[i] - positions[j])
                    if d <= (r_i + r_j) * connectivity_factor:
                        coordination[i] += 1

        vert_coord: np.ndarray = coordination[vertices]
        vert_dist = distances[vertices]
        # Add random noise for tie-breaking
        noise: np.ndarray[tuple[Any, ...], np.dtype[np.float64]] = rng.random(
            len(vertices)
        )
        order = np.lexsort((noise, -vert_dist, vert_coord))
        sorted_vertices: np.ndarray[tuple[Any, ...], np.dtype[Any]] = vertices[order]

        # Prepare variables for removal logic
        remove_counts: dict[str, int] | None = None
        total_to_remove_this: int

        if target_composition is not None:
            assert final_counts is not None
            current_counts: Counter[str] = get_composition_counts(symbols)
            rc: dict[str, int] = {}
            for el in set(symbols) | set(final_counts.keys()):
                cur: int = current_counts.get(el, 0)
                fin: int = final_counts.get(el, 0)
                rc[el] = max(0, cur - fin)
            remove_counts = {k: v for k, v in rc.items() if v > 0}
            total_to_remove_this = sum(remove_counts.values())
            if total_to_remove_this != remaining_to_remove:
                raise ValueError(
                    f"Cannot preserve target composition: need to remove "
                    f"{total_to_remove_this} atoms this round but "
                    f"{remaining_to_remove} remaining. "
                    f"Target: {target_composition}, current: {symbols}"
                )
        else:
            remove_counts = None
            total_to_remove_this = remaining_to_remove

        remove_indices: list[int]
        to_remove_this: int

        if target_composition is not None:
            remove_indices = []
            assert remove_counts is not None
            for el, count in remove_counts.items():
                el_verts: list[int] = [
                    int(i) for i in sorted_vertices if symbols[i] == el
                ]
                remove_indices.extend(el_verts[:count])
            if not remove_indices and total_to_remove_this > 0:
                return None
            to_remove_this = min(total_to_remove_this, len(remove_indices), 1)
            remove_indices = remove_indices[:to_remove_this]
        else:
            max_remove_this: int = max(1, len(sorted_vertices) // 2)
            to_remove_this = min(remaining_to_remove, max_remove_this)
            remove_indices = sorted_vertices[:to_remove_this].tolist()

        safe_candidates: list[int] = _identify_safe_removal_candidates(
            current,
            remove_indices,
            connectivity_factor,
            max_to_check=len(remove_indices),
        )

        if not safe_candidates:
            return None

        remove_indices = safe_candidates[:to_remove_this]

        keep: np.ndarray[tuple[Any, ...], np.dtype[Any]] = np.setdiff1d(
            np.arange(len(current)), remove_indices
        )
        new_symbols: list[Symbols | str] = [current.symbols[i] for i in keep]
        new_positions: Any | np.ndarray[tuple[Any, ...], np.dtype[Any]] = positions[
            keep
        ]
        new_cluster = Atoms(
            symbols=new_symbols,
            positions=new_positions,
            cell=current.get_cell(),
            pbc=current.get_pbc(),
        )

        is_valid, err = validate_cluster_structure(
            new_cluster,
            min_distance_factor,
            connectivity_factor,
            check_clashes=True,
            check_connectivity=_should_check_connectivity(new_cluster),
        )
        if not is_valid:
            return None

        clear_convex_hull_cache()
        current = new_cluster
        total_removed += len(remove_indices)
        if not remove_indices:
            break

    if target_composition is not None:
        actual: Counter[str] = get_composition_counts(current.get_chemical_symbols())
        if actual != final_counts:
            raise ValueError(
                f"Vertex removal did not preserve exact composition. "
                f"Expected {final_composition} (counts: {final_counts}), "
                f"got counts {actual}."
            )

    return current


def grow_template_via_facets(
    seed_atoms: Atoms,
    target_composition: list[str],
    placement_radius_scaling: float,
    cell_side: float,
    rng: np.random.Generator,
    min_distance_factor: float = MIN_DISTANCE_FACTOR_DEFAULT,
    connectivity_factor: float = CONNECTIVITY_FACTOR,
    template_name: str | None = None,
) -> Atoms | None:
    """Grow a template seed to target composition by placing atoms on all facets.

    One hull per round, one position per facet, single validation per round.
    Falls back to grow_from_seed when seed has <4 atoms (no 3D hull).

    Args:
        seed_atoms: The template seed to grow from.
        target_composition: Target composition as list of element symbols.
        placement_radius_scaling: Scaling for placement shell radius.
        cell_side: Cubic cell side length.
        rng: Random number generator.
        min_distance_factor: Factor for minimum distance checks.
        connectivity_factor: Factor for connectivity threshold.
        template_name: Optional name of template type (e.g., "cuboctahedron").
                       Used to enable smart facet filtering for specific templates.

    Returns:
        Grown Atoms with target composition, or None on failure.
    """
    base: Atoms = seed_atoms.copy()
    base_composition = base.get_chemical_symbols()

    is_feasible, _ = _check_composition_feasibility(
        base_composition, target_composition, operation="grow"
    )
    if not is_feasible:
        return None

    base_counts: Counter[str] = get_composition_counts(base_composition)
    target_counts: Counter[str] = get_composition_counts(target_composition)
    atoms_to_add: list[str] = list((target_counts - base_counts).elements())

    if not atoms_to_add:
        base.set_cell([cell_side, cell_side, cell_side])
        base.center()
        return base

    if len(base) < 4:
        return grow_from_seed(
            seed_atoms=base,
            target_composition=target_composition,
            placement_radius_scaling=placement_radius_scaling,
            cell_side=cell_side,
            rng=rng,
            min_distance_factor=min_distance_factor,
            connectivity_factor=connectivity_factor,
        )

    current: Atoms = base.copy()
    to_add: list[str] = list(atoms_to_add)
    radii_to_add: dict[str, float] = {s: get_covalent_radius(s) for s in set(to_add)}

    max_round_retries = 3
    round_retry_count = 0

    while to_add:
        symbols = current.get_chemical_symbols()
        current_radii: np.ndarray[tuple[Any, ...], np.dtype[Any]] = np.array(
            [get_covalent_radius(s) for s in symbols]
        )
        max_existing: float = (
            float(np.max(current_radii)) if len(current_radii) > 0 else 0.0
        )
        avg_new = float(np.mean([radii_to_add[s] for s in to_add]))
        bond_distance, min_dist, max_conn = compute_bond_distance_params(
            max_existing,
            avg_new,
            connectivity_factor,
            min_distance_factor,
            placement_radius_scaling,
        )

        new_atom_symbol: str | None = to_add[0] if to_add else None
        use_smart_filtering: bool = template_name == "cuboctahedron"

        candidates: list[np.ndarray[tuple[Any, ...], np.dtype[Any]]] = (
            _generate_batch_positions_on_convex_hull(
                current,
                n_candidates=0,
                bond_distance=bond_distance,
                rng=rng,
                min_connectivity_dist=min_dist,
                max_connectivity_dist=max_conn,
                use_all_facets=True,
                min_distance_factor=min_distance_factor,
                new_atom_symbol=new_atom_symbol,
                smart_facet_filtering=use_smart_filtering,
                connectivity_factor=connectivity_factor,
            )
        )
        if not candidates:
            logger.debug(
                f"grow_template_via_facets: no candidates generated for {template_name}, "
                f"n_atoms={len(current)}, to_add={len(to_add)}, "
                f"bond_distance={bond_distance:.3f}, max_conn={max_conn:.3f}"
                f" (discovery failure: candidate discarded; not a per-structure fallback)"
            )
            return None

        placed_count = 0
        candidate_idx = 0

        while placed_count < len(to_add) and candidate_idx < len(candidates):
            sym: str = to_add[placed_count]
            pos: np.ndarray[tuple[Any, ...], np.dtype[Any]] = candidates[candidate_idx]
            candidate_idx += 1

            test_atom = Atom(sym, pos)
            has_clash = False
            for existing_atom in current:
                dist: np.floating[Any] = np.linalg.norm(
                    test_atom.position - existing_atom.position
                )
                r_new: float = get_covalent_radius(sym)
                r_existing: float = get_covalent_radius(existing_atom.symbol)
                min_allowed: float = (r_new + r_existing) * min_distance_factor
                if dist < min_allowed:
                    has_clash = True
                    break

            if not has_clash:
                current.append(test_atom)
                placed_count += 1

        to_add = to_add[placed_count:]

        if placed_count == 0 and to_add and round_retry_count < max_round_retries:
            round_retry_count += 1
            logger.debug(
                f"grow_template_via_facets: no atoms placed, retry {round_retry_count}/{max_round_retries}, "
                f"candidates={len(candidates)}, to_add={len(to_add)}"
            )
            clear_convex_hull_cache()
            continue
        elif placed_count == 0 and to_add:
            logger.debug(
                f"grow_template_via_facets: failed to place any atoms after {max_round_retries} retries, "
                f"candidates={len(candidates)}, to_add={len(to_add)}"
                f" (discovery failure: candidate discarded; not a per-structure fallback)"
            )
            return None

        round_retry_count = 0

        is_valid, err = validate_cluster_structure(
            current,
            min_distance_factor,
            connectivity_factor,
            check_clashes=True,
            check_connectivity=_should_check_connectivity(current),
        )
        if not is_valid:
            logger.debug(
                f"grow_template_via_facets: validation failed after placing {placed_count} atoms, "
                f"n_atoms={len(current)}, error: {err}"
            )
            return None

        clear_convex_hull_cache()

    current.set_cell([cell_side, cell_side, cell_side])
    current.center()

    if not _verify_exact_composition(current, target_composition):
        expected: Counter[str] = get_composition_counts(target_composition)
        actual: Counter[str] = get_composition_counts(current.get_chemical_symbols())
        raise ValueError(
            f"grow_template_via_facets produced wrong composition: "
            f"expected {target_composition} (counts {expected}), "
            f"got counts {actual}"
        )

    return current


def _create_balanced_base_composition(
    composition: list[str], base_n_atoms: int
) -> list[str]:
    """Create a balanced base composition by cycling through elements.

    For multi-element compositions, this ensures the base template has
    a balanced distribution of elements, making it easier to adjust to
    the target composition by adding, removing, or switching labels.

    Args:
        composition: Target composition list
        base_n_atoms: Number of atoms in the base template

    Returns:
        List of element symbols with balanced distribution

    Raises:
        ValueError: If composition is empty
    """
    if not composition:
        raise ValueError(
            f"Cannot create balanced base composition from empty composition "
            f"for {base_n_atoms} atoms"
        )

    if len(composition) == 1:
        return composition * base_n_atoms

    return [composition[i % len(composition)] for i in range(base_n_atoms)]


def _assign_balanced_composition_if_multi(
    cluster: Atoms, composition: list[str]
) -> None:
    """Assign balanced composition to cluster if composition has multiple elements.

    For multi-element compositions, assigns a balanced distribution of elements
    to the cluster. Single-element compositions are left unchanged.

    Args:
        cluster: The Atoms object to assign composition to
        composition: Target composition list
    """
    if len(composition) > 1:
        base_symbols: list[str] = _create_balanced_base_composition(
            composition, len(cluster)
        )
        cluster.set_chemical_symbols(base_symbols)


def _deduplicate_positions(
    positions: list[np.ndarray] | list[list[float]], bond_length: float
) -> list[np.ndarray]:
    """Remove duplicate positions from a list using bond-length-based tolerance.

    Args:
        positions: List of position arrays or lists
        bond_length: Typical bond length for calculating tolerance

    Returns:
        List of unique positions as numpy arrays
    """
    position_tolerance: float = bond_length * POSITION_COMPARISON_TOLERANCE_FACTOR
    unique_positions: list[np.ndarray] = []
    for pos in positions:
        pos_array: np.ndarray[tuple[Any, ...], np.dtype[Any]] = np.array(pos)
        if not any(
            np.allclose(pos_array, up, atol=position_tolerance)
            for up in unique_positions
        ):
            unique_positions.append(pos_array)
    return unique_positions


def _validate_n_atoms(n_atoms: int, expected: int | None, template_name: str) -> bool:
    """Validate that n_atoms matches expected value for a template.

    Args:
        n_atoms: Number of atoms to validate
        expected: Expected number of atoms (None means no specific requirement)
        template_name: Name of template for error messages

    Returns:
        True if valid, False otherwise (logs debug message if invalid)
    """
    if n_atoms <= 0:
        return False
    # Silently skip - this is expected during template discovery
    # Debug logging here creates excessive noise
    return expected is None or n_atoms == expected


def _generate_custom_template(
    template_name: str,
    composition: list[str],
    n_atoms: int,
    position_generator: Callable[
        [list[str], int, float, float], list[np.ndarray] | list[list[float]]
    ],
    rng: np.random.Generator | None = None,
    connectivity_factor: float = CONNECTIVITY_FACTOR,
    n_atoms_validator: Callable[[int], tuple[bool, str | None]] | None = None,
    post_process: Callable[[Atoms, list[str], int], Atoms] | None = None,
    expected_n_atoms: int | None = None,
) -> Atoms | None:
    """Generate a custom template structure using a position generator function.

    This wrapper handles the common boilerplate for custom template generators:
    validation, RNG setup, error handling, composition assignment, and adjustment.

    Args:
        template_name: Name of template type (e.g., "tetrahedron")
        composition: List of element symbols
        n_atoms: Target number of atoms
        position_generator: Function that takes (composition, n_atoms, bond_length, connectivity_factor)
            and returns list of positions
        rng: Optional random number generator
        connectivity_factor: Factor for connectivity threshold
        n_atoms_validator: Optional function that takes n_atoms and returns (is_valid, error_msg)
        post_process: Optional function to modify cluster after creation (takes cluster, composition, n_atoms)
        expected_n_atoms: Optional expected atom count for validation (uses _validate_n_atoms)

    Returns:
        Atoms object with template structure, or None if generation fails
    """
    if expected_n_atoms is not None:
        if not _validate_n_atoms(n_atoms, expected_n_atoms, template_name):
            return None
    elif n_atoms <= 0:
        return None

    if n_atoms_validator is not None:
        is_valid, _ = n_atoms_validator(n_atoms)
        if not is_valid:
            return None

    rng = ensure_rng_or_create(rng)

    try:
        base_element: str = _get_base_element(composition)
        a: float = _get_typical_bond_length(composition)
        positions: (
            list[np.ndarray[tuple[Any, ...], np.dtype[Any]]] | list[list[float]]
        ) = position_generator(composition, n_atoms, a, connectivity_factor)

        cluster = Atoms([base_element] * len(positions), positions=positions)

        if post_process is not None:
            cluster = post_process(cluster, composition, n_atoms)

        _assign_balanced_composition_if_multi(cluster, composition)

        adjusted = _adjust_template_to_target(
            cluster,
            n_atoms,
            composition,
            rng,
            template_name,
            connectivity_factor,
            MIN_DISTANCE_FACTOR_DEFAULT,
        )
        if adjusted is None:
            return None
        return adjusted

    except (ValueError, RuntimeError, AttributeError):
        return None


def _rescale_cluster_to_bond_length(
    atoms: Atoms,
    composition: list[str],
    connectivity_factor: float,
) -> None:
    """Rescale ASE-generated cluster so nn distances match covalent-based bond length.

    ASE Icosahedron, Decahedron, Octahedron use atomic radii. We validate with
    covalent radii × connectivity_factor. Rescaling ensures nn distances align
    with our connectivity model so structures pass validation.

    Modifies atoms in place. Keeps center of mass fixed.

    Args:
        atoms: ASE-generated cluster (e.g. Icosahedron, Decahedron, Octahedron).
        composition: Target composition (used for typical bond length).
        connectivity_factor: Connectivity factor; kept for API consistency.
    """
    if len(atoms) < 2:
        return
    a: float = _get_typical_bond_length(composition)
    positions: Any | np.ndarray[tuple[Any, ...], np.dtype[Any]] = atoms.get_positions()
    n: int = len(positions)
    min_dists: list[np.floating[Any]] = [
        min(np.linalg.norm(positions[i] - positions[j]) for j in range(n) if j != i)
        for i in range(n)
    ]
    current_scale = float(np.mean(min_dists))
    if current_scale <= 0:
        return
    target_scale: float = a * min(1.0, connectivity_factor * 0.95)
    scale: float = target_scale / current_scale
    com = np.mean(positions, axis=0)
    new_positions = com + (positions - com) * scale
    atoms.set_positions(new_positions)


def _register_ase_template(
    template_name: str,
    find_params_func: Callable[[int], Any],
    generate_base_func: Callable[[str, Any], Atoms],
) -> None:
    """Register an ASE-based template generator in the template registry.

    Args:
        template_name: Name of the template type
        find_params_func: Function to find parameters for the template
        generate_base_func: Function to generate base cluster from parameters
    """
    _TEMPLATE_REGISTRY[template_name] = {
        "find_params": find_params_func,
        "generate_base": generate_base_func,
    }


_register_ase_template(
    "icosahedron",
    find_params_func=_find_icosahedron_shells,
    generate_base_func=lambda elem, params: Icosahedron(symbol=elem, noshells=params),
)

_register_ase_template(
    "decahedron",
    find_params_func=_find_decahedron_params,
    generate_base_func=lambda elem, params: Decahedron(
        symbol=elem, p=params[0], q=params[1], r=params[2]
    ),
)

_register_ase_template(
    "octahedron",
    find_params_func=_find_octahedron_params,
    generate_base_func=lambda elem, params: Octahedron(
        symbol=elem, length=params[0], cutoff=params[1]
    ),
)


def _generate_ase_template_with_common_pattern(
    composition: list[str],
    n_atoms: int,
    rng: np.random.Generator | None,
    template_name: str,
    find_params_func: Callable[[int], Any],
    generate_base_func: Callable[[str, Any], Atoms],
    connectivity_factor: float = CONNECTIVITY_FACTOR,
) -> Atoms | None:
    """Helper function for ASE-based template generators with common pattern.

    This function handles the common pattern used by icosahedron, decahedron,
    and octahedron generators:
    1. Validate n_atoms
    2. Ensure RNG exists
    3. Find parameters (with early return if None)
    4. Generate base cluster
    5. Assign balanced composition if multi-element
    6. Adjust atom count and composition

    Note on scaling: ASE's Icosahedron, Decahedron, and Octahedron scale using
    atomic radii. We rescale via _rescale_cluster_to_bond_length so nn distances
    match covalent-based bond length (a = 2×avg covalent radius), ensuring
    structures stay within (r_i + r_j) × connectivity_factor before validation.

    Args:
        composition: List of element symbols
        n_atoms: Target number of atoms
        rng: Optional random number generator
        template_name: Name of template type (for error messages)
        find_params_func: Function to find parameters for the template (returns params or None)
        generate_base_func: Function to generate base cluster from params (returns Atoms)
        connectivity_factor: Factor for connectivity threshold (based on covalent radii)

    Returns:
        Atoms object with template structure, or None if generation fails
    """
    if n_atoms <= 0:
        return None

    rng = ensure_rng_or_create(rng)

    params = find_params_func(n_atoms)
    if params is None:
        return None

    try:
        base_element: str = _get_base_element(composition)
        cluster: Atoms = generate_base_func(base_element, params)

        _rescale_cluster_to_bond_length(cluster, composition, connectivity_factor)
        _assign_balanced_composition_if_multi(cluster, composition)

        adjusted = _adjust_template_to_target(
            cluster,
            n_atoms,
            composition,
            rng,
            template_name,
            connectivity_factor,
            MIN_DISTANCE_FACTOR_DEFAULT,
        )
        if adjusted is None:
            return None
        return adjusted

    except (ValueError, RuntimeError, AttributeError):
        return None


def _set_template_info(atoms: Atoms, template_name: str) -> None:
    """Set template type in atoms info for tracking.

    Args:
        atoms: The Atoms object to set info on
        template_name: Name of template type (e.g., "icosahedron")
    """
    if atoms.info is None:
        atoms.info = {}
    atoms.info["template_type"] = template_name


def _adjust_template_to_target(
    cluster: Atoms,
    target_n_atoms: int,
    composition: list[str],
    rng: np.random.Generator,
    template_name: str,
    connectivity_factor: float = CONNECTIVITY_FACTOR,
    min_distance_factor: float = MIN_DISTANCE_FACTOR_DEFAULT,
    cell_side: float | None = None,
    placement_radius_scaling: float = PLACEMENT_RADIUS_SCALING_DEFAULT,
) -> Atoms | None:
    """Shared helper to adjust template cluster to target atom count and composition.

    Handles three cases: grow (add atoms), shrink (remove atoms), or exact match.
    This consolidates the common adjustment logic used throughout template generation.

    Args:
        cluster: The base template cluster
        target_n_atoms: Target number of atoms
        composition: Target composition
        rng: Random number generator
        template_name: Name of template type (for error messages)
        connectivity_factor: Factor for connectivity threshold (default: CONNECTIVITY_FACTOR)
        min_distance_factor: Factor for minimum distance checks
            (default: MIN_DISTANCE_FACTOR_DEFAULT)
        cell_side: Cell side length (defaults to cluster cell or VACUUM_DEFAULT * 2)
        placement_radius_scaling: Scaling for placement radius (for growth).

    Returns:
        Adjusted Atoms object, or None if adjustment fails
    """
    base_count: int = len(cluster)

    if cell_side is None:
        cell_side = (
            cluster.cell.lengths()[0] if cluster.cell.any() else VACUUM_DEFAULT * 2
        )

    if base_count < target_n_atoms:
        target_composition: list[str] = _cycle_composition_to_length(
            composition, target_n_atoms
        )

        try:
            grown = grow_template_via_facets(
                seed_atoms=cluster,
                target_composition=target_composition,
                placement_radius_scaling=placement_radius_scaling,
                cell_side=cell_side,
                rng=rng,
                min_distance_factor=min_distance_factor,
                connectivity_factor=connectivity_factor,
                template_name=template_name,
            )
            if grown is None:
                logger.debug(
                    f"Failed to add atoms to {template_name} template "
                    f"while maintaining connectivity"
                    f" (discovery failure: candidate discarded; not a per-structure fallback)"
                )
                return None
            if len(grown) != target_n_atoms:
                logger.debug(
                    f"{template_name} template has {len(grown)} atoms after growth, "
                    f"expected {target_n_atoms}"
                )
                return None
            _set_template_info(grown, template_name)
            return grown
        except ValueError as e:
            logger.debug(
                f"Failed to grow {template_name} template from {base_count} "
                f"to {target_n_atoms} atoms: {e}"
            )
            return None
    elif base_count > target_n_atoms:
        n_remove: int = base_count - target_n_atoms
        adjusted = remove_atoms_from_vertices(
            cluster,
            n_remove,
            target_composition=composition,
            connectivity_factor=connectivity_factor,
            min_distance_factor=min_distance_factor,
            rng=rng,
        )

        if adjusted is None:
            return None

        cluster = adjusted
        _set_template_info(cluster, template_name)
    else:
        cluster = _assign_exact_composition(
            cluster, composition, target_n_atoms, rng=rng
        )
        _set_template_info(cluster, template_name)

    return cluster


def _generate_ase_template_from_registry(
    template_name: str,
    composition: list[str],
    n_atoms: int,
    rng: np.random.Generator | None = None,
    connectivity_factor: float = CONNECTIVITY_FACTOR,
) -> Atoms | None:
    """Helper to generate ASE-based template from registry.

    Args:
        template_name: Name of template type (e.g., "icosahedron")
        composition: List of element symbols
        n_atoms: Target number of atoms
        rng: Optional random number generator
        connectivity_factor: Factor for connectivity threshold

    Returns:
        Atoms object with template structure, or None if generation fails
    """
    config = _TEMPLATE_REGISTRY.get(template_name)
    if config is None:
        logger.warning(f"{template_name.capitalize()} template not registered")
        return None

    find_params = cast(Callable[[int], Any], config["find_params"])
    generate_base = cast(Callable[[str, Any], Atoms], config["generate_base"])

    return _generate_ase_template_with_common_pattern(
        composition=composition,
        n_atoms=n_atoms,
        rng=rng,
        template_name=template_name,
        find_params_func=find_params,
        generate_base_func=generate_base,
        connectivity_factor=connectivity_factor,
    )


def generate_icosahedron(
    composition: list[str],
    n_atoms: int,
    rng: np.random.Generator | None = None,
    connectivity_factor: float = CONNECTIVITY_FACTOR,
) -> Atoms | None:
    """Generate an icosahedral cluster.

    Uses ASE's Icosahedron generator and adjusts atom count by adding/removing
    surface atoms if needed.

    Args:
        composition: List of element symbols (cycled to match n_atoms)
        n_atoms: Target number of atoms
        rng: Optional random number generator for reproducibility
        connectivity_factor: Factor for connectivity threshold

    Returns:
        Atoms object with icosahedral structure, or None if generation fails
    """
    return _generate_ase_template_from_registry(
        "icosahedron", composition, n_atoms, rng, connectivity_factor
    )


def generate_decahedron(
    composition: list[str],
    n_atoms: int,
    rng: np.random.Generator | None = None,
    connectivity_factor: float = CONNECTIVITY_FACTOR,
) -> Atoms | None:
    """Generate a decahedral cluster.

    Uses ASE's Decahedron generator and adjusts atom count by adding/removing
    surface atoms if needed.

    Args:
        composition: List of element symbols (cycled to match n_atoms)
        n_atoms: Target number of atoms
        rng: Optional random number generator for reproducibility
        connectivity_factor: Factor for connectivity threshold

    Returns:
        Atoms object with decahedral structure, or None if generation fails
    """
    return _generate_ase_template_from_registry(
        "decahedron", composition, n_atoms, rng, connectivity_factor
    )


def generate_octahedron(
    composition: list[str],
    n_atoms: int,
    rng: np.random.Generator | None = None,
    connectivity_factor: float = CONNECTIVITY_FACTOR,
) -> Atoms | None:
    """Generate an octahedral cluster.

    Uses ASE's Octahedron generator and adjusts atom count by adding/removing
    surface atoms if needed.

    Args:
        composition: List of element symbols (cycled to match n_atoms)
        n_atoms: Target number of atoms
        rng: Optional random number generator for reproducibility
        connectivity_factor: Factor for connectivity threshold

    Returns:
        Atoms object with octahedral structure, or None if generation fails
    """
    return _generate_ase_template_from_registry(
        "octahedron", composition, n_atoms, rng, connectivity_factor
    )


def generate_tetrahedron(
    composition: list[str],
    n_atoms: int,
    rng: np.random.Generator | None = None,
    connectivity_factor: float = CONNECTIVITY_FACTOR,
) -> Atoms | None:
    """Generate a tetrahedral cluster with the specified number of atoms.

    Creates a regular tetrahedron with atoms at vertices. Only supports 4 atoms
    (the vertices of a regular tetrahedron).

    Args:
        composition: List of element symbols (cycled to match n_atoms)
        n_atoms: Target number of atoms (must be 4)
        rng: Optional random number generator for reproducibility

    Returns:
        Atoms object with tetrahedral structure, or None if generation fails
        (e.g., n_atoms != 4)
    """

    def _generate_tetrahedron_positions(
        comp: list[str], n: int, bond_length: float, cf: float
    ) -> list[np.ndarray]:
        return [
            np.array([0.0, 0.0, 0.0]),
            np.array([bond_length, 0.0, 0.0]),
            np.array([bond_length / 2, bond_length * np.sqrt(3) / 2, 0.0]),
            np.array(
                [
                    bond_length / 2,
                    bond_length / (2 * np.sqrt(3)),
                    bond_length * np.sqrt(2 / 3),
                ]
            ),
        ]

    return _generate_custom_template(
        template_name="tetrahedron",
        composition=composition,
        n_atoms=n_atoms,
        position_generator=_generate_tetrahedron_positions,
        rng=rng,
        connectivity_factor=connectivity_factor,
        expected_n_atoms=4,
    )


def generate_cube(
    composition: list[str],
    n_atoms: int,
    rng: np.random.Generator | None = None,
    connectivity_factor: float = CONNECTIVITY_FACTOR,
) -> Atoms | None:
    """Generate a cubic cluster with the specified number of atoms.

    Creates cubic structures (n×n×n cubes) for perfect cube sizes only.
    Only supports perfect cubes (8, 27, 64, 125, etc.).

    Args:
        composition: List of element symbols (cycled to match n_atoms)
        n_atoms: Target number of atoms (must be a perfect cube: n³)
        rng: Optional random number generator for reproducibility

    Returns:
        Atoms object with cubic structure, or None if generation fails
        (e.g., n_atoms is not a perfect cube)
    """

    def _validate_cube(n: int) -> tuple[bool, str | None]:
        """Validate that n_atoms is a perfect cube."""
        cube_root = round(n ** (1 / 3))
        if cube_root**3 == n:
            return True, None
        return False, (
            f"generate_cube only supports perfect cubes (n³), got {n}. "
            f"Returning None instead of falling back to other template."
        )

    def _generate_cube_positions(
        comp: list[str], n: int, bond_length: float, cf: float
    ) -> list[np.ndarray]:
        """Generate positions for n×n×n cubic lattice."""
        cube_root = round(n ** (1 / 3))
        return [
            np.array([i * bond_length, j * bond_length, k * bond_length])
            for i in range(cube_root)
            for j in range(cube_root)
            for k in range(cube_root)
        ]

    return _generate_custom_template(
        template_name="cube",
        composition=composition,
        n_atoms=n_atoms,
        position_generator=_generate_cube_positions,
        rng=rng,
        connectivity_factor=connectivity_factor,
        n_atoms_validator=_validate_cube,
    )


def generate_cuboctahedron(
    composition: list[str],
    n_atoms: int,
    rng: np.random.Generator | None = None,
    connectivity_factor: float = CONNECTIVITY_FACTOR,
) -> Atoms | None:
    """Generate a cuboctahedral cluster with the specified number of atoms.

    Cuboctahedron has 12 vertices. For 13 atoms, adds a center atom.

    Args:
        composition: List of element symbols (cycled to match n_atoms)
        n_atoms: Target number of atoms (12 or 13 for perfect structures)
        rng: Optional random number generator for reproducibility

    Returns:
        Atoms object with cuboctahedral structure, or None if generation fails
    """

    def _generate_cuboctahedron_positions(
        comp: list[str], n: int, bond_length: float, cf: float
    ) -> list[np.ndarray]:
        s: float = bond_length * cf / 2.0
        positions = []
        for sign1 in [-1, 1]:
            for sign2 in [-1, 1]:
                positions.append([sign1 * s, sign2 * s, 0.0])
                positions.append([sign1 * s, 0.0, sign2 * s])
                positions.append([0.0, sign1 * s, sign2 * s])
        return _deduplicate_positions(positions, bond_length)

    def _post_process_cuboctahedron(cluster: Atoms, comp: list[str], n: int) -> Atoms:
        """Add center atom for 13-atom cuboctahedron."""
        if n == 13:
            base_element: str = _get_base_element(comp)
            center_pos: np.ndarray[tuple[Any, ...], np.dtype[Any]] = np.array(
                [0.0, 0.0, 0.0]
            )
            cluster.append(Atom(base_element, center_pos))
        return cluster

    return _generate_custom_template(
        template_name="cuboctahedron",
        composition=composition,
        n_atoms=n_atoms,
        position_generator=_generate_cuboctahedron_positions,
        rng=rng,
        connectivity_factor=connectivity_factor,
        post_process=_post_process_cuboctahedron,
    )


def generate_truncated_octahedron(
    composition: list[str],
    n_atoms: int,
    rng: np.random.Generator | None = None,
    connectivity_factor: float = CONNECTIVITY_FACTOR,
) -> Atoms | None:
    """Generate a truncated octahedral cluster with the specified number of atoms.

    Truncated octahedron has 24 vertices (6 square faces, 8 hexagonal faces).
    Only supports 24 atoms (the vertices of a truncated octahedron).

    Args:
        composition: List of element symbols (cycled to match n_atoms)
        n_atoms: Target number of atoms (must be 24)
        rng: Optional random number generator for reproducibility

    Returns:
        Atoms object with truncated octahedral structure, or None if generation fails
        (e.g., n_atoms != 24 or position generation doesn't yield exactly 24 positions)
    """

    def _generate_truncated_octahedron_positions(
        comp: list[str], n: int, bond_length: float, cf: float
    ) -> list[np.ndarray]:
        s: float = bond_length * cf / 2.0
        positions: list[list[float]] = []
        for x_sign in [-1, 1]:
            for y_sign in [-1, 1]:
                for z_sign in [-1, 1]:
                    minus_count: int = sum([x_sign < 0, y_sign < 0, z_sign < 0])
                    if minus_count % 2 == 0:
                        for perm in [
                            [2 * s * x_sign, s * y_sign, 0],
                            [2 * s * x_sign, 0, s * z_sign],
                            [s * x_sign, 2 * s * y_sign, 0],
                            [s * x_sign, 0, 2 * s * z_sign],
                            [0, 2 * s * y_sign, s * z_sign],
                            [0, s * y_sign, 2 * s * z_sign],
                        ]:
                            if not any(np.allclose(perm, p) for p in positions):
                                positions.append(perm)
        unique_positions: list[np.ndarray[tuple[Any, ...], np.dtype[Any]]] = (
            _deduplicate_positions(positions, bond_length)
        )
        if len(unique_positions) != 24:
            raise ValueError(
                f"generate_truncated_octahedron requires exactly 24 positions, "
                f"got {len(unique_positions)}"
            )
        return unique_positions[:24]

    return _generate_custom_template(
        template_name="truncated_octahedron",
        composition=composition,
        n_atoms=n_atoms,
        position_generator=_generate_truncated_octahedron_positions,
        rng=rng,
        connectivity_factor=connectivity_factor,
        expected_n_atoms=24,
    )


_TEMPLATE_GENERATORS: dict[str, Callable[..., Atoms | None]] = {
    "icosahedron": generate_icosahedron,
    "decahedron": generate_decahedron,
    "cuboctahedron": generate_cuboctahedron,
    "truncated_octahedron": generate_truncated_octahedron,
    "octahedron": generate_octahedron,
    "cube": generate_cube,
    "tetrahedron": generate_tetrahedron,
}


def generate_template_structure(
    composition: list[str],
    n_atoms: int,
    template_type: str = "auto",
    rng: np.random.Generator | None = None,
    connectivity_factor: float = CONNECTIVITY_FACTOR,
) -> Atoms | None:
    """Generate a template structure of the specified type.

    Args:
        composition: List of element symbols
        n_atoms: Target number of atoms
        template_type: Type of template. Can be:
            - "auto": Automatically select best template type
            - "icosahedron": Icosahedral structure
            - "decahedron": Decahedral structure
            - "octahedron": Octahedral structure
            - "tetrahedron": Tetrahedral structure
            - "cube": Cubic structure
            - "cuboctahedron": Cuboctahedral structure
            - "truncated_octahedron": Truncated octahedral structure
        rng: Optional random number generator

    Returns:
        Atoms object with template structure, or None if generation fails
    """
    if template_type == "auto":
        preferred_order: list[str] = [
            "icosahedron",
            "decahedron",
            "cuboctahedron",
            "truncated_octahedron",
            "octahedron",
            "cube",
            "tetrahedron",
        ]
        for template_name in preferred_order:
            gen_func = _TEMPLATE_GENERATORS.get(template_name)
            if gen_func is not None:
                result: Atoms | None = gen_func(
                    composition, n_atoms, rng, connectivity_factor
                )
                if result is not None:
                    return result
        return None

    gen_func = _TEMPLATE_GENERATORS.get(template_type)
    if gen_func is None:
        logger.warning(f"Unknown template type: {template_type}")
        return None
    return gen_func(composition, n_atoms, rng, connectivity_factor)


def _find_valid_template_types(
    n_atoms: int, rng: np.random.Generator | None = None
) -> list[str]:
    """Find all template types that can successfully generate a structure with n_atoms.

    Args:
        n_atoms: Target number of atoms
        rng: Optional random number generator

    Returns:
        List of template type names that can generate this size
    """
    rng = ensure_rng_or_create(rng)

    valid_types = []
    test_composition: list[str] = ["Pt"] * n_atoms
    sorted_template_types: list[str] = sorted(_TEMPLATE_GENERATORS.keys())

    for template_type in sorted_template_types:
        gen_func: Callable[..., Atoms | None] = _TEMPLATE_GENERATORS[template_type]
        try:
            result: Atoms | None = gen_func(test_composition, n_atoms, rng)
            if result is not None and len(result) == n_atoms:
                valid_types.append(template_type)
        except (ValueError, RuntimeError, TypeError):
            continue

    return sorted(valid_types)


def _generate_template_with_atom_adjustment(
    base_template_type: str,
    base_n_atoms: int,
    target_n_atoms: int,
    composition: list[str],
    rng: np.random.Generator,
    cell_side: float | None = None,
    placement_radius_scaling: float = PLACEMENT_RADIUS_SCALING_DEFAULT,
    min_distance_factor: float = MIN_DISTANCE_FACTOR_DEFAULT,
    connectivity_factor: float = CONNECTIVITY_FACTOR,
) -> Atoms | None:
    """Generate a template structure and adjust atom count to match target.

    Uses seed growth functions to add/remove atoms from the surface.

    Args:
        base_template_type: Template type to start from
        base_n_atoms: Number of atoms in the base template
        target_n_atoms: Target number of atoms
        composition: Target composition
        rng: Random number generator
        cell_side: Cell side length (defaults to VACUUM_DEFAULT * 2)
        placement_radius_scaling: Scaling for atom placement
        min_distance_factor: Factor for minimum distance checks
        connectivity_factor: Factor for connectivity threshold

    Returns:
        Atoms object with target composition, or None if generation fails
    """
    if cell_side is None:
        cell_side = VACUUM_DEFAULT * 2

    if target_n_atoms <= 0:
        return None

    # Generate base template
    if not composition:
        raise ValueError(
            f"Cannot generate template with empty composition for {target_n_atoms} atoms"
        )
    elif len(composition) >= base_n_atoms:
        base_composition = composition[:base_n_atoms]
    else:
        base_composition = _cycle_composition_to_length(composition, base_n_atoms)

    gen_func: Callable[..., Atoms | None] | None = _TEMPLATE_GENERATORS.get(
        base_template_type
    )
    if gen_func is None:
        return None

    base_cluster: Atoms | None = gen_func(base_composition, base_n_atoms, rng)
    if base_cluster is None:
        return None

    base_cluster.set_cell([cell_side, cell_side, cell_side])
    base_cluster.center()

    n_diff: int = target_n_atoms - base_n_atoms

    if n_diff == 0:
        result: Atoms = _assign_exact_composition(
            base_cluster, composition, target_n_atoms, rng=rng
        )
        return result

    if n_diff < 0:
        n_remove: int = -n_diff
        removal_ratio: float = n_remove / base_n_atoms
        if removal_ratio >= 0.5:
            return None

        max_removal_attempts: int = min(3, base_n_atoms)
        for attempt in range(max_removal_attempts):
            attempt_rng: Generator = (
                rng if attempt == 0 else np.random.default_rng(rng.integers(0, 2**31))
            )
            adjusted = remove_atoms_from_vertices(
                base_cluster,
                n_remove,
                target_composition=composition,
                connectivity_factor=connectivity_factor,
                min_distance_factor=min_distance_factor,
                rng=attempt_rng,
            )
            if adjusted is None:
                if attempt < max_removal_attempts - 1:
                    continue
                return None
            return adjusted
        return None

    adjusted = _adjust_template_to_target(
        cluster=base_cluster,
        target_n_atoms=target_n_atoms,
        composition=composition,
        rng=rng,
        template_name=base_template_type,
        connectivity_factor=connectivity_factor,
        min_distance_factor=min_distance_factor,
        cell_side=cell_side,
        placement_radius_scaling=placement_radius_scaling,
    )
    if adjusted is not None:
        return adjusted
    return None


def _validate_template_geometry(atoms: Atoms) -> bool:
    """Validate that a template structure has reasonable geometry.

    Filters out templates with atoms that are unreasonably far apart or too close.
    This ensures templates are physically reasonable starting structures.

    Args:
        atoms: The Atoms object to validate

    Returns:
        True if geometry is reasonable, False otherwise
    """
    if len(atoms) <= 1:
        return True

    positions: Any | np.ndarray[tuple[Any, ...], np.dtype[Any]] = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    distances = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            dist: np.floating[Any] = np.linalg.norm(positions[i] - positions[j])
            distances.append(dist)

    if not distances:
        return True

    min_dist = min(distances)
    max_dist = max(distances)

    if len(atoms) <= 3:
        max_covalent_sum = 0.0
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                r_i = get_covalent_radius(symbols[i])
                r_j = get_covalent_radius(symbols[j])
                max_covalent_sum = max(max_covalent_sum, r_i + r_j)

        # For 2-atom clusters, use strict criteria: within 1.2x sum of covalent radii
        max_reasonable_distance: float = (
            BOND_DISTANCE_MULTIPLIER_2ATOM * max_covalent_sum
            if len(atoms) == 2
            else BOND_DISTANCE_MULTIPLIER_3ATOM * max_covalent_sum
        )

        if max_dist > max_reasonable_distance:
            return False

    min_covalent_sum = float("inf")
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            r_i = get_covalent_radius(symbols[i])
            r_j = get_covalent_radius(symbols[j])
            min_covalent_sum = min(min_covalent_sum, r_i + r_j)

    min_allowed_distance: float = min_covalent_sum * MIN_DISTANCE_FACTOR_DEFAULT
    return not (min_dist < min_allowed_distance)


def _validate_and_add_template(
    atoms: Atoms,
    results: list[Atoms],
    template_type: str,
    template_description: str,
    min_distance_factor: float,
    connectivity_factor: float,
    logger_instance: Any = None,
) -> bool:
    """Validate a template structure and add it to results if valid.

    This helper consolidates the common validation pattern used in template
    generation functions. It performs geometry validation and cluster structure
    validation, then adds valid templates to the results list.

    Args:
        atoms: The Atoms object to validate
        results: List to append valid templates to
        template_type: Type of template (e.g., "icosahedron")
        template_description: Description string for logging (e.g., "for 13 atoms")
        min_distance_factor: Factor for minimum distance checks
        connectivity_factor: Factor for connectivity threshold
        logger_instance: Logger instance (defaults to module logger)

    Returns:
        True if template was added to results, False otherwise
    """
    if logger_instance is None:
        logger_instance = logger

    if not _validate_template_geometry(atoms):
        return False

    validated_atoms, is_valid, error_message = validate_cluster(
        atoms,
        composition=None,
        min_distance_factor=min_distance_factor,
        connectivity_factor=connectivity_factor,
        check_clashes=True,
        check_connectivity=None,
        sort_atoms=False,
        raise_on_failure=False,
        source="",
    )

    if is_valid:
        _set_template_info(validated_atoms, template_type)
        results.append(validated_atoms)
        return True
    else:
        return False


def generate_template_matches(
    composition: list[str],
    n_atoms: int,
    rng: np.random.Generator | None = None,
    cell_side: float | None = None,
    placement_radius_scaling: float = PLACEMENT_RADIUS_SCALING_DEFAULT,
    min_distance_factor: float = MIN_DISTANCE_FACTOR_DEFAULT,
    connectivity_factor: float = CONNECTIVITY_FACTOR,
    include_exact: bool = True,
    include_near: bool = True,
) -> list[Atoms]:
    """Generate template structures for the target size.

    Provides both exact and near-match template generation in a single interface.
    Exact matches when n_atoms is a magic number; near matches by adjusting from
    the nearest magic number.

    Args:
        composition: Target composition
        n_atoms: Target number of atoms
        rng: Optional random number generator
        cell_side: Cell side length (needed for near matches with growth)
        placement_radius_scaling: Scaling for atom placement (needed for near matches)
        min_distance_factor: Factor for minimum distance checks
        connectivity_factor: Factor for connectivity threshold
        include_exact: If True, generate exact matches when n_atoms is a magic number
        include_near: If True, generate near matches from nearest magic number

    Returns:
        List of Atoms objects with template structures
    """
    rng = ensure_rng_or_create(rng)
    results: list[Atoms] = []

    nearest_magic: int | None = get_nearest_magic_number(n_atoms)
    if nearest_magic is None:
        return results

    is_exact_match: bool = nearest_magic == n_atoms

    if include_exact and is_exact_match:
        valid_types = _find_valid_template_types(n_atoms, rng)
        for template_type in valid_types:
            try:
                atoms = _TEMPLATE_GENERATORS[template_type](
                    composition, n_atoms, rng, connectivity_factor
                )
                if atoms is not None and len(atoms) == n_atoms:
                    assigned = _assign_exact_composition(
                        atoms, composition, n_atoms, rng=rng
                    )
                    if assigned is not None:
                        _validate_and_add_template(
                            atoms=assigned,
                            results=results,
                            template_type=template_type,
                            template_description=f"for {n_atoms} atoms",
                            min_distance_factor=min_distance_factor,
                            connectivity_factor=connectivity_factor,
                        )
            except (ValueError, RuntimeError, AttributeError, TypeError, KeyError) as e:
                logger.debug(
                    "Template generation failed for %s (n_atoms=%s): %s: %s",
                    template_type,
                    n_atoms,
                    type(e).__name__,
                    e,
                )

    if include_near and not is_exact_match:
        valid_types = _find_valid_template_types(nearest_magic, rng)
        for template_type in valid_types:
            try:
                adjusted: Atoms | None = _generate_template_with_atom_adjustment(
                    base_template_type=template_type,
                    base_n_atoms=nearest_magic,
                    target_n_atoms=n_atoms,
                    composition=composition,
                    rng=rng,
                    cell_side=cell_side,
                    placement_radius_scaling=placement_radius_scaling,
                    min_distance_factor=min_distance_factor,
                    connectivity_factor=connectivity_factor,
                )
                if adjusted is not None and len(adjusted) == n_atoms:
                    _validate_and_add_template(
                        atoms=adjusted,
                        results=results,
                        template_type=template_type,
                        template_description=f"({nearest_magic} -> {n_atoms})",
                        min_distance_factor=min_distance_factor,
                        connectivity_factor=connectivity_factor,
                    )
            except (ValueError, RuntimeError, AttributeError, TypeError, KeyError) as e:
                logger.debug(
                    "Template adjustment failed for %s (base=%s, target=%s): %s: %s",
                    template_type,
                    nearest_magic,
                    n_atoms,
                    type(e).__name__,
                    e,
                )

    return results
