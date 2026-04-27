"""Initial cluster structure generation with seed-based growth strategies.

This module provides functions for creating initial cluster structures for global
optimization, including intelligent seed selection from previous runs and adaptive
growth strategies based on available candidates.
"""

from __future__ import annotations

import glob
import itertools
import logging
import os
import re
import sqlite3
import threading
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, vdw_radii

from scgo.database import get_connection as db_connection
from scgo.database.cache import get_global_cache
from scgo.database.metadata import get_metadata as _get_db_metadata
from scgo.utils.helpers import (
    extract_energy_from_atoms,
    get_cluster_formula,
    get_composition_counts,
)
from scgo.utils.logging import get_logger
from scgo.utils.parallel_workers import resolve_n_jobs_to_workers
from scgo.utils.validation import validate_composition

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
    SEED_BASE_PCT,
    SEED_COMBINATION_STRATEGY_COUNT,
    SEED_PREFACTOR,
    TEMPLATE_BASE_PCT,
    TEMPLATE_BASE_WEIGHTS,
    TEMPLATE_DIVERSITY_BOOST_FACTOR,
    TEMPLATE_PREFACTOR,
    TEMPLATE_ROTATION_CANDIDATES,
    VACUUM_DEFAULT,
)
from .random_spherical import grow_from_seed, random_spherical
from .seed_combiners import combine_and_grow
from .templates import generate_template_matches

logger = get_logger(__name__)

CandidateEntry = tuple[tuple[str, ...], float, Atoms]


class InitStrategy(Enum):
    """Initialization strategies used by allocation and generation logic."""

    TEMPLATE = "template"
    SEED_GROWTH = "seed+growth"
    RANDOM_SPHERICAL = "random_spherical"


def is_composition_subset(
    subset_counts: Counter[str] | dict[str, int],
    target_counts: Counter[str] | dict[str, int],
) -> bool:
    """Check if subset_counts is a subset of target_counts.

    A composition is considered a subset if all element counts in subset_counts
    are less than or equal to the corresponding counts in target_counts.

    Args:
        subset_counts: Counter or dict of element counts for the subset composition.
        target_counts: Counter or dict of element counts for the target composition.

    Returns:
        True if subset_counts is a subset of target_counts, False otherwise.
    """
    return all(
        subset_counts.get(el, 0) <= target_counts.get(el, 0) for el in subset_counts
    )


# Cache namespaces for direct use with UnifiedCache
_COMPOSITION_CACHE_NS = "composition"
_TEMPLATE_ROTATIONS_CACHE_NS = "template_rotations"
_DB_CANDIDATES_CACHE_NS = "db_candidates"

# Lock to protect mtime caching for database files
_DB_MTIME_LOCK = threading.Lock()

# Cache of canonical mtimes for database files to ensure stable cache keys
# Opening a database can change its mtime, so we cache the first mtime we see
_DB_CANONICAL_MTIME: dict[str, float] = {}

# Soft cap on relaxed candidates per formula (prevents memory blow-up)
_MAX_CANDIDATES_PER_FORMULA = 10000

# Bump this version to invalidate older caches when the candidate selection
# behavior changes (e.g., toggling how 'final_unique_minimum' is interpreted).
_FIND_SMALLER_CANDIDATES_CACHE_VERSION = 3


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


def _safe_mtime(path: str) -> float:
    """Return file mtime or 0.0 on error. Used for cache signature without loading."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _load_candidates_from_file(db_file: str, mtime: float) -> list[CandidateEntry]:
    """Load candidates from file. Cached by filename and mtime (thread-safe)."""
    cache_key = (db_file, mtime)

    def compute_candidates():
        entries: list[CandidateEntry] = []
        try:
            with db_connection(db_file) as da:
                all_relaxed = da.get_all_relaxed_candidates()
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
            logger.warning(
                f"Failed to load database {db_file}: {type(e).__name__}: {e}"
            )
            return []

        for row in all_relaxed:
            try:
                symbols = tuple(row.get_chemical_symbols())
                energy = extract_energy_from_atoms(row)
                if energy is None or not np.isfinite(energy):
                    continue
            except (AttributeError, TypeError):
                continue

            try:
                atoms_copy = row.copy()
                if len(atoms_copy) != len(row):
                    continue
            except (AttributeError, TypeError, RuntimeError) as e:
                logger.debug(
                    "Skipped candidate: failed to copy Atoms - %s: %s",
                    type(e).__name__,
                    e,
                )
                continue

            entries.append((symbols, energy, atoms_copy))

        return entries

    return get_global_cache().get_or_compute(
        _DB_CANDIDATES_CACHE_NS, cache_key, compute_candidates
    )


def invalidate_db_canonical_mtime(db_file: str | None = None) -> None:
    """Invalidate cached canonical mtime(s) for database files.

    If ``db_file`` is None, clears the entire canonical mtime cache. Otherwise
    removes the single entry for ``db_file`` if present. Thread-safe.
    """
    with _DB_MTIME_LOCK:
        if db_file is None:
            _DB_CANONICAL_MTIME.clear()
        else:
            _DB_CANONICAL_MTIME.pop(db_file, None)


def _load_db_candidates(db_file: str) -> tuple[float, list[CandidateEntry]]:
    """Load relaxed candidates from an ASE database file with thread-safe caching.

    Args:
        db_file: Path to database file

    Returns:
        Tuple of (modification_time, list of candidate entries). Returns (0.0, [])
        if file cannot be accessed or is invalid.
    """
    # Use cached mtime if available to ensure stable cache keys
    # (opening a database can modify its mtime, so we need consistency)
    with _DB_MTIME_LOCK:
        cur_mtime = _safe_mtime(db_file)
        if db_file in _DB_CANONICAL_MTIME:
            cached = _DB_CANONICAL_MTIME[db_file]
            if cached != cur_mtime:
                # File changed since canonical mtime was set — refresh it so
                # per-file caches keyed by mtime do not return stale content.
                logger.debug(
                    f"Refreshing canonical mtime for {db_file}: cached={cached} current={cur_mtime}"
                )
                if cur_mtime == 0.0:
                    return 0.0, []
                _DB_CANONICAL_MTIME[db_file] = cur_mtime
                mtime = cur_mtime
            else:
                mtime = cached
        else:
            if cur_mtime == 0.0:
                return 0.0, []
            _DB_CANONICAL_MTIME[db_file] = cur_mtime
            mtime = cur_mtime

    # Use thread-safe cache via helper
    entries = _load_candidates_from_file(db_file, mtime)
    return mtime, entries


def _parse_composition_from_path(path: str) -> list[str] | None:
    """Parse composition from directory path name.

    Attempts to extract composition information from directory names like:
    - "Pt10_searches" -> ["Pt"] * 10
    - "AuPt_searches" -> ["Au", "Pt"]
    - "Au2Pt_searches" -> ["Au", "Au", "Pt"]
    - "Pt4_searches" -> ["Pt"] * 4

    Args:
        path: File or directory path that may contain composition info.

    Returns:
        List of element symbols if composition can be parsed, None otherwise.
    """
    # Look for pattern like "Pt10_searches", "AuPt_searches", etc.
    # Extract the part before "_searches"
    parts = path.split("/")
    for part in parts:
        if "_searches" in part:
            comp_str = part.replace("_searches", "")
            # Try to parse composition string
            # Pattern: element symbols optionally followed by numbers
            # e.g., "Pt10", "AuPt", "Au2Pt", "Pt4Au3"
            composition = []
            # Match element symbols (1-2 letters) followed by optional numbers
            pattern = r"([A-Z][a-z]?)(\d*)"
            matches = re.findall(pattern, comp_str)
            if matches:
                for element, count_str in matches:
                    count = int(count_str) if count_str else 1
                    composition.extend([element] * count)
                return composition
    return None


def _could_path_contain_relevant_candidates(
    db_path: str, target_counts: Counter[str]
) -> bool:
    """Check if a database path could potentially contain relevant candidates.

    Uses directory name hints to skip databases that definitely won't have
    sub-compositions of the target. This is a fast pre-filter before loading
    the database.

    Args:
        db_path: Path to database file.
        target_counts: Counter of target composition element counts.

    Returns:
        True if the path might contain relevant candidates, False if it can
        be safely skipped.
    """
    parsed_comp = _parse_composition_from_path(db_path)
    if parsed_comp is None:
        # Can't parse composition from path, so we can't skip it
        # (might be in a different directory structure)
        return True

    parsed_counts = get_composition_counts(parsed_comp)

    # Path relevant if elements are subset of target and counts <= target
    for element, count in parsed_counts.items():
        if element not in target_counts:
            # Path contains elements not in target - skip it
            return False
        if count > target_counts[element]:
            return False

    # Path composition is a subset or could contain relevant sub-clusters
    return True


def _compute_files_signature(files: list[str]) -> tuple[tuple[str, float], ...]:
    """Compute a signature for a list of files based on modification times.

    Args:
        files: List of file paths.

    Returns:
        Tuple of (file_path, mtime) tuples, sorted by file path.
    """
    return tuple(sorted((f, _safe_mtime(f)) for f in files))


def _find_smaller_candidates(
    target_composition: list[str],
    db_glob_pattern: str,
) -> dict[str, list[tuple[float, Atoms]]]:
    """Find all relaxed database candidates that are sub-compositions of target.

    This function scans database files, finds all relaxed structures that are
    sub-compositions of the target, groups them by their chemical formula,
    and returns them sorted by energy (lowest first).

    Note:
        Seed selection uses only structures explicitly tagged with
        ``final_unique_minimum`` (final minima). Non-final relaxed candidates
        are intentionally ignored for seed/template selection — this behavior
        is relied upon by the initialization and tests.

    Uses directory name hints to skip databases that definitely won't contain
    relevant candidates, significantly speeding up the search.

    Args:
        target_composition: The composition of the desired final cluster.
        db_glob_pattern: Glob pattern to find database files (e.g., '**/*.db').

    Returns:
        A dictionary where keys are chemical formulas (e.g., "Pt3") and
        values are lists of (energy, Atoms) tuples for all suitable candidates
        of that formula found in the databases.

    """
    cwd = os.getcwd()
    matches = glob.glob(os.path.join(cwd, db_glob_pattern), recursive=True)
    candidates_by_formula: dict[str, list[tuple[float, Atoms]]] = {}
    target_counts = get_composition_counts(target_composition)
    n_target_atoms = len(target_composition)

    logger.debug(
        f"Found {len(matches)} total database files matching pattern '{db_glob_pattern}'"
    )

    # Pre-filter database files using directory name hints
    # This can skip a large number of irrelevant databases
    filtered_matches = [
        db_file
        for db_file in matches
        if _could_path_contain_relevant_candidates(db_file, target_counts)
    ]

    if len(filtered_matches) < len(matches):
        logger.debug(
            f"Filtered {len(matches)} -> {len(filtered_matches)} database files "
            f"using directory name hints (skipped {len(matches) - len(filtered_matches)} irrelevant files)"
        )

    # Compute signature once and include it in the cache key
    # This eliminates the need for post-load verification (TOCTOU protection via cache key itself)
    signature_tuple = _compute_files_signature(filtered_matches)
    # Include an explicit version token so changes to candidate selection
    # logic invalidate previous cached results.
    cache_key = (
        _FIND_SMALLER_CANDIDATES_CACHE_VERSION,
        tuple(sorted(target_counts.items())),
        db_glob_pattern,
        signature_tuple,
    )
    cached_entry = get_global_cache().get(_COMPOSITION_CACHE_NS, cache_key)
    if cached_entry is not None:
        stale_cache = False
        for entries in cached_entry.values():
            for _energy, atom in entries:
                if not _get_db_metadata(atom, "final_unique_minimum", False):
                    stale_cache = True
                    break
            if stale_cache:
                break

        if not stale_cache:
            logger.debug(
                f"Using cached seed candidates for {get_cluster_formula(target_composition)}"
            )
            return {
                formula: [(energy, atom.copy()) for energy, atom in entries]
                for formula, entries in cached_entry.items()
            }
        else:
            logger.debug(
                "Cached seed candidates appear stale (non-final entries present); recomputing"
            )

    if filtered_matches:
        logger.info(
            f"Loading seed candidates from {len(filtered_matches)} database files"
        )

    # Process databases one at a time instead of loading all into memory first
    # This prevents memory bloat and ensures file handles are released promptly
    total_candidates_found = 0
    for idx, db_file in enumerate(filtered_matches):
        if idx > 0 and idx % 50 == 0:
            logger.debug(
                f"Processed {idx}/{len(filtered_matches)} seed database files, found {total_candidates_found} candidates so far"
            )

        try:
            _mtime, entries = _load_db_candidates(db_file)
        except (
            sqlite3.DatabaseError,
            sqlite3.OperationalError,
            OSError,
            ValueError,
            RuntimeError,
        ) as e:
            logger.warning(
                "Failed to load seed database %s: %s",
                os.path.basename(db_file),
                e,
            )
            continue

        # Process entries from this database immediately, then release them
        for symbols, energy, atoms in entries:
            # Only consider structures explicitly tagged as final unique minima
            # for seed generation. This ensures that seeds come from validated
            # final structures rather than transient intermediate candidates.
            if not _get_db_metadata(atoms, "final_unique_minimum", False):
                continue

            if len(symbols) >= n_target_atoms:
                continue

            row_counts = get_composition_counts(list(symbols))
            # Check if row_counts is a subset of target_counts
            if not is_composition_subset(row_counts, target_counts):
                continue

            formula = get_cluster_formula(list(symbols))
            if formula not in candidates_by_formula:
                candidates_by_formula[formula] = []
            candidates_by_formula[formula].append((energy, atoms))
            total_candidates_found += 1

    # Post-process per formula: sort by energy (ascending) and apply a soft cap
    # on the number of stored candidates to avoid unbounded memory usage.
    processed: dict[str, list[tuple[float, Atoms]]] = {}
    for formula, entries in candidates_by_formula.items():
        # Sort by energy (low to high) so lower-energy seeds are sampled more
        # often, while still allowing stochastic selection via Boltzmann
        # weighting in the caller.
        sorted_entries = sorted(entries, key=lambda e: e[0])
        deduped_entries = _deduplicate_seed_candidates(sorted_entries)
        if len(deduped_entries) < len(sorted_entries):
            logger.debug(
                f"Deduplicated seed candidates for {formula}: "
                f"{len(sorted_entries)} -> {len(deduped_entries)} unique structures"
            )
        if len(deduped_entries) > _MAX_CANDIDATES_PER_FORMULA:
            deduped_entries = deduped_entries[:_MAX_CANDIDATES_PER_FORMULA]
        processed[formula] = deduped_entries

    # Thread-safe cache update (signature is in cache key, so no re-verification needed)
    get_global_cache().set(_COMPOSITION_CACHE_NS, cache_key, processed)

    return {
        formula: [(energy, atom.copy()) for energy, atom in entries]
        for formula, entries in processed.items()
    }


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


def _get_structure_signature(atoms: Atoms, precision: int = 4) -> tuple[float, ...]:
    """Create a signature based on sorted interatomic distances.

    This signature uniquely identifies structures with identical geometry,
    regardless of atom ordering or composition assignment.

    Args:
        atoms: Atoms object to compute signature for
        precision: Decimal precision for rounding distances

    Returns:
        Tuple of sorted interatomic distances (rounded to precision)
    """
    positions = atoms.get_positions()
    distances = [
        np.linalg.norm(positions[i] - positions[j])
        for i in range(len(positions))
        for j in range(i + 1, len(positions))
    ]
    return tuple(np.round(np.sort(distances), precision))


def _deduplicate_seed_candidates(
    entries: list[tuple[float, Atoms]],
    precision: int = 4,
    energy_bin: float | None = None,
) -> list[tuple[float, Atoms]]:
    """Deduplicate seed candidates by geometry signature.

    Entries are assumed to be for a single chemical formula. We keep the
    lowest-energy representative for each unique structure signature.

    Uses an energy-bin prefilter to reduce the number of expensive structure
    signatures computed when many near-identical seeds exist.

    Args:
        entries: List of (energy, Atoms) tuples for a single formula.
        precision: Decimal precision used for structure signatures.
        energy_bin: Energy bin size (same units as energy) for prefiltering.
            If None, a bin size is chosen to split the energy range into
            approximately 100 bins.

    Returns:
        Deduplicated list of (energy, Atoms) tuples.
    """
    if len(entries) <= 1:
        return entries

    if energy_bin is None:
        energies = [energy for energy, _ in entries]
        if not energies:
            return entries
        energy_range = max(energies) - min(energies)
        energy_bin = energy_range / 100.0 if energy_range > 0 else 0.0
    elif energy_bin <= 0:
        energy_bin = 0.0

    if energy_bin == 0.0:
        unique: dict[tuple[float, ...], tuple[float, Atoms]] = {}
        for energy, atoms in entries:
            signature = _get_structure_signature(atoms, precision=precision)
            if signature not in unique:
                unique[signature] = (energy, atoms)
        return list(unique.values())

    # Energy-bin prefilter: only compare structures within the same bin.
    binned: dict[int, list[tuple[float, Atoms]]] = {}
    for energy, atoms in entries:
        energy_key = int(round(energy / energy_bin))
        binned.setdefault(energy_key, []).append((energy, atoms))

    deduped: list[tuple[float, Atoms]] = []
    for bucket in binned.values():
        unique: dict[tuple[float, ...], tuple[float, Atoms]] = {}
        for energy, atoms in bucket:
            signature = _get_structure_signature(atoms, precision=precision)
            if signature not in unique:
                unique[signature] = (energy, atoms)
        deduped.extend(unique.values())

    return deduped


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


def _calculate_target_allocations(
    n_templates: int, n_seed_combinations: int, n_structures: int
) -> dict[str, int]:
    """Calculate target counts for each strategy based on logarithmic scaling."""
    targets = {"template": 0, "seed": 0}

    if n_templates > 0:
        template_scaling = TEMPLATE_BASE_PCT * np.log(
            1 + n_templates * TEMPLATE_PREFACTOR
        )
        target_template_raw = int(n_structures * template_scaling)
        targets["template"] = min(
            target_template_raw,
            2 * n_templates,  # Cap at 2 per template
            n_structures,
        )

    if n_seed_combinations > 0:
        seed_scaling = SEED_BASE_PCT * np.log(1 + n_seed_combinations * SEED_PREFACTOR)
        target_seed_raw = int(n_structures * seed_scaling)
        targets["seed"] = min(
            target_seed_raw,
            2 * n_seed_combinations,  # Cap at 2 per combination
            n_structures,
        )

    return targets


def _distribute_remaining(
    targets: dict[str, int],
    remaining: int,
    n_templates: int,
    n_seed_combinations: int,
) -> dict[str, int]:
    """Distribute remaining slots to templates and seeds up to caps."""
    if remaining <= 0:
        return targets

    # Prefer filling up to caps
    if n_templates > 0:
        template_cap = 2 * n_templates
        if targets["template"] < template_cap:
            add = min(remaining, template_cap - targets["template"])
            targets["template"] += add
            remaining -= add

    if remaining > 0 and n_seed_combinations > 0:
        seed_cap = 2 * n_seed_combinations
        if targets["seed"] < seed_cap:
            add = min(remaining, seed_cap - targets["seed"])
            targets["seed"] += add
            remaining -= add

    return targets


def _apply_guarantees(
    targets: dict[str, int],
    n_templates: int,
    n_seed_combinations: int,
    n_structures: int,
) -> dict[str, int]:
    """Apply minimum guarantees when structures >= options."""
    min_template = 0
    min_seed = 0

    if n_structures >= n_templates + n_seed_combinations:
        min_template = n_templates
        min_seed = n_seed_combinations
        targets["template"] = max(targets["template"], min_template)
        targets["seed"] = max(targets["seed"], min_seed)

    total_requested = targets["template"] + targets["seed"]

    if total_requested > n_structures:
        # Scale down if requested more than available
        if min_template > 0 or min_seed > 0:
            guaranteed = min_template + min_seed
            if guaranteed <= n_structures:
                excess_template = targets["template"] - min_template
                excess_seed = targets["seed"] - min_seed
                excess_total = excess_template + excess_seed

                if excess_total > 0:
                    available = n_structures - guaranteed
                    scale = available / excess_total
                    targets["template"] = min_template + int(excess_template * scale)
                    targets["seed"] = min_seed + int(excess_seed * scale)
                else:
                    targets["template"] = min_template
                    targets["seed"] = min_seed
            else:
                scale = n_structures / total_requested
                targets["template"] = int(targets["template"] * scale)
                targets["seed"] = int(targets["seed"] * scale)
        else:
            scale = n_structures / total_requested
            targets["template"] = int(targets["template"] * scale)
            targets["seed"] = int(targets["seed"] * scale)

        # If we have space left due to rounding, distribute it
        current_total = targets["template"] + targets["seed"]
        if current_total < n_structures:
            targets = _distribute_remaining(
                targets, n_structures - current_total, n_templates, n_seed_combinations
            )

    return targets


def _generate_allocations_list(
    targets: dict[str, int],
    n_structures: int,
    templates: list[Atoms],
    n_seed_combinations: int,
    rng: np.random.Generator,
) -> list[tuple[str, int | None]]:
    """Generate the list of allocation tuples from target counts."""
    allocations: list[tuple[str, int | None]] = []
    n_templates = len(templates)
    template_usage_count = [0] * n_templates

    # 1. Template allocations
    if n_templates > 0:
        # If we have enough structures, guarantee one of each template
        if n_structures >= n_templates:
            indices = list(range(n_templates))
            rng.shuffle(indices)
            for idx in indices:
                allocations.append((InitStrategy.TEMPLATE.value, idx))
                template_usage_count[idx] += 1

        # Fill remaining template slots based on target
        current_count = len(allocations)
        needed = targets["template"] - current_count

        if needed > 0:
            for _ in range(needed):
                # Weighted random selection favoring less used templates
                weights = [1.0 / (1 + c) for c in template_usage_count]
                probs = np.array(weights) / sum(weights)
                idx = rng.choice(n_templates, p=probs)
                allocations.append((InitStrategy.TEMPLATE.value, idx))
                template_usage_count[idx] += 1

    # 2. Seed allocations
    remaining = n_structures - len(allocations)
    seed_count = min(targets["seed"], remaining)

    if n_seed_combinations > 0 and n_structures >= n_templates + n_seed_combinations:
        seed_count = max(seed_count, n_seed_combinations)
        seed_count = min(seed_count, remaining)

    allocations.extend([(InitStrategy.SEED_GROWTH.value, None)] * seed_count)

    # 3. Random allocations
    remaining = n_structures - len(allocations)
    allocations.extend([(InitStrategy.RANDOM_SPHERICAL.value, None)] * remaining)

    return allocations


def _allocate_strategies_metropolis(
    n_structures: int,
    templates: list[Atoms],
    n_seed_formulas: int,
    n_seed_combinations: int,
    rng: np.random.Generator,
    n_atoms: int = 0,
) -> list[tuple[str, int | None]]:
    """Allocate structures across strategies using logarithmic scaling with caps.

    Allocation strategy uses symmetric logarithmic scaling for templates and seeds,
    allocating more structures when many options are available, while avoiding over-allocation:

    - Templates: Logarithmically scaled allocation (base_pct * log(1 + n_templates * prefactor))
                 capped at 2 per template. All templates are used at least once when
                 n_structures >= n_templates + n_seed_combinations.
    - Seed+growth: Logarithmically scaled allocation (base_pct * log(1 + n_combinations * prefactor))
                   capped at 2 per seed combination (symmetric to templates). At least
                   n_seed_combinations are allocated when n_structures >= n_templates + n_seed_combinations,
                   ensuring each combination can be used at least once. Returns 0 if no seeds available.
    - Random_spherical: Fills remaining slots as fallback.

    The logarithmic scaling ensures gradual increases in allocation as more options
    become available, preventing over-allocation while prioritizing structured initialization
    over random placement. Template and seed allocation are symmetric and consistent.

    Args:
        n_structures: Number of structures to generate
        templates: List of available template Atoms objects
        n_seed_formulas: Number of seed formula types available
        n_seed_combinations: Number of valid seed combinations
        rng: Random number generator

    Returns:
        List of (strategy_name, template_index) tuples for each structure.
        template_index is None for non-template strategies, or index into templates list.
    """
    n_templates = len(templates)

    # 1. Calculate initial targets
    targets = _calculate_target_allocations(
        n_templates, n_seed_combinations, n_structures
    )

    # 2. Apply guarantees and scaling
    targets = _apply_guarantees(targets, n_templates, n_seed_combinations, n_structures)

    # 3. Generate actual allocations list
    allocations = _generate_allocations_list(
        targets, n_structures, templates, n_seed_combinations, rng
    )

    # Calculate scaling values for debug message
    template_count = sum(1 for s, _ in allocations if s == InitStrategy.TEMPLATE.value)
    seed_count = sum(1 for s, _ in allocations if s == InitStrategy.SEED_GROWTH.value)
    random_count = sum(
        1 for s, _ in allocations if s == InitStrategy.RANDOM_SPHERICAL.value
    )

    total_allocated = template_count + seed_count + random_count

    # Only log for batch initialization (n_structures > 1)
    # Single-structure calls are typically for templates/seeds, not user-visible initialization
    if n_structures > 1:
        # Include discovery info when logging allocation
        logger.info(
            f"Initialization for {n_atoms}-atom clusters: "
            f"{n_templates} template(s), {n_seed_formulas} seed formula(s), "
            f"{n_seed_combinations} seed combination(s) available"
        )
        logger.info(
            f"Strategy allocation ({total_allocated} structures): "
            f"{template_count} template, {seed_count} seed+growth, {random_count} random"
        )
        if logger.isEnabledFor(logging.DEBUG):
            template_scaling_val = (
                TEMPLATE_BASE_PCT * np.log(1 + n_templates * TEMPLATE_PREFACTOR)
                if n_templates > 0
                else 0.0
            )
            seed_scaling_val = (
                SEED_BASE_PCT * np.log(1 + n_seed_combinations * SEED_PREFACTOR)
                if n_seed_combinations > 0
                else 0.0
            )
            logger.debug(
                f"  Detail: template {template_count}/{targets['template']} (scaling: {template_scaling_val:.3f}), "
                f"seed {seed_count}/{targets['seed']} (scaling: {seed_scaling_val:.3f}), "
                f"random {random_count}"
            )

    return allocations


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
    """Create multiple initial clusters with logarithmic scaling strategy allocation.

    This function generates N structures and allocates them across different
    initialization strategies using logarithmic scaling that ensures all available
    templates are sampled while maintaining diversity.

    For "smart" mode, the allocation uses logarithmic scaling:
    - Templates: Allocation scales logarithmically with number of templates
      (base_pct * log(1 + n_templates * prefactor)). Guarantees all templates
      are sampled at least once (if n_structures >= n_templates), capped at 2 per template.
    - Seed+growth: Allocation scales logarithmically with number of seed combinations
      (base_pct * log(1 + n_combinations * prefactor)) if seeds are available, otherwise 0.
    - Random_spherical: Fills remaining slots as fallback.

    The logarithmic scaling ensures gradual increases in allocation as more options
    become available, prioritizing structured initialization (templates/seeds) over
    random placement when many options are available.

    The allocation pattern is reproducible: for a given seed and n_structures,
    the same distribution of strategies will be used.

    Args:
        composition: Target list of element symbols.
        n_structures: Number of structures to generate.
        rng: Numpy Generator for randomness (used for actual structure generation).
        placement_radius_scaling: Scale factor for radii in random placement.
        min_distance_factor: Scale factor for minimum distance checks.
        vacuum: Extra padding for the generated simulation cell.
        previous_search_glob: Glob pattern to find database files.
        mode: Base mode for initialization:
            - "smart": Uses Metropolis allocation based on available templates/seeds
            - "template": All structures use template strategy
            - "seed+growth": All structures use seed+growth strategy
            - "random_spherical": All structures use random_spherical strategy
        connectivity_factor: Factor to multiply sum of covalent radii for
            connectivity threshold.
        n_jobs: Number of parallel workers. Default: 1 (sequential execution).
            Use >1 for parallel execution. Special values:
            - -1: Use all available CPU cores
            - -2: Use all available CPU cores except one
            Parallel execution maintains reproducibility through deterministic seeds.

    Returns:
        List of Atoms objects, one for each requested structure.

    Raises:
        ValueError: If n_structures < 1 or invalid mode/parameters.
    """
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
    # Handle different modes
    if mode == "smart":
        # Discover available strategies and use Metropolis allocation
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

        # Allocate strategies using Metropolis algorithm
        allocations = _allocate_strategies_metropolis(
            n_structures=n_structures,
            templates=discovery["templates"],
            n_seed_formulas=discovery["n_seed_formulas"],
            n_seed_combinations=discovery["n_seed_combinations"],
            rng=rng,
            n_atoms=n_atoms,
        )
        discovery_templates = discovery["templates"]
    elif mode == "template":
        allocations = [("template", None)] * n_structures
    elif mode == "seed+growth":
        allocations = [("seed+growth", None)] * n_structures
    elif mode == "random_spherical":
        allocations = [("random_spherical", None)] * n_structures
    else:
        raise ValueError(
            f'Unsupported mode: "{mode}". '
            f'Supported modes: "smart", "template", "seed+growth", "random_spherical"'
        )

    batch_base_seed = rng.integers(0, 2**31)
    structure_assignments = []
    for i, (strategy, template_index) in enumerate(allocations):
        # Create deterministic seed for this structure
        # Use a hash-like combination to ensure different structures get different seeds
        structure_seed = (batch_base_seed + i * 7919) % (2**31)  # 7919 is a prime
        structure_assignments.append((i, strategy, template_index, structure_seed))

    def _generate_structure_with_strategy(
        comp: list[str],
        strat: str,
        structure_rng: np.random.Generator,
        template_index: int | None = None,
    ) -> tuple[Atoms, str, str | None]:
        """Generate a single structure using a specific strategy.

        This is the core implementation that both create_initial_cluster() and
        create_initial_cluster_batch() use to avoid circular dependencies.
        """
        cell_side_local = compute_cell_side(comp, vacuum=vacuum)
        n_atoms_local = len(comp)

        # Define strategy execution functions
        def _run_template_strategy() -> Atoms | None:
            if n_atoms_local == 2:
                return None
            return _try_template_generation(
                composition=comp,
                n_atoms=n_atoms_local,
                cell_side=cell_side_local,
                rng=structure_rng,
                placement_radius_scaling=placement_radius_scaling,
                min_distance_factor=min_distance_factor,
                connectivity_factor=connectivity_factor,
                template_index=template_index,
                discovery_templates=discovery_templates,
            )

        def _run_seed_growth_strategy() -> Atoms | None:
            if n_atoms_local <= 2:
                return None
            return _try_seed_growth(
                composition=comp,
                cell_side=cell_side_local,
                rng=structure_rng,
                placement_radius_scaling=placement_radius_scaling,
                min_distance_factor=min_distance_factor,
                connectivity_factor=connectivity_factor,
                candidates_by_formula=precomputed_candidates_by_formula,
                valid_combinations=valid_seed_combinations,
            )

        def _run_random_spherical_strategy() -> Atoms:
            return random_spherical(
                composition=comp,
                cell_side=cell_side_local,
                rng=structure_rng,
                placement_radius_scaling=placement_radius_scaling,
                min_distance_factor=min_distance_factor,
                connectivity_factor=connectivity_factor,
            )

        # Strategy dispatch table
        _STRATEGIES = {
            "template": _run_template_strategy,
            "seed+growth": _run_seed_growth_strategy,
            "random_spherical": _run_random_spherical_strategy,
        }

        # Determine strategy sequence based on mode
        if strat == InitStrategy.RANDOM_SPHERICAL.value:
            # Explicit random mode: only use random placement
            atoms = _run_random_spherical_strategy()
            validated_atoms, _, _ = validate_cluster(
                atoms,
                composition=comp,
                min_distance_factor=min_distance_factor,
                connectivity_factor=connectivity_factor,
                sort_atoms=False,
                raise_on_failure=True,
                source=InitStrategy.RANDOM_SPHERICAL.value,
            )
            return validated_atoms, InitStrategy.RANDOM_SPHERICAL.value, None
        elif strat == InitStrategy.TEMPLATE.value:
            # Explicit template mode: try templates, fallback to random
            strategy_sequence = [
                InitStrategy.TEMPLATE.value,
                InitStrategy.RANDOM_SPHERICAL.value,
            ]
        elif strat == InitStrategy.SEED_GROWTH.value:
            # Explicit seed+growth mode: try seed+growth, fallback to random
            strategy_sequence = [
                InitStrategy.SEED_GROWTH.value,
                InitStrategy.RANDOM_SPHERICAL.value,
            ]
        else:
            raise ValueError(
                f'Unknown strategy: "{strat}". '
                f'Valid strategies are: "{InitStrategy.TEMPLATE.value}", "{InitStrategy.SEED_GROWTH.value}", "{InitStrategy.RANDOM_SPHERICAL.value}"'
            )

        # Execute strategies in sequence
        strategy_functions = [(name, _STRATEGIES[name]) for name in strategy_sequence]
        return _try_strategies_in_order(
            strategies=strategy_functions,
            composition=comp,
            connectivity_factor=connectivity_factor,
            min_distance_factor=min_distance_factor,
            return_strategy=True,
        )

    # Helper function to generate a single structure
    # Captures all parameters via closure for parallel execution
    def _generate_structure(
        assignment: tuple[int, str, int | None, int],
    ) -> tuple[int, Atoms, str, str | None]:
        idx, strategy, template_index, structure_seed = assignment
        structure_rng = np.random.default_rng(structure_seed)
        atoms, used_strategy, fallback_from = _generate_structure_with_strategy(
            comp=composition,
            strat=strategy,
            structure_rng=structure_rng,
            template_index=template_index,
        )
        return (idx, atoms, used_strategy, fallback_from)

    max_workers = resolve_n_jobs_to_workers(n_jobs)
    # No point creating more workers than tasks
    max_workers = min(max_workers, n_structures)

    # Generate structures (parallel or sequential)
    results: list[Atoms | None] = [None] * n_structures
    failed_indices: list[int] = []
    fallback_info: dict[int, tuple[str, str | None]] = {}

    if max_workers == 1:
        # Sequential execution
        for assignment in structure_assignments:
            idx, atoms, used_strategy, fallback_from = _generate_structure(assignment)
            if atoms is not None:
                results[idx] = atoms
                fallback_info[idx] = (used_strategy, fallback_from)
            else:
                failed_indices.append(idx)
    else:
        # Parallel execution using ThreadPoolExecutor
        try:
            with ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="scgo_init_batch",
            ) as executor:
                # Submit all tasks
                future_to_index = {
                    executor.submit(_generate_structure, assignment): assignment[0]
                    for assignment in structure_assignments
                }

                # Collect results as they complete, but fill by index to maintain order
                for future in as_completed(future_to_index):
                    try:
                        idx, atoms, used_strategy, fallback_from = future.result()
                        if atoms is not None:
                            results[idx] = atoms
                            fallback_info[idx] = (used_strategy, fallback_from)
                        else:
                            failed_indices.append(idx)
                    except KeyboardInterrupt:
                        # Attempt to cancel remaining futures and re-raise
                        for f in future_to_index:
                            f.cancel()
                        raise
                    except (RuntimeError, ValueError) as e:
                        # Treat expected generation errors as per-row failures
                        idx = future_to_index.get(future)
                        logger.exception(
                            "Structure generation raised an exception: %s",
                            type(e).__name__,
                        )
                        if idx is not None:
                            failed_indices.append(idx)
        except KeyboardInterrupt:
            logger.warning(
                "Structure generation interrupted by user (KeyboardInterrupt)"
            )
            raise

    # Log generation failures
    if failed_indices:
        logger.warning(
            f"Failed to generate structures at indices: {failed_indices}. "
            f"Generated {sum(1 for r in results if r is not None)}/{n_structures} structures"
        )

    if logger.isEnabledFor(logging.DEBUG):
        template_to_random = sum(
            1
            for used, fallback in fallback_info.values()
            if used == InitStrategy.RANDOM_SPHERICAL.value
            and fallback == InitStrategy.TEMPLATE.value
        )
        seed_to_random = sum(
            1
            for used, fallback in fallback_info.values()
            if used == InitStrategy.RANDOM_SPHERICAL.value
            and fallback == InitStrategy.SEED_GROWTH.value
        )
        logger.debug(
            "Fallbacks to random_spherical: template=%d, seed+growth=%d",
            template_to_random,
            seed_to_random,
        )

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
