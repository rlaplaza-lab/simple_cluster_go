"""Database candidate discovery for cluster initialization.

This module handles scanning database files for suitable seed candidates
based on composition subsets and structural validation.
"""

from __future__ import annotations

import glob
import os
import re
import sqlite3
import threading
from collections import Counter

import numpy as np
from ase import Atoms

from scgo.database import get_connection as db_connection
from scgo.database.cache import get_global_cache
from scgo.database.metadata import get_metadata as _get_db_metadata
from scgo.utils.helpers import (
    get_cluster_formula,
    get_composition_counts,
)
from scgo.utils.logging import get_logger

from .initialization_config import (
    _COMPOSITION_CACHE_NS,
    _FIND_SMALLER_CANDIDATES_CACHE_VERSION,
    _MAX_CANDIDATES_PER_FORMULA,
)

logger = get_logger(__name__)

CandidateEntry = tuple[tuple[str, ...], float, Atoms]

# Lock to protect mtime caching for database files
_DB_MTIME_LOCK = threading.Lock()

# Cache of canonical mtimes for database files to ensure stable cache keys
_DB_CANONICAL_MTIME: dict[str, float] = {}


def is_composition_subset(
    subset_counts: Counter[str] | dict[str, int],
    target_counts: Counter[str] | dict[str, int],
) -> bool:
    """Check if subset_counts is a subset of target_counts."""
    return all(
        subset_counts.get(el, 0) <= target_counts.get(el, 0) for el in subset_counts
    )


def _safe_mtime(path: str) -> float:
    """Return file mtime or 0.0 on error. Used for cache signature without loading."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _load_candidates_from_file(db_file: str, mtime: float) -> list[CandidateEntry]:
    """Load relaxed candidates from a single database file with mtime caching."""
    try:
        with db_connection(db_file) as da:
            # Note: We load all relaxed candidates; the caller handles filtering
            # for final_unique_minimum if needed.
            candidates = da.get_all_relaxed_candidates()
            results: list[CandidateEntry] = []
            for atoms in candidates:
                symbols = tuple(atoms.get_chemical_symbols())
                # Match logic from extract_energy_from_atoms
                energy = _get_db_metadata(atoms, "raw_score", None)
                if energy is None:
                    energy = _get_db_metadata(atoms, "potential_energy", 0.0)

                # If energy is still something like None or non-finite, we might want to skip,
                # but for consistency with previous implementation:
                results.append((symbols, energy, atoms))
            return results
    except (sqlite3.DatabaseError, sqlite3.OperationalError, OSError, ValueError) as e:
        logger.debug(f"Failed to load candidates from {db_file}: {e}")
        return []


def invalidate_db_canonical_mtime(db_file: str | None = None) -> None:
    """Invalidate the canonical mtime cache for one or all database files."""
    with _DB_MTIME_LOCK:
        if db_file is None:
            _DB_CANONICAL_MTIME.clear()
        else:
            _DB_CANONICAL_MTIME.pop(db_file, None)


def _load_db_candidates(db_file: str) -> tuple[float, list[CandidateEntry]]:
    """Load candidates from database with mtime-based caching."""
    mtime = _safe_mtime(db_file)

    with _DB_MTIME_LOCK:
        if db_file in _DB_CANONICAL_MTIME:
            canonical_mtime = _DB_CANONICAL_MTIME[db_file]
        else:
            canonical_mtime = mtime
            _DB_CANONICAL_MTIME[db_file] = mtime

    cache_ns = "db_candidates"
    cache_key = (db_file, canonical_mtime)

    # Cache candidates per file to avoid re-reading large DBs
    cached = get_global_cache().get(cache_ns, cache_key)
    if cached is not None:
        return canonical_mtime, cached

    candidates = _load_candidates_from_file(db_file, canonical_mtime)
    get_global_cache().set(cache_ns, cache_key, candidates)
    return canonical_mtime, candidates


def _parse_composition_from_path(path: str) -> list[str] | None:
    """Parse chemical composition from a directory path (e.g., 'Pt3Au2_searches')."""
    parts = path.split(os.sep)
    for part in parts:
        if "_searches" in part:
            comp_str = part.replace("_searches", "")
            pattern = r"([A-Z][a-z]?)(\d*)"
            matches = re.findall(pattern, comp_str)
            if matches:
                composition = []
                for symbol, count_str in matches:
                    count = int(count_str) if count_str else 1
                    composition.extend([symbol] * count)
                return composition
    return None


def _could_path_contain_relevant_candidates(
    path: str, target_counts: Counter[str]
) -> bool:
    """Check if a path might contain candidates that are subsets of target."""
    path_comp = _parse_composition_from_path(path)
    if path_comp is None:
        return True  # Cannot determine, assume it might

    path_counts = get_composition_counts(path_comp)
    return is_composition_subset(path_counts, target_counts)


def _compute_files_signature(files: list[str]) -> tuple[tuple[str, float], ...]:
    """Compute a signature for a list of files based on their mtimes."""
    return tuple(sorted((f, _safe_mtime(f)) for f in files))


def get_structure_signature(atoms: Atoms, precision: int = 4) -> tuple[float, ...]:
    """Create a signature based on sorted interatomic distances."""
    positions = atoms.get_positions()
    distances = [
        np.linalg.norm(positions[i] - positions[j])
        for i in range(len(positions))
        for j in range(i + 1, len(positions))
    ]
    return tuple(np.round(np.sort(distances), precision))


def deduplicate_seed_candidates(
    entries: list[tuple[float, Atoms]],
    precision: int = 4,
    energy_bin: float | None = None,
) -> list[tuple[float, Atoms]]:
    """Deduplicate seed candidates by geometry signature."""
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
            signature = get_structure_signature(atoms, precision=precision)
            if signature not in unique:
                unique[signature] = (energy, atoms)
        return list(unique.values())

    binned: dict[int, list[tuple[float, Atoms]]] = {}
    for energy, atoms in entries:
        energy_key = int(round(energy / energy_bin))
        binned.setdefault(energy_key, []).append((energy, atoms))

    deduped: list[tuple[float, Atoms]] = []
    for bucket in binned.values():
        unique_bucket: dict[tuple[float, ...], tuple[float, Atoms]] = {}
        for energy, atoms in bucket:
            signature = get_structure_signature(atoms, precision=precision)
            if signature not in unique_bucket:
                unique_bucket[signature] = (energy, atoms)
        deduped.extend(unique_bucket.values())

    return deduped


def _find_smaller_candidates(
    target_composition: list[str],
    db_glob_pattern: str,
) -> dict[str, list[tuple[float, Atoms]]]:
    """Find all relaxed database candidates that are sub-compositions of target."""
    cwd = os.getcwd()
    matches = glob.glob(os.path.join(cwd, db_glob_pattern), recursive=True)
    candidates_by_formula: dict[str, list[tuple[float, Atoms]]] = {}
    target_counts = get_composition_counts(target_composition)
    n_target_atoms = len(target_composition)

    filtered_matches = [
        db_file
        for db_file in matches
        if _could_path_contain_relevant_candidates(db_file, target_counts)
    ]

    signature_tuple = _compute_files_signature(filtered_matches)
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
            return {
                formula: [(energy, atom.copy()) for energy, atom in entries]
                for formula, entries in cached_entry.items()
            }

    for db_file in filtered_matches:
        try:
            _mtime, entries = _load_db_candidates(db_file)
        except (sqlite3.Error, OSError, RuntimeError):
            continue

        for symbols, energy, atoms in entries:
            if not _get_db_metadata(atoms, "final_unique_minimum", False):
                continue

            if len(symbols) >= n_target_atoms:
                continue

            row_counts = get_composition_counts(list(symbols))
            if not is_composition_subset(row_counts, target_counts):
                continue

            formula = get_cluster_formula(list(symbols))
            if formula not in candidates_by_formula:
                candidates_by_formula[formula] = []
            candidates_by_formula[formula].append((energy, atoms))

    processed: dict[str, list[tuple[float, Atoms]]] = {}
    for formula, entries in candidates_by_formula.items():
        sorted_entries = sorted(entries, key=lambda e: e[0])
        deduped_entries = deduplicate_seed_candidates(sorted_entries)

        if len(deduped_entries) > _MAX_CANDIDATES_PER_FORMULA:
            deduped_entries = deduped_entries[:_MAX_CANDIDATES_PER_FORMULA]
        processed[formula] = deduped_entries

    get_global_cache().set(_COMPOSITION_CACHE_NS, cache_key, processed)

    return {
        formula: [(energy, atom.copy()) for energy, atom in entries]
        for formula, entries in processed.items()
    }
