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

from scgo.database.cache import get_global_cache
from scgo.database.helpers import extract_minima_from_database_file
from scgo.metadata.atoms import get_tag as _get_db_tag
from scgo.metadata.run_dir import resolve_run_id_from_db_path
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


def _load_candidates_from_file(db_file: str) -> list[CandidateEntry]:
    """Load minima from a single database file.

    Args:
        db_file: Path to the database file to read.

    Returns:
        List of ``(symbols, energy, atoms)`` entries, or an empty list if the
        file could not be read. Each atoms object is stamped with provenance
        tags (``scgo_source_db`` / ``scgo_source_run_id``) so reused seeds are
        traceable back to the originating database.
    """
    try:
        run_id = resolve_run_id_from_db_path(db_file)
        minima = extract_minima_from_database_file(
            db_file,
            run_id or "",
            require_final=False,
        )
        results: list[CandidateEntry] = []
        for energy, atoms in minima:
            symbols = tuple(atoms.get_chemical_symbols())
            if db_file:
                atoms.info.setdefault("key_value_pairs", {})["scgo_source_db"] = db_file
            if run_id is not None:
                atoms.info.setdefault("key_value_pairs", {})["scgo_source_run_id"] = (
                    run_id
                )
            results.append((symbols, energy, atoms))
        return results
    except (sqlite3.DatabaseError, OSError, ValueError) as e:
        logger.debug("Failed to load candidates from %s: %s", db_file, e)
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

    candidates = _load_candidates_from_file(db_file)
    get_global_cache().set(cache_ns, cache_key, candidates)
    return canonical_mtime, candidates


def _symbols_from_formula_segment(segment: str) -> list[str] | None:
    """Parse one chemical-formula segment, or None if not a pure formula token."""
    if not segment or not segment[0].isupper():
        return None
    pattern = r"([A-Z][a-z]?)(\d*)"
    matches = list(re.finditer(pattern, segment))
    if not matches:
        return None
    covered = "".join(m.group(0) for m in matches)
    if covered != segment:
        return None
    composition: list[str] = []
    for match in matches:
        symbol = match.group(1)
        count_str = match.group(2)
        count = int(count_str) if count_str else 1
        composition.extend([symbol] * count)
    return composition


def _parse_composition_from_path(path: str) -> list[str] | None:
    """Parse mobile composition from a ``*_searches`` directory path.

    Accepts component path keys (``Pt5_OH_OH_graphite_searches``) and pure
    cluster formulas (``Pt5_searches``, ``Au2Pt3_searches``). Non-formula tokens
    such as surface names (``graphite``, ``slab``) are skipped.
    """
    parts = path.split(os.sep)
    for part in parts:
        if not part.endswith("_searches"):
            continue
        comp_str = part[: -len("_searches")]
        if not comp_str:
            continue
        composition: list[str] = []
        parsed_any = False
        for segment in comp_str.split("_"):
            symbols = _symbols_from_formula_segment(segment)
            if symbols is None:
                continue
            composition.extend(symbols)
            parsed_any = True
        if parsed_any:
            return composition
    return None


def _path_relevance_status(
    path: str,
    target_counts: Counter[str],
) -> tuple[bool, bool]:
    """Return (is_relevant, is_parseable) for candidate discovery path filtering."""
    path_comp = _parse_composition_from_path(path)
    if path_comp is None:
        return False, False

    path_counts = get_composition_counts(path_comp)
    return is_composition_subset(path_counts, target_counts), True


def _compute_files_signature(files: list[str]) -> tuple[tuple[str, float], ...]:
    """Compute a signature for a list of files based on their mtimes."""
    return tuple(sorted((f, _safe_mtime(f)) for f in files))


def get_structure_signature(atoms: Atoms, precision: int = 4) -> tuple[float, ...]:
    """Create a signature based on sorted interatomic distances."""
    from scipy.spatial.distance import pdist

    positions = atoms.get_positions()
    if len(positions) <= 1:
        return ()
    distances = pdist(positions)
    return tuple(np.round(np.sort(distances), precision))


def deduplicate_seed_candidates(
    entries: list[tuple[float, Atoms]],
    precision: int = 4,
    energy_bin: float | None = None,
) -> list[tuple[float, Atoms]]:
    """Deduplicate seed candidates by geometry signature.

    Candidates are first grouped into energy bins (width ``energy_bin``) and
    then reduced to one entry per interatomic-distance signature within each
    bin. When ``energy_bin`` is ``None`` it defaults to one hundredth of the
    energy range; a bin width of zero deduplicates by signature only.

    Args:
        entries: List of ``(energy, atoms)`` candidates.
        precision: Decimal places used when rounding the distance signature.
        energy_bin: Optional energy bin width; ``None`` derives it from the
            energy range and non-positive values disable binning.

    Returns:
        Deduplicated list of ``(energy, atoms)`` candidates (unordered).
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


def _postprocess_candidate_bucket(
    candidates_by_formula: dict[str, list[tuple[float, Atoms]]],
) -> dict[str, list[tuple[float, Atoms]]]:
    """Sort, deduplicate and cap a formula->candidates bucket.

    Mirrors the ordering used by the historical ``_find_smaller_candidates``:
    candidates are grouped by formula, sorted by energy, deduplicated by
    geometry signature (energy-binned), re-sorted by energy, and truncated to
    ``_MAX_CANDIDATES_PER_FORMULA`` entries per formula.
    """
    processed: dict[str, list[tuple[float, Atoms]]] = {}
    for formula, entries in candidates_by_formula.items():
        sorted_entries = sorted(entries, key=lambda e: e[0])
        deduped_entries = deduplicate_seed_candidates(sorted_entries)

        # Deduplication groups by energy bin and geometry signature, so the
        # surviving entries come back in an arbitrary (dict/bucket) order.
        # Re-sort by energy so truncation keeps the lowest-energy candidates.
        deduped_entries = sorted(deduped_entries, key=lambda e: e[0])

        if len(deduped_entries) > _MAX_CANDIDATES_PER_FORMULA:
            deduped_entries = deduped_entries[:_MAX_CANDIDATES_PER_FORMULA]
        processed[formula] = deduped_entries
    return processed


def _copy_candidate_buckets(
    buckets: tuple[
        dict[str, list[tuple[float, Atoms]]],
        dict[str, list[tuple[float, Atoms]]],
    ],
) -> tuple[
    dict[str, list[tuple[float, Atoms]]],
    dict[str, list[tuple[float, Atoms]]],
]:
    """Deep-copy the atoms in a cached ``(sub, exact)`` tuple for callers."""
    sub, exact = buckets
    return (
        {
            formula: [(energy, atom.copy()) for energy, atom in entries]
            for formula, entries in sub.items()
        },
        {
            formula: [(energy, atom.copy()) for energy, atom in entries]
            for formula, entries in exact.items()
        },
    )


def _discover_all_candidates(
    target_composition: list[str],
    db_glob_pattern: str,
) -> tuple[
    dict[str, list[tuple[float, Atoms]]],
    dict[str, list[tuple[float, Atoms]]],
]:
    """Scan DB candidates once and split into sub- and exact-match buckets.

    The single glob scan + mtime cache is reused for both tiers. Entries must
    be tagged ``final_unique_minimum``. Two buckets are produced:

    - ``sub``: strict sub-compositions with strictly fewer atoms than the
      target (the historical seed tier).
    - ``exact``: full composition matches (identical element counts) with the
      same number of atoms as the target.

    Both buckets are grouped by formula, sorted by energy, deduplicated by
    geometry, and truncated to ``_MAX_CANDIDATES_PER_FORMULA`` entries per
    formula.

    Args:
        target_composition: Target composition as a list of element symbols.
        db_glob_pattern: Glob pattern (relative to the current working
            directory) used to locate database files.

    Returns:
        Tuple of ``(sub_dict, exact_dict)``. Atoms objects are copied so callers
        can mutate them safely.
    """
    cwd = os.getcwd()
    matches = glob.glob(os.path.join(cwd, db_glob_pattern), recursive=True)
    sub_by_formula: dict[str, list[tuple[float, Atoms]]] = {}
    exact_by_formula: dict[str, list[tuple[float, Atoms]]] = {}
    target_counts = get_composition_counts(target_composition)
    n_target_atoms = len(target_composition)

    skipped_unparseable = 0
    filtered_matches: list[str] = []
    for db_file in matches:
        is_relevant, is_parseable = _path_relevance_status(db_file, target_counts)
        if not is_parseable:
            skipped_unparseable += 1
            logger.debug(
                "Cannot parse composition from path %s; skipping candidate discovery scan",
                db_file,
            )
            continue
        if is_relevant:
            filtered_matches.append(db_file)

    if matches:
        logger.info(
            "Candidate discovery: %d DB path(s) matched glob, "
            "%d relevant after composition filter, "
            "%d skipped (unparseable *_searches path)",
            len(matches),
            len(filtered_matches),
            skipped_unparseable,
        )

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
        for bucket in cached_entry:
            for entries in bucket.values():
                for _energy, atom in entries:
                    if not _get_db_tag(atom, "final_unique_minimum", False):
                        stale_cache = True
                        break
                if stale_cache:
                    break
            if stale_cache:
                break

        if not stale_cache:
            return _copy_candidate_buckets(cached_entry)

    for db_file in filtered_matches:
        try:
            _mtime, entries = _load_db_candidates(db_file)
        except (sqlite3.Error, OSError, RuntimeError) as e:
            logger.debug(
                "Failed to load candidates from %s during discovery scan: %s: %s",
                db_file,
                type(e).__name__,
                e,
            )
            continue

        for symbols, energy, atoms in entries:
            if not _get_db_tag(atoms, "final_unique_minimum", False):
                continue

            n_symbols = len(symbols)
            row_counts = get_composition_counts(list(symbols))

            if n_symbols < n_target_atoms and is_composition_subset(
                row_counts, target_counts
            ):
                formula = get_cluster_formula(list(symbols))
                sub_by_formula.setdefault(formula, []).append((energy, atoms))
            elif n_symbols == n_target_atoms and dict(row_counts) == dict(
                target_counts
            ):
                formula = get_cluster_formula(list(symbols))
                exact_by_formula.setdefault(formula, []).append((energy, atoms))
            # Larger-than-target candidates are excluded (unchanged behaviour).

    sub = _postprocess_candidate_bucket(sub_by_formula)
    exact = _postprocess_candidate_bucket(exact_by_formula)

    get_global_cache().set(_COMPOSITION_CACHE_NS, cache_key, (sub, exact))

    return _copy_candidate_buckets((sub, exact))


def _find_smaller_candidates(
    target_composition: list[str],
    db_glob_pattern: str,
) -> dict[str, list[tuple[float, Atoms]]]:
    """Find all relaxed database candidates that are sub-compositions of target.

    Only entries tagged as ``final_unique_minimum`` and holding strictly fewer
    atoms than the target are kept. Results are grouped by cluster formula,
    sorted by energy, deduplicated by geometry, and truncated to
    ``_MAX_CANDIDATES_PER_FORMULA`` entries per formula.

    This is the historical seed tier; the exact-match tier is exposed separately
    via :func:`_find_exact_candidates`.

    Args:
        target_composition: Target composition as a list of element symbols.
        db_glob_pattern: Glob pattern (relative to the current working
            directory) used to locate database files.

    Returns:
        Mapping of cluster formula to ``(energy, atoms)`` candidates, with the
        atoms objects copied so callers can mutate them safely.
    """
    sub, _exact = _discover_all_candidates(target_composition, db_glob_pattern)
    return sub


def _find_exact_candidates(
    target_composition: list[str],
    db_glob_pattern: str,
) -> dict[str, list[tuple[float, Atoms]]]:
    """Find relaxed database candidates with the exact target composition.

    Only entries tagged as ``final_unique_minimum`` whose element counts match
    the target exactly (and have the same atom count) are kept. Results are
    grouped by cluster formula, sorted by energy, deduplicated by geometry, and
    truncated to ``_MAX_CANDIDATES_PER_FORMULA`` entries per formula.

    Args:
        target_composition: Target composition as a list of element symbols.
        db_glob_pattern: Glob pattern (relative to the current working
            directory) used to locate database files.

    Returns:
        Mapping of cluster formula to ``(energy, atoms)`` candidates, with the
        atoms objects copied so callers can mutate them safely.
    """
    _sub, exact = _discover_all_candidates(target_composition, db_glob_pattern)
    return exact
