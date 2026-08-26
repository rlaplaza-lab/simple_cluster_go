"""Database discovery service for SCGO.

Centralizes finding and indexing database files across run directories.
Path lists are not memoized: GO may write a new ``run_*/ga_go.db`` after an
earlier same-process discovery (e.g. loading previous results), and TS must
see that file on reload.
"""

from __future__ import annotations

import glob
import os
import sqlite3
from pathlib import Path

from scgo.database.connection import get_connection
from scgo.database.registry import get_registry
from scgo.database.streaming import relaxed_rows_where_clause
from scgo.metadata.db_stamp import is_scgo_db
from scgo.metadata.run_dir import load_run_dir_record, resolve_run_id_from_db_path
from scgo.utils.helpers import get_cluster_formula, get_composition_counts
from scgo.utils.logging import get_logger

logger = get_logger(__name__)

_discovery_by_base: dict[str, DatabaseDiscovery] = {}


def _filter_scgo_databases(db_files: list[Path]) -> list[Path]:
    """Keep only databases marked as SCGO."""
    return [p for p in db_files if is_scgo_db(p)]


class DatabaseDiscovery:
    """Service for discovering and indexing database files."""

    def __init__(self, base_dir: str | Path):
        """Initialize database discovery.

        Args:
            base_dir: Base directory to search (usually output directory)
        """
        self.base_dir = Path(base_dir)
        self._registry = get_registry(self.base_dir)
        logger.debug("Initialized DatabaseDiscovery for %s", self.base_dir)

    def find_databases(
        self,
        composition: list[str] | None = None,
        run_id: str | None = None,
        db_filename: str = "*.db",
        use_cache: bool = True,
    ) -> list[Path]:
        """Find databases matching criteria.

        ``use_cache`` is accepted for API compatibility but ignored. Registry
        hits are always merged with a ``run_*/`` filesystem scan so a DB on
        disk that is not yet registered is still found.
        """
        _ = use_cache
        by_resolved: dict[str, Path] = {}

        if db_filename == "*.db":
            registry_files = self._registry.find_databases(
                composition=composition,
                run_id=run_id,
            )
            logger.debug("Registry found %d databases", len(registry_files))
            filtered = _filter_scgo_databases(registry_files)
            if len(filtered) != len(registry_files):
                logger.debug(
                    "Dropped %d non-SCGO paths from registry results",
                    len(registry_files) - len(filtered),
                )
            for path in filtered:
                by_resolved[str(path.resolve())] = path

        pattern = self._build_glob_pattern(run_id, db_filename)
        full_pattern = str(self.base_dir / pattern)
        glob_files = [Path(p) for p in glob.glob(full_pattern, recursive=True)]

        logger.debug(
            "Found %d databases matching pattern: %s", len(glob_files), pattern
        )

        if composition:
            glob_files = self._filter_by_composition(glob_files, composition)
            logger.debug(
                "After composition filter: %d databases remain", len(glob_files)
            )

        orig_count = len(glob_files)
        glob_files = _filter_scgo_databases(glob_files)
        if len(glob_files) != orig_count:
            logger.debug(
                "Filtered non-SCGO DBs: %d -> %d databases", orig_count, len(glob_files)
            )

        for path in glob_files:
            by_resolved[str(path.resolve())] = path

        return list(by_resolved.values())

    def _build_glob_pattern(
        self,
        run_id: str | None,
        db_filename: str,
    ) -> str:
        """Build glob pattern for database search."""
        if run_id:
            return f"{run_id}/{db_filename}"
        return f"run_*/{db_filename}"

    def _get_first_relaxed_candidate(self, db) -> object | None:
        """Get one relaxed candidate via SQL."""
        where_sql = relaxed_rows_where_clause()
        try:
            with db.c.managed_connection() as conn:
                cur = conn.execute(
                    f"SELECT id FROM systems WHERE {where_sql} ORDER BY id ASC LIMIT 1"
                )
                row = cur.fetchone()
            rowid = row[0] if row else None
            if rowid is None:
                return None
            return db.get_atoms(rowid)
        except (AttributeError, sqlite3.DatabaseError, TypeError, ValueError) as e:
            logger.debug("Failed relaxed-candidate probe: %s", e)
            return None

    def _filter_by_composition(
        self,
        db_files: list[Path],
        composition: list[str],
    ) -> list[Path]:
        """Filter database files by composition."""
        target_counts = get_composition_counts(composition)
        target_formula = get_cluster_formula(composition)
        filtered = []
        run_formula_cache: dict[str, str | None] = {}

        for db_path in db_files:
            try:
                run_id = resolve_run_id_from_db_path(
                    str(db_path), base_dir=str(self.base_dir)
                )
                if run_id:
                    if run_id not in run_formula_cache:
                        record = load_run_dir_record(str(self.base_dir / run_id))
                        run_formula_cache[run_id] = record.formula if record else None
                    known_formula = run_formula_cache[run_id]
                    if known_formula is not None and known_formula == target_formula:
                        filtered.append(db_path)
                        continue
                    # Metadata formula is often mobile-only (e.g. ``Pt5``) while
                    # TS loads slab+mobile (``C150Pt5``). Fall through to the
                    # atom composition probe instead of rejecting the DB.
                with get_connection(db_path) as db:
                    first_candidate = self._get_first_relaxed_candidate(db)

                    if not first_candidate:
                        continue

                    symbols = first_candidate.get_chemical_symbols()
                    cand_counts = get_composition_counts(symbols)

                    if cand_counts == target_counts:
                        filtered.append(db_path)

            except (
                sqlite3.DatabaseError,
                OSError,
                ValueError,
                KeyError,
                AttributeError,
            ) as e:
                logger.debug("Error checking composition for %s: %s", db_path, e)
                continue

        return filtered


def _get_discovery(base_dir: str | Path) -> DatabaseDiscovery:
    """Return a cached :class:`~scgo.database.discovery.DatabaseDiscovery` for *base_dir*."""
    key = os.path.abspath(str(base_dir))
    if key not in _discovery_by_base:
        _discovery_by_base[key] = DatabaseDiscovery(key)
    return _discovery_by_base[key]


def clear_discovery_cache(base_dir: str | Path | None = None) -> None:
    """Drop process-wide database discovery instances for *base_dir* (or all).

    Args:
        base_dir: When set, clear only that resolved path; otherwise clear all.
    """
    if base_dir is None:
        _discovery_by_base.clear()
        return
    _discovery_by_base.pop(os.path.abspath(str(base_dir)), None)


def list_discovered_db_paths_with_run(
    base_dir: str | Path,
    *,
    composition: list[str] | None = None,
    use_cache: bool = True,
    db_filename: str | None = None,
) -> list[tuple[str, str | None]]:
    """List DB paths via :class:`~scgo.database.discovery.DatabaseDiscovery` with run parsed from layout.

    Returns tuples ``(absolute_path, run_id)``. ``run_id`` is ``None`` if the path
    is not under a recognizable ``run_*`` directory.

    ``use_cache`` is accepted for API compatibility; discovery always rescans.
    """
    _ = use_cache
    base_s = os.path.abspath(str(base_dir))
    discovery = _get_discovery(base_s)
    filename_pattern = db_filename if db_filename else "*.db"
    db_paths = discovery.find_databases(
        composition=composition,
        db_filename=filename_pattern,
    )

    out: list[tuple[str, str | None]] = []
    for db_path in db_paths:
        db_path_str = os.path.abspath(str(db_path))
        run_id = resolve_run_id_from_db_path(db_path_str, base_dir=base_s)
        if not run_id:
            logger.warning(
                "Could not resolve run_id for database %s under %s",
                db_path_str,
                base_s,
            )
        out.append((db_path_str, run_id))
    return out
