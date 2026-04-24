"""Database discovery service for SCGO.

Centralizes logic for finding and indexing database files across
run directories with caching for performance.
"""

from __future__ import annotations

import contextlib
import glob
import os
import sqlite3
from pathlib import Path

from scgo.database.connection import open_db
from scgo.database.constants import SYSTEMS_JSON_COLUMN
from scgo.database.registry import DatabaseRegistry
from scgo.database.schema import is_scgo_database
from scgo.utils.helpers import get_composition_counts
from scgo.utils.logging import get_logger

logger = get_logger(__name__)


def _filter_scgo_databases(db_files: list[Path]) -> list[Path]:
    """Keep only databases marked as SCGO."""
    return [p for p in db_files if is_scgo_database(p)]


class DatabaseDiscovery:
    """Service for discovering and indexing database files.

    Provides centralized database finding with caching to avoid
    repeated filesystem scans.

    Example:
        >>> discovery = DatabaseDiscovery("output")
        >>>
        >>> # Find all databases for Pt3
        >>> db_files = discovery.find_databases(composition=["Pt", "Pt", "Pt"])
        >>>
        >>> # Find databases for specific run
        >>> db_files = discovery.find_databases(run_id="run_20260204_120000")
        >>>
        >>> # Clear cache when filesystem changes
        >>> discovery.clear_cache()
    """

    def __init__(self, base_dir: str | Path):
        """Initialize database discovery.

        Args:
            base_dir: Base directory to search (usually output directory)
        """
        self.base_dir = Path(base_dir)
        self._cache: dict[str, list[Path]] = {}
        self._metadata_cache: dict[Path, dict] = {}

        # Persistent registry for fast lookups (fail-fast on invalid registries)
        self._registry = DatabaseRegistry(self.base_dir)
        logger.debug("Using registry for fast database discovery")
        logger.debug(f"Initialized DatabaseDiscovery for {self.base_dir}")

    def find_databases(
        self,
        composition: list[str] | None = None,
        run_id: str | None = None,
        trial_id: int | None = None,
        db_filename: str = "*.db",
        use_cache: bool = True,
    ) -> list[Path]:
        """Find databases matching criteria.

        Args:
            composition: Filter by composition (e.g., ["Pt", "Pt", "Pt"])
            run_id: Filter by specific run (e.g., "run_20260204_120000")
            trial_id: Filter by specific trial number
            db_filename: Database filename pattern (default "*.db")
            use_cache: Whether to use cached results (default True)

        Returns:
            List of Path objects for matching databases

        Example:
            >>> # Find all Pt3 databases
            >>> db_files = discovery.find_databases(composition=["Pt"]*3)
            >>>
            >>> # Find specific run and trial
            >>> db_files = discovery.find_databases(
            ...     run_id="run_20260204_120000",
            ...     trial_id=1,
            ...     db_filename="ga_go.db"
            ... )
        """
        # Build cache key
        cache_key = self._build_cache_key(composition, run_id, trial_id, db_filename)

        # Check cache
        if use_cache and cache_key in self._cache:
            logger.debug("Using cached results for: %s", cache_key)
            return self._cache[cache_key]

        if db_filename == "*.db":
            db_files = self._registry.find_databases(
                composition=composition,
                run_id=run_id,
                trial_id=trial_id,
            )
            logger.debug("Registry found %d databases", len(db_files))

            if db_files:
                filtered = _filter_scgo_databases(db_files)
                if len(filtered) != len(db_files):
                    logger.debug(
                        "Dropped %d non-SCGO paths from registry results",
                        len(db_files) - len(filtered),
                    )
                db_files = filtered
                if db_files:
                    if use_cache:
                        self._cache[cache_key] = db_files
                    return db_files

            logger.debug("Registry returned no databases; running filesystem scan")

        # Build glob pattern
        pattern = self._build_glob_pattern(run_id, trial_id, db_filename)

        # Find matching files
        full_pattern = str(self.base_dir / pattern)
        db_files = [Path(p) for p in glob.glob(full_pattern, recursive=True)]

        logger.debug("Found %d databases matching pattern: %s", len(db_files), pattern)

        # Filter by composition if specified
        if composition:
            db_files = self._filter_by_composition(db_files, composition)
            logger.debug("After composition filter: %d databases remain", len(db_files))

        orig_count = len(db_files)
        db_files = _filter_scgo_databases(db_files)
        if len(db_files) != orig_count:
            logger.debug(
                "Filtered non-SCGO DBs: %d -> %d databases", orig_count, len(db_files)
            )

        # Cache results
        if use_cache:
            self._cache[cache_key] = db_files

        return db_files

    def get_database_info(self, db_path: Path) -> dict:
        """Get metadata about a database.

        Args:
            db_path: Path to database file

        Returns:
            dict: Database metadata (run_id, trial_id, count, etc.)
        """
        # Check cache
        if db_path in self._metadata_cache:
            return self._metadata_cache[db_path]

        info = {
            "path": str(db_path),
            "exists": db_path.exists(),
            "size_mb": 0,
            "run_id": None,
            "trial_id": None,
            "structure_count": 0,
        }

        if not db_path.exists():
            self._metadata_cache[db_path] = info
            return info

        # Get file size
        info["size_mb"] = db_path.stat().st_size / (1024 * 1024)

        # Parse run_id and trial_id from path
        parts = db_path.parts
        for _part in parts:
            if _part.startswith("run_"):
                info["run_id"] = _part
            if _part.startswith("trial_"):
                with contextlib.suppress(IndexError, ValueError):
                    trial_num = int(_part.split("_")[1])
                    info["trial_id"] = trial_num

        # Count structures
        try:
            with open_db(db_path) as db:
                candidates = db.get_all_relaxed_candidates()
                info["structure_count"] = len(candidates)
        except (
            sqlite3.DatabaseError,
            sqlite3.OperationalError,
            OSError,
            ValueError,
        ) as e:
            logger.debug("Failed to count structures in %s: %s", db_path, e)

        # Cache result
        self._metadata_cache[db_path] = info
        return info

    def get_statistics(self) -> dict:
        """Get statistics about discovered databases.

        Returns:
            dict: Statistics (total_databases, total_structures, etc.)
        """
        all_dbs = self.find_databases(use_cache=False)

        stats = {
            "total_databases": len(all_dbs),
            "total_size_mb": 0,
            "total_structures": 0,
            "by_run": {},
            "by_composition": {},
        }

        for db_path in all_dbs:
            info = self.get_database_info(db_path)
            stats["total_size_mb"] += info["size_mb"]
            stats["total_structures"] += info["structure_count"]

            # Count by run
            run_id = info.get("run_id")
            if run_id:
                stats["by_run"][run_id] = stats["by_run"].get(run_id, 0) + 1

        return stats

    def clear_cache(self) -> None:
        """Clear all caches.

        Call this if filesystem has changed (new runs, databases added/removed).
        """
        self._cache.clear()
        self._metadata_cache.clear()
        logger.debug("Cleared database discovery caches")

    def _build_cache_key(
        self,
        composition: list[str] | None,
        run_id: str | None,
        trial_id: int | None,
        db_filename: str,
    ) -> str:
        """Build unique cache key from parameters."""
        comp_str = "-".join(sorted(composition)) if composition else "any"
        run_str = run_id or "any"
        trial_str = str(trial_id) if trial_id is not None else "any"
        return f"{comp_str}:{run_str}:{trial_str}:{db_filename}"

    def _build_glob_pattern(
        self,
        run_id: str | None,
        trial_id: int | None,
        db_filename: str,
    ) -> str:
        """Build glob pattern for database search."""
        if run_id and trial_id is not None:
            return f"{run_id}/trial_{trial_id}/{db_filename}"
        if run_id:
            return f"{run_id}/**/{db_filename}"
        return f"run_*/**/{db_filename}"

    def _get_first_relaxed_candidate(self, db) -> object | None:
        """Get one relaxed candidate via SQL (``json_extract`` on the systems JSON column)."""
        try:
            with db.c.managed_connection() as conn:
                cur = conn.execute(
                    f"SELECT id FROM systems WHERE json_extract({SYSTEMS_JSON_COLUMN}, '$.relaxed') = 1 "
                    "ORDER BY id ASC LIMIT 1"
                )
                row = cur.fetchone()
            rowid = row[0] if row else None
            if rowid is None:
                return None
            return db.get_atoms(rowid)
        except (
            AttributeError,
            sqlite3.DatabaseError,
            sqlite3.OperationalError,
            TypeError,
            ValueError,
        ) as e:
            logger.debug("Failed relaxed-candidate probe: %s", e)
            return None

    def _filter_by_composition(
        self,
        db_files: list[Path],
        composition: list[str],
    ) -> list[Path]:
        """Filter database files by composition.

        Checks if database contains structures with matching composition.
        """
        target_counts = get_composition_counts(composition)
        filtered = []

        for db_path in db_files:
            try:
                with open_db(db_path) as db:
                    first_candidate = self._get_first_relaxed_candidate(db)

                    if not first_candidate:
                        continue

                    # Check first candidate's composition
                    symbols = first_candidate.get_chemical_symbols()
                    cand_counts = get_composition_counts(symbols)

                    if cand_counts == target_counts:
                        filtered.append(db_path)

            except (
                sqlite3.DatabaseError,
                sqlite3.OperationalError,
                OSError,
                ValueError,
                KeyError,
                AttributeError,
            ) as e:
                logger.debug("Error checking composition for %s: %s", db_path, e)
                continue

        return filtered


def find_databases_simple(
    base_dir: str | Path,
    db_pattern: str = "**/*.db",
    composition: list[str] | None = None,
) -> list[Path]:
    """Database discovery with cache disabled (``use_cache=False``).

    Only the last path segment of ``db_pattern`` is used as ``db_filename`` (e.g.
    ``*.db`` from ``"**/*.db"``). The search follows :meth:`DatabaseDiscovery.find_databases`
    layout rules from ``base_dir``, not a raw recursive ``**`` glob from ``db_pattern``.

    Args:
        base_dir: Directory to search
        db_pattern: Whose last component is the filename pattern (default ``"**/*.db"``)
        composition: Optional composition filter

    Returns:
        List of matching database paths

    Example:
        >>> db_files = find_databases_simple("output", "**/*ga_go.db")
    """
    discovery = DatabaseDiscovery(base_dir)
    pattern_parts = db_pattern.split("/")
    db_filename = pattern_parts[-1] if pattern_parts else "*.db"

    return discovery.find_databases(
        composition=composition, db_filename=db_filename, use_cache=False
    )


def list_discovered_db_paths_with_run_trial(
    base_dir: str | Path,
    *,
    composition: list[str] | None = None,
    use_cache: bool = True,
) -> list[tuple[str, str, int | None]]:
    """List DB paths via :class:`DatabaseDiscovery` with run/trial parsed from layout.

    Returns tuples ``(absolute_path, run_id, trial_id)``. ``run_id`` is empty if
    the path is not under ``run_*``; ``trial_id`` is None if not under ``trial_*``.
    """
    base_s = os.path.abspath(str(base_dir))
    discovery = DatabaseDiscovery(base_s)
    out: list[tuple[str, str, int | None]] = []
    for db_path in discovery.find_databases(
        composition=composition, use_cache=use_cache
    ):
        db_path_str = os.path.abspath(str(db_path))
        rel = os.path.relpath(db_path_str, base_s)
        parts = rel.split(os.sep)
        run_id = parts[0] if parts and parts[0].startswith("run_") else ""
        trial_id = None
        if len(parts) >= 2 and parts[1].startswith("trial_"):
            with contextlib.suppress(ValueError, IndexError):
                trial_id = int(parts[1].split("_")[1])
        out.append((db_path_str, run_id, trial_id))
    return out
