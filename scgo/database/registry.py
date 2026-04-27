"""Database registry for fast database lookups.

Simplified in-memory registry for database discovery without filesystem scanning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scgo.utils.helpers import get_composition_counts
from scgo.utils.logging import get_logger

logger = get_logger(__name__)


class DatabaseRegistry:
    """In-memory registry of databases for quick discovery."""

    def __init__(self, base_dir: str | Path):
        """Initialize registry.

        Args:
            base_dir: Base directory (e.g., "output")
        """
        self.base_dir = Path(base_dir).resolve()
        self._data: dict[str, Any] = {"version": "1.0", "databases": {}}

    def register_database(
        self,
        db_path: Path,
        composition: list[str] | None = None,
        run_id: str | None = None,
        trial_id: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Register a database in the index.

        Args:
            db_path: Path to database file
            composition: Composition (e.g., ["Pt", "Pt"])
            run_id: Run identifier
            trial_id: Trial number
            metadata: Additional metadata to store
        """
        # Build database entry
        db_path_resolved = db_path.resolve()
        base_dir_resolved = self.base_dir.resolve()

        try:
            db_key = str(db_path_resolved.relative_to(base_dir_resolved))
        except ValueError as e:
            logger.warning(
                f"Cannot register database: path mismatch between "
                f"db_path={db_path_resolved} and base_dir={base_dir_resolved}: {e}"
            )
            return

        entry = {
            "path": db_key,
            "absolute_path": str(db_path_resolved),
            "composition": composition or [],
            "composition_str": self._make_composition_key(composition or []),
            "run_id": run_id,
            "trial_id": trial_id,
            "metadata": metadata or {},
        }

        self._data["databases"][db_key] = entry
        logger.debug("Registered database: %s", db_key)

    def unregister_database(self, db_path: Path) -> bool:
        """Remove database from registry.

        Args:
            db_path: Path to database file

        Returns:
            True if database was in registry and removed
        """
        try:
            db_key = str(db_path.resolve().relative_to(self.base_dir.resolve()))
        except ValueError:
            return False

        if db_key in self._data["databases"]:
            del self._data["databases"][db_key]
            logger.debug("Unregistered database: %s", db_key)
            return True
        return False

    def find_databases(
        self,
        composition: list[str] | None = None,
        run_id: str | None = None,
        trial_id: int | None = None,
    ) -> list[Path]:
        """Find databases matching criteria.

        Args:
            composition: Filter by composition
            run_id: Filter by run ID
            trial_id: Filter by trial number

        Returns:
            List of matching database paths
        """
        matches = []
        comp_key = self._make_composition_key(composition) if composition else None

        for entry in self._data["databases"].values():
            if comp_key and entry["composition_str"] != comp_key:
                continue

            if run_id and entry["run_id"] != run_id:
                continue

            if trial_id is not None and entry["trial_id"] != trial_id:
                continue

            db_path = self.base_dir / entry["path"]
            if db_path.exists():
                matches.append(db_path)

        return matches

    def get_all_databases(self) -> list[Path]:
        """Get all registered databases.

        Returns:
            List of all database paths
        """
        paths = []
        for entry in self._data["databases"].values():
            db_path = self.base_dir / entry["path"]
            if db_path.exists():
                paths.append(db_path)
        return paths

    def get_database_entry(self, db_path: Path) -> dict | None:
        """Get registry entry for a database.

        Args:
            db_path: Path to database

        Returns:
            Registry entry dict or None if not found
        """
        try:
            db_key = str(db_path.resolve().relative_to(self.base_dir.resolve()))
        except ValueError:
            return None
        return self._data["databases"].get(db_key)

    def clear(self) -> None:
        """Clear all registry entries."""
        self._data["databases"] = {}
        logger.info("Cleared registry")

    def rebuild_from_filesystem(
        self,
        pattern: str = "**/*.db",
    ) -> int:
        """Rebuild registry by scanning filesystem.

        Args:
            pattern: Glob pattern for database files

        Returns:
            Number of databases registered
        """
        db_files = list(self.base_dir.glob(pattern))
        logger.info("Scanning %s database files...", len(db_files))

        registered = 0
        for db_path in db_files:
            try:
                # Simple registration without metadata detection
                self.register_database(db_path)
                registered += 1
            except (ValueError, OSError) as e:
                # Skip files that cannot be registered (filesystem issues)
                logger.warning("Failed to register %s: %s", db_path, e)

        logger.info("Registered %s databases", registered)
        return registered

    def invalidate_stale_entries(self) -> int:
        """Remove entries for databases that no longer exist.

        Returns:
            Number of entries removed
        """
        stale_keys = []
        for db_key, entry in self._data["databases"].items():
            db_path = self.base_dir / entry["path"]
            if not db_path.exists():
                stale_keys.append(db_key)

        for key in stale_keys:
            del self._data["databases"][key]

        if stale_keys:
            logger.info("Removed %s stale entries", len(stale_keys))

        return len(stale_keys)

    @staticmethod
    def _make_composition_key(composition: list[str]) -> str:
        """Make canonical composition key for indexing.

        Args:
            composition: List of element symbols

        Returns:
            Canonical composition string (e.g., "Pt2" or "PdPt")
        """
        if not composition:
            return ""

        counts = get_composition_counts(composition)
        # Sort by element symbol for canonical form
        sorted_elements = sorted(counts.keys())
        parts = [f"{elem}{counts[elem]}" for elem in sorted_elements]
        return "".join(parts)


# Global registry instance cache
_global_registries: dict[Path, DatabaseRegistry] = {}


def get_registry(base_dir: str | Path) -> DatabaseRegistry:
    """Get or create a registry for a base directory.

    Args:
        base_dir: Base directory for the registry

    Returns:
        DatabaseRegistry instance (cached)

    Example:
        >>> registry = get_registry("output")
        >>> db_files = registry.find_databases(composition=["Pt", "Pt"])
    """
    base_path = Path(base_dir).resolve()

    if base_path not in _global_registries:
        _global_registries[base_path] = DatabaseRegistry(base_path)
    return _global_registries[base_path]


def clear_registry_cache() -> None:
    """Clear the global registry cache."""
    _global_registries.clear()
