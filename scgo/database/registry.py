"""Database registry for fast database lookups.

Simplified in-memory registry for database discovery: lookups never scan the
filesystem, they only stat the registered paths.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from scgo.utils.helpers import get_composition_counts
from scgo.utils.logging import get_logger

logger = get_logger(__name__)

# Maximum number of registry instances retained; oldest is evicted on insertion
# beyond this cap to bound memory in long-running, multi-directory sessions.
_REGISTRY_MAX_SIZE = 16


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
    ) -> None:
        """Register a database in the index.

        Databases outside ``base_dir`` cannot be keyed relative to it, so they
        are skipped with a warning instead of being registered.

        Args:
            db_path: Path to database file
            composition: Composition (e.g., ["Pt", "Pt"])
            run_id: Run identifier
        """
        # Build database entry
        db_path_resolved = db_path.resolve()
        base_dir_resolved = self.base_dir.resolve()

        try:
            db_key = str(db_path_resolved.relative_to(base_dir_resolved))
        except ValueError as e:
            logger.warning(
                "Cannot register database: path mismatch between db_path=%s and "
                "base_dir=%s: %s",
                db_path_resolved,
                base_dir_resolved,
                e,
            )
            return

        entry = {
            "path": db_key,
            "absolute_path": str(db_path_resolved),
            "composition": composition or [],
            "composition_str": self._make_composition_key(composition or []),
            "run_id": run_id,
        }

        self._data["databases"][db_key] = entry
        logger.debug("Registered database: %s", db_key)

    def find_databases(
        self,
        composition: list[str] | None = None,
        run_id: str | None = None,
    ) -> list[Path]:
        """Find databases matching criteria.

        Args:
            composition: Filter by composition
            run_id: Filter by run ID

        Returns:
            List of matching database paths, skipping entries whose file no
            longer exists on disk
        """
        matches = []
        comp_key = self._make_composition_key(composition) if composition else None

        for entry in self._data["databases"].values():
            if comp_key and entry["composition_str"] != comp_key:
                continue

            if run_id and entry["run_id"] != run_id:
                continue

            db_path = self.base_dir / entry["path"]
            if db_path.exists():
                matches.append(db_path)

        return matches

    def get_all_databases(self) -> list[Path]:
        """Get all registered databases.

        Returns:
            List of registered database paths that still exist on disk
        """
        paths = []
        for entry in self._data["databases"].values():
            db_path = self.base_dir / entry["path"]
            if db_path.exists():
                paths.append(db_path)
        return paths

    def clear(self) -> None:
        """Clear all registry entries."""
        self._data["databases"] = {}
        logger.info("Cleared registry")

    @staticmethod
    def _make_composition_key(composition: list[str]) -> str:
        """Make canonical composition key for indexing.

        Args:
            composition: List of element symbols

        Returns:
            Canonical composition string with explicit counts
            (e.g., "Pt2" or "Pd1Pt1")
        """
        if not composition:
            return ""

        counts = get_composition_counts(composition)
        # Sort by element symbol for canonical form
        sorted_elements = sorted(counts.keys())
        parts = [f"{elem}{counts[elem]}" for elem in sorted_elements]
        return "".join(parts)


# Global registry instance cache
_global_registries: OrderedDict[Path, DatabaseRegistry] = OrderedDict()
_global_registries_lock = threading.Lock()


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

    with _global_registries_lock:
        if base_path in _global_registries:
            # Mark as most-recently-used.
            _global_registries.move_to_end(base_path)
            return _global_registries[base_path]

        registry = DatabaseRegistry(base_path)
        _global_registries[base_path] = registry
        _global_registries.move_to_end(base_path)
        if len(_global_registries) > _REGISTRY_MAX_SIZE:
            _global_registries.popitem(last=False)
        return registry


def clear_registry() -> None:
    """Discard all cached registry instances (useful for tests/sessions)."""
    with _global_registries_lock:
        _global_registries.clear()
