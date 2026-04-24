"""Database registry for fast O(1) database lookups.

Maintains a persistent index of all databases for fast discovery
without filesystem scanning.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

from scgo.database.connection import open_db
from scgo.database.constants import SYSTEMS_JSON_COLUMN
from scgo.utils.helpers import get_composition_counts
from scgo.utils.logging import get_logger

logger = get_logger(__name__)

REGISTRY_LOCK_FILENAME = ".scgo_db_registry.lock"


@contextlib.contextmanager
def _registry_process_lock(base_dir: Path, *, exclusive: bool):
    """Serialize registry updates across processes (fcntl flock on Linux HPC)."""
    base_dir.mkdir(parents=True, exist_ok=True)
    lock_path = base_dir / REGISTRY_LOCK_FILENAME
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        flag = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(fd, flag)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


class DatabaseRegistry:
    """Persistent JSON registry of databases for quick discovery."""

    REGISTRY_FILENAME = ".scgo_db_registry.json"

    @staticmethod
    def _safe_file_size_mb(path: Path) -> float:
        """Return file size in MiB, falling back to 0 on filesystem races/errors."""
        try:
            return path.stat().st_size / (1024 * 1024)
        except OSError:
            return 0.0

    def __init__(self, base_dir: str | Path):
        """Initialize registry.

        Args:
            base_dir: Base directory (e.g., "output")
        """
        self.base_dir = Path(base_dir).resolve()
        self.registry_path = self.base_dir / self.REGISTRY_FILENAME
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {"version": "1.0", "databases": {}}
        self._load()

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
        # Open SQLite outside the cross-process lock (avoid holding flock during I/O).
        if not run_id or not composition:
            detected = self._detect_metadata(db_path)
            run_id = run_id or detected.get("run_id")
            composition = composition or detected.get("composition")

        if trial_id is None:
            for part in db_path.resolve().parts:
                if part.startswith("trial_"):
                    with contextlib.suppress(ValueError, IndexError):
                        trial_id = int(part.split("_", 1)[1])
                        break

        with self._lock, _registry_process_lock(self.base_dir, exclusive=True):
            self._load()

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
                "size_mb": self._safe_file_size_mb(db_path),
                "metadata": metadata or {},
            }

            self._data["databases"][db_key] = entry
            self._save()

            logger.debug("Registered database: %s", db_key)

    def unregister_database(self, db_path: Path) -> bool:
        """Remove database from registry.

        Args:
            db_path: Path to database file

        Returns:
            True if database was in registry and removed
        """
        with self._lock, _registry_process_lock(self.base_dir, exclusive=True):
            self._load()
            try:
                db_key = str(db_path.resolve().relative_to(self.base_dir.resolve()))
            except ValueError:
                return False

            if db_key in self._data["databases"]:
                del self._data["databases"][db_key]
                self._save()
                logger.debug("Unregistered database: %s", db_key)
                return True
            return False

    def find_databases(
        self,
        composition: list[str] | None = None,
        run_id: str | None = None,
        trial_id: int | None = None,
    ) -> list[Path]:
        """Find databases matching criteria (O(n) but faster than filesystem scan).

        Args:
            composition: Filter by composition
            run_id: Filter by run ID
            trial_id: Filter by trial number

        Returns:
            List of matching database paths
        """
        with self._lock, _registry_process_lock(self.base_dir, exclusive=False):
            self._load()
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
                else:
                    logger.debug(
                        "Stale registry entry: %s (file not found)", entry["path"]
                    )

            return matches

    def get_all_databases(self) -> list[Path]:
        """Get all registered databases.

        Returns:
            List of all database paths
        """
        with self._lock, _registry_process_lock(self.base_dir, exclusive=False):
            self._load()
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
        with self._lock, _registry_process_lock(self.base_dir, exclusive=False):
            self._load()
            try:
                db_key = str(db_path.resolve().relative_to(self.base_dir.resolve()))
            except ValueError:
                return None
            return self._data["databases"].get(db_key)

    def rebuild_from_filesystem(
        self,
        pattern: str = "**/*.db",
        force: bool = False,
    ) -> int:
        """Rebuild registry by scanning filesystem.

        Args:
            pattern: Glob pattern for database files
            force: If True, clear existing registry first

        Returns:
            Number of databases registered
        """
        if force:
            with self._lock, _registry_process_lock(self.base_dir, exclusive=True):
                self._load()
                self._data["databases"] = {}
                self._save()

        db_files = list(self.base_dir.glob(pattern))
        logger.info("Scanning %s database files...", len(db_files))

        registered = 0
        for db_path in db_files:
            try:
                self.register_database(db_path)
                registered += 1
            except (ValueError, OSError, sqlite3.DatabaseError) as e:
                # Skip files that cannot be registered (filesystem/DB issues)
                logger.warning("Failed to register %s: %s", db_path, e)

        logger.info("Registered %s databases", registered)
        return registered

    def invalidate_stale_entries(self) -> int:
        """Remove entries for databases that no longer exist.

        Returns:
            Number of entries removed
        """
        with self._lock, _registry_process_lock(self.base_dir, exclusive=True):
            self._load()
            stale_keys = []
            for db_key, entry in self._data["databases"].items():
                db_path = self.base_dir / entry["path"]
                if not db_path.exists():
                    stale_keys.append(db_key)

            for key in stale_keys:
                del self._data["databases"][key]

            if stale_keys:
                self._save()
                logger.info("Removed %s stale entries", len(stale_keys))

            return len(stale_keys)

    def get_statistics(self) -> dict:
        """Get registry statistics.

        Returns:
            Statistics dict with counts, sizes, etc.
        """
        with self._lock, _registry_process_lock(self.base_dir, exclusive=False):
            self._load()
            stats = {
                "total_databases": len(self._data["databases"]),
                "total_size_mb": 0.0,
                "by_run": {},
                "by_composition": {},
                "stale_entries": 0,
            }

            for entry in self._data["databases"].values():
                db_path = self.base_dir / entry["path"]

                if not db_path.exists():
                    stats["stale_entries"] += 1
                    continue

                stats["total_size_mb"] += entry.get("size_mb", 0)

                run_id = entry.get("run_id")
                if run_id:
                    stats["by_run"][run_id] = stats["by_run"].get(run_id, 0) + 1

                comp_str = entry.get("composition_str", "unknown")
                stats["by_composition"][comp_str] = (
                    stats["by_composition"].get(comp_str, 0) + 1
                )

            return stats

    def clear(self) -> None:
        """Clear all registry entries."""
        with self._lock, _registry_process_lock(self.base_dir, exclusive=True):
            self._load()
            self._data["databases"] = {}
            self._save()
            logger.info("Cleared registry")

    def _load(self) -> None:
        """Load registry from disk."""
        if not self.registry_path.exists():
            logger.debug("No registry file at %s, starting fresh", self.registry_path)
            return

        try:
            with open(self.registry_path) as f:
                self._data = json.load(f)
            logger.debug(
                "Loaded registry with %s entries", len(self._data["databases"])
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load registry: %s, starting fresh", e)
            self._data = {"version": "1.0", "databases": {}}

    def _save(self) -> None:
        """Save registry to disk (atomic write).

        Writes to a temporary file and atomically replaces the registry file to
        avoid partial/corrupt registry state if multiple processes write
        concurrently or a write is interrupted.
        """
        tmp_path = self.registry_path.with_suffix(".tmp")
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "w") as f:
                json.dump(self._data, f, indent=2)
            # Atomic replace on POSIX filesystems
            os.replace(tmp_path, self.registry_path)
            logger.debug("Saved registry with %s entries", len(self._data["databases"]))
        except (OSError, TypeError) as e:
            logger.error("Failed to save registry: %s", e)
            if tmp_path.exists():
                with contextlib.suppress(OSError):
                    tmp_path.unlink()

    def _detect_metadata(self, db_path: Path) -> dict:
        """Auto-detect metadata from database path and content.

        Args:
            db_path: Path to database file

        Returns:
            Dict with detected run_id, composition, etc.
        """
        metadata: dict[str, Any] = {}

        # Parse from path
        parts = db_path.parts
        for part in parts:
            if part.startswith("run_"):
                metadata["run_id"] = part
            if part.startswith("trial_"):
                try:
                    trial_num = int(part.split("_")[1])
                    metadata["trial_id"] = trial_num
                except (IndexError, ValueError):
                    pass

        # Try to detect composition from first readable structure (strict mode).
        # Under strict/default policy we surface DB/ASE errors instead of
        # silently swallowing them so callers can correct invalid DBs.
        if self._safe_file_size_mb(db_path) > 0:
            # Prefer a 'relaxed' systems row; fall back to any row. Use a
            # direct SQL read for the row id to avoid calling fragile ASE
            # helper methods that vary across ASE versions.
            with open_db(db_path) as db:
                conn = getattr(db.c, "connection", None)

                def _fetch_rowid_from_conn(_conn, relaxed_first: bool = True):
                    if relaxed_first:
                        # Prefer relaxed rows via the systems JSON column (JSON1).
                        cur = _conn.execute(
                            f"SELECT id FROM systems WHERE json_extract({SYSTEMS_JSON_COLUMN}, '$.relaxed') = 1 ORDER BY id ASC LIMIT 1"
                        )
                        r = cur.fetchone()
                        if r:
                            return r[0]

                    cur = _conn.execute(
                        "SELECT id FROM systems ORDER BY id ASC LIMIT 1"
                    )
                    r = cur.fetchone()
                    return r[0] if r else None

                rowid = None
                if conn is not None:
                    rowid = _fetch_rowid_from_conn(conn, relaxed_first=True)
                else:
                    with db.c.managed_connection() as _conn:
                        rowid = _fetch_rowid_from_conn(_conn, relaxed_first=True)

                if rowid is not None:
                    atoms_obj = db.get_atoms(id=rowid)
                    symbols = atoms_obj.get_chemical_symbols()
                    metadata["composition"] = symbols

        return metadata

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
_registry_lock = threading.Lock()


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

    with _registry_lock:
        if base_path not in _global_registries:
            _global_registries[base_path] = DatabaseRegistry(base_path)
        return _global_registries[base_path]


def clear_registry_cache() -> None:
    """Clear the global registry cache."""
    with _registry_lock:
        _global_registries.clear()
